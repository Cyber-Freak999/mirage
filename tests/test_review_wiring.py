"""Tests for review-gate wiring: drift ids, scheduler sweep, API decisions."""

import sqlite3

import numpy as np
import pytest
from flask import Flask

import app.retrain.drift as drift_mod
import app.retrain.review_gate as review_mod
from app.retrain.drift import DriftMonitor


@pytest.fixture
def isolated_dbs(tmp_path, monkeypatch):
    """Redirect drift/review DBs to tmp paths."""
    monkeypatch.setattr(drift_mod, "DRIFT_DB", tmp_path / "drift.db")
    monkeypatch.setattr(review_mod, "REVIEW_DB", tmp_path / "review.db")
    return tmp_path


def test_drift_result_carries_id(isolated_dbs) -> None:
    """check_drift on shifted data returns a result whose id matches drift_checks."""
    rng = np.random.RandomState(42)
    X_ref = rng.normal(size=(300, 25)).astype(float)
    names = [f"feat_{i}" for i in range(25)]

    monitor = DriftMonitor(top_k_features=25, min_samples=50)
    monitor.set_reference(X_ref, names, {n: 0.04 for n in names})

    X_cur = rng.normal(size=(300, 25)).astype(float)
    X_cur[:, 4] = rng.normal(loc=10.0, scale=1.0, size=300)
    result = monitor.check_drift(X_cur, feature_names=names)

    assert result.triggered
    assert result.id is not None
    conn = sqlite3.connect(str(isolated_dbs / "drift.db"))
    row = conn.execute("SELECT max_psi FROM drift_checks WHERE id = ?", (result.id,)).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == pytest.approx(result.max_psi)


def test_sweep_processes_approved_review(isolated_dbs, tmp_path, monkeypatch) -> None:
    """An approved review is picked up by the sweep and retrained."""
    import app.api as api_mod
    import app.honeypot as honeypot_mod
    from app.retrain.champion_challenger import ValidationResult
    from app.retrain.review_gate import ReviewGate
    from app.retrain.scheduler import RetrainingConfig, RetrainingScheduler

    monkeypatch.setattr(honeypot_mod, "DB_PATH", tmp_path / "honeypot.db")
    honeypot_mod.init_db()

    gate = ReviewGate()
    review = gate.create_review(
        drift_result_id=7, max_psi=0.5, psi_scores={"a": 0.5}, sample_size=10, sample_data={"count": 10}
    )
    assert gate.approve(review.id, "tester") is True

    calls: dict = {}

    def fake_retrain(X_new, y_new, **kwargs):
        calls["n"] = len(X_new)
        calls["kwargs"] = kwargs
        return None, ValidationResult(
            timestamp=0.0,
            challenger_version="v-test",
            champion_version="v-old",
            challenger_metrics={},
            champion_metrics={},
            promoted=False,
            reason="test",
        )

    monkeypatch.setattr("app.retrain.scheduler.run_full_retrain", fake_retrain)
    monkeypatch.setattr(api_mod, "reload_model", lambda: None)

    scheduler = RetrainingScheduler(RetrainingConfig())
    processed = scheduler.sweep_reviews()

    assert processed == [review.id]
    assert gate.get_review(review.id).status.value == "done"
    assert gate.should_proceed(review.id) is False
    # Second sweep must not retrain again.
    assert scheduler.sweep_reviews() == []
    assert calls["n"] == 0  # honeypot DB empty; retrain ran on zero new rows


def test_api_approve_reject_roundtrip(isolated_dbs, monkeypatch) -> None:
    """Approve/reject endpoints: 200 once, 409 on re-decide, 404 unknown."""
    from app.api import bp
    from app.retrain.review_gate import ReviewGate

    monkeypatch.setenv("MIRAGE_API_KEY", "test-key")
    headers = {"X-API-Key": "test-key"}

    gate = ReviewGate()
    review = gate.create_review(drift_result_id=3, max_psi=0.4, psi_scores={}, sample_size=5, sample_data={})

    app = Flask(__name__)
    app.register_blueprint(bp)
    client = app.test_client()

    resp = client.post(f"/api/reviews/{review.id}/approve", json={"reviewer": "tester"}, headers=headers)
    assert resp.status_code == 200

    resp = client.post(f"/api/reviews/{review.id}/approve", json={"reviewer": "tester"}, headers=headers)
    assert resp.status_code == 409

    resp = client.post("/api/reviews/99999/reject", json={"reviewer": "tester"}, headers=headers)
    assert resp.status_code == 404
