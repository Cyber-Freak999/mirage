"""End-to-end adaptive loop test plus drift/gate unit tests (spec section 16)."""

import sqlite3

import numpy as np
import pytest
from flask import Flask

from app.retrain.drift import compute_psi


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    """Redirect every loop artifact (DBs, models, reference, validation) to tmp."""
    import app.honeypot as honeypot_mod
    import app.model.ensemble as ensemble_mod
    import app.retrain.champion_challenger as cc_mod
    import app.retrain.drift as drift_mod
    import app.retrain.review_gate as review_mod

    models = tmp_path / "models"
    models.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    monkeypatch.setattr(honeypot_mod, "DB_PATH", tmp_path / "honeypot.db")
    monkeypatch.setattr(ensemble_mod, "MODEL_DIR", models)
    monkeypatch.setattr(cc_mod, "VALIDATION_DB", tmp_path / "validation.db")
    monkeypatch.setattr(cc_mod, "VALIDATION_SET_PATH", tmp_path / "validation_set.npz")
    monkeypatch.setattr(drift_mod, "DRIFT_DB", tmp_path / "drift.db")
    monkeypatch.setattr(drift_mod, "REFERENCE_DIR", reference)
    monkeypatch.setattr(review_mod, "REVIEW_DB", tmp_path / "review.db")
    return tmp_path


def _seed_reference_and_validation(tmp_path) -> None:
    """Persist a synthetic drift reference and held-out validation set."""
    from app.retrain.drift import DriftMonitor
    from app.schema.features import FEATURE_NAMES

    rng = np.random.RandomState(42)
    X_ref = rng.normal(size=(300, 25)).astype(float)
    DriftMonitor(top_k_features=8, min_samples=50).set_reference(
        X_ref, FEATURE_NAMES, dict.fromkeys(FEATURE_NAMES, 0.04)
    )
    X_val = rng.normal(size=(60, 25)).astype(float)
    y_val = np.array([0, 1] * 30)
    np.savez(tmp_path / "validation_set.npz", X=X_val, y=y_val)


def _seed_captures(n: int = 250) -> None:
    """Insert a burst of heavy-payload attack captures (far from reference)."""
    import app.honeypot as honeypot_mod

    honeypot_mod.init_db()
    conn = sqlite3.connect(str(honeypot_mod.DB_PATH))
    rows = [
        (
            1_700_000_000.0 + i,
            "10.0.0.9",
            "POST",
            "/login",
            "",
            "evil-bot/1.0",
            "username=" + "x" * 5000,
            "sqli",
            0,
            '{"user-agent": "evil-bot/1.0", "content-type": "application/x-www-form-urlencoded"}',
            "application/x-www-form-urlencoded",
        )
        for i in range(n)
    ]
    conn.executemany(
        "INSERT INTO requests (timestamp, source_ip, method, path, query_string,"
        " user_agent, raw_request, attack_type, decoy_indicator, headers_json, content_type)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _fake_public(monkeypatch) -> None:
    """Serve small balanced synthetic public data to run_full_retrain."""
    import app.retrain.champion_challenger as cc_mod

    rng = np.random.RandomState(7)
    X_public = rng.normal(size=(120, 25)).astype(np.float32)
    X_public[60:, 10] += 5.0  # attacks carry heavier payloads
    y_public = np.array([0] * 60 + [1] * 60)

    def fake_load(cicids_dir=None, unsw_dir=None, *args, **kwargs):
        return X_public, y_public

    monkeypatch.setattr(cc_mod, "load_combined_datasets", fake_load)


def test_loop_end_to_end(loop_env, tmp_path, monkeypatch) -> None:
    """Burst -> drift -> review -> approve -> retrain -> promote -> reload."""
    import app.api as api_mod
    from app.api import bp
    from app.model.ensemble import load_model
    from app.retrain.scheduler import RetrainingConfig, RetrainingScheduler

    _seed_reference_and_validation(tmp_path)
    _seed_captures()
    _fake_public(monkeypatch)
    (tmp_path / "cicids").mkdir()
    (tmp_path / "unsw").mkdir()

    scheduler = RetrainingScheduler(
        RetrainingConfig(
            min_samples=200,
            k_folds=2,
            cicids_dir=tmp_path / "cicids",
            unsw_dir=tmp_path / "unsw",
            cicids_sample=1.0,
            unsw_sample=1.0,
        )
    )

    # 1. Capture burst triggers drift + creates a linked review.
    drift_result = scheduler.check_drift_and_queue()
    assert drift_result is not None and drift_result.triggered
    assert drift_result.id is not None
    pending = scheduler.review_gate.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0].drift_result_id == drift_result.id
    review_id = pending[0].id

    # 2. API approval.
    app = Flask(__name__)
    app.register_blueprint(bp)
    client = app.test_client()
    resp = client.post(f"/api/reviews/{review_id}/approve", json={"reviewer": "e2e"})
    assert resp.status_code == 200

    # 3. Sweep retrains through the gate.
    monkeypatch.setattr(api_mod, "reload_model", lambda: load_model("champion"))
    processed = scheduler.sweep_reviews()
    assert processed == [review_id]

    # 4. Promotion persisted and serving the new version.
    champion = load_model("champion")
    assert champion.version is not None
    assert (tmp_path / "models" / "champion.json").exists()
    reloaded = api_mod.reload_model()
    assert reloaded.version.version_id == champion.version.version_id

    # 5. Review consumed; second sweep is a no-op.
    assert scheduler.sweep_reviews() == []


def test_compute_psi_quiet_on_identical() -> None:
    """PSI is ~0 for identical distributions."""
    rng = np.random.RandomState(42)
    ref = rng.normal(size=500)
    assert compute_psi(ref, ref) == pytest.approx(0.0, abs=1e-6)


def test_compute_psi_fires_on_shift() -> None:
    """PSI exceeds the 0.25 convention threshold on a shifted distribution."""
    rng = np.random.RandomState(42)
    ref = rng.normal(size=500)
    cur = rng.normal(loc=5.0, size=500)
    assert compute_psi(ref, cur) > 0.25


def test_should_promote_tie_and_loss() -> None:
    """Gate promotes on F1 tie (>=), rejects a worse challenger."""
    from app.retrain.champion_challenger import ChampionChallenger

    gate = ChampionChallenger()
    assert gate._should_promote({"f1": 0.5}, {"f1": 0.5}) is True
    assert gate._should_promote({"f1": 0.4}, {"f1": 0.5}) is False
