"""Tests for dashboard production mode: admin gate and real confidence data."""

import sqlite3

import pytest


def test_check_admin_key_accepts_correct(monkeypatch) -> None:
    """Correct admin key unlocks."""
    from dashboard.app import check_admin_key

    monkeypatch.setenv("MIRAGE_ADMIN_KEY", "s3cret")
    assert check_admin_key("s3cret") is True


def test_check_admin_key_rejects_wrong(monkeypatch) -> None:
    """Wrong admin key stays locked."""
    from dashboard.app import check_admin_key

    monkeypatch.setenv("MIRAGE_ADMIN_KEY", "s3cret")
    assert check_admin_key("nope") is False


def test_check_admin_key_locked_without_env(monkeypatch) -> None:
    """No configured key means locked (fail closed)."""
    from dashboard.app import check_admin_key

    monkeypatch.delenv("MIRAGE_ADMIN_KEY", raising=False)
    assert check_admin_key("anything") is False


def test_confidence_history_reads_logged_scores(tmp_path, monkeypatch) -> None:
    """Confidence tab reflects logged /api/score events in order."""
    import time

    import dashboard.app as dash_mod

    now = time.time()
    db = tmp_path / "dashboard.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE score_log (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " timestamp REAL NOT NULL, score REAL NOT NULL,"
        " prediction INTEGER NOT NULL, model_version TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO score_log (timestamp, score, prediction, model_version) VALUES (?, ?, ?, ?)",
        [(now - 300.0, 0.9, 1, "v1"), (now - 200.0, 0.2, 0, "v1"), (now - 100.0, 0.95, 1, "v2")],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(dash_mod, "DASHBOARD_DB", db)

    df = dash_mod.get_confidence_history(hours=24 * 365)

    # Confidence mirrors the API: max(score, 1 - score).
    assert list(df["confidence"]) == pytest.approx([0.9, 0.8, 0.95])
    assert list(df["prediction"]) == [1, 0, 1]
    assert list(df["model_version"]) == ["v1", "v1", "v2"]


def test_confidence_history_empty_without_table(tmp_path, monkeypatch) -> None:
    """Missing score log yields an empty frame, not a crash."""
    import dashboard.app as dash_mod

    monkeypatch.setattr(dash_mod, "DASHBOARD_DB", tmp_path / "missing.db")

    df = dash_mod.get_confidence_history()

    assert df.empty
