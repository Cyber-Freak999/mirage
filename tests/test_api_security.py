"""Tests for API hardening: key auth, error hygiene, body limits, rate limiting."""

import pytest

from app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App with isolated honeypot DB and no API key configured."""
    import app.honeypot as honeypot_mod

    monkeypatch.setattr(honeypot_mod, "DB_PATH", tmp_path / "honeypot.db")
    monkeypatch.delenv("MIRAGE_API_KEY", raising=False)
    monkeypatch.delenv("MIRAGE_SECRET_KEY", raising=False)
    return create_app().test_client()


def test_reload_without_key_configured_returns_503(client) -> None:
    """Mutating endpoints fail closed when no API key is configured."""
    resp = client.post("/api/model/reload")
    assert resp.status_code == 503
    assert "not configured" in resp.get_json()["error"]


def test_reload_with_wrong_key_returns_401(client, monkeypatch) -> None:
    """Wrong API key is rejected."""
    monkeypatch.setenv("MIRAGE_API_KEY", "correct-key")
    resp = client.post("/api/model/reload", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_reload_with_correct_key_reloads(client, monkeypatch) -> None:
    """Correct API key reloads (or reports no model without leaking internals)."""
    monkeypatch.setenv("MIRAGE_API_KEY", "correct-key")
    resp = client.post("/api/model/reload", headers={"X-API-Key": "correct-key"})
    assert resp.status_code in (200, 500)
    if resp.status_code == 500:
        assert "Traceback" not in resp.get_data(as_text=True)


def test_score_error_hides_internals(client) -> None:
    """Scoring failures return a generic message, not exception strings."""
    resp = client.post("/api/score", json={"method": "GET", "path": "/", "headers": "not-a-dict", "body": ""})
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["error"] == "Scoring failed"
    assert "AttributeError" not in resp.get_data(as_text=True)


def test_score_shape_unchanged(client) -> None:
    """Successful scores keep the documented response shape."""
    resp = client.post(
        "/api/score",
        json={"method": "GET", "path": "/", "query_string": "", "headers": {}, "body": ""},
    )
    assert resp.status_code == 200
    assert set(resp.get_json()) == {
        "score",
        "prediction",
        "label",
        "confidence",
        "base_scores",
        "model_version",
        "timestamp",
    }


def test_oversize_body_rejected(client) -> None:
    """Bodies over MAX_CONTENT_LENGTH get 413."""
    big = "x" * (2 * 1024 * 1024)
    resp = client.post("/api/score", json={"method": "POST", "path": "/", "headers": {}, "body": big})
    assert resp.status_code == 413


def test_review_endpoints_require_key(client, monkeypatch) -> None:
    """Review decisions are mutating endpoints: 401 without the key."""
    monkeypatch.setenv("MIRAGE_API_KEY", "correct-key")
    assert client.post("/api/reviews/1/approve", json={}).status_code == 401
    assert client.post("/api/reviews/1/reject", json={}).status_code == 401


def test_honeypot_rate_limit_logs_everything(client, tmp_path, monkeypatch) -> None:
    """Burst over the limit yields 429s but every hit is still logged (spec 13)."""
    import sqlite3

    from app.honeypot import reset_rate_limit

    monkeypatch.setenv("MIRAGE_RATE_LIMIT_PER_MIN", "5")
    reset_rate_limit()

    statuses = [client.get("/login").status_code for _ in range(8)]
    assert statuses[:5] == [200] * 5
    assert statuses[5:] == [429] * 3

    conn = sqlite3.connect(str(tmp_path / "honeypot.db"))
    count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    conn.close()
    assert count == 8


def test_score_is_logged_for_dashboard(client, tmp_path, monkeypatch) -> None:
    """Each /api/score call appends a row to the dashboard score log."""
    import sqlite3

    import app.api as api_mod

    monkeypatch.setattr(api_mod, "DASHBOARD_DB", tmp_path / "dashboard.db")
    resp = client.post(
        "/api/score",
        json={"method": "GET", "path": "/", "query_string": "", "headers": {}, "body": ""},
    )
    assert resp.status_code == 200

    conn = sqlite3.connect(str(tmp_path / "dashboard.db"))
    row = conn.execute("SELECT score, prediction, model_version FROM score_log").fetchone()
    conn.close()
    assert row is not None
    assert row[0] == pytest.approx(resp.get_json()["score"])
