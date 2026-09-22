"""Tests for the capture-data purge script (retention policy)."""

import sqlite3
import time

SCHEMA = """
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    source_ip TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    query_string TEXT,
    user_agent TEXT,
    raw_request TEXT,
    attack_type TEXT NOT NULL DEFAULT 'unknown',
    decoy_indicator INTEGER NOT NULL DEFAULT 0,
    headers_json TEXT,
    content_type TEXT
)
"""


def _seed(db_path, now: float) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA)
    conn.executemany(
        "INSERT INTO requests (timestamp, source_ip, method, path, user_agent,"
        " raw_request, attack_type, headers_json, content_type)"
        " VALUES (?, '1.2.3.4', 'POST', '/login', 'UA', 'body-secret', 'sqli', '{\"a\":\"b\"}', 'text/plain')",
        [(now - 40 * 86400,), (now - 5 * 86400,)],
    )
    conn.commit()
    conn.close()


def test_purge_dry_run_reports_without_modifying(tmp_path) -> None:
    """Dry run counts purgeable rows but changes nothing."""
    from scripts.purge_captures import main

    db = tmp_path / "honeypot.db"
    _seed(db, time.time())

    assert main(["--db", str(db), "--days", "30", "--dry-run"]) == 0

    conn = sqlite3.connect(str(db))
    remaining = conn.execute("SELECT COUNT(*) FROM requests WHERE raw_request IS NOT NULL").fetchone()[0]
    conn.close()
    assert remaining == 2


def test_purge_nulls_only_old_payloads(tmp_path) -> None:
    """Real run nulls raw_request/headers_json on old rows; fresh rows kept."""
    from scripts.purge_captures import main

    db = tmp_path / "honeypot.db"
    _seed(db, time.time())

    assert main(["--db", str(db), "--days", "30"]) == 0

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT raw_request, headers_json, method, path FROM requests ORDER BY timestamp").fetchall()
    conn.close()

    assert rows[0]["raw_request"] is None
    assert rows[0]["headers_json"] is None
    assert rows[1]["raw_request"] == "body-secret"
    # Structured columns survive for aggregation.
    assert rows[0]["method"] == "POST"
    assert rows[0]["path"] == "/login"


def test_purged_rows_still_convert(tmp_path) -> None:
    """Purged rows still yield finite features via the degraded path."""
    import numpy as np

    from app.schema.capture import rows_to_features
    from scripts.purge_captures import main

    db = tmp_path / "honeypot.db"
    _seed(db, time.time())
    assert main(["--db", str(db), "--days", "30"]) == 0

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM requests ORDER BY timestamp").fetchall()
    conn.close()

    X = rows_to_features(rows)
    assert X.shape == (2, 25)
    assert bool(np.isfinite(X).all())
