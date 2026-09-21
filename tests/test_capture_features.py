"""Tests for the shared honeypot capture-to-features converter."""

import json
import sqlite3

import numpy as np

from app.schema.capture import rows_to_features

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

LEGACY_SCHEMA = SCHEMA.replace(",\n    headers_json TEXT,\n    content_type TEXT", "")


def _rows(conn: sqlite3.Connection):
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM requests ORDER BY id").fetchall()


def _conn_with(schema: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(schema)
    return conn


def test_post_form_body_populates_payload() -> None:
    """Urlencoded SQLi body yields payload length and SQL keyword hits."""
    conn = _conn_with(SCHEMA)
    conn.execute(
        "INSERT INTO requests (timestamp, source_ip, method, path, query_string,"
        " user_agent, raw_request, attack_type, headers_json, content_type)"
        " VALUES (1.0, '1.2.3.4', 'POST', '/login', '', 'UA',"
        " 'username=admin%27+OR+1%3D1--&password=x', 'sqli', '{}',"
        " 'application/x-www-form-urlencoded')"
    )
    X = rows_to_features(_rows(conn))

    assert X.shape == (1, 25)
    assert X[0][10] > 0  # payload_length
    assert X[0][12] >= 1  # sql_keyword_count


def test_path_excludes_query() -> None:
    """Path features measure the path only; query flags separately."""
    conn = _conn_with(SCHEMA)
    conn.execute(
        "INSERT INTO requests (timestamp, source_ip, method, path, query_string,"
        " user_agent, raw_request, attack_type, headers_json, content_type)"
        " VALUES (1.0, '1.2.3.4', 'GET', '/login', 'user=a', 'UA', '', 'sqli', '{}', '')"
    )
    X = rows_to_features(_rows(conn))

    assert X[0][4] == float(len("/login"))  # path_length
    assert X[0][5] == 1.0  # has_query


def test_headers_preserved() -> None:
    """Stored headers drive header_count; cookies drive cookie_count."""
    headers = {"user-agent": "UA", "content-type": "text/html", "cookie": "a=1; b=2"}
    conn = _conn_with(SCHEMA)
    conn.execute(
        "INSERT INTO requests (timestamp, source_ip, method, path, query_string,"
        " user_agent, raw_request, attack_type, headers_json, content_type)"
        " VALUES (1.0, '1.2.3.4', 'GET', '/', '', 'UA', '', 'unknown', ?, '')",
        (json.dumps(headers),),
    )
    X = rows_to_features(_rows(conn))

    assert X[0][23] == 3.0  # header_count
    assert X[0][21] == 2.0  # cookie_count


def test_legacy_row_still_converts() -> None:
    """Rows logged before the schema migration still yield finite features."""
    conn = _conn_with(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO requests (timestamp, source_ip, method, path, query_string,"
        " user_agent, raw_request, attack_type)"
        " VALUES (1.0, '1.2.3.4', 'GET', '/search?q=x', 'q=x', 'UA', 'q=x', 'sqli')"
    )
    X = rows_to_features(_rows(conn))

    assert X.shape == (1, 25)
    assert np.isfinite(X).all()
