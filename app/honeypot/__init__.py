"""Honeypot endpoints — SQLi-style and RCE/upload-style entry points."""

import json
import os
import sqlite3
import time
from collections import deque
from pathlib import Path

from flask import Flask

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "honeypot.db"

MAX_HEADERS_BYTES = 4096
MAX_TRACKED_IPS = 10_000
RATE_LIMIT_WINDOW_SECONDS = 60.0

_rate_hits: dict[str, deque[float]] = {}


def reset_rate_limit():
    """Clear all rate-limiter state (tests)."""
    _rate_hits.clear()


def _rate_limit() -> int:
    """Per-minute request cap for honeypot endpoints (env `MIRAGE_RATE_LIMIT_PER_MIN`)."""
    return int(os.environ.get("MIRAGE_RATE_LIMIT_PER_MIN", 60))


def init_db():
    """Create the honeypot logs table if it does not exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            source_ip TEXT NOT NULL,
            method TEXT NOT NULL,
            path TEXT NOT NULL,
            query_string TEXT,
            user_agent TEXT,
            raw_request TEXT,
            attack_type TEXT NOT NULL DEFAULT 'unknown',
            headers_json TEXT,
            content_type TEXT
        )
        """
    )
    _migrate(conn)
    conn.commit()
    conn.close()


def _migrate(conn: sqlite3.Connection):
    """Add capture-fidelity columns to pre-migration databases."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
    if "headers_json" not in columns:
        conn.execute("ALTER TABLE requests ADD COLUMN headers_json TEXT")
    if "content_type" not in columns:
        conn.execute("ALTER TABLE requests ADD COLUMN content_type TEXT")


def register_blueprints(app: Flask):
    """Register all honeypot blueprints on the given Flask app."""
    from . import rce, sqlii

    app.register_blueprint(sqlii.bp)
    app.register_blueprint(rce.bp)


def capture_context() -> dict[str, str]:
    """Extract structured capture fields from the active Flask request.

    Uses ``request.path`` (no query string — that lives in ``query_string``)
    and preserves the full header set plus content type for faithful
    feature extraction downstream.

    Returns:
        Dict with method/path/query_string/user_agent/raw_request/
        headers_json/content_type keys.
    """
    from flask import request as flask_request

    return {
        "method": flask_request.method,
        "path": flask_request.path,
        "query_string": flask_request.query_string.decode("utf-8", errors="replace")
        if flask_request.query_string
        else "",
        "user_agent": flask_request.headers.get("User-Agent", ""),
        "raw_request": flask_request.get_data(as_text=True),
        "headers_json": json.dumps(dict(flask_request.headers))[:MAX_HEADERS_BYTES],
        "content_type": flask_request.content_type or "",
    }


def log_request(source_ip: str, capture: dict[str, str], attack_type: str):
    """Persist one honeypot capture with structured fields."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        INSERT INTO requests
        (timestamp, source_ip, method, path, query_string,
         user_agent, raw_request, attack_type,
         headers_json, content_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(),
            source_ip,
            capture["method"],
            capture["path"],
            capture["query_string"],
            capture["user_agent"],
            capture["raw_request"],
            attack_type,
            capture["headers_json"],
            capture["content_type"],
        ),
    )
    conn.commit()
    conn.close()


def register_rate_limit(app: Flask):
    """Throttle honeypot endpoints per IP without dropping logs.

    Runs as an ``after_request`` handler so the view (and its logging) always
    executes first — spec section 13 requires all traffic to be logged
    regardless of rate limiting. Over-limit responses are replaced with 429.

    Note: counters are per-process memory, so multi-worker deployments
    (gunicorn) enforce the limit approximately per worker.
    """

    @app.after_request
    def _throttle(response):
        from flask import request as flask_request

        if flask_request.blueprint not in ("sqlii", "rce"):
            return response
        now = time.time()
        ip = flask_request.remote_addr or "unknown"
        hits = _rate_hits.get(ip)
        if hits is None:
            if len(_rate_hits) >= MAX_TRACKED_IPS:
                _rate_hits.pop(next(iter(_rate_hits)))
            hits = _rate_hits[ip] = deque()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while hits and hits[0] <= cutoff:
            hits.popleft()
        hits.append(now)
        if len(hits) > _rate_limit():
            return app.response_class("Rate limit exceeded", status=429)
        return response


__all__ = [
    "init_db",
    "register_blueprints",
    "capture_context",
    "log_request",
    "register_rate_limit",
    "reset_rate_limit",
]
