"""Honeypot endpoints — SQLi-style and RCE/upload-style entry points."""

import sqlite3
from pathlib import Path

from flask import Flask

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "honeypot.db"


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
            decoy_indicator INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def register_blueprints(app: Flask):
    """Register all honeypot blueprints on the given Flask app."""
    from . import rce, sqlii

    app.register_blueprint(sqlii.bp)
    app.register_blueprint(rce.bp)


__all__ = ["init_db", "register_blueprints"]
