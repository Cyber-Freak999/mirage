"""SQLi-style honeypot entry points."""

import sqlite3
import time

from flask import Blueprint, jsonify, request

from . import DB_PATH

bp = Blueprint("sqlii", __name__)


def _log_request(source_ip, method, path, query_string, user_agent, raw_request, attack_type):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        INSERT INTO requests
        (timestamp, source_ip, method, path, query_string,
         user_agent, raw_request, attack_type, decoy_indicator)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (time.time(), source_ip, method, path, query_string, user_agent, raw_request, attack_type, 0),
    )
    conn.commit()
    conn.close()


@bp.route("/login", methods=["GET", "POST"])
def login():
    """Simulated login form — classic SQLi target."""
    source_ip = request.remote_addr or "unknown"
    method = request.method
    path = request.full_path or request.path
    query_string = request.query_string.decode("utf-8", errors="replace") if request.query_string else ""
    user_agent = request.headers.get("User-Agent", "")
    raw_request = request.get_data(as_text=True)

    _log_request(source_ip, method, path, query_string, user_agent, raw_request, "sqli")

    if method == "POST":
        data = request.form
        # Simulate a vulnerable check — always "accept" but log the payload
        return jsonify(
            {
                "status": "credential_attempted",
                "message": f"Attempted login with username: {data.get('username', '')}",
            }
        ), 200

    # GET — serve a simple login form HTML
    return jsonify(
        {
            "form": (
                "<form method='POST' action='/login'>"
                "<input name='username' placeholder='username'/>"
                "<input type='password' name='password' placeholder='password'/>"
                "<button type='submit'>Login</button></form>"
            )
        }
    ), 200


@bp.route("/search", methods=["GET", "POST"])
def search():
    """Simulated search/filter page — another SQLi entry point."""
    source_ip = request.remote_addr or "unknown"
    method = request.method
    path = request.full_path or request.path
    query_string = request.query_string.decode("utf-8", errors="replace") if request.query_string else ""
    user_agent = request.headers.get("User-Agent", "")
    raw_request = request.get_data(as_text=True)

    _log_request(source_ip, method, path, query_string, user_agent, raw_request, "sqli")

    if method == "POST":
        data = request.form
        query = data.get("query", "")
        # Simulate query execution — never truly vulnerable, just log
        return jsonify(
            {
                "results": f"Fake results for: {query}",
                "query_logged": query,
            }
        ), 200

    return jsonify(
        {
            "form": (
                "<form method='POST' action='/search'>"
                "<input name='query' placeholder='search...'/>"
                "<button type='submit'>Search</button></form>"
            )
        }
    ), 200
