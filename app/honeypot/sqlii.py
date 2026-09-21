"""SQLi-style honeypot entry points."""

from flask import Blueprint, jsonify, request

from . import capture_context, log_request

bp = Blueprint("sqlii", __name__)


@bp.route("/login", methods=["GET", "POST"])
def login():
    """Simulated login form — classic SQLi target."""
    source_ip = request.remote_addr or "unknown"
    ctx = capture_context()
    method = ctx["method"]

    log_request(source_ip, ctx, "sqli")

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
    ctx = capture_context()
    method = ctx["method"]

    log_request(source_ip, ctx, "sqli")

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
