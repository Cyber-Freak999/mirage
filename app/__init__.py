"""Mirage - Adaptive IDS Honeypot & Detection System."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from .api import bp as api_bp
from .honeypot import init_db, register_rate_limit
from .honeypot import register_blueprints as register_honeypot_blueprints

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

DEV_SECRET_KEY = "dev-only-insecure"


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    if config:
        app.config.update(config)

    app.config.setdefault("JSON_SORT_KEYS", False)

    secret_key = os.environ.get("MIRAGE_SECRET_KEY")
    if not secret_key:
        logger.warning("MIRAGE_SECRET_KEY not set - using insecure dev default")
        secret_key = DEV_SECRET_KEY
    app.secret_key = secret_key
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MIRAGE_MAX_CONTENT_LENGTH", 1_000_000))

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    init_db()

    logger.info("Registering honeypot blueprints...")
    register_honeypot_blueprints(app)

    logger.info("Registering API blueprint...")
    app.register_blueprint(api_bp)

    logger.info("Registering honeypot rate limiter...")
    register_rate_limit(app)

    @app.route("/")
    def index():
        return {
            "name": "Mirage Adaptive IDS",
            "version": "0.1.0",
            "description": "Honeypot-driven adaptive intrusion detection system",
            "endpoints": {
                "honeypot": ["/login", "/search", "/upload", "/admin/diagnostics"],
                "api": ["/api/health", "/api/score", "/api/batch_score", "/api/model/info", "/api/model/versions"],
            },
        }

    logger.info("Application created successfully")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
