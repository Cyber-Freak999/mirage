"""Mirage - Adaptive IDS Honeypot & Detection System."""

import logging
from pathlib import Path

from flask import Flask

from .api import bp as api_bp
from .honeypot import init_db
from .honeypot import register_blueprints as register_honeypot_blueprints

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    if config:
        app.config.update(config)

    app.config.setdefault("JSON_SORT_KEYS", False)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    init_db()

    logger.info("Registering honeypot blueprints...")
    register_honeypot_blueprints(app)

    logger.info("Registering API blueprint...")
    app.register_blueprint(api_bp)

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
