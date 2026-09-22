"""Flask API for real-time intrusion detection scoring."""

import hmac
import logging
import os
import time
from functools import wraps
from pathlib import Path

from flask import Blueprint, jsonify, request

from ..model.ensemble import StackingEnsemble, load_model
from ..schema.features import extract_features

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__, url_prefix="/api")

_model: StackingEnsemble = None
_model_version: str = "unknown"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DASHBOARD_DB = DATA_DIR / "dashboard.db"


def log_score(score: float, prediction: int, model_version: str):
    """Append one scoring event for the dashboard confidence trend.

    Best-effort: logging failures are swallowed so scoring never breaks.
    """
    try:
        import sqlite3

        conn = sqlite3.connect(str(DASHBOARD_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS score_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                score REAL NOT NULL,
                prediction INTEGER NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO score_log (timestamp, score, prediction, model_version) VALUES (?, ?, ?, ?)",
            (time.time(), score, prediction, model_version),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.warning("Failed to log score event", exc_info=True)


def require_api_key(view):
    """Require a valid ``X-API-Key`` header for mutating endpoints.

    Fails closed (503) when ``MIRAGE_API_KEY`` is not configured; 401 on
    mismatch. Scoring and read-only endpoints stay public by design.
    """

    @wraps(view)
    def wrapper(*args, **kwargs):
        expected = os.environ.get("MIRAGE_API_KEY")
        if not expected:
            return jsonify({"error": "API key not configured"}), 503
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, expected):
            return jsonify({"error": "Invalid API key"}), 401
        return view(*args, **kwargs)

    return wrapper


def get_model() -> StackingEnsemble:
    global _model, _model_version
    if _model is None:
        try:
            _model = load_model("champion")
            _model_version = _model.version.version_id if _model.version else "unknown"
            logger.info(f"Loaded champion model: {_model_version}")
        except FileNotFoundError:
            logger.warning("No champion model found. API will return errors until model is trained.")
            _model = None
    return _model


def reload_model():
    global _model, _model_version
    _model = None
    _model = get_model()
    return _model


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model = get_model()
    return jsonify(
        {
            "status": "healthy" if model else "no_model",
            "model_version": _model_version,
            "timestamp": time.time(),
        }
    )


@bp.route("/score", methods=["POST"])
def score():
    """Score a single request for intrusion detection."""
    model = get_model()
    if model is None:
        return jsonify(
            {
                "error": "Model not loaded",
                "message": "No trained model available. Train a model first.",
            }
        ), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    method = data.get("method", "GET")
    path = data.get("path", "/")
    query_string = data.get("query_string", "")
    headers = data.get("headers", {})
    body = data.get("body", "")

    try:
        features = extract_features(method, path, query_string, headers, body)
        score = float(model.predict_proba(features.reshape(1, -1))[0])
        prediction = int(score >= model.decision_threshold)

        rf_score, xgb_score = model.get_base_predictions(features.reshape(1, -1))

        log_score(score, prediction, _model_version)
        return jsonify(
            {
                "score": score,
                "prediction": prediction,
                "label": "attack" if prediction == 1 else "benign",
                "confidence": max(score, 1 - score),
                "base_scores": {
                    "random_forest": float(rf_score[0]),
                    "xgboost": float(xgb_score[0]),
                },
                "model_version": _model_version,
                "timestamp": time.time(),
            }
        )
    except Exception:
        logger.exception("Scoring failed")
        return jsonify({"error": "Scoring failed"}), 500


@bp.route("/batch_score", methods=["POST"])
def batch_score():
    """Score multiple requests at once."""
    model = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "requests" not in data:
        return jsonify({"error": "Expected 'requests' array"}), 400

    requests_data = data["requests"]
    if not isinstance(requests_data, list):
        return jsonify({"error": "'requests' must be an array"}), 400

    results = []
    for req in requests_data:
        method = req.get("method", "GET")
        path = req.get("path", "/")
        query_string = req.get("query_string", "")
        headers = req.get("headers", {})
        body = req.get("body", "")

        try:
            features = extract_features(method, path, query_string, headers, body)
            score = float(model.predict_proba(features.reshape(1, -1))[0])
            prediction = int(score >= model.decision_threshold)

            results.append(
                {
                    "score": score,
                    "prediction": prediction,
                    "label": "attack" if prediction == 1 else "benign",
                    "confidence": max(score, 1 - score),
                }
            )
        except Exception:
            logger.exception("Batch item scoring failed")
            results.append({"error": "Scoring failed"})

    return jsonify(
        {
            "results": results,
            "model_version": _model_version,
            "timestamp": time.time(),
        }
    )


@bp.route("/model/info", methods=["GET"])
def model_info():
    """Get model metadata."""
    model = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify(
        {
            "version": _model_version,
            "k_folds": model.k_folds,
            "training_samples": model.version.training_samples if model.version else None,
            "class_distribution": model.version.class_distribution if model.version else None,
            "validation_metrics": model.version.validation_metrics if model.version else None,
            "top_features": model.get_drift_tracking_features(8),
        }
    )


@bp.route("/reviews/<int:review_id>/approve", methods=["POST"])
@require_api_key
def review_approve(review_id: int):
    """Approve a drift review, queueing it for retraining on the next sweep."""
    # TODO(Tier3): require API key auth.
    from ..retrain.review_gate import ReviewGate

    data = request.get_json(silent=True) or {}
    gate = ReviewGate()
    if gate.get_review(review_id) is None:
        return jsonify({"error": f"Review {review_id} not found"}), 404
    if not gate.approve(review_id, data.get("reviewer", "api"), data.get("notes", "")):
        return jsonify({"error": f"Review {review_id} is already decided"}), 409
    return jsonify({"status": "approved", "review_id": review_id})


@bp.route("/reviews/<int:review_id>/reject", methods=["POST"])
@require_api_key
def review_reject(review_id: int):
    """Reject a drift review, skipping retraining for that batch."""
    # TODO(Tier3): require API key auth.
    from ..retrain.review_gate import ReviewGate

    data = request.get_json(silent=True) or {}
    gate = ReviewGate()
    if gate.get_review(review_id) is None:
        return jsonify({"error": f"Review {review_id} not found"}), 404
    if not gate.reject(review_id, data.get("reviewer", "api"), data.get("notes", "")):
        return jsonify({"error": f"Review {review_id} is already decided"}), 409
    return jsonify({"status": "rejected", "review_id": review_id})


@bp.route("/model/reload", methods=["POST"])
@require_api_key
def reload():
    """Reload the champion model (e.g., after retraining)."""
    try:
        reload_model()
        return jsonify(
            {
                "status": "reloaded",
                "model_version": _model_version,
            }
        )
    except Exception:
        logger.exception("Model reload failed")
        return jsonify({"error": "Model reload failed"}), 500


@bp.route("/model/versions", methods=["GET"])
def list_versions():
    """List all available model versions."""
    from ..model.ensemble import list_model_versions

    versions = list_model_versions()
    return jsonify(
        {
            "versions": [v.to_dict() for v in versions],
            "current_champion": _model_version,
        }
    )
