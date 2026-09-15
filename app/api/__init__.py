"""Flask API for real-time intrusion detection scoring."""

import logging
import time

from flask import Blueprint, jsonify, request

from ..model.ensemble import StackingEnsemble, load_model
from ..schema.features import extract_features

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__, url_prefix="/api")

_model: StackingEnsemble = None
_model_version: str = "unknown"


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
    except Exception as e:
        logger.exception("Scoring failed")
        return jsonify({"error": str(e)}), 500


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
        except Exception as e:
            results.append({"error": str(e)})

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


@bp.route("/model/reload", methods=["POST"])
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
    except Exception as e:
        logger.exception("Model reload failed")
        return jsonify({"error": str(e)}), 500


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
