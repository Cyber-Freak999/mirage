"""Stacking ensemble implementation with k-fold cross-validated stacking."""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ..schema.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_K_FOLDS = 5
DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

DEFAULT_META_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
}


@dataclass
class ModelVersion:
    """Metadata for a model version."""

    version_id: str
    timestamp: float
    k_folds: int
    rf_params: dict
    xgb_params: dict
    meta_params: dict
    training_samples: int
    class_distribution: dict[int, int]
    training_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    decision_threshold: float = 0.5
    is_champion: bool = False

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "k_folds": self.k_folds,
            "rf_params": self.rf_params,
            "xgb_params": self.xgb_params,
            "meta_params": self.meta_params,
            "training_samples": self.training_samples,
            "class_distribution": self.class_distribution,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "feature_importance": self.feature_importance,
            "decision_threshold": self.decision_threshold,
            "is_champion": self.is_champion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(**data)


class StackingEnsemble:
    """Stacking ensemble with Random Forest + XGBoost base learners and Logistic Regression meta-learner.

    Uses k-fold cross-validated stacking to prevent leakage:
    - Base learners trained on k-1 folds, predict on held-out fold
    - Meta-learner trained on out-of-fold predictions only
    - Base learners retrained on full data for deployment
    """

    def __init__(
        self,
        k_folds: int = DEFAULT_K_FOLDS,
        rf_params: dict | None = None,
        xgb_params: dict | None = None,
        meta_params: dict | None = None,
        random_state: int = 42,
    ):
        self.k_folds = k_folds
        self.rf_params = {**DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.meta_params = {**DEFAULT_META_PARAMS, **(meta_params or {})}
        self.random_state = random_state

        self.rf = RandomForestClassifier(**self.rf_params)
        self.xgb = XGBClassifier(**self.xgb_params)
        self.meta = LogisticRegression(**self.meta_params)

        self.rf_fitted = False
        self.xgb_fitted = False
        self.meta_fitted = False
        self.feature_names = FEATURE_NAMES
        self.version: ModelVersion | None = None
        self._oof_predictions: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """Train the stacking ensemble with k-fold cross-validated stacking."""
        logger.info(f"Training stacking ensemble with {self.k_folds}-fold CV on {len(X)} samples")

        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist(), strict=False))
        logger.info(f"Class distribution: {class_dist}")

        if self.xgb_params.get("scale_pos_weight", 1.0) == 1.0 and class_dist.get(0, 0) > 0:
            scale_pos_weight = class_dist[0] / max(class_dist.get(1, 1), 1)
            self.xgb.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"Set XGBoost scale_pos_weight={scale_pos_weight:.2f}")

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)

        oof_rf = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold + 1}/{self.k_folds}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            rf_fold = clone(self.rf)
            rf_fold.fit(X_train, y_train)
            oof_rf[val_idx] = rf_fold.predict_proba(X_val)[:, 1]

            xgb_fold = clone(self.xgb)
            xgb_fold.fit(X_train, y_train)
            oof_xgb[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]

        self._oof_predictions = np.column_stack([oof_rf, oof_xgb])

        self.meta.fit(self._oof_predictions, y)
        self.meta_fitted = True

        logger.info("Retraining base learners on full dataset...")
        self.rf.fit(X, y)
        self.rf_fitted = True
        self.xgb.fit(X, y)
        self.xgb_fitted = True

        rf_importance = dict(zip(self.feature_names, self.rf.feature_importances_, strict=False))
        xgb_importance = dict(zip(self.feature_names, self.xgb.feature_importances_, strict=False))

        combined_importance = {}
        for feat in self.feature_names:
            combined_importance[feat] = (rf_importance.get(feat, 0) + xgb_importance.get(feat, 0)) / 2

        top_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)[:8]
        logger.info(f"Top 8 features by importance: {top_features}")

        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
        auc = roc_auc_score(y, y_pred_proba)

        version_id = f"v{int(time.time())}"
        self.version = ModelVersion(
            version_id=version_id,
            timestamp=time.time(),
            k_folds=self.k_folds,
            rf_params=self.rf_params,
            xgb_params=self.xgb_params,
            meta_params=self.meta_params,
            training_samples=len(X),
            class_distribution=class_dist,
            training_metrics={
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
            },
            feature_importance=combined_importance,
            is_champion=True,
        )

        logger.info(
            "Training complete. Training-set metrics: " f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )
        return self

    @property
    def decision_threshold(self) -> float:
        """Deployment decision threshold (defaults to 0.5 until calibrated)."""
        return self.version.decision_threshold if self.version else 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get attack probability predictions."""
        if not self.rf_fitted or not self.xgb_fitted or not self.meta_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        rf_proba = self.rf.predict_proba(X)[:, 1]
        xgb_proba = self.xgb.predict_proba(X)[:, 1]
        meta_input = np.column_stack([rf_proba, xgb_proba])
        return np.asarray(self.meta.predict_proba(meta_input)[:, 1])

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Get binary predictions at the model's (or an explicit) threshold."""
        effective = self.decision_threshold if threshold is None else threshold
        return (self.predict_proba(X) >= effective).astype(int)

    def get_base_predictions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get individual base learner predictions for debugging/analysis."""
        if not self.rf_fitted or not self.xgb_fitted:
            raise RuntimeError("Base learners not fitted")
        return self.rf.predict_proba(X)[:, 1], self.xgb.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Get combined feature importance."""
        if not self.version:
            return {}
        return self.version.feature_importance

    def get_drift_tracking_features(self, top_n: int = 8) -> list[str]:
        """Get top N features by importance for drift monitoring."""
        importance = self.get_feature_importance()
        return [feat for feat, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def train_initial_model(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = DEFAULT_K_FOLDS,
    rf_params: dict | None = None,
    xgb_params: dict | None = None,
    meta_params: dict | None = None,
    random_state: int = 42,
) -> StackingEnsemble:
    """Train initial model from scratch.

    Args:
        X: Feature matrix.
        y: Binary labels.
        k_folds: Stacking folds.
        rf_params: Random Forest overrides (merged over defaults).
        xgb_params: XGBoost overrides (merged over defaults).
        meta_params: Meta-learner overrides (merged over defaults).
        random_state: Seed applied to all three learners unless overridden
            in the corresponding params dict.
    """
    rf = {**DEFAULT_RF_PARAMS, **(rf_params or {}), "random_state": (rf_params or {}).get("random_state", random_state)}
    xgb = {
        **DEFAULT_XGB_PARAMS,
        **(xgb_params or {}),
        "random_state": (xgb_params or {}).get("random_state", random_state),
    }
    meta = {
        **DEFAULT_META_PARAMS,
        **(meta_params or {}),
        "random_state": (meta_params or {}).get("random_state", random_state),
    }
    ensemble = StackingEnsemble(
        k_folds=k_folds,
        rf_params=rf,
        xgb_params=xgb,
        meta_params=meta,
        random_state=random_state,
    )
    ensemble.fit(X, y)
    return ensemble


def save_model(ensemble: StackingEnsemble, version_id: str | None = None) -> Path:
    """Save model and metadata to disk."""
    if not ensemble.version:
        raise ValueError("Model has no version metadata")

    if version_id:
        ensemble.version.version_id = version_id

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{ensemble.version.version_id}.pkl"
    meta_path = MODEL_DIR / f"{ensemble.version.version_id}.json"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "rf": ensemble.rf,
                "xgb": ensemble.xgb,
                "meta": ensemble.meta,
                "k_folds": ensemble.k_folds,
                "rf_params": ensemble.rf_params,
                "xgb_params": ensemble.xgb_params,
                "meta_params": ensemble.meta_params,
                "random_state": ensemble.random_state,
                "feature_names": ensemble.feature_names,
            },
            f,
        )

    with open(meta_path, "w") as f:
        json.dump(ensemble.version.to_dict(), f, indent=2)

    champion_path = MODEL_DIR / "champion.pkl"
    champion_meta_path = MODEL_DIR / "champion.json"
    with open(champion_path, "wb") as f:
        pickle.dump(
            {
                "rf": ensemble.rf,
                "xgb": ensemble.xgb,
                "meta": ensemble.meta,
                "k_folds": ensemble.k_folds,
                "rf_params": ensemble.rf_params,
                "xgb_params": ensemble.xgb_params,
                "meta_params": ensemble.meta_params,
                "random_state": ensemble.random_state,
                "feature_names": ensemble.feature_names,
            },
            f,
        )
    with open(champion_meta_path, "w") as f:
        json.dump(ensemble.version.to_dict(), f, indent=2)

    logger.info(f"Model saved: {model_path}")
    return model_path


def load_model(version_id: str = "champion") -> StackingEnsemble:
    """Load model from disk."""
    model_path = MODEL_DIR / f"{version_id}.pkl"
    meta_path = MODEL_DIR / f"{version_id}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    ensemble = StackingEnsemble(
        k_folds=data["k_folds"],
        rf_params=data["rf_params"],
        xgb_params=data["xgb_params"],
        meta_params=data["meta_params"],
        random_state=data["random_state"],
    )
    ensemble.rf = data["rf"]
    ensemble.xgb = data["xgb"]
    ensemble.meta = data["meta"]
    ensemble.rf_fitted = True
    ensemble.xgb_fitted = True
    ensemble.meta_fitted = True
    ensemble.feature_names = data["feature_names"]

    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)
        ensemble.version = ModelVersion.from_dict(meta_data)

    logger.info(f"Model loaded: {version_id}")
    return ensemble


def list_model_versions() -> list[ModelVersion]:
    """List all saved model versions."""
    versions = []
    for meta_path in MODEL_DIR.glob("*.json"):
        if meta_path.name == "champion.json":
            continue
        try:
            with open(meta_path) as f:
                data = json.load(f)
            versions.append(ModelVersion.from_dict(data))
        except Exception as e:
            logger.warning(f"Failed to load {meta_path}: {e}")
    return sorted(versions, key=lambda v: v.timestamp, reverse=True)
