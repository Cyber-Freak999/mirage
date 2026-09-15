"""Tests for truthful model metrics (training vs held-out validation)."""

from pathlib import Path

import numpy as np
import pytest

from app.model.ensemble import ModelVersion, train_initial_model
from app.retrain import champion_challenger as cc


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable synthetic dataset with 25 features."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=0.0, scale=1.0, size=(100, 25))
    X1 = rng.normal(loc=3.0, scale=1.0, size=(100, 25))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100, dtype=int)
    return X, y


def test_fit_stores_training_metrics_not_validation(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_data

    ensemble = train_initial_model(X, y, k_folds=2)

    assert set(ensemble.version.training_metrics) == {"precision", "recall", "f1", "auc"}
    assert ensemble.version.validation_metrics == {}


def test_validate_writes_held_out_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = synthetic_data
    shuffle = np.random.default_rng(7).permutation(len(X))
    X, y = X[shuffle], y[shuffle]
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    monkeypatch.setattr(cc, "VALIDATION_DB", tmp_path / "validation.db")
    import app.model.ensemble as model_mod

    monkeypatch.setattr(model_mod, "MODEL_DIR", tmp_path / "models")

    ensemble = train_initial_model(X_train, y_train, k_folds=2)
    training_metrics = dict(ensemble.version.training_metrics)
    assert ensemble.version.validation_metrics == {}

    result = cc.ChampionChallenger().validate(ensemble, X_val=X_val, y_val=y_val)

    expected = cc.evaluate_model(ensemble, X_val, y_val)
    assert result.promoted  # first model auto-promotes
    assert ensemble.version.validation_metrics == expected
    assert ensemble.version.training_metrics == training_metrics

    import json

    champion_meta = json.loads((tmp_path / "models" / "champion.json").read_text())
    assert champion_meta["validation_metrics"] == expected
    assert champion_meta["training_metrics"] == training_metrics


def test_model_version_from_dict_backcompat() -> None:
    old_shape = {
        "version_id": "v0000000001",
        "timestamp": 0.0,
        "k_folds": 5,
        "rf_params": {},
        "xgb_params": {},
        "meta_params": {},
        "training_samples": 10,
        "class_distribution": {"0": 5, "1": 5},
        "validation_metrics": {"f1": 0.5},
        "feature_importance": {},
        "is_champion": False,
    }

    version = ModelVersion.from_dict(old_shape)

    assert version.version_id == "v0000000001"
    assert version.training_metrics == {}
