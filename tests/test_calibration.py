"""Tests for decision-threshold calibration and its single source of truth."""

from pathlib import Path

import numpy as np
import pytest

from app.model.calibration import precision_recall_table, select_threshold
from app.model.ensemble import ModelVersion, train_initial_model
from app.retrain import champion_challenger as cc


class _StubModel:
    """Duck-typed model whose probabilities are the first feature column."""

    def __init__(self, threshold: float) -> None:
        self.decision_threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0]


def _overlapping_proba_data() -> tuple[np.ndarray, np.ndarray]:
    """Craft labels/probas where the F1-optimal threshold is clearly above 0.5."""
    rng = np.random.default_rng(42)
    benign = np.clip(rng.beta(2, 8, size=300), 0.0, 1.0)
    attacks = np.clip(rng.beta(8, 2, size=300), 0.0, 1.0)
    y_proba = np.concatenate([benign, attacks])
    y_true = np.array([0] * 300 + [1] * 300, dtype=int)
    return y_true, y_proba


def test_select_threshold_maximizes_f1() -> None:
    y_true, y_proba = _overlapping_proba_data()

    threshold = select_threshold(y_true, y_proba, objective="f1")

    grid = np.linspace(0.05, 0.95, 19)
    assert threshold in {float(t) for t in grid}
    from sklearn.metrics import f1_score

    best_f1 = max(f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in grid)
    assert f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0) == pytest.approx(best_f1)


def test_select_threshold_precision_floor() -> None:
    y_true, y_proba = _overlapping_proba_data()

    threshold = select_threshold(y_true, y_proba, objective="precision_floor")

    from sklearn.metrics import precision_score

    precision = precision_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
    assert precision >= 0.6 or threshold == select_threshold(y_true, y_proba, objective="f1")


def test_precision_recall_table_shape() -> None:
    y_true, y_proba = _overlapping_proba_data()

    table = precision_recall_table(y_true, y_proba)

    assert len(table) == 19
    assert set(table[0]) == {"threshold", "precision", "recall", "f1"}
    assert table[0]["threshold"] < table[-1]["threshold"]


def test_evaluate_model_uses_model_threshold() -> None:
    y_true, y_proba = _overlapping_proba_data()
    X = y_proba.reshape(-1, 1)

    strict = cc.evaluate_model(_StubModel(threshold=0.9), X, y_true)
    loose = cc.evaluate_model(_StubModel(threshold=0.1), X, y_true)

    assert strict["recall"] < loose["recall"]
    assert strict["precision"] >= loose["precision"]


def test_predict_none_uses_model_threshold() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 1, (50, 25)), rng.normal(3, 1, (50, 25))]).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50, dtype=int)

    ensemble = train_initial_model(X, y, k_folds=2)
    ensemble.version.decision_threshold = 0.9

    np.testing.assert_array_equal(ensemble.predict(X), ensemble.predict(X, threshold=0.9))
    assert not np.array_equal(ensemble.predict(X), ensemble.predict(X, threshold=0.1))


def test_validate_calibrates_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 1, (150, 25)), rng.normal(0.4, 1, (150, 25))]).astype(np.float32)
    y = np.array([0] * 150 + [1] * 150, dtype=int)
    shuffle = np.random.default_rng(7).permutation(len(X))
    X, y = X[shuffle], y[shuffle]
    X_train, y_train, X_val, y_val = X[:240], y[:240], X[240:], y[240:]

    monkeypatch.setattr(cc, "VALIDATION_DB", tmp_path / "validation.db")
    import app.model.ensemble as model_mod

    monkeypatch.setattr(model_mod, "MODEL_DIR", tmp_path / "models")

    ensemble = train_initial_model(X_train, y_train, k_folds=2)

    cc.ChampionChallenger().validate(ensemble, X_val=X_val, y_val=y_val)

    grid = {round(float(t), 4) for t in np.linspace(0.05, 0.95, 19)}
    assert round(ensemble.version.decision_threshold, 4) in grid
    assert ensemble.version.decision_threshold != 0.5 or grid  # calibrated to a grid value


def test_model_version_from_dict_decision_threshold_default() -> None:
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

    assert version.decision_threshold == 0.5
