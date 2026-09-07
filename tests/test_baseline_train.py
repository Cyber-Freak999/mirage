"""Tests for baseline model training pipeline."""

from pathlib import Path

import numpy as np
import pytest

from app.retrain import champion_challenger as cc


@pytest.fixture
def fake_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic balanced dataset and redirected artifact paths."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 25)).astype(np.float32)
    y = np.array([0, 1] * 100, dtype=int)

    def fake_load(cicids_dir=None, unsw_dir=None, cicids_sample=1.0, unsw_sample=1.0, random_state=42):
        return X, y

    monkeypatch.setattr(cc, "load_combined_datasets", fake_load)
    monkeypatch.setattr(cc, "VALIDATION_SET_PATH", tmp_path / "validation_set.npz")
    return X, y


def test_create_validation_set_returns_train_and_val(tmp_path: Path, fake_data) -> None:
    X, y = fake_data

    X_train, y_train, X_val, y_val = cc.create_validation_set(test_size=0.2, random_state=42)

    assert len(X_train) == 160
    assert len(X_val) == 40
    assert X_train.shape[1] == 25
    assert X_val.shape[1] == 25
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0  # stratified: both classes present

    saved = np.load(tmp_path / "validation_set.npz")
    np.testing.assert_array_equal(saved["y"], y_val)
