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


def test_train_cli_missing_data_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import app.model.train as train_mod

    monkeypatch.setattr(
        cc,
        "load_combined_datasets",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("No datasets loaded")),
    )
    monkeypatch.setattr(cc, "VALIDATION_SET_PATH", tmp_path / "validation_set.npz")

    result = train_mod.main(["--cicids-dir", str(tmp_path / "missing-cicids")])

    assert result == 1


def test_train_cli_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_data) -> None:
    import app.model.train as train_mod
    from app.model import ensemble as model_mod

    monkeypatch.setattr(cc, "VALIDATION_DB", tmp_path / "validation.db")
    monkeypatch.setattr(model_mod, "MODEL_DIR", tmp_path / "models")

    result = train_mod.main(["--k-folds", "2"])

    assert result == 0
    assert (tmp_path / "models" / "champion.pkl").exists()
    assert (tmp_path / "models" / "champion.json").exists()
    assert (tmp_path / "validation_set.npz").exists()

    import json

    champion_meta = json.loads((tmp_path / "models" / "champion.json").read_text())
    assert champion_meta["training_samples"] == 160
    assert champion_meta["is_champion"] is True
    assert champion_meta["validation_metrics"]["f1"] >= 0.0

    conn = __import__("sqlite3").connect(str(tmp_path / "validation.db"))
    row = conn.execute("SELECT champion_version, promoted FROM validations ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row[0] == "none"
    assert row[1] == 1
