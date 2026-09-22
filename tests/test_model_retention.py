"""Tests for opt-in model retention pruning (champion + last N kept)."""

import json

import pytest

from app.model.ensemble import ModelVersion


def _version(version_id: str, timestamp: float) -> ModelVersion:
    return ModelVersion(
        version_id=version_id,
        timestamp=timestamp,
        k_folds=2,
        rf_params={},
        xgb_params={},
        meta_params={},
        training_samples=100,
        class_distribution={0: 50, 1: 50},
    )


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """MODEL_DIR with 5 versions (v1 oldest .. v5 newest) + champion copy of v2."""
    import app.model.ensemble as ensemble_mod

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(ensemble_mod, "MODEL_DIR", models)
    for i in range(1, 6):
        v = _version(f"v{i}", float(i))
        (models / f"v{i}.pkl").write_bytes(b"fake-model")
        (models / f"v{i}.json").write_text(json.dumps(v.to_dict()))
    (models / "champion.pkl").write_bytes(b"fake-model")
    (models / "champion.json").write_text(json.dumps(_version("v2", 2.0).to_dict()))
    return models


def test_prune_dry_run_removes_nothing(model_dir) -> None:
    """Dry run reports candidates but deletes nothing."""
    from app.model.ensemble import prune_models

    removed = prune_models(keep_last_n=2, dry_run=True)

    assert sorted(removed) == ["v1", "v3"]
    assert len(list(model_dir.glob("v*.pkl"))) == 5


def test_prune_keeps_champion_and_newest(model_dir) -> None:
    """Real run keeps the champion (v2, not newest) plus the N newest."""
    from app.model.ensemble import prune_models

    removed = prune_models(keep_last_n=2, dry_run=False)

    assert sorted(removed) == ["v1", "v3"]
    remaining = sorted(p.name for p in model_dir.glob("v*.pkl"))
    assert remaining == ["v2.pkl", "v4.pkl", "v5.pkl"]
    assert (model_dir / "champion.pkl").exists()
    assert (model_dir / "champion.json").exists()


def test_prune_without_champion_file(model_dir) -> None:
    """Missing champion.json still prunes to the N newest without crashing."""
    from app.model.ensemble import prune_models

    (model_dir / "champion.pkl").unlink()
    (model_dir / "champion.json").unlink()

    removed = prune_models(keep_last_n=2, dry_run=False)

    assert sorted(removed) == ["v1", "v2", "v3"]
    assert sorted(p.name for p in model_dir.glob("v*.pkl")) == ["v4.pkl", "v5.pkl"]
