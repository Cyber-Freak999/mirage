"""Tests for the model-honesty evaluation harness."""

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.evaluate as ev
from app.schema.features import FEATURE_NAMES


def test_zero_feature_columns() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, len(FEATURE_NAMES))).astype(np.float32)

    zeroed = ev.zero_feature_columns(X, ["method_get", "method_post"])

    idx_get = FEATURE_NAMES.index("method_get")
    idx_post = FEATURE_NAMES.index("method_post")
    np.testing.assert_array_equal(zeroed[:, idx_get], np.zeros(20))
    np.testing.assert_array_equal(zeroed[:, idx_post], np.zeros(20))
    np.testing.assert_array_equal(zeroed[:, :idx_get], X[:, :idx_get])
    assert zeroed.shape == X.shape


def test_zero_feature_columns_noop_on_copy() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, len(FEATURE_NAMES))).astype(np.float32)
    original = X.copy()

    ev.zero_feature_columns(X, ["path_length"])

    np.testing.assert_array_equal(X, original)


def test_build_temporal_dirs(tmp_path: Path) -> None:
    for name in (
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "README.txt").write_text("x", encoding="utf-8")

    train_dir, eval_dir = ev.build_temporal_dirs(tmp_path, tmp_path / "work")

    train_files = sorted(p.name for p in train_dir.iterdir())
    eval_files = sorted(p.name for p in eval_dir.iterdir())
    assert train_files == [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
    ]
    assert eval_files == [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning.pcap_ISCX.csv",
    ]
    assert (train_dir / train_files[0]).is_symlink()


def test_run_study_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=0.0, size=(120, 25)).astype(np.float32)
    X1 = rng.normal(loc=2.0, size=(120, 25)).astype(np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * 120 + [1] * 120, dtype=int)

    def fake_load(cicids_dir=None, unsw_dir=None, cicids_sample=1.0, unsw_sample=1.0, random_state=42):
        return X, y

    monkeypatch.setattr(ev, "load_combined_datasets", fake_load)

    result = ev.run_study(
        mode="zero-features",
        cicids_dir=tmp_path,
        unsw_dir=tmp_path,
        zero_features=["method_get", "method_post"],
        k_folds=2,
    )

    assert result["mode"] == "zero-features"
    assert result["train_counts"]["rows"] > 0
    assert result["eval_counts"]["rows"] > 0
    assert set(result["metrics_at_default"]) == {"precision", "recall", "f1", "auc"}
    assert 0.05 <= result["calibrated_threshold"] <= 0.95
    assert len(result["pr_table"]) == 19
    assert len(result["feature_importance_top8"]) == 8
    assert result["zeroed_features"] == ["method_get", "method_post"]

    json.dumps(result)


def test_main_writes_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=0.0, size=(120, 25)).astype(np.float32)
    X1 = rng.normal(loc=2.0, size=(120, 25)).astype(np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * 120 + [1] * 120, dtype=int)

    monkeypatch.setattr(ev, "load_combined_datasets", lambda *a, **k: (X, y))

    out_path = tmp_path / "result.json"
    rc = ev.main(
        [
            "--mode",
            "zero-features",
            "--zero-features",
            "method_get",
            "--k-folds",
            "2",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    saved = json.loads(out_path.read_text())
    assert saved["mode"] == "zero-features"
