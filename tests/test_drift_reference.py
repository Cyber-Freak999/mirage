"""Tests for drift reference built from real training data with name-based columns."""

import numpy as np

from app.retrain.drift import DriftMonitor


def test_set_reference_maps_columns_by_name() -> None:
    """Tracked feature stores its own column, not the positional rank column."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(50, 25)).astype(float)
    # Give column 7 a distinctive distribution.
    X[:, 7] = rng.normal(loc=100.0, scale=1.0, size=50)
    names = [f"feat_{i}" for i in range(25)]
    importance = {f"feat_{i}": 0.01 for i in range(25)}
    importance["feat_7"] = 0.9  # top-ranked

    monitor = DriftMonitor(top_k_features=3)
    monitor.set_reference(X, names, importance)

    assert monitor.tracked_features[0] == "feat_7"
    np.testing.assert_array_equal(monitor.reference_distributions["feat_7"], X[:, 7])


def test_check_drift_uses_name_mapping() -> None:
    """PSI is high for the shifted named column, low elsewhere."""
    rng = np.random.RandomState(42)
    X_ref = rng.normal(size=(300, 25)).astype(float)
    names = [f"feat_{i}" for i in range(25)]
    importance = {f"feat_{i}": 0.04 for i in range(25)}

    monitor = DriftMonitor(top_k_features=25, min_samples=50)
    monitor.set_reference(X_ref, names, importance)

    X_cur = rng.normal(size=(300, 25)).astype(float)
    X_cur[:, 4] = rng.normal(loc=10.0, scale=1.0, size=300)  # shift column 4

    result = monitor.check_drift(X_cur, feature_names=names)

    assert result.psi_scores["feat_4"] > 0.25
    assert result.psi_scores["feat_0"] < 0.25


def test_load_reference_roundtrip(tmp_path) -> None:
    """set_reference persists to disk; a fresh monitor loads it back."""
    import app.retrain.drift as drift_mod

    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    monkey_npz = ref_dir / "reference_distributions.npz"
    monkey_json = ref_dir / "tracked_features.json"
    orig_ref = drift_mod.REFERENCE_DIR
    drift_mod.REFERENCE_DIR = ref_dir
    try:
        rng = np.random.RandomState(42)
        X = rng.normal(size=(50, 25)).astype(float)
        names = [f"feat_{i}" for i in range(25)]
        importance = {f"feat_{i}": 0.04 for i in range(25)}

        monitor = DriftMonitor(top_k_features=5)
        monitor.set_reference(X, names, importance)
        assert monkey_npz.exists()
        assert monkey_json.exists()

        fresh = DriftMonitor(top_k_features=5)
        assert fresh.load_reference() is True
        assert fresh.tracked_features == monitor.tracked_features
    finally:
        drift_mod.REFERENCE_DIR = orig_ref
