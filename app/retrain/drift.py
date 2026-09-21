"""Feature distribution drift monitoring using PSI (Population Stability Index)."""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DRIFT_DB = DATA_DIR / "drift.db"
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def init_drift_db():
    """Initialize drift monitoring database."""
    conn = sqlite3.connect(str(DRIFT_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            sample_size INTEGER NOT NULL,
            psi_scores TEXT NOT NULL,
            max_psi REAL NOT NULL,
            triggered BOOLEAN NOT NULL,
            tracked_features TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            feature_name TEXT NOT NULL,
            mean REAL NOT NULL,
            std REAL NOT NULL,
            min_val REAL NOT NULL,
            max_val REAL NOT NULL,
            sample_size INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, epsilon: float = 1e-6) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Args:
        reference: Reference distribution (training data)
        current: Current distribution (new data)
        bins: Number of bins for histogram
        epsilon: Small value to avoid log(0)

    Returns:
        PSI value (0 = no drift, >0.25 = significant drift)
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Use percentiles for bin edges to handle outliers
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, percentiles)

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    ref_perc = ref_hist / len(reference)
    cur_perc = cur_hist / len(current)

    ref_perc = np.maximum(ref_perc, epsilon)
    cur_perc = np.maximum(cur_perc, epsilon)

    psi = np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc))
    return float(psi)


@dataclass
class DriftResult:
    """Result of a drift check."""

    timestamp: float
    sample_size: int
    psi_scores: dict[str, float]
    max_psi: float
    triggered: bool
    tracked_features: list[str]
    id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DriftMonitor:
    """Monitors feature distribution drift using PSI on top-k important features."""

    def __init__(
        self,
        psi_threshold: float = 0.25,
        top_k_features: int = 8,
        min_samples: int = 200,
        reference_window: int = 10000,
    ):
        self.psi_threshold = psi_threshold
        self.top_k_features = top_k_features
        self.min_samples = min_samples
        self.reference_window = reference_window

        self.reference_distributions: dict[str, np.ndarray] = {}
        self.tracked_features: list[str] = []

        init_drift_db()

    def set_reference(self, X: np.ndarray, feature_names: list[str], importance: dict[str, float]):
        """Set reference distributions from training data.

        Columns are resolved by feature name (via ``feature_names``, which must
        list the schema names in column order), not by importance rank position.

        Args:
            X: Training feature matrix with columns in ``feature_names`` order.
            feature_names: Schema feature name per column of ``X``.
            importance: Feature-name → importance score mapping.
        """
        self.tracked_features = sorted(importance.keys(), key=lambda f: importance.get(f, 0), reverse=True)[
            : self.top_k_features
        ]

        logger.info(f"Tracking drift on top {len(self.tracked_features)} features: {self.tracked_features}")

        col_index = {name: idx for idx, name in enumerate(feature_names)}
        for feat in self.tracked_features:
            idx = col_index.get(feat)
            if idx is not None and idx < X.shape[1]:
                self.reference_distributions[feat] = X[:, idx].copy()
            else:
                logger.warning(f"No column for tracked feature {feat!r}; skipping")

        self._save_reference(feature_names)

    def _save_reference(self, feature_names: list[str]):
        """Save reference distributions to disk."""
        ref_path = REFERENCE_DIR / "reference_distributions.npz"
        feat_path = REFERENCE_DIR / "tracked_features.json"

        np.savez(ref_path, **self.reference_distributions)
        with open(feat_path, "w") as f:
            json.dump(
                {
                    "features": self.tracked_features,
                    "feature_names": feature_names,
                    "timestamp": time.time(),
                },
                f,
            )

    def load_reference(self) -> bool:
        """Load reference distributions from disk."""
        ref_path = REFERENCE_DIR / "reference_distributions.npz"
        feat_path = REFERENCE_DIR / "tracked_features.json"

        if not ref_path.exists() or not feat_path.exists():
            return False

        data = np.load(ref_path)
        self.reference_distributions = {k: data[k] for k in data.files}

        with open(feat_path) as f:
            meta = json.load(f)
        self.tracked_features = meta["features"]

        logger.info(f"Loaded reference for {len(self.tracked_features)} features")
        return True

    def check_drift(self, X: np.ndarray, feature_names: list[str] | None = None) -> DriftResult:
        """Check for drift in current batch against reference.

        Args:
            X: Current batch feature matrix.
            feature_names: Optional schema names in column order; when given,
                tracked features resolve by name, otherwise positionally
                (backward compatible).
        """
        if not self.tracked_features or not self.reference_distributions:
            if not self.load_reference():
                raise RuntimeError("No reference distributions available. Train model first.")

        if X.shape[0] < self.min_samples:
            return DriftResult(
                timestamp=time.time(),
                sample_size=X.shape[0],
                psi_scores={},
                max_psi=0.0,
                triggered=False,
                tracked_features=self.tracked_features,
            )

        col_index = {name: idx for idx, name in enumerate(feature_names)} if feature_names else {}
        psi_scores = {}
        for i, feat in enumerate(self.tracked_features):
            if feature_names is not None:
                idx = col_index.get(feat)
                if idx is None or idx >= X.shape[1] or feat not in self.reference_distributions:
                    psi_scores[feat] = 0.0
                    continue
                cur = X[:, idx]
            else:
                if i >= X.shape[1] or feat not in self.reference_distributions:
                    psi_scores[feat] = 0.0
                    continue
                cur = X[:, i]

            ref = self.reference_distributions[feat]
            psi_scores[feat] = compute_psi(ref, cur)

        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        triggered = max_psi >= self.psi_threshold

        result = DriftResult(
            timestamp=time.time(),
            sample_size=X.shape[0],
            psi_scores=psi_scores,
            max_psi=max_psi,
            triggered=triggered,
            tracked_features=self.tracked_features,
        )

        result.id = self._log_drift_check(result)

        if triggered:
            logger.warning(f"DRIFT DETECTED: max PSI = {max_psi:.4f} (threshold: {self.psi_threshold})")
            for feat, psi in sorted(psi_scores.items(), key=lambda x: x[1], reverse=True):
                if psi > 0.1:
                    logger.warning(f"  {feat}: PSI = {psi:.4f}")

        return result

    def _log_drift_check(self, result: DriftResult) -> int:
        """Log drift check result to database.

        Returns:
            The inserted row id, for linking review entries.
        """
        conn = sqlite3.connect(str(DRIFT_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute(
            """
            INSERT INTO drift_checks (timestamp, sample_size, psi_scores, max_psi, triggered, tracked_features)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.timestamp,
                result.sample_size,
                json.dumps(result.psi_scores),
                result.max_psi,
                result.triggered,
                json.dumps(result.tracked_features),
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def get_drift_history(self, limit: int = 100) -> list[DriftResult]:
        """Get recent drift check history."""
        conn = sqlite3.connect(str(DRIFT_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM drift_checks ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append(
                DriftResult(
                    timestamp=row["timestamp"],
                    sample_size=row["sample_size"],
                    psi_scores=json.loads(row["psi_scores"]),
                    max_psi=row["max_psi"],
                    triggered=bool(row["triggered"]),
                    tracked_features=json.loads(row["tracked_features"]),
                )
            )
        return results
