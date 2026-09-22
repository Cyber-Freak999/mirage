"""Champion/challenger validation for model promotion."""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ..model.calibration import select_threshold
from ..model.ensemble import StackingEnsemble, load_model, save_model
from ..schema.datasets import load_combined_datasets

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
VALIDATION_DB = DATA_DIR / "validation.db"
VALIDATION_SET_PATH = DATA_DIR / "validation_set.npz"


@dataclass
class ValidationResult:
    """Result of champion/challenger validation."""

    timestamp: float
    challenger_version: str
    champion_version: str
    challenger_metrics: dict[str, float]
    champion_metrics: dict[str, float]
    promoted: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ValidationResult":
        return cls(
            timestamp=row["timestamp"],
            challenger_version=row["challenger_version"],
            champion_version=row["champion_version"],
            challenger_metrics=json.loads(row["challenger_metrics"]),
            champion_metrics=json.loads(row["champion_metrics"]),
            promoted=bool(row["promoted"]),
            reason=row["reason"],
        )


def init_validation_db():
    """Initialize validation database."""
    conn = sqlite3.connect(str(VALIDATION_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            challenger_version TEXT NOT NULL,
            champion_version TEXT NOT NULL,
            challenger_metrics TEXT NOT NULL,
            champion_metrics TEXT NOT NULL,
            promoted BOOLEAN NOT NULL,
            reason TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_validation_set(
    cicids_dir: Path | None = None,
    unsw_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cicids_sample: float = 1.0,
    unsw_sample: float = 1.0,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create and persist a fixed held-out validation set.

    Splits combined public data into stratified train/validation portions,
    persists the validation portion to ``VALIDATION_SET_PATH``, and returns
    ``(X_train, y_train, X_val, y_val)`` so training can use the same
    split without label leakage through the validation set.

    The persisted set is the fixed gate all promotions are judged against;
    it is never silently replaced — pass ``overwrite=True`` to regenerate.

    Args:
        cicids_dir: Directory containing CICIDS2017 CSV files.
        unsw_dir: Directory containing UNSW-NB15 CSV files.
        test_size: Fraction of data held out for validation.
        random_state: Random seed for sampling and splitting.
        cicids_sample: Fraction of CICIDS2017 rows to load.
        unsw_sample: Fraction of UNSW-NB15 rows to load.
        overwrite: Allow replacing an existing persisted validation set.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).

    Raises:
        FileExistsError: If the validation set exists and ``overwrite`` is false.
    """
    from sklearn.model_selection import train_test_split

    if VALIDATION_SET_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"Validation set {VALIDATION_SET_PATH} exists;"
            " pass overwrite=True to regenerate (invalidates prior gate comparisons)"
        )

    X, y = load_combined_datasets(cicids_dir, unsw_dir, cicids_sample, unsw_sample, random_state)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    np.savez(VALIDATION_SET_PATH, X=X_val, y=y_val)
    logger.info(
        f"Created validation set: {len(X_val)} samples ({y_val.sum()} attacks); " f"{len(X_train)} samples for training"
    )
    return X_train, y_train, X_val, y_val


def load_validation_set() -> tuple[np.ndarray, np.ndarray]:
    """Load the fixed held-out validation set."""
    if not VALIDATION_SET_PATH.exists():
        raise FileNotFoundError("Validation set not found. Run create_validation_set() first.")

    data = np.load(VALIDATION_SET_PATH)
    return data["X"], data["y"]


def evaluate_model(model: StackingEnsemble, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Evaluate model on validation set at its deployment threshold."""
    y_pred_proba = model.predict_proba(X)
    threshold = getattr(model, "decision_threshold", 0.5)
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, y_pred_proba)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


class ChampionChallenger:
    """Champion/challenger validation gate."""

    def __init__(self):
        init_validation_db()

    def validate(
        self,
        challenger: StackingEnsemble,
        champion: StackingEnsemble | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        calibrate: bool = True,
    ) -> ValidationResult:
        """Validate challenger against champion on held-out set.

        Args:
            challenger: Candidate model.
            champion: Reigning model; loaded from disk when ``None``.
            X_val: Held-out features; loaded from the persisted set when ``None``.
            y_val: Held-out labels; loaded from the persisted set when ``None``.
            calibrate: When true, tune the challenger's decision threshold on the
                held-out set before evaluating, so the gate compares models at
                their deployment thresholds.

        Returns:
            The validation result including the promotion decision.
        """
        if X_val is None or y_val is None:
            X_val, y_val = load_validation_set()

        assert challenger.version is not None, "challenger has no version metadata"
        logger.info(f"Validating challenger {challenger.version.version_id} on {len(X_val)} samples")

        if calibrate:
            challenger.version.decision_threshold = select_threshold(y_val, challenger.predict_proba(X_val))
            logger.info(f"Calibrated challenger decision_threshold={challenger.version.decision_threshold:.2f}")

        challenger_metrics = evaluate_model(challenger, X_val, y_val)
        challenger.version.validation_metrics = dict(challenger_metrics)

        if champion is None:
            try:
                champion = load_model("champion")
            except FileNotFoundError:
                logger.info("No champion model exists. Promoting challenger as first model.")
                result = ValidationResult(
                    timestamp=time.time(),
                    challenger_version=challenger.version.version_id,
                    champion_version="none",
                    challenger_metrics=challenger_metrics,
                    champion_metrics={},
                    promoted=True,
                    reason="First model - no champion to compare",
                )
                self._log_validation(result)
                save_model(challenger)
                return result

        champion_metrics = evaluate_model(champion, X_val, y_val)

        promoted = self._should_promote(challenger_metrics, champion_metrics)

        if promoted:
            reason = f"Challenger beats champion: F1 {challenger_metrics['f1']:.4f} > {champion_metrics['f1']:.4f}"
        else:
            reason = (
                "Challenger does not beat champion: "
                f"F1 {challenger_metrics['f1']:.4f} <= {champion_metrics['f1']:.4f}"
            )

        result = ValidationResult(
            timestamp=time.time(),
            challenger_version=challenger.version.version_id,
            champion_version=champion.version.version_id if champion.version else "unknown",
            challenger_metrics=challenger_metrics,
            champion_metrics=champion_metrics,
            promoted=promoted,
            reason=reason,
        )

        self._log_validation(result)

        if promoted:
            logger.info(f"CHALLENGER PROMOTED: {reason}")
            save_model(challenger)
        else:
            logger.info(f"Challenger rejected: {reason}")

        return result

    def _should_promote(self, challenger: dict, champion: dict) -> bool:
        """Determine if challenger should be promoted.

        Promotion criteria: challenger must meet or beat champion on F1.
        Can be extended to require improvement on multiple metrics.
        """
        return bool(challenger["f1"] >= champion["f1"])

    def _log_validation(self, result: ValidationResult):
        """Log validation result to database."""
        conn = sqlite3.connect(str(VALIDATION_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            INSERT INTO validations
            (timestamp, challenger_version, champion_version, challenger_metrics,
             champion_metrics, promoted, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.timestamp,
                result.challenger_version,
                result.champion_version,
                json.dumps(result.challenger_metrics),
                json.dumps(result.champion_metrics),
                result.promoted,
                result.reason,
            ),
        )
        conn.commit()
        conn.close()

    def get_validation_history(self, limit: int = 50) -> list[ValidationResult]:
        """Get validation history."""
        conn = sqlite3.connect(str(VALIDATION_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM validations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [ValidationResult.from_row(row) for row in rows]


def run_full_retrain(
    X_new: np.ndarray,
    y_new: np.ndarray,
    cicids_dir: Path | None = None,
    unsw_dir: Path | None = None,
    k_folds: int = 5,
    cicids_sample: float = 1.0,
    unsw_sample: float = 1.0,
    random_state: int = 42,
) -> tuple[StackingEnsemble, ValidationResult]:
    """Run full retraining pipeline: train challenger, validate, promote if better.

    Args:
        X_new: New honeypot feature matrix (attack-only, label 1).
        y_new: Labels for the new data.
        cicids_dir: Directory containing CICIDS2017 CSV files.
        unsw_dir: Directory containing UNSW-NB15 CSV files.
        k_folds: Stacking folds for the challenger.
        cicids_sample: Fraction of CICIDS2017 rows to load (cap for small hosts).
        unsw_sample: Fraction of UNSW-NB15 rows to load (cap for small hosts).
        random_state: Seed for sampling, splitting, and training.
    """
    from ..model.ensemble import train_initial_model

    if cicids_dir and cicids_dir.exists() or unsw_dir and unsw_dir.exists():
        X_public, y_public = load_combined_datasets(cicids_dir, unsw_dir, cicids_sample, unsw_sample)
        X_combined = np.vstack([X_public, X_new])
        y_combined = np.concatenate([y_public, y_new])
    else:
        X_combined = X_new
        y_combined = y_new

    logger.info(f"Retraining on {len(X_combined)} total samples ({y_combined.sum()} attacks)")

    challenger = train_initial_model(X_combined, y_combined, k_folds=k_folds, random_state=random_state)

    validator = ChampionChallenger()
    result = validator.validate(challenger)

    return challenger, result
