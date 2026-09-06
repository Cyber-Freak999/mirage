"""Champion/challenger validation for model promotion."""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

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
) -> tuple[np.ndarray, np.ndarray]:
    """Create and save a fixed held-out validation set."""
    from sklearn.model_selection import train_test_split

    X, y = load_combined_datasets(cicids_dir, unsw_dir, random_state=random_state)
    _, X_val, _, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    np.savez(VALIDATION_SET_PATH, X=X_val, y=y_val)
    logger.info(f"Created validation set: {len(X_val)} samples ({y_val.sum()} attacks)")
    return X_val, y_val


def load_validation_set() -> tuple[np.ndarray, np.ndarray]:
    """Load the fixed held-out validation set."""
    if not VALIDATION_SET_PATH.exists():
        raise FileNotFoundError("Validation set not found. Run create_validation_set() first.")

    data = np.load(VALIDATION_SET_PATH)
    return data["X"], data["y"]


def evaluate_model(model: StackingEnsemble, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Evaluate model on validation set."""
    y_pred_proba = model.predict_proba(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)

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
    ) -> ValidationResult:
        """Validate challenger against champion on held-out set."""
        if X_val is None or y_val is None:
            X_val, y_val = load_validation_set()

        logger.info(f"Validating challenger {challenger.version.version_id} on {len(X_val)} samples")

        challenger_metrics = evaluate_model(challenger, X_val, y_val)

        if champion is None:
            try:
                champion = load_model("champion")
            except FileNotFoundError:
                logger.info("No champion model exists. Promoting challenger as first model.")
                return ValidationResult(
                    timestamp=time.time(),
                    challenger_version=challenger.version.version_id,
                    champion_version="none",
                    challenger_metrics=challenger_metrics,
                    champion_metrics={},
                    promoted=True,
                    reason="First model - no champion to compare",
                )

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
        return challenger["f1"] >= champion["f1"]

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
) -> tuple[StackingEnsemble, ValidationResult]:
    """Run full retraining pipeline: train challenger, validate, promote if better."""
    from ..model.ensemble import train_initial_model

    if cicids_dir and cicids_dir.exists() or unsw_dir and unsw_dir.exists():
        X_public, y_public = load_combined_datasets(cicids_dir, unsw_dir)
        X_combined = np.vstack([X_public, X_new])
        y_combined = np.concatenate([y_public, y_new])
    else:
        X_combined = X_new
        y_combined = y_new

    logger.info(f"Retraining on {len(X_combined)} total samples ({y_combined.sum()} attacks)")

    challenger = train_initial_model(X_combined, y_combined, k_folds=k_folds)

    validator = ChampionChallenger()
    result = validator.validate(challenger)

    return challenger, result
