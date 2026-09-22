"""APScheduler-based retraining scheduler with persistent job store."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from ..model.ensemble import load_model
from .champion_challenger import ChampionChallenger, ValidationResult, run_full_retrain
from .drift import DriftMonitor, DriftResult
from .review_gate import ReviewGate

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SCHEDULER_DB = DATA_DIR / "scheduler.db"

DEFAULT_MIN_SAMPLES = 200
DEFAULT_MAX_HOURS = 24


@dataclass
class RetrainingConfig:
    """Configuration for retraining scheduler."""

    min_samples: int = DEFAULT_MIN_SAMPLES
    max_hours: int = DEFAULT_MAX_HOURS
    psi_threshold: float = 0.25
    top_k_features: int = 8
    review_timeout_hours: int = 72
    k_folds: int = 5
    cicids_dir: Path | None = None
    unsw_dir: Path | None = None
    cicids_sample: float = 0.05
    unsw_sample: float = 0.25


class RetrainingScheduler:
    """Manages the adaptive retraining loop."""

    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.drift_monitor = DriftMonitor(
            psi_threshold=config.psi_threshold,
            top_k_features=config.top_k_features,
            min_samples=config.min_samples,
        )
        self.review_gate = ReviewGate(timeout_hours=config.review_timeout_hours)
        self.validator = ChampionChallenger()

        self._scheduler: BackgroundScheduler | None = None
        self._running = False
        self._last_check_time = 0
        self._samples_since_check = 0

        self._load_drift_reference()

    def _load_drift_reference(self):
        """Load drift reference from disk, falling back to champion metadata."""
        if self.drift_monitor.load_reference():
            return
        try:
            champion = load_model("champion")
            if champion.version and champion.version.feature_importance:
                logger.warning(
                    "No persisted drift reference found; drift checks will fail until a "
                    "training run persists one. Champion importance keys noted but no "
                    "training matrix is available at runtime, so no placeholder is set."
                )
        except FileNotFoundError:
            logger.warning("No champion model found for drift reference")

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{SCHEDULER_DB}")}
        executors = {"default": ThreadPoolExecutor(max_workers=2)}
        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 300,
        }

        self._scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC",
        )

        self._scheduler.add_job(
            self._check_and_retrain,
            "interval",
            minutes=15,
            id="retrain_check",
            replace_existing=True,
        )

        self._scheduler.start()
        self._running = True
        logger.info("Retraining scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=True)
            self._running = False
            logger.info("Retraining scheduler stopped")

    def _check_and_retrain(self):
        """Main check function: collect new data, check drift, trigger retrain if needed."""
        if self._running:
            logger.debug("Running scheduled drift check")
            try:
                self.check_drift_and_queue()
            except Exception:
                logger.exception("Drift check failed")

    def check_drift_and_queue(self) -> DriftResult | None:
        """Check for drift on new honeypot data."""
        import sqlite3

        from ..honeypot import DB_PATH

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(
            "SELECT * FROM requests WHERE timestamp > ? ORDER BY timestamp",
            (self._last_check_time,),
        )
        new_rows = cursor.fetchall()
        conn.close()

        if not new_rows:
            self.sweep_reviews()
            return None

        self._samples_since_check += len(new_rows)
        self._last_check_time = new_rows[-1]["timestamp"] if new_rows else self._last_check_time

        should_check = (
            self._samples_since_check >= self.config.min_samples
            or (time.time() - self._last_check_time) >= self.config.max_hours * 3600
        )

        if not should_check:
            logger.debug(f"Accumulated {self._samples_since_check} samples, waiting for threshold")
            return None

        logger.info(f"Checking drift on {len(new_rows)} new samples")

        X_new = self._rows_to_features(new_rows)

        drift_result = self.drift_monitor.check_drift(X_new)

        if drift_result.triggered:
            sample_data = {
                "count": len(new_rows),
                "time_range": [new_rows[0]["timestamp"], new_rows[-1]["timestamp"]],
                "attack_types": list(set(r["attack_type"] for r in new_rows)),
            }

            review = self.review_gate.create_review(
                drift_result_id=drift_result.id or 0,
                max_psi=drift_result.max_psi,
                psi_scores=drift_result.psi_scores,
                sample_size=drift_result.sample_size,
                sample_data=sample_data,
            )

            logger.warning(f"Drift triggered review {review.id}. Waiting for approval or timeout.")

        self._samples_since_check = 0
        self.sweep_reviews()
        return drift_result

    def _rows_to_features(self, rows) -> np.ndarray:
        """Convert database rows to feature vectors."""
        from ..schema.capture import rows_to_features

        return rows_to_features(rows)

    def sweep_reviews(self) -> list[int]:
        """Process reviews ready to proceed (approved or auto-proceeded).

        Called on every scheduled tick so manual decisions (API/dashboard)
        take effect within one interval and timed-out reviews auto-proceed
        per the 72-hour policy. Manual decisions apply on the next sweep.

        Returns:
            IDs of reviews for which retraining ran.
        """
        processed = []
        for review in self.review_gate.get_actionable_reviews():
            if review.id is None:
                continue
            if not self.review_gate.should_proceed(review.id):
                continue
            logger.info(f"Sweep: retraining for review {review.id} (status auto-proceeded or approved)")
            self.process_review(review.id, approved=True, reviewer="system-sweep")
            self.review_gate.mark_done(review.id)
            processed.append(review.id)
        return processed

    def process_review(self, review_id: int, approved: bool, reviewer: str = "system") -> ValidationResult | None:
        """Process a review decision and run retraining if approved."""
        if approved:
            self.review_gate.approve(review_id, reviewer)
        else:
            self.review_gate.reject(review_id, reviewer)

        if not self.review_gate.should_proceed(review_id):
            logger.info(f"Review {review_id} not approved. Skipping retraining.")
            return None

        logger.info(f"Starting retraining for review {review_id}")

        import sqlite3

        from ..honeypot import DB_PATH

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM requests ORDER BY timestamp")
        all_rows = cursor.fetchall()
        conn.close()

        X_new = self._rows_to_features(all_rows)
        y_new = np.ones(len(all_rows))

        challenger, result = run_full_retrain(
            X_new,
            y_new,
            cicids_dir=self.config.cicids_dir,
            unsw_dir=self.config.unsw_dir,
            k_folds=self.config.k_folds,
            cicids_sample=self.config.cicids_sample,
            unsw_sample=self.config.unsw_sample,
        )

        if result.promoted:
            from ..api import reload_model

            reload_model()
            logger.info("Model reloaded after promotion")

        return result

    def get_status(self) -> dict:
        """Get scheduler status."""
        pending = self.review_gate.get_pending_reviews()
        drift_history = self.drift_monitor.get_drift_history(10)
        validation_history = self.validator.get_validation_history(10)

        return {
            "running": self._running,
            "samples_since_check": self._samples_since_check,
            "last_check_time": self._last_check_time,
            "pending_reviews": len(pending),
            "reviews": [r.to_dict() for r in pending],
            "recent_drift_checks": [d.to_dict() for d in drift_history],
            "recent_validations": [v.to_dict() for v in validation_history],
        }


_scheduler_instance: RetrainingScheduler | None = None


def start_scheduler(config: RetrainingConfig | None = None) -> RetrainingScheduler:
    """Start the global retraining scheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = RetrainingScheduler(config or RetrainingConfig())
    _scheduler_instance.start()
    return _scheduler_instance


def stop_scheduler():
    """Stop the global retraining scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop()
        _scheduler_instance = None


def get_scheduler() -> RetrainingScheduler | None:
    """Get the global scheduler instance."""
    return _scheduler_instance
