"""Review gate for human-in-the-loop validation before retraining."""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
REVIEW_DB = DATA_DIR / "review.db"
REVIEW_TIMEOUT_HOURS = 72


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_PROCEEDED = "auto_proceeded"


@dataclass
class ReviewEntry:
    """Entry in the review queue."""

    id: int | None
    timestamp: float
    drift_result_id: int
    max_psi: float
    psi_scores: dict[str, float]
    sample_size: int
    sample_data: dict  # Serialized batch metadata
    status: ReviewStatus
    reviewer: str | None = None
    review_timestamp: float | None = None
    review_notes: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ReviewEntry":
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            drift_result_id=row["drift_result_id"],
            max_psi=row["max_psi"],
            psi_scores=json.loads(row["psi_scores"]),
            sample_size=row["sample_size"],
            sample_data=json.loads(row["sample_data"]),
            status=ReviewStatus(row["status"]),
            reviewer=row["reviewer"],
            review_timestamp=row["review_timestamp"],
            review_notes=row["review_notes"],
        )


def init_review_db():
    """Initialize review gate database."""
    conn = sqlite3.connect(str(REVIEW_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            drift_result_id INTEGER NOT NULL,
            max_psi REAL NOT NULL,
            psi_scores TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            sample_data TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            reviewer TEXT,
            review_timestamp REAL,
            review_notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


class ReviewGate:
    """Manages human review queue for drift-triggered retraining."""

    def __init__(self, timeout_hours: int = REVIEW_TIMEOUT_HOURS):
        self.timeout_hours = timeout_hours
        init_review_db()

    def create_review(
        self,
        drift_result_id: int,
        max_psi: float,
        psi_scores: dict[str, float],
        sample_size: int,
        sample_data: dict,
    ) -> ReviewEntry:
        """Create a new review entry."""
        entry = ReviewEntry(
            id=None,
            timestamp=time.time(),
            drift_result_id=drift_result_id,
            max_psi=max_psi,
            psi_scores=psi_scores,
            sample_size=sample_size,
            sample_data=sample_data,
            status=ReviewStatus.PENDING,
        )

        conn = sqlite3.connect(str(REVIEW_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute(
            """
            INSERT INTO review_queue (timestamp, drift_result_id, max_psi, psi_scores, sample_size, sample_data, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp,
                entry.drift_result_id,
                entry.max_psi,
                json.dumps(entry.psi_scores),
                entry.sample_size,
                json.dumps(entry.sample_data),
                entry.status.value,
            ),
        )
        entry.id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Created review entry {entry.id} for drift check {drift_result_id} (PSI={max_psi:.4f})")
        return entry

    def get_pending_reviews(self) -> list[ReviewEntry]:
        """Get all pending reviews, checking for timeout."""
        self._check_timeouts()

        conn = sqlite3.connect(str(REVIEW_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY timestamp ASC",
            (ReviewStatus.PENDING.value,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [ReviewEntry.from_row(row) for row in rows]

    def get_review(self, review_id: int) -> ReviewEntry | None:
        """Get a specific review by ID."""
        conn = sqlite3.connect(str(REVIEW_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM review_queue WHERE id = ?", (review_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return ReviewEntry.from_row(row)
        return None

    def approve(self, review_id: int, reviewer: str, notes: str = "") -> bool:
        """Approve a review for retraining."""
        return self._update_status(review_id, ReviewStatus.APPROVED, reviewer, notes)

    def reject(self, review_id: int, reviewer: str, notes: str = "") -> bool:
        """Reject a review (skip retraining)."""
        return self._update_status(review_id, ReviewStatus.REJECTED, reviewer, notes)

    def _update_status(
        self,
        review_id: int,
        status: ReviewStatus,
        reviewer: str,
        notes: str,
    ) -> bool:
        conn = sqlite3.connect(str(REVIEW_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute(
            """
            UPDATE review_queue
            SET status = ?, reviewer = ?, review_timestamp = ?, review_notes = ?
            WHERE id = ? AND status = ?
            """,
            (status.value, reviewer, time.time(), notes, review_id, ReviewStatus.PENDING.value),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()

        if updated:
            logger.info(f"Review {review_id} {status.value} by {reviewer}")
        return updated

    def _check_timeouts(self):
        """Auto-proceed reviews that have timed out."""
        timeout_seconds = self.timeout_hours * 3600
        cutoff = time.time() - timeout_seconds

        conn = sqlite3.connect(str(REVIEW_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute(
            """
            UPDATE review_queue
            SET status = ?, review_timestamp = ?, review_notes = ?
            WHERE status = ? AND timestamp < ?
            """,
            (
                ReviewStatus.AUTO_PROCEEDED.value,
                time.time(),
                "Auto-proceeded after 72-hour timeout",
                ReviewStatus.PENDING.value,
                cutoff,
            ),
        )
        conn.commit()
        count = cursor.rowcount
        conn.close()

        if count > 0:
            logger.warning(f"Auto-proceeded {count} timed-out reviews")

    def should_proceed(self, review_id: int) -> bool:
        """Check if retraining should proceed for a review."""
        entry = self.get_review(review_id)
        if not entry:
            return False

        if entry.status == ReviewStatus.APPROVED:
            return True
        if entry.status == ReviewStatus.AUTO_PROCEEDED:
            return True
        if entry.status == ReviewStatus.REJECTED:
            return False
        return False

    def get_review_stats(self) -> dict:
        """Get review queue statistics."""
        conn = sqlite3.connect(str(REVIEW_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT status, COUNT(*) as count FROM review_queue GROUP BY status")
        rows = cursor.fetchall()
        conn.close()

        stats = {row["status"]: row["count"] for row in rows}
        return stats
