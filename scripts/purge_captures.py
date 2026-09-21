"""Purge old raw capture payloads per the retention policy.

Nulls ``raw_request``/``headers_json`` on honeypot rows older than N days
(default 30). Structured columns (method/path/query/attack_type) and all
training artifacts are kept — purged rows still convert to features via the
degraded path. See README "Capture-data retention".
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB = DATA_ROOT / "honeypot.db"
DEFAULT_DAYS = 30


def count_purgeable(db_path: Path, cutoff: float) -> int:
    """Count rows with payloads older than ``cutoff``.

    Args:
        db_path: Honeypot SQLite database.
        cutoff: UNIX timestamp; rows older than this qualify.

    Returns:
        Number of rows that would be purged.
    """
    conn = sqlite3.connect(str(db_path))
    count = conn.execute(
        "SELECT COUNT(*) FROM requests WHERE timestamp < ?"
        " AND (raw_request IS NOT NULL OR headers_json IS NOT NULL)",
        (cutoff,),
    ).fetchone()[0]
    conn.close()
    return count


def purge(db_path: Path, cutoff: float) -> int:
    """Null raw payloads on rows older than ``cutoff``.

    Args:
        db_path: Honeypot SQLite database.
        cutoff: UNIX timestamp; rows older than this are purged.

    Returns:
        Number of rows purged.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "UPDATE requests SET raw_request = NULL, headers_json = NULL"
        " WHERE timestamp < ? AND (raw_request IS NOT NULL OR headers_json IS NOT NULL)",
        (cutoff,),
    )
    purged = cursor.rowcount
    conn.commit()
    conn.close()
    return purged


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for purge_captures."""
    parser = argparse.ArgumentParser(description="Purge old raw honeypot payloads (retention policy)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Honeypot SQLite database")
    parser.add_argument("--days", type=float, default=DEFAULT_DAYS, help="Purge payloads older than DAYS")
    parser.add_argument("--dry-run", action="store_true", help="Report the count without modifying anything")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the purge from the command line."""
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1
    cutoff = time.time() - args.days * 86400
    if args.dry_run:
        logger.info(f"Would purge {count_purgeable(args.db, cutoff)} rows older than {args.days} days")
        return 0
    purged = purge(args.db, cutoff)
    logger.info(f"Purged raw payloads on {purged} rows older than {args.days} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
