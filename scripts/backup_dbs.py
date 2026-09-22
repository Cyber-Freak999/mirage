"""Hot-backup SQLite databases with timestamped, pruned backup sets.

Uses the SQLite backup API for consistent snapshots without stopping any
service. Backs up the known Mirage databases present in --db-dir; missing
ones are skipped. Oldest sets beyond --keep-last are deleted.
"""

import argparse
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
KNOWN_DBS = ("honeypot.db", "validation.db", "drift.db", "review.db", "scheduler.db", "dashboard.db")
DEFAULT_KEEP_LAST = 7


def backup_one(source: Path, dest: Path) -> None:
    """Copy one SQLite database via the online backup API.

    Args:
        source: Live database file.
        dest: Destination path (parent must exist).
    """
    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for backup_dbs."""
    parser = argparse.ArgumentParser(description="Hot-backup Mirage SQLite databases")
    parser.add_argument("--db-dir", type=Path, default=DATA_ROOT, help="Directory holding live databases")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "backups", help="Backup set destination")
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST, help="Backup sets to retain")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing anything")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the backup from the command line."""
    args = build_parser().parse_args(argv)
    if args.keep_last < 1:
        logger.error("--keep-last must be >= 1")
        return 1
    present = [name for name in KNOWN_DBS if (args.db_dir / name).exists()]
    missing = [name for name in KNOWN_DBS if name not in present]
    for name in missing:
        logger.info(f"Skipping missing database: {name}")
    stamp = time.strftime("backup-%Y%m%dT%H%M%S", time.gmtime())
    if args.dry_run:
        logger.info(f"Would back up {len(present)} database(s) to {args.out_dir / stamp}")
        return 0
    dest_dir = args.out_dir / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in present:
        backup_one(args.db_dir / name, dest_dir / name)
    logger.info(f"Backed up {len(present)} database(s) to {dest_dir}")

    sets = sorted(p for p in args.out_dir.iterdir() if p.is_dir() and p.name.startswith("backup-"))
    for stale in sets[: max(len(sets) - args.keep_last, 0)]:
        shutil.rmtree(stale)
        logger.info(f"Pruned old backup set: {stale.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
