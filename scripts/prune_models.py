"""Prune archived model versions beyond the retention window.

Keeps the live champion plus the N newest versioned pairs; deletes the rest.
Strictly opt-in (spec section 9 never deletes by default) — use ``--dry-run``
to preview. Champion ``champion.pkl/json`` files are never touched.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KEEP_LAST = 3


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for prune_models."""
    parser = argparse.ArgumentParser(description="Prune archived model versions (keep champion + newest N)")
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST, help="Newest versioned pairs to keep")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without deleting anything")
    parser.add_argument("--model-dir", type=Path, default=None, help="Model directory (default: data/models)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run pruning from the command line."""
    args = build_parser().parse_args(argv)
    if args.keep_last < 0:
        logger.error("--keep-last must be >= 0")
        return 1
    if args.model_dir is not None:
        import app.model.ensemble as ensemble_mod

        ensemble_mod.MODEL_DIR = args.model_dir
    from app.model.ensemble import prune_models

    removed = prune_models(keep_last_n=args.keep_last, dry_run=args.dry_run)
    if args.dry_run:
        logger.info(f"Would remove {len(removed)} version(s)")
    else:
        logger.info(f"Removed {len(removed)} version(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
