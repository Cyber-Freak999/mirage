"""Download dataset CSVs into data/raw for offline training data preparation."""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_csv(url: str, dest: Path) -> Path:
    """Download a single CSV file.

    Args:
        url: Source URL (http/https/file).
        dest: Destination path; parent directories are created if missing.

    Returns:
        Destination path of the downloaded file.
    """
    if dest.suffix.lower() != ".csv":
        raise ValueError(f"Destination must end in .csv, got: {dest.suffix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    logger.info(f"Downloaded {dest} ({dest.stat().st_size} bytes)")
    return dest


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fetch_csv."""
    parser = argparse.ArgumentParser(description="Download a dataset CSV into data/raw")
    parser.add_argument("--url", required=True, help="Source URL of the CSV file")
    parser.add_argument("--dest", type=Path, required=True, help="Destination .csv path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run fetch_csv from the command line."""
    args = build_parser().parse_args(argv)
    try:
        fetch_csv(args.url, args.dest)
        return 0
    except OSError as exc:
        logger.error(f"Download failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
