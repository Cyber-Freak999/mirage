"""Entry point for retrainer service."""

import logging
import signal
import sys
import time
from pathlib import Path

from .scheduler import RetrainingConfig, start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

scheduler = None


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    stop_scheduler()
    sys.exit(0)


def build_config() -> RetrainingConfig:
    """Build the retrainer config with real dataset dirs and OOM-safe caps.

    Falls back to honeypot-only (all-label-1) training with a loud warning
    when the public dataset dirs are absent — never silently.
    """
    cicids_dir = DATA_ROOT / "raw" / "cicids2017"
    unsw_dir = DATA_ROOT / "raw" / "unsw-nb15"
    if not cicids_dir.exists():
        logger.warning(f"CICIDS dir {cicids_dir} not found - skipping CICIDS2017")
        cicids_dir = None
    if not unsw_dir.exists():
        logger.warning(f"UNSW dir {unsw_dir} not found - skipping UNSW-NB15")
        unsw_dir = None
    if cicids_dir is None and unsw_dir is None:
        logger.warning("No public datasets found - retraining on honeypot-only all-label-1 data")
    return RetrainingConfig(
        min_samples=200,
        max_hours=24,
        psi_threshold=0.25,
        top_k_features=8,
        review_timeout_hours=72,
        k_folds=5,
        cicids_dir=cicids_dir,
        unsw_dir=unsw_dir,
    )


def main():
    global scheduler

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    config = build_config()

    logger.info("Starting Mirage retrainer service...")
    scheduler = start_scheduler(config)

    try:
        while True:
            time.sleep(60)
            status = scheduler.get_status()
            logger.info(
                f"Scheduler status: {status['pending_reviews']} pending reviews, "
                f"{status['samples_since_check']} samples since last check"
            )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        stop_scheduler()


if __name__ == "__main__":
    main()
