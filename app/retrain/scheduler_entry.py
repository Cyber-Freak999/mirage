"""Entry point for retrainer service."""

import logging
import signal
import sys
import time

from .scheduler import RetrainingConfig, start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

scheduler = None


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    stop_scheduler()
    sys.exit(0)


def main():
    global scheduler

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    config = RetrainingConfig(
        min_samples=200,
        max_hours=24,
        psi_threshold=0.25,
        top_k_features=8,
        review_timeout_hours=72,
        k_folds=5,
    )

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
