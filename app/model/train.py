"""Command-line entry point for training the Mirage baseline model."""

import argparse
import logging
import sys
from pathlib import Path

from ..retrain.champion_challenger import ChampionChallenger, create_validation_set
from .ensemble import save_model, train_initial_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for baseline training."""
    parser = argparse.ArgumentParser(description="Train the Mirage baseline stacking ensemble")
    parser.add_argument(
        "--cicids-dir",
        type=Path,
        default=DATA_ROOT / "raw" / "cicids2017",
        help="Directory containing CICIDS2017 CSV files (default: data/raw/cicids2017)",
    )
    parser.add_argument(
        "--unsw-dir",
        type=Path,
        default=DATA_ROOT / "raw" / "unsw-nb15",
        help="Directory containing UNSW-NB15 CSV files (default: data/raw/unsw-nb15)",
    )
    parser.add_argument("--cicids-sample", type=float, default=0.1, help="Fraction of CICIDS2017 rows to load")
    parser.add_argument("--unsw-sample", type=float, default=0.5, help="Fraction of UNSW-NB15 rows to load")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of stacking folds")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction held out for validation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load public data, train the stacking ensemble, and persist the champion model."""
    args = build_parser().parse_args(argv)

    if not args.cicids_dir.exists():
        logger.warning(f"Directory {args.cicids_dir} not found - skipping CICIDS2017")
    cicids_dir = args.cicids_dir if args.cicids_dir.exists() else None
    unsw_dir = args.unsw_dir if args.unsw_dir.exists() else None

    try:
        X_train, y_train, X_val, y_val = create_validation_set(
            cicids_dir=cicids_dir,
            unsw_dir=unsw_dir,
            test_size=args.val_size,
            random_state=args.random_state,
            cicids_sample=args.cicids_sample,
            unsw_sample=args.unsw_sample,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Could not load datasets: {exc}")
        return 1

    logger.info(f"Training on {len(X_train)} samples ({y_train.sum()} attacks), validating on {len(X_val)}")

    ensemble = train_initial_model(X_train, y_train, k_folds=args.k_folds)

    result = ChampionChallenger().validate(ensemble, X_val=X_val, y_val=y_val)

    if result.promoted:
        save_model(ensemble)
        try:
            from ..retrain.drift import DriftMonitor
            from ..schema.features import FEATURE_NAMES

            DriftMonitor(top_k_features=8).set_reference(X_train, FEATURE_NAMES, ensemble.version.feature_importance)
            logger.info("Persisted drift reference from training data")
        except Exception:
            logger.exception("Failed to persist drift reference")

    v = result.challenger_metrics
    logger.info(
        f"Baseline ready: version={ensemble.version.version_id} "
        f"promoted={result.promoted} val_f1={v['f1']:.4f} val_precision={v['precision']:.4f} "
        f"val_recall={v['recall']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
