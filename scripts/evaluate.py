"""Offline model-honesty evaluation harness.

Runs temporal-split, cross-dataset, and feature-ablation studies on the public
datasets using the existing loaders, and reports held-out metrics at the default
and calibrated decision thresholds plus a precision/recall table.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from app.model.calibration import precision_recall_table, select_threshold
from app.model.ensemble import train_initial_model
from app.schema.datasets import load_cicids2017, load_combined_datasets, load_unsw_nb15
from app.schema.features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

TEMPORAL_TRAIN_DAYS = ("Monday", "Tuesday")
TEMPORAL_EVAL_DAYS = ("Thursday", "Friday")


def build_temporal_dirs(cicids_dir: Path, work_dir: Path) -> tuple[Path, Path]:
    """Split per-day CICIDS2017 files into train/eval symlink directories.

    Args:
        cicids_dir: Directory containing the official per-day CSV files.
        work_dir: Scratch directory for the two symlink subdirectories.

    Returns:
        ``(train_dir, eval_dir)`` — train holds Mon/Tue symlinks, eval Thu/Fri;
        files matching neither group are skipped with a warning.
    """
    train_dir = work_dir / "temporal_train"
    eval_dir = work_dir / "temporal_eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(cicids_dir.glob("*.csv")):
        if csv_file.name.startswith(TEMPORAL_TRAIN_DAYS):
            target = train_dir
        elif csv_file.name.startswith(TEMPORAL_EVAL_DAYS):
            target = eval_dir
        else:
            logger.warning(f"Skipping unrecognized day file: {csv_file.name}")
            continue
        link = target / csv_file.name
        if not link.exists():
            link.symlink_to(csv_file.resolve())

    return train_dir, eval_dir


def zero_feature_columns(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Return a copy of ``X`` with the named feature columns zeroed.

    Args:
        X: Feature matrix with columns ordered like ``FEATURE_NAMES``.
        names: Feature names to zero.

    Returns:
        A new array; the input is not modified.
    """
    indices = [FEATURE_NAMES.index(name) for name in names]
    out = X.copy()
    out[:, indices] = 0.0
    return out


def _counts(y: np.ndarray) -> dict[str, int]:
    return {"rows": int(len(y)), "attacks": int(y.sum())}


def _metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "auc": float(auc)}


def _train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    k_folds: int = 5,
) -> dict:
    """Train an ensemble and report held-out metrics at default and calibrated thresholds."""
    ensemble = train_initial_model(X_train, y_train, k_folds=k_folds)
    y_proba = ensemble.predict_proba(X_eval)
    threshold = select_threshold(y_eval, y_proba)

    importance = ensemble.get_feature_importance()
    top8 = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:8]

    return {
        "train_counts": _counts(y_train),
        "eval_counts": _counts(y_eval),
        "metrics_at_default": _metrics(y_eval, y_proba, 0.5),
        "metrics_at_calibrated": _metrics(y_eval, y_proba, threshold),
        "calibrated_threshold": float(threshold),
        "pr_table": precision_recall_table(y_eval, y_proba),
        "feature_importance_top8": [[name, float(value)] for name, value in top8],
    }


def run_study(
    mode: str,
    cicids_dir: Path,
    unsw_dir: Path,
    cicids_sample: float = 0.1,
    unsw_sample: float = 0.5,
    zero_features: list[str] | None = None,
    k_folds: int = 5,
    random_state: int = 42,
    work_dir: Path | None = None,
) -> dict:
    """Run one honesty study and return its results as a JSON-serializable dict.

    Args:
        mode: ``temporal`` (train Mon/Tue, eval Thu/Fri on CICIDS), ``cross``
            (train each dataset, eval on the other), or ``zero-features``
            (combined train with named feature columns zeroed).
        cicids_dir: Directory containing CICIDS2017 CSV files.
        unsw_dir: Directory containing UNSW-NB15 CSV files.
        cicids_sample: Sampling fraction for CICIDS2017.
        unsw_sample: Sampling fraction for UNSW-NB15.
        zero_features: Feature names to zero in ``zero-features`` mode.
        k_folds: Stacking folds for the trained ensemble.
        random_state: Random seed for sampling and splitting.
        work_dir: Scratch directory for symlink dirs (temporal mode only).

    Returns:
        Study results including counts, metrics, PR table, and importances.
    """
    logger.info(f"Running study: mode={mode}")

    if mode == "temporal":
        work = work_dir or Path(tempfile.mkdtemp(prefix="mirage_temporal_"))
        train_dir, eval_dir = build_temporal_dirs(cicids_dir, work)
        X_train, y_train = load_cicids2017(train_dir, sample_frac=cicids_sample, random_state=random_state)
        X_eval, y_eval = load_cicids2017(eval_dir, sample_frac=cicids_sample, random_state=random_state)
        result = _train_and_evaluate(X_train, y_train, X_eval, y_eval, k_folds=k_folds)
        result["mode"] = mode
        return result

    if mode == "cross":
        result = {"mode": mode}
        logger.info("Cross sub-run: train CICIDS2017, eval UNSW-NB15")
        X_train, y_train = load_cicids2017(cicids_dir, sample_frac=cicids_sample, random_state=random_state)
        X_eval, y_eval = load_unsw_nb15(unsw_dir, sample_frac=unsw_sample, random_state=random_state)
        result["cicids_to_unsw"] = _train_and_evaluate(X_train, y_train, X_eval, y_eval, k_folds=k_folds)

        logger.info("Cross sub-run: train UNSW-NB15, eval CICIDS2017")
        X_train, y_train = load_unsw_nb15(unsw_dir, sample_frac=unsw_sample, random_state=random_state)
        X_eval, y_eval = load_cicids2017(cicids_dir, sample_frac=cicids_sample, random_state=random_state)
        result["unsw_to_cicids"] = _train_and_evaluate(X_train, y_train, X_eval, y_eval, k_folds=k_folds)
        return result

    if mode == "zero-features":
        names = zero_features or []
        X, y = load_combined_datasets(
            cicids_dir=cicids_dir,
            unsw_dir=unsw_dir,
            cicids_sample=cicids_sample,
            unsw_sample=unsw_sample,
            random_state=random_state,
        )
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        if names:
            X_train = zero_feature_columns(X_train, names)
            X_eval = zero_feature_columns(X_eval, names)
        result = _train_and_evaluate(X_train, y_train, X_eval, y_eval, k_folds=k_folds)
        result["mode"] = mode
        result["zeroed_features"] = names
        return result

    raise ValueError(f"Unknown mode: {mode!r} (expected temporal, cross, or zero-features)")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the evaluation harness."""
    parser = argparse.ArgumentParser(description="Run a model-honesty evaluation study")
    parser.add_argument("--mode", choices=["temporal", "cross", "zero-features"], required=True)
    parser.add_argument(
        "--cicids-dir", type=Path, default=DATA_ROOT / "raw" / "cicids2017", help="CICIDS2017 CSV directory"
    )
    parser.add_argument(
        "--unsw-dir", type=Path, default=DATA_ROOT / "raw" / "unsw-nb15", help="UNSW-NB15 CSV directory"
    )
    parser.add_argument("--cicids-sample", type=float, default=0.1, help="CICIDS2017 sampling fraction")
    parser.add_argument("--unsw-sample", type=float, default=0.5, help="UNSW-NB15 sampling fraction")
    parser.add_argument(
        "--zero-features",
        type=str,
        default="method_get,method_post,method_other",
        help="Comma-separated feature names to zero in zero-features mode",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="Stacking folds")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the JSON results")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one study from the CLI and print/save JSON results."""
    args = build_parser().parse_args(argv)

    result = run_study(
        mode=args.mode,
        cicids_dir=args.cicids_dir,
        unsw_dir=args.unsw_dir,
        cicids_sample=args.cicids_sample,
        unsw_sample=args.unsw_sample,
        zero_features=[name.strip() for name in args.zero_features.split(",") if name.strip()],
        k_folds=args.k_folds,
    )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        logger.info(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
