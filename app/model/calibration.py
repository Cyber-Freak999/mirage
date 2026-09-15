"""Decision-threshold calibration utilities."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)
DEFAULT_PRECISION_FLOOR = 0.6


def _metrics_at(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = "f1",
    grid: np.ndarray | None = None,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
) -> float:
    """Pick the decision threshold that optimizes an objective on labeled scores.

    Args:
        y_true: Binary ground-truth labels.
        y_proba: Positive-class probabilities for the same rows.
        objective: ``"f1"`` maximizes F1; ``"precision_floor"`` picks the lowest
            threshold (max recall) whose precision meets ``precision_floor``,
            falling back to the max-F1 threshold when the floor is unreachable.
        grid: Candidate thresholds; defaults to 0.05..0.95 in steps of 0.05.
        precision_floor: Minimum precision for the ``precision_floor`` objective.

    Returns:
        The selected threshold as a float.
    """
    candidates = THRESHOLD_GRID if grid is None else np.asarray(grid, dtype=float)

    scored = {float(t): _metrics_at(y_true, y_proba, float(t)) for t in candidates}

    if objective == "f1":
        return max(scored, key=lambda t: scored[t]["f1"])
    if objective == "precision_floor":
        eligible = [t for t, m in scored.items() if m["precision"] >= precision_floor]
        if eligible:
            return min(eligible)
        return max(scored, key=lambda t: scored[t]["f1"])
    raise ValueError(f"Unknown objective: {objective!r} (expected 'f1' or 'precision_floor')")


def precision_recall_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Build a precision/recall/F1 table across candidate thresholds.

    Args:
        y_true: Binary ground-truth labels.
        y_proba: Positive-class probabilities for the same rows.
        thresholds: Candidate thresholds; defaults to 0.05..0.95 in steps of 0.05.

    Returns:
        Rows of ``{"threshold", "precision", "recall", "f1"}`` sorted by threshold.
    """
    candidates = THRESHOLD_GRID if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    for t in sorted(float(t) for t in candidates):
        m = _metrics_at(y_true, y_proba, t)
        rows.append({"threshold": t, **m})
    return rows
