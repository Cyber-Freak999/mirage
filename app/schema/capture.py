"""Shared converter from honeypot capture rows to the 25-feature schema.

Single source of truth for capture → features, used by the retraining
scheduler and the dashboard drift panel. Handles both current-schema rows
(with ``headers_json``/``content_type``) and legacy rows logged before the
migration (those convert with degraded header fidelity).
"""

import json
import logging
from collections.abc import Iterable
from sqlite3 import Row

import numpy as np

from .features import extract_features

logger = logging.getLogger(__name__)


def _col(row: Row, name: str, default: str = "") -> str:
    """Return a text column value, tolerating partial selects and NULLs."""
    try:
        value = row[name]
    except (IndexError, KeyError):
        return default
    return value if value else default


def rows_to_features(rows: Iterable[Row]) -> np.ndarray:
    """Convert honeypot DB rows to a ``(N, 25)`` feature matrix.

    Args:
        rows: SQLite rows from the ``requests`` table (current or legacy schema).

    Returns:
        Feature matrix with one 25-feature row per input row.
    """
    features = []
    for row in rows:
        columns = row.keys()
        headers: dict[str, str] = {}
        if "headers_json" in columns and _col(row, "headers_json"):
            try:
                stored = json.loads(_col(row, "headers_json"))
                if isinstance(stored, dict):
                    headers = {str(k).lower(): str(v) for k, v in stored.items()}
            except (json.JSONDecodeError, TypeError):
                logger.debug("Unparseable headers_json; falling back to user_agent column")
        if "user-agent" not in headers and _col(row, "user_agent"):
            headers["user-agent"] = _col(row, "user_agent")
        if _col(row, "content_type"):
            headers.setdefault("content-type", _col(row, "content_type"))

        body = _col(row, "raw_request")

        features.append(
            extract_features(
                method=_col(row, "method", "GET"),
                path=_col(row, "path", "/"),
                query_string=_col(row, "query_string"),
                headers=headers,
                body=body,
            )
        )

    if not features:
        return np.zeros((0, 25), dtype=np.float32)
    return np.array(features)
