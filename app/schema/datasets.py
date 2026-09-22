"""Public dataset loaders mapped to unified 25-feature schema.

Flow-to-feature mappers replace the synthetic HTTP request fabrication
(forming method from arbitrary flow thresholds) with direct mappings from
network-flow columns to the 25-feature schema.  Public-dataset method
features are constant 0.0 — there is no true HTTP method in flow data.
Honeypot data continues to use real HTTP extraction via ``extract_features``.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_features

logger = logging.getLogger(__name__)

UNSW_FLOW_HEADERS = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "Label",
]

_HEADER_TOKENS = ("attack_cat", "Label", "label")

UNSW_CHUNK_SIZE = 100_000
CICIDS_CHUNK_SIZE = 100_000

HTTP_DERIVED_FEATURES = {
    "method": "GET",
    "path": "/",
    "query": "",
    "headers": {},
    "body": "",
}


def _is_unsw_header_line(line: str) -> bool:
    """Return whether ``line`` looks like a header row (vs. headless flow data)."""
    return any(token in line for token in _HEADER_TOKENS)


def _unsw_label_column(frame: pd.DataFrame) -> str | None:
    """Return the preferred label column with values in ``frame``, or None."""
    for candidate in ("Label", "attack_cat"):
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    return None


def _binary_labels(series: pd.Series, label_col: str) -> np.ndarray:
    """Convert an UNSW label series to binary 0/1 attack labels."""
    if label_col == "Label":
        return series.astype(int).values
    return series.apply(lambda x: 0 if str(x).lower() == "normal" else 1).values


def _cicids_labels(frame: pd.DataFrame) -> np.ndarray:
    """Return binary labels for a CICIDS frame: 0 for BENIGN, 1 otherwise."""
    return frame["Label"].apply(lambda x: 0 if "BENIGN" in str(x).upper() else 1).values


def load_cicids2017(data_dir: Path, sample_frac: float = 1.0, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load CICIDS2017 dataset and map to unified schema.

    Note: CICIDS2017 is network-flow based, not HTTP-request based.
    This function creates synthetic HTTP-like features from flow features
    as a best-effort mapping. For true HTTP features, use PCAP-based extraction.

    The official ``*.pcap_ISCX.csv`` files carry a leading space on many
    column names (e.g. `` Label``) and occasionally leave numeric cells
    blank; names are stripped on load and blank numerics are coerced to zero.
    Files are read in chunks of :data:`CICIDS_CHUNK_SIZE` rows and sampled
    per chunk, so only ``sample_frac`` of the rows is ever materialized in
    memory.

    Args:
        data_dir: Directory containing CICIDS2017 CSV files
        sample_frac: Fraction of data to sample (for memory)
        random_state: Random seed for sampling

    Returns:
        Tuple of (features_array, labels_array)
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CICIDS2017 CSV files found in {data_dir}")

    rng = np.random.RandomState(random_state)
    feature_parts, label_parts = [], []
    for csv_file in csv_files:
        try:
            file_features, file_labels = [], []
            for chunk in pd.read_csv(csv_file, encoding="utf-8-sig", chunksize=CICIDS_CHUNK_SIZE):
                chunk.columns = chunk.columns.str.strip()
                if "Label" not in chunk.columns:
                    break
                sampled = _sample_chunk(chunk, sample_frac, rng).replace([np.inf, -np.inf], np.nan).fillna(0)
                if len(sampled):
                    # Direct flow-to-feature mapping; method features are 0.0.
                    file_features.append(_cicids_frame_to_features(sampled))
                    file_labels.append(_cicids_labels(sampled))
            if not file_features:
                continue
            features = np.concatenate(file_features)
            labels = np.concatenate(file_labels)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
            continue
        feature_parts.append(features)
        label_parts.append(labels)
        logger.info(f"Loaded {csv_file.name}: {len(features)} sampled rows")

    if not feature_parts:
        raise ValueError("No valid CICIDS2017 CSV files loaded")

    return np.concatenate(feature_parts), np.concatenate(label_parts)


def _sample_chunk(frame: pd.DataFrame, sample_frac: float, rng: np.random.RandomState) -> pd.DataFrame:
    """Select an exact-size random draw from ``frame`` (paired with ``rng``)."""
    if sample_frac >= 1.0:
        return frame
    n_keep = int(round(len(frame) * sample_frac))
    if n_keep <= 0:
        return frame.iloc[0:0]
    indices = rng.choice(len(frame), size=n_keep, replace=False)
    return frame.iloc[indices].reset_index(drop=True)


def _extract_request_features(frame: pd.DataFrame, row_to_http: Callable[[pd.Series], dict]) -> np.ndarray:
    """Build the unified 25-feature matrix for every row in ``frame``."""
    features = np.zeros((len(frame), 25), dtype=np.float32)
    for i, row in frame.iterrows():
        synthetic_request = row_to_http(row)
        features[i] = extract_features(
            method=synthetic_request["method"],
            path=synthetic_request["path"],
            query_string=synthetic_request["query"],
            headers=synthetic_request["headers"],
            body=synthetic_request["body"],
        )
    return features


def load_unsw_nb15(data_dir: Path, sample_frac: float = 1.0, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load UNSW-NB15 dataset and map to unified schema.

    Note: UNSW-NB15 has some HTTP-specific features but is primarily flow-based.
    This creates synthetic HTTP features from available fields.

    The official ``UNSW-NB15_*.csv`` flow files ship without a header row and
    with a UTF-8 BOM; they are read with explicit column names. Files that
    carry a header row (the ``UNSW_NB15_training-set.csv`` / ``testing-set.csv``
    layout) are only used when no headless flow files are present. Other
    bundled reference files (e.g. the ground-truth or features-listing CSVs)
    are skipped with a warning.

    Files are read in chunks of :data:`UNSW_CHUNK_SIZE` rows, sampled per
    chunk, and converted to features immediately, so only ``sample_frac`` of
    the rows is ever materialized in memory.

    Args:
        data_dir: Directory containing UNSW-NB15 CSV files
        sample_frac: Fraction of data to sample
        random_state: Random seed

    Returns:
        Tuple of (features_array, labels_array)
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No UNSW-NB15 CSV files found in {data_dir}")

    headless_files, headed_files = [], []
    for csv_file in csv_files:
        with open(csv_file, encoding="utf-8-sig", errors="replace") as fh:
            first_line = next((line.rstrip("\n") for line in fh if line.strip()), "")
        (headed_files if _is_unsw_header_line(first_line) else headless_files).append(csv_file)

    rng = np.random.RandomState(random_state)
    feature_parts, label_parts = [], []
    for csv_file in headless_files or headed_files:
        try:
            read_kwargs: dict = {"encoding": "utf-8-sig", "chunksize": UNSW_CHUNK_SIZE}
            if headless_files:
                # Headless flow files mix port spellings (ints, hex) across chunks;
                # read everything as text and coerce per-field downstream.
                read_kwargs.update(header=None, names=UNSW_FLOW_HEADERS, dtype=str)
            label_col, file_features, file_labels = None, [], []
            for chunk in pd.read_csv(csv_file, **read_kwargs):
                if label_col is None:
                    label_col = _unsw_label_column(chunk)
                    if label_col is None:
                        logger.warning(f"Skipping {csv_file.name}: no UNSW label column")
                        break
                sampled = _sample_chunk(chunk, sample_frac, rng)
                if len(sampled):
                    # Direct flow-to-feature mapping; method features are 0.0.
                    file_features.append(_unsw_frame_to_features(sampled))
                    file_labels.append(_binary_labels(sampled[label_col], label_col))
            if not file_features:
                continue
            features = np.concatenate(file_features)
            labels = np.concatenate(file_labels)
        except Exception as e:
            logger.warning(f"Skipping {csv_file.name}: failed to parse ({e})")
            continue
        feature_parts.append(features)
        label_parts.append(labels)
        logger.info(f"Loaded {csv_file.name}: {len(features)} sampled rows")

    if not feature_parts:
        raise ValueError("No valid UNSW-NB15 flow files loaded")

    return np.concatenate(feature_parts), np.concatenate(label_parts)


def _cicids_row_to_http(row: pd.Series) -> dict:
    """Convert CICIDS flow row to synthetic HTTP request for feature extraction."""
    total_bytes = row.get("Total Length of Fwd Packets", 0) + row.get("Total Length of Bwd Packets", 0)
    fwd_pkts = row.get("Total Fwd Packets", 1)

    method = "POST" if total_bytes > 1000 else "GET"
    path_depth = int(row.get("Fwd Packet Length Mean", 1)) % 5 + 1
    path = "/" + "/".join([f"path{j}" for j in range(path_depth)])

    param_count = int(row.get("Flow Packets/s", 1)) % 10
    query_params = "&".join([f"param{k}={row.get('Flow IAT Mean', 'value')}" for k in range(param_count)])
    query = query_params if param_count > 0 else ""

    body = ""
    if method == "POST":
        body = "data=" + "x" * int(min(total_bytes / max(fwd_pkts, 1), 1000))

    headers = {
        "user-agent": f"Mozilla/5.0 (Flow-{row.get('Protocol', 'TCP')})",
        "content-type": "application/x-www-form-urlencoded" if method == "POST" else "",
    }

    return {
        "method": method,
        "path": path,
        "query": query,
        "headers": headers,
        "body": body,
    }


def _unsw_row_to_http(row: pd.Series) -> dict:
    """Convert UNSW-NB15 row to synthetic HTTP request for feature extraction."""
    service = str(row.get("service", "http")).lower()
    is_http = "http" in service or "web" in service

    method = "GET"
    if row.get("Spkts", 0) > row.get("Dpkts", 0):
        method = "POST"

    trans_depth = int(row.get("trans_depth", 1))
    path = "/" + "/".join([f"resource{j}" for j in range(max(1, trans_depth))])

    res_body_len = int(row.get("res_bdy_len", 0))
    query = ""
    body = ""

    if is_http and res_body_len > 0:
        param_count = min(int(row.get("Sload", 1)) % 5 + 1, 10)
        query = "&".join([f"field{k}=value{k}" for k in range(param_count)])
        if method == "POST":
            body = "payload=" + "y" * min(res_body_len, 500)

    headers = {
        "user-agent": f"UNSW-Agent/{row.get('proto', 'tcp')}",
        "content-type": "application/x-www-form-urlencoded" if method == "POST" else "",
    }

    return {
        "method": method,
        "path": path,
        "query": query,
        "headers": headers,
        "body": body,
    }


def _to_float(value: object, default: float = 0.0) -> float:
    """Safely coerce a flow cell to float, falling back to ``default``."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):  # NaN / inf guard
        return default
    return result


def _row_get(row: pd.Series, *names: str, default: object = 0) -> object:
    """Return the first present column value from ``names``, else ``default``.

    UNSW-NB15 ships two layouts (headless flow files with ``Spkts``/``Sload``
    casing and headed train/test files with ``spkts``/``sload`` casing), so
    mappers accept both variants.
    """
    for name in names:
        if name in row:
            return row.get(name, default)
    return default


def _cicids_frame_to_features(frame: pd.DataFrame) -> np.ndarray:
    """Build the unified 25-feature matrix for every row in ``frame``."""
    features = np.zeros((len(frame), 25), dtype=np.float32)
    for i, (_, row) in enumerate(frame.iterrows()):
        features[i] = _cicids_row_to_features(row)
    return features


def _unsw_frame_to_features(frame: pd.DataFrame) -> np.ndarray:
    """Build the unified 25-feature matrix for every row in ``frame``."""
    features = np.zeros((len(frame), 25), dtype=np.float32)
    for i, (_, row) in enumerate(frame.iterrows()):
        features[i] = _unsw_row_to_features(row)
    return features


def _cicids_row_to_features(row: pd.Series) -> np.ndarray:
    """Map a CICIDS2017 flow row directly to the unified 25-feature vector.

    Method features (indices 0, 1, 2) are always 0.0 because CICIDS is
    flow-based, not HTTP-request based — there is no true HTTP method.
    """
    import math

    features = np.zeros(25, dtype=np.float32)

    # ---- 0-2: HTTP method (constant 0.0 for flow data) ----
    # features[0] = method_get  -> already 0.0
    # features[1] = method_post -> already 0.0
    # features[2] = method_other-> already 0.0

    # ---- 3: path_depth ----
    # Bucket Fwd Packet Length Mean into 1-5 segments
    fwd_pkt_len_mean = _to_float(row.get("Fwd Packet Length Mean", 1), 1.0)
    features[3] = min(max(int(fwd_pkt_len_mean) % 5 + 1, 1), 5)

    # ---- 4: path_length ----
    # Total Length of Fwd Packets, log-scaled
    total_fwd = _to_float(row.get("Total Length of Fwd Packets", 0))
    features[4] = math.log1p(total_fwd) if total_fwd > 0 else 0.0

    # ---- 5: has_query ----
    # Proxy: 1.0 if Flow Packets/s suggests a request with params
    flow_pkt_s = _to_float(row.get("Flow Packets/s", 1), 1.0)
    features[5] = 1.0 if flow_pkt_s > 10 else 0.0

    # ---- 6: query_length ----
    # Flow Packets/s, log-scaled
    features[6] = math.log1p(flow_pkt_s) if flow_pkt_s > 0 else 0.0

    # ---- 7: param_count ----
    # Total Fwd Packets, capped at 20
    total_fwd_pkts = _to_float(row.get("Total Fwd Packets", 1), 1.0)
    features[7] = min(total_fwd_pkts, 20.0)

    # ---- 8: param_name_entropy ----
    # Fwd IAT Std normalized (clamped)
    fwd_iat_std = _to_float(row.get("Fwd IAT Std", 0))
    features[8] = min(fwd_iat_std / 1000.0, 10.0)

    # ---- 9: param_value_entropy ----
    # Bwd IAT Std normalized (clamped)
    bwd_iat_std = _to_float(row.get("Bwd IAT Std", 0))
    features[9] = min(bwd_iat_std / 1000.0, 10.0)

    # ---- 10: payload_length ----
    # Total Length of Bwd Packets
    total_bwd = _to_float(row.get("Total Length of Bwd Packets", 0))
    features[10] = total_bwd

    # ---- 11: special_char_density ----
    # Proxy: Fwd PSH Flags / Total Fwd Packets (flag density)
    fwd_psh = _to_float(row.get("Fwd PSH Flags", 0))
    total_fwd_pkts_count = _to_float(row.get("Total Fwd Packets", 1), 1.0)
    features[11] = fwd_psh / total_fwd_pkts_count if total_fwd_pkts_count > 0 else 0.0

    # ---- 12-15: attack keyword counts (0.0 for flow data) ----
    # Already zeros from np.zeros initialization

    # ---- 17: digit_ratio ----
    # Flow Bytes/s normalized digit-like density proxy
    flow_bytes_s = _to_float(row.get("Flow Bytes/s", 0))
    # Approximate: count digits in Flow Bytes/s string representation
    # Simplified: just use the value scaled
    features[17] = min(flow_bytes_s / 1e7, 1.0) if flow_bytes_s > 0 else 0.0

    # ---- 19: longest_param_len ----
    # Max Packet Length
    max_pkt_len = _to_float(row.get("Max Packet Length", 0))
    features[19] = max_pkt_len

    # ---- 20: avg_param_len ----
    # Average Packet Size
    avg_pkt_size = _to_float(row.get("Avg Fwd Segment Size", row.get("Average Packet Size", 0)))
    features[20] = avg_pkt_size if avg_pkt_size > 0 else 0.0

    # ---- 23: header_count ----
    # TCP flag count: ACK + SYN + FIN
    ack = _to_float(row.get("ACK Flag Count", 0))
    syn = _to_float(row.get("SYN Flag Count", 0))
    fin = _to_float(row.get("FIN Flag Count", 0))
    features[23] = ack + syn + fin

    return features


def _unsw_row_to_features(row: pd.Series) -> np.ndarray:
    """Map a UNSW-NB15 flow row directly to the unified 25-feature vector.

    Method features (indices 0, 1, 2) are always 0.0 because UNSW is
    flow-based, not HTTP-request based — there is no true HTTP method.
    """
    import math

    features = np.zeros(25, dtype=np.float32)

    # ---- 0-2: HTTP method (constant 0.0 for flow data) ----
    # Already 0.0 from np.zeros

    # ---- 3: path_depth ----
    # trans_depth directly maps
    trans_depth = _to_float(_row_get(row, "trans_depth", default=1), 1.0)
    features[3] = max(1.0, trans_depth)

    # ---- 4: path_length ----
    # sbytes + dbytes, log-scaled
    sbytes = _to_float(_row_get(row, "sbytes", default=0))
    dbytes = _to_float(_row_get(row, "dbytes", default=0))
    features[4] = math.log1p(sbytes + dbytes) if (sbytes + dbytes) > 0 else 0.0

    # ---- 5: has_query ----
    # 1.0 if res_bdy_len > 0 suggests HTTP body / query present
    res_bdy_len = _to_float(_row_get(row, "res_bdy_len", "response_body_len", default=0))
    features[5] = 1.0 if res_bdy_len > 0 else 0.0

    # ---- 6: query_length ----
    # res_bdy_len
    features[6] = float(res_bdy_len) if res_bdy_len > 0 else 0.0

    # ---- 7: param_count ----
    # ct_flw_http_mthd: count of HTTP method identifiers in flow
    ct_flw = _to_float(_row_get(row, "ct_flw_http_mthd", default=0))
    features[7] = ct_flw if ct_flw > 0 else 0.0

    # ---- 8: param_name_entropy ----
    # Sjit (source jitter) normalized
    sjit = _to_float(_row_get(row, "Sjit", "sjit", default=0))
    features[8] = min(sjit / 100.0, 10.0)

    # ---- 9: param_value_entropy ----
    # Djit (dest jitter) normalized
    djit = _to_float(_row_get(row, "Djit", "djit", default=0))
    features[9] = min(djit / 100.0, 10.0)

    # ---- 10: payload_length ----
    # sbytes (source bytes)
    features[10] = sbytes if sbytes > 0 else 0.0

    # ---- 11: special_char_density ----
    # sloss / (sbytes + 1) — loss ratio proxy
    sloss = _to_float(_row_get(row, "sloss", default=0))
    sbytes_val = _to_float(_row_get(row, "sbytes", default=0))
    features[11] = sloss / (sbytes_val + 1) if sbytes_val >= 0 else 0.0

    # ---- 17: digit_ratio ----
    # Sload normalized
    sload = _to_float(_row_get(row, "Sload", "sload", default=0))
    features[17] = min(sload / 1000.0, 1.0) if sload > 0 else 0.0

    # ---- 19: longest_param_len ----
    # tcprtt (round-trip time as length proxy)
    tcprtt = _to_float(_row_get(row, "tcprtt", default=0))
    features[19] = tcprtt if tcprtt > 0 else 0.0

    # ---- 20: avg_param_len ----
    # dur (duration)
    dur = _to_float(_row_get(row, "dur", default=0))
    features[20] = dur if dur > 0 else 0.0

    # ---- 23: header_count ----
    # ct_state_ttl — connection state count
    ct_ttl = _to_float(_row_get(row, "ct_state_ttl", default=0))
    features[23] = ct_ttl if ct_ttl > 0 else 0.0

    return features


def load_combined_datasets(
    cicids_dir: Path | None = None,
    unsw_dir: Path | None = None,
    cicids_sample: float = 1.0,
    unsw_sample: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and combine both public datasets with honeypot schema."""
    all_features = []
    all_labels = []

    if cicids_dir and cicids_dir.exists():
        logger.info("Loading CICIDS2017...")
        X_cicids, y_cicids = load_cicids2017(cicids_dir, cicids_sample, random_state)
        all_features.append(X_cicids)
        all_labels.append(y_cicids)
        logger.info(f"CICIDS2017: {len(X_cicids)} samples, {y_cicids.sum()} attacks")

    if unsw_dir and unsw_dir.exists():
        logger.info("Loading UNSW-NB15...")
        X_unsw, y_unsw = load_unsw_nb15(unsw_dir, unsw_sample, random_state)
        all_features.append(X_unsw)
        all_labels.append(y_unsw)
        logger.info(f"UNSW-NB15: {len(X_unsw)} samples, {y_unsw.sum()} attacks")

    if not all_features:
        raise ValueError("No datasets loaded")

    X_combined = np.vstack(all_features)
    y_combined = np.concatenate(all_labels)

    return X_combined, y_combined
