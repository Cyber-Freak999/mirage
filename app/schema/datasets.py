"""Public dataset loaders mapped to unified 25-feature schema."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_features

logger = logging.getLogger(__name__)

CICIDS_FEATURE_MAP = {
    "Flow Duration": None,
    "Total Fwd Packets": None,
    "Total Backward Packets": None,
    "Total Length of Fwd Packets": None,
    "Total Length of Bwd Packets": None,
    "Fwd Packet Length Max": None,
    "Fwd Packet Length Min": None,
    "Fwd Packet Length Mean": None,
    "Fwd Packet Length Std": None,
    "Bwd Packet Length Max": None,
    "Bwd Packet Length Min": None,
    "Bwd Packet Length Mean": None,
    "Bwd Packet Length Std": None,
    "Flow Bytes/s": None,
    "Flow Packets/s": None,
    "Flow IAT Mean": None,
    "Flow IAT Std": None,
    "Flow IAT Max": None,
    "Flow IAT Min": None,
    "Fwd IAT Total": None,
    "Fwd IAT Mean": None,
    "Fwd IAT Std": None,
    "Fwd IAT Max": None,
    "Fwd IAT Min": None,
    "Bwd IAT Total": None,
    "Bwd IAT Mean": None,
    "Bwd IAT Std": None,
    "Bwd IAT Max": None,
    "Bwd IAT Min": None,
    "Fwd PSH Flags": None,
    "Bwd PSH Flags": None,
    "Fwd URG Flags": None,
    "Bwd URG Flags": None,
    "Fwd Header Length": None,
    "Bwd Header Length": None,
    "Fwd Packets/s": None,
    "Bwd Packets/s": None,
    "Min Packet Length": None,
    "Max Packet Length": None,
    "Packet Length Mean": None,
    "Packet Length Std": None,
    "Packet Length Variance": None,
    "FIN Flag Count": None,
    "SYN Flag Count": None,
    "RST Flag Count": None,
    "PSH Flag Count": None,
    "ACK Flag Count": None,
    "URG Flag Count": None,
    "CWE Flag Count": None,
    "ECE Flag Count": None,
    "Down/Up Ratio": None,
    "Average Packet Size": None,
    "Avg Fwd Segment Size": None,
    "Avg Bwd Segment Size": None,
    "Fwd Header Length.1": None,
    "Fwd Avg Bytes/Bulk": None,
    "Fwd Avg Packets/Bulk": None,
    "Fwd Avg Bulk Rate": None,
    "Bwd Avg Bytes/Bulk": None,
    "Bwd Avg Packets/Bulk": None,
    "Bwd Avg Bulk Rate": None,
    "Subflow Fwd Packets": None,
    "Subflow Fwd Bytes": None,
    "Subflow Bwd Packets": None,
    "Subflow Bwd Bytes": None,
    "Init_Win_bytes_forward": None,
    "Init_Win_bytes_backward": None,
    "act_data_pkt_fwd": None,
    "min_seg_size_forward": None,
    "Active Mean": None,
    "Active Std": None,
    "Active Max": None,
    "Active Min": None,
    "Idle Mean": None,
    "Idle Std": None,
    "Idle Max": None,
    "Idle Min": None,
    "Label": "label",
}

UNSW_FEATURE_MAP = {
    "srcip": None,
    "sport": None,
    "dstip": None,
    "dsport": None,
    "proto": None,
    "state": None,
    "dur": None,
    "sbytes": None,
    "dbytes": None,
    "sttl": None,
    "dttl": None,
    "sloss": None,
    "dloss": None,
    "service": None,
    "Sload": None,
    "Dload": None,
    "Spkts": None,
    "Dpkts": None,
    "swin": None,
    "dwin": None,
    "stcpb": None,
    "dtcpb": None,
    "smeansz": None,
    "dmeansz": None,
    "trans_depth": None,
    "res_bdy_len": None,
    "Sjit": None,
    "Djit": None,
    "Stime": None,
    "Ltime": None,
    "Sintpkt": None,
    "Dintpkt": None,
    "tcprtt": None,
    "synack": None,
    "ackdat": None,
    "is_sm_ips_ports": None,
    "ct_state_ttl": None,
    "ct_flw_http_mthd": None,
    "ct_ftp_cmd": None,
    "ct_srv_src": None,
    "ct_srv_dst": None,
    "ct_dst_ltm": None,
    "ct_src_ltm": None,
    "ct_src_dport_ltm": None,
    "ct_dst_sport_ltm": None,
    "ct_dst_src_ltm": None,
    "attack_cat": "label",
    "Label": "label_binary",
}

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
                sampled = _sample_chunk(chunk, sample_frac, rng).fillna(0)
                if len(sampled):
                    file_features.append(_extract_request_features(sampled, _cicids_row_to_http))
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
            read_kwargs = {"encoding": "utf-8-sig", "chunksize": UNSW_CHUNK_SIZE}
            if headless_files:
                read_kwargs.update(header=None, names=UNSW_FLOW_HEADERS)
            label_col, file_features, file_labels = None, [], []
            for chunk in pd.read_csv(csv_file, **read_kwargs):
                if label_col is None:
                    label_col = _unsw_label_column(chunk)
                    if label_col is None:
                        logger.warning(f"Skipping {csv_file.name}: no UNSW label column")
                        break
                sampled = _sample_chunk(chunk, sample_frac, rng)
                if len(sampled):
                    file_features.append(_extract_request_features(sampled, _unsw_row_to_http))
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
