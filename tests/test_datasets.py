"""Tests for public dataset loaders against real UNSW-NB15 and CICIDS2017 layouts."""

from pathlib import Path

import numpy as np
import pytest

from app.schema.datasets import load_cicids2017, load_unsw_nb15

CANONICAL_HEADERS = [
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


def _canonical_rows(n_rows: int) -> list[list]:
    """Build synthetic 49-field UNSW flow rows matching the real headless layout."""
    rows = []
    for i in range(n_rows):
        attack = i % 2 == 1
        rows.append(
            [
                "10.0.0.1",
                "443",
                "10.0.0.2",
                "80",
                "tcp",
                "FIN",
                "0.5",
                "1000",
                "2000",
                "64",
                "63",
                "0",
                "0",
                "http",
                "8000.0",
                "16000.0",
                "3",
                "4",
                "0",
                "0",
                "0",
                "0",
                "333",
                "500",
                "1",
                "1000",
                "0.01",
                "0.02",
                "1623",
                "1623",
                "0.1",
                "0.2",
                "0.3",
                "0.1",
                "0.2",
                "0",
                "1",
                "1",
                "1",
                "0",
                "2",
                "3",
                "4",
                "5",
                "1",
                "2",
                "3",
                "Exploits" if attack else "",
                "1" if attack else "0",
            ]
        )
    return rows


def _write_headless_csv(path: Path, rows: list[list], add_bom: bool = True) -> None:
    """Write a canonical flow CSV WITHOUT a header row (real UNSW layout)."""
    text = "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"
    if add_bom:
        text = "\ufeff" + text
    path.write_text(text, encoding="utf-8")


def test_load_unsw_nb15_skips_nonflow_files(tmp_path: Path) -> None:
    rows = _canonical_rows(20)
    _write_headless_csv(tmp_path / "UNSW-NB15_1.csv", rows)
    (tmp_path / "NUSW-NB15_GT.csv").write_text(
        "Start time,Last time,Attack category,Attack subcategory,Protocol\n" "1,2,Reconnaissance,HTTP,tcp\n",
        encoding="utf-8",
    )

    X, y = load_unsw_nb15(tmp_path)

    assert X.shape == (20, 25)
    assert X.dtype == np.float32
    assert y.tolist() == [0, 1] * 10


def test_load_unsw_nb15_headless_with_bom(tmp_path: Path) -> None:
    rows = _canonical_rows(4)
    _write_headless_csv(tmp_path / "UNSW-NB15_1.csv", rows, add_bom=True)

    X, y = load_unsw_nb15(tmp_path)

    assert len(y) == 4
    assert y.tolist() == [0, 1, 0, 1]


def test_load_unsw_nb15_headed_modern_layout(tmp_path: Path) -> None:
    (tmp_path / "UNSW_NB15_training-set.csv").write_text(
        "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,"
        "sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,"
        "tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,"
        "ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,"
        "ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,"
        "ct_srv_dst,is_sm_ips_ports,attack_cat,label\n"
        "1,0.1,udp,http,FIN,2,1,100,50,10,64,63,0,0,0,0,0.1,0.1,0,0,0,0,0,0,"
        "0.1,0.1,0.1,100,50,1,100,2,2,3,1,2,3,0,0,1,1,0,0,Normal,0\n"
        "2,0.2,tcp,http,CON,4,2,500,250,20,64,63,0,0,0,0,0.2,0.2,0,0,0,0,0,0,"
        "0.2,0.1,0.1,250,125,2,200,3,3,4,1,2,3,1,0,1,1,0,0,Generic,1\n",
        encoding="utf-8",
    )

    X, y = load_unsw_nb15(tmp_path)

    assert y.tolist() == [0, 1]
    assert X.shape == (2, 25)


def test_load_unsw_nb15_sampling(tmp_path: Path) -> None:
    rows = _canonical_rows(100)
    _write_headless_csv(tmp_path / "UNSW-NB15_1.csv", rows)

    X, y = load_unsw_nb15(tmp_path, sample_frac=0.5, random_state=42)

    assert len(y) == 50


def test_load_unsw_nb15_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No UNSW-NB15 CSV"):
        load_unsw_nb15(tmp_path)


def test_load_unsw_nb15_dir_with_only_junk_raises(tmp_path: Path) -> None:
    (tmp_path / "NUSW-NB15_GT.csv").write_text("Start time,Attack category\n1,Reconnaissance\n")
    (tmp_path / "NUSW-NB15_features.csv").write_text("No.,Name,Type\n1,srcip,nominal\n")

    with pytest.raises(ValueError, match="No valid UNSW-NB15"):
        load_unsw_nb15(tmp_path)


CICIDS_HEADER = [
    " Destination Port",
    "Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Mean",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Label",
]


def _cicids_row(i: int) -> str:
    """Build a synthetic CICIDS2017 row; odd indices are attacks."""
    benign = i % 2 == 0
    label = "BENIGN" if benign else "DDoS"
    return "80,1000,2,1,200,100,1,1000.0,10,5.0," + label


def _write_cicids_csv(path: Path, n_rows: int, blank_numeric: bool = False) -> None:
    rows = [_cicids_row(i) for i in range(n_rows)]
    if blank_numeric:
        rows[1] = rows[1].replace("1,1000.0,", ",1000.0,")
    path.write_text(",".join(CICIDS_HEADER) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _cicids_row_with_inf(i: int) -> str:
    """Build a synthetic CICIDS2017 row with an ``inf`` Flow Packets/s cell."""
    label = "BENIGN" if i % 2 == 0 else "DDoS"
    return "80,1000,2,1,200,100,1,1000.0,inf,5.0," + label


def test_load_cicids2017_coerces_inf_to_zero(tmp_path: Path) -> None:
    rows = [_cicids_row_with_inf(i) for i in range(4)]
    (tmp_path / "Monday-WorkingHours.csv").write_text(
        ",".join(CICIDS_HEADER) + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    X, y = load_cicids2017(tmp_path)

    assert X.shape == (4, 25)
    assert np.isfinite(X).all()
    assert y.tolist() == [0, 1, 0, 1]


def test_load_cicids2017_strips_header_whitespace(tmp_path: Path) -> None:
    _write_cicids_csv(tmp_path / "Monday-WorkingHours.csv", 4)

    X, y = load_cicids2017(tmp_path)

    assert y.tolist() == [0, 1, 0, 1]
    assert X.shape == (4, 25)


def test_load_cicids2017_sampling(tmp_path: Path) -> None:
    _write_cicids_csv(tmp_path / "Monday-WorkingHours.csv", 100)

    X, y = load_cicids2017(tmp_path, sample_frac=0.5, random_state=42)

    assert len(y) == 50


def test_load_cicids2017_handles_blank_numeric(tmp_path: Path) -> None:
    _write_cicids_csv(tmp_path / "Monday-WorkingHours.csv", 4, blank_numeric=True)

    X, y = load_cicids2017(tmp_path)

    assert X.shape == (4, 25)
    assert y.tolist() == [0, 1, 0, 1]


def test_load_cicids2017_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No CICIDS2017 CSV"):
        load_cicids2017(tmp_path)
