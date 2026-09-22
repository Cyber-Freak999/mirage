"""Tests for SQLite hot-backup script."""

import sqlite3
import time


def _seed(db_path, table: str = "t") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute(f"INSERT INTO {table} (v) VALUES ('hello')")
    conn.commit()
    conn.close()


def test_backup_copies_present_dbs(tmp_path) -> None:
    """Existing DBs are copied with data intact; missing ones skipped."""
    from scripts.backup_dbs import main

    data = tmp_path / "data"
    data.mkdir()
    _seed(data / "honeypot.db")
    _seed(data / "validation.db")

    assert main(["--db-dir", str(data), "--out-dir", str(tmp_path / "backups")]) == 0

    backed = sorted((tmp_path / "backups").iterdir())
    assert len(backed) == 1
    for name in ("honeypot.db", "validation.db"):
        conn = sqlite3.connect(str(backed[0] / name))
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "hello"
        conn.close()


def test_backup_dry_run_changes_nothing(tmp_path) -> None:
    """Dry run reports without writing anything."""
    from scripts.backup_dbs import main

    data = tmp_path / "data"
    data.mkdir()
    _seed(data / "honeypot.db")
    out = tmp_path / "backups"

    assert main(["--db-dir", str(data), "--out-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_backup_prunes_old_sets(tmp_path) -> None:
    """keep-last retains only the newest backup sets."""
    import shutil

    from scripts.backup_dbs import main

    data = tmp_path / "data"
    data.mkdir()
    _seed(data / "honeypot.db")
    out = tmp_path / "backups"
    out.mkdir()
    for i in range(3):
        old = out / f"backup-2000010{i}"
        old.mkdir()
        shutil.copy(str(data / "honeypot.db"), str(old / "honeypot.db"))
        time.sleep(0.01)

    assert main(["--db-dir", str(data), "--out-dir", str(out), "--keep-last", "2"]) == 0
    assert len(list(out.iterdir())) == 2
