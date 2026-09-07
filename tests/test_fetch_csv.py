"""Tests for the CSV fetch helper."""

from pathlib import Path

from scripts.fetch_csv import fetch_csv

SAMPLE_CSV = b"a,b,c\n1,2,3\n4,5,6\n"


def test_fetch_csv_downloads_to_nested_dest(tmp_path: Path) -> None:
    src = tmp_path / "source.csv"
    src.write_bytes(SAMPLE_CSV)

    dest = tmp_path / "nested" / "dir" / "data.csv"

    result = fetch_csv(src.as_uri(), dest)

    assert result == dest
    assert dest.read_bytes() == SAMPLE_CSV


def test_fetch_csv_requires_csv_ext() -> None:
    import pytest

    from scripts.fetch_csv import fetch_csv

    with pytest.raises(ValueError, match="must end in .csv"):
        fetch_csv("file:///tmp/nope.txt", Path("/tmp/opencode/out.txt"))
