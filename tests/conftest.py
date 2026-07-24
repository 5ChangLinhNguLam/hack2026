"""Fixture dùng chung: trỏ vào data/ ở repo root, skip nếu thiếu dataset."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _require_trip(name: str) -> Path:
    d = DATA_DIR / name
    if not d.is_dir() or not list(d.glob(f"{name}.json*")):
        pytest.skip(f"{name} không có trong data/ — bỏ qua test cần dataset")
    return d


@pytest.fixture(scope="session")
def t01_dir() -> Path:
    return _require_trip("T01-Sample")


@pytest.fixture(scope="session")
def t02_dir() -> Path:
    return _require_trip("T02-Sample")


@pytest.fixture(scope="session")
def t01_loader(t01_dir):
    from tripkit import TripLoader

    return TripLoader(t01_dir)
