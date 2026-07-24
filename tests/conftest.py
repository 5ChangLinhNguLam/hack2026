"""Fixture dùng chung: trỏ vào data/ ở repo root, skip nếu thiếu dataset."""

import gzip
import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Trip giả lập kiểu T0Xd: cố tình khác 600/1800 — code không được hardcode số frame
N_FAKE_FRAMES = 7


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


def make_trip(root: Path, name: str, frames: list, **top) -> Path:
    """Ghi 1 trip giả lập tối thiểu (chỉ JSON, không ảnh) để test loader."""
    trip = root / name
    trip.mkdir()
    raw = {
        "trip_id": name,
        "metadata": {"trip_id": name, "fps": 20, "map": "Town04", "speed_limit_kmh": 60},
        "events_log": [],
        "frames": frames,
        **top,
    }
    with gzip.open(trip / f"{name}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(raw, f)
    return trip


def make_redacted_frame(i: int) -> dict:
    """Frame kiểu T0Xd: chỉ còn ego speed/accel, không GT (fact #8)."""
    return {
        "frame_id": i,
        "world_frame": 12345 + i,
        "timestamp": i / 20.0,
        "ego": {"speed_kmh": 30.0 + i, "longitudinal_accel": 0.1, "lateral_accel": 0.0},
        "targets": [],
        "events_active": [],
    }


@pytest.fixture()
def redacted_trip_dir(tmp_path) -> Path:
    """Trip kiểu T0Xd: không GT, không có thư mục ảnh/depth/label."""
    frames = [make_redacted_frame(i) for i in range(N_FAKE_FRAMES)]
    return make_trip(tmp_path, "T99d", frames, events_log=[{"t": 0.1, "type": "lead_brake"}])
