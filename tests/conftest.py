"""Fixture dùng chung: trỏ vào data/ ở repo root, skip nếu thiếu dataset.

Đặt biến môi trường ``TRIPKIT_DATA_DIR`` để trỏ sang chỗ khác mà không cần
copy hay symlink dataset vào ``data/``::

    TRIPKIT_DATA_DIR=D:/.../Practice_Dataset python -m pytest tests/

Ưu tiên biến môi trường vì symlink trên Windows cần quyền admin, còn copy thì
mất vài GB — cả hai đều là rào cản không cần thiết chỉ để chạy test.
"""

import gzip
import json
import os
from pathlib import Path

import pytest

_ENV_DATA_DIR = "TRIPKIT_DATA_DIR"
DATA_DIR = (
    Path(os.environ[_ENV_DATA_DIR]).expanduser().resolve()
    if os.environ.get(_ENV_DATA_DIR)
    else Path(__file__).resolve().parents[1] / "data"
)

# Trip giả lập kiểu T0Xd: cố tình khác 600/1800 — code không được hardcode số frame
N_FAKE_FRAMES = 7


def _require_trip(name: str) -> Path:
    d = DATA_DIR / name
    if not d.is_dir() or not list(d.glob(f"{name}.json*")):
        pytest.skip(
            f"{name} không có trong {DATA_DIR} — bỏ qua test cần dataset "
            f"(đặt {_ENV_DATA_DIR} để trỏ sang thư mục khác)"
        )
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


def make_full_trip(root: Path, name: str, n_frames: int, json_name: str = "") -> Path:
    """Trip giả lập ĐẦY ĐỦ modality (ảnh 640x360, depth, calib, label) — để
    test validate/replay trên số frame bất kỳ, không phụ thuộc data/ thật."""
    import cv2
    import numpy as np

    trip = root / name
    for sub in ("kitti/image_2", "kitti/image_3", "kitti/depth",
                "kitti/calib", "kitti/label_2", "driver"):
        (trip / sub).mkdir(parents=True)
    img = np.zeros((360, 640, 3), np.uint8)
    calib_txt = "P0: 320 0 320 0 0 320 180 0 0 0 1 0\n"
    for i in range(n_frames):
        cv2.imwrite(str(trip / "kitti" / "image_2" / f"{i:06d}.jpg"), img)
        cv2.imwrite(str(trip / "kitti" / "image_3" / f"{i:06d}.jpg"), img)
        cv2.imwrite(str(trip / "driver" / f"frame_{i:06d}.jpg"), img)
        (trip / "kitti" / "calib" / f"{i:06d}.txt").write_text(calib_txt)
        (trip / "kitti" / "label_2" / f"{i:06d}.txt").write_text("")
        if i % 5 == 0:
            np.save(str(trip / "kitti" / "depth" / f"{i:06d}.npy"),
                    np.ones((360, 640), np.float32))
    (trip / "kitti" / "calibration_info.txt").write_text(json.dumps({
        "fov_deg": 90, "baseline_m": 0.3, "image_width": 640, "image_height": 360,
        "K_left": [[320, 0, 320], [0, 320, 180], [0, 0, 1]],
    }))
    frames = [make_redacted_frame(i) for i in range(n_frames)]
    raw = {
        "trip_id": name,
        "metadata": {"trip_id": name, "fps": 20, "map": "Town04", "speed_limit_kmh": 60},
        "events_log": [],
        "frames": frames,
    }
    with gzip.open(trip / (json_name or f"{name}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(raw, f)
    return trip


@pytest.fixture()
def full_trip_dir(tmp_path) -> Path:
    """Trip đầy đủ modality với N_FAKE_FRAMES=7 frame (cố tình khác 600/1800)."""
    return make_full_trip(tmp_path, "T92d", N_FAKE_FRAMES)
