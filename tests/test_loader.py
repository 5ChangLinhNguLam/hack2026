"""Phase 1 — TripLoader: JSON + index + Calib + has_gt().

DoD (spec mục 4):
- loader.n_frames == 600 trên T01-Sample
- math.isinf(min_ttc) với 1 frame biết trước (T01 frame 300)
- thiếu key GT không crash (trip giả lập kiểu T0Xd, 7 frame)
"""

import gzip
import json
import math

import pytest

from tripkit import Calib, TripLoader


# ---------------------------------------------------------------------- #
# JSON + thuộc tính cấp trip (trên dữ liệu thật T01-Sample)
# ---------------------------------------------------------------------- #
def test_n_frames_t01(t01_loader):
    assert t01_loader.n_frames == 600


def test_trip_id_fps_metadata(t01_loader):
    assert t01_loader.trip_id == "T01-Sample"
    assert t01_loader.fps == 20.0
    assert t01_loader.metadata["map"]
    assert t01_loader.metadata["speed_limit_kmh"] == 40


def test_prefers_decompressed_json(t01_loader):
    # T01 là trip duy nhất có sẵn .json — phải được ưu tiên hơn .json.gz (fact #3)
    assert t01_loader.json_path.suffix == ".json"


def test_gz_fallback(t02_dir):
    loader = TripLoader(t02_dir)
    assert loader.json_path.name.endswith(".json.gz")
    assert loader.n_frames == 600
    assert loader.trip_id == "T02-Sample"


def test_bare_infinity_preserved(t01_loader):
    # Frame 300 của T01 đã kiểm chứng min_ttc = Infinity (không target trong cone)
    assert math.isinf(t01_loader.raw_frame(300)["min_ttc"])
    # và phải tồn tại frame có min_ttc hữu hạn (T01 có 7 frame near-miss)
    assert any(
        math.isfinite(t01_loader.raw_frame(i).get("min_ttc", math.inf))
        for i in range(t01_loader.n_frames)
    )


def test_has_gt_true_on_sample(t01_loader):
    assert t01_loader.has_gt() is True
    assert t01_loader.trip_aggregate is not None
    assert t01_loader.driver_summary is not None
    assert t01_loader.events_log


# ---------------------------------------------------------------------- #
# Calib — load 1 lần/trip từ calibration_info.txt (fact #6)
# ---------------------------------------------------------------------- #
def test_calib_values(t01_loader):
    c = t01_loader.calib
    assert isinstance(c, Calib)
    assert c.fx == pytest.approx(320.0)
    assert c.fy == pytest.approx(320.0)
    assert c.cx == pytest.approx(320.0)
    assert c.cy == pytest.approx(180.0)
    assert c.baseline_m == pytest.approx(0.3)
    assert (c.width, c.height) == (640, 360)
    assert c.depth_factor == pytest.approx(96.0)


def test_calib_cached_object(t01_loader):
    assert t01_loader.calib is t01_loader.calib


# ---------------------------------------------------------------------- #
# Index đường dẫn — 2 pattern tên ảnh (fact #4), depth keyframe (fact #5)
# ---------------------------------------------------------------------- #
def test_image_path_patterns(t01_loader):
    assert t01_loader.driver_path(0).name == "frame_000000.jpg"
    assert t01_loader.left_path(0).name == "000000.jpg"
    assert t01_loader.left_path(0).parent.name == "image_2"
    assert t01_loader.right_path(0).parent.name == "image_3"
    for p in (t01_loader.driver_path(0), t01_loader.left_path(599), t01_loader.right_path(599)):
        assert p.exists()


def test_depth_path_keyframe_only(t01_loader):
    assert t01_loader.depth_path(0) is not None and t01_loader.depth_path(0).exists()
    assert t01_loader.depth_path(595).exists()
    assert t01_loader.depth_path(3) is None
    assert t01_loader.depth_path(599) is None


def test_label_path(t01_loader):
    assert t01_loader.label_path(323).exists()  # frame có nhãn Pedestrian
    assert t01_loader.label_path(0).exists()    # file rỗng 0 byte vẫn tồn tại


def test_frame_id_out_of_range(t01_loader):
    with pytest.raises(IndexError):
        t01_loader.raw_frame(600)
    with pytest.raises(IndexError):
        t01_loader.left_path(-1)


# ---------------------------------------------------------------------- #
# Trip chấm điểm giả lập: GT bị xoá, số frame bất kỳ (fact #1, #8)
# ---------------------------------------------------------------------- #
N_FAKE_FRAMES = 7  # cố tình khác 600/1800 — code không được hardcode số frame


@pytest.fixture()
def redacted_trip_dir(tmp_path):
    """Trip kiểu T0Xd: chỉ còn ego speed/accel, targets/events rỗng, không GT."""
    trip = tmp_path / "T99d"
    trip.mkdir()
    frames = [
        {
            "frame_id": i,
            "world_frame": 12345 + i,
            "timestamp": i / 20.0,
            "ego": {"speed_kmh": 30.0 + i, "longitudinal_accel": 0.1, "lateral_accel": 0.0},
            "targets": [],
            "events_active": [],
        }
        for i in range(N_FAKE_FRAMES)
    ]
    raw = {
        "trip_id": "T99d",
        "metadata": {"trip_id": "T99d", "fps": 20, "map": "Town04", "speed_limit_kmh": 60},
        "events_log": [{"t": 0.1, "type": "lead_brake"}],
        "frames": frames,
    }
    with gzip.open(trip / "T99d.json.gz", "wt", encoding="utf-8") as f:
        json.dump(raw, f)
    return trip


def test_redacted_trip_does_not_crash(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    assert loader.trip_id == "T99d"
    assert loader.n_frames == N_FAKE_FRAMES
    assert loader.fps == 20.0
    assert loader.has_gt() is False
    assert loader.trip_aggregate is None
    assert loader.driver_summary is None
    f0 = loader.raw_frame(0)
    assert f0["ego"]["speed_kmh"] == 30.0
    assert "min_ttc" not in f0  # thiếu key GT — không crash, không bịa giá trị


def test_missing_json_raises(tmp_path):
    empty = tmp_path / "T98d"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        TripLoader(empty)
