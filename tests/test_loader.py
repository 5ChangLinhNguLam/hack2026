"""Phase 1 — TripLoader: JSON + index + Calib + has_gt().

DoD (spec mục 4):
- loader.n_frames == 600 trên T01-Sample
- math.isinf(min_ttc) với 1 frame biết trước (T01 frame 300)
- thiếu key GT không crash (trip giả lập kiểu T0Xd, 7 frame)
"""

import math

import pytest

from conftest import N_FAKE_FRAMES, make_redacted_frame, make_trip
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


# ---------------------------------------------------------------------- #
# Che GT một phần: has_gt() và frame().gt phải luôn nhất quán (fact #8)
# ---------------------------------------------------------------------- #
def test_gt_masked_with_empty_dict_consistent(tmp_path):
    # Trip chấm điểm che GT bằng dict rỗng/null thay vì xoá key
    frame = make_redacted_frame(0)
    frame["driver"] = {}
    frame["min_ttc"] = None
    trip = make_trip(tmp_path, "T97d", [frame])
    loader = TripLoader(trip)
    assert loader.has_gt() is False
    assert loader.frame(0).gt is None  # nhất quán với has_gt()


def test_gt_partial_driver_state_only(tmp_path):
    frame = make_redacted_frame(0)
    frame["driver"] = {"state": "alert"}
    trip = make_trip(tmp_path, "T96d", [frame])
    loader = TripLoader(trip)
    assert loader.has_gt() is True
    assert loader.frame(0).gt == {"driver": {"state": "alert"}}


def test_has_gt_from_driver_summary_only(tmp_path):
    # GT cấp trip còn nhưng GT mức frame bị xoá → has_gt True, frame().gt None
    frames = [make_redacted_frame(0)]
    trip = make_trip(tmp_path, "T95d", frames, driver_summary={"subject_id": "9"})
    loader = TripLoader(trip)
    assert loader.has_gt() is True
    assert loader.frame(0).gt is None
