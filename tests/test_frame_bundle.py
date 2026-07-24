"""Phase 2 — FrameBundle: ảnh lazy + depth + KITTI labels.

DoD (spec mục 4):
- shape ảnh (360,640,3); depth shape (360,640) float32
- frame(3).depth is None
- label frame rỗng → []
"""

import math

import numpy as np
import pytest

from tripkit import KittiLabel, TripLoader


# ---------------------------------------------------------------------- #
# Ảnh lazy-load (fact #4)
# ---------------------------------------------------------------------- #
def test_image_shapes(t01_loader):
    b = t01_loader.frame(0)
    for img in (b.left(), b.right(), b.driver()):
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8


def test_image_cache_on_by_default(t01_loader):
    b = t01_loader.frame(0)
    assert b.left() is b.left()


def test_image_cache_off(t01_loader):
    b = t01_loader.frame(0, cache_images=False)
    assert b.left() is not b.left()


def test_images_are_lazy(redacted_trip_dir):
    # Trip không có thư mục ảnh: tạo bundle vẫn OK, chỉ crash khi truy cập ảnh
    loader = TripLoader(redacted_trip_dir)
    b = loader.frame(0)
    with pytest.raises(FileNotFoundError):
        b.driver()


# ---------------------------------------------------------------------- #
# Depth — chỉ keyframe i%5==0 (fact #5)
# ---------------------------------------------------------------------- #
def test_depth_keyframe(t01_loader):
    b = t01_loader.frame(0)
    assert b.depth is not None
    assert b.depth.shape == (360, 640)
    assert b.depth.dtype == np.float32


def test_depth_none_on_non_keyframe(t01_loader):
    assert t01_loader.frame(3).depth is None


def test_depth_nearest(t01_loader):
    d5, exact5 = t01_loader.depth_nearest(5)
    assert exact5 is True
    d7, exact7 = t01_loader.depth_nearest(7)
    assert exact7 is False
    assert np.array_equal(d5, d7)  # 7 → keyframe 5
    _, exact0 = t01_loader.depth_nearest(0)
    assert exact0 is True


# ---------------------------------------------------------------------- #
# KITTI labels — file rỗng → [] (fact #7)
# ---------------------------------------------------------------------- #
def test_labels_empty_file(t01_loader):
    assert t01_loader.frame(0).labels == []


def test_labels_parse_known_frame(t01_loader):
    # T01 frame 323: "Pedestrian 0.00 0 0.00 0*4 1.70 0.60 0.60 -2.42 1.50 10.18 0.00"
    labels = t01_loader.frame(323).labels
    assert len(labels) == 1
    lb = labels[0]
    assert isinstance(lb, KittiLabel)
    assert lb.type == "Pedestrian"
    assert (lb.height, lb.width, lb.length) == (1.70, 0.60, 0.60)
    assert (lb.x, lb.y, lb.z) == (-2.42, 1.50, 10.18)
    # bbox 2D / alpha / rotation_y luôn 0 trong dataset này — không dùng làm nhãn 2D
    assert (lb.bbox_left, lb.bbox_top, lb.bbox_right, lb.bbox_bottom) == (0, 0, 0, 0)


def test_labels_missing_file_ok(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    assert loader.frame(0).labels == []


# ---------------------------------------------------------------------- #
# GT trong bundle — None ở trip chấm điểm (fact #8)
# ---------------------------------------------------------------------- #
def test_gt_present_on_sample(t01_loader):
    b = t01_loader.frame(300)
    assert b.gt is not None
    assert b.gt["driver"]["state"] == "alert"       # T01: distracted 0-15s -> alert 15-30s
    assert math.isinf(b.gt["min_ttc"])
    assert b.gt["behavior_flags"]["harsh_brake"] is False
    assert t01_loader.frame(0).gt["driver"]["state"] == "distracted"


def test_gt_none_on_redacted(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    b = loader.frame(0)
    assert b.gt is None
    assert b.ego["speed_kmh"] == 30.0
    assert b.targets == []
    assert b.depth is None
    assert b.timestamp == 0.0


def test_bundle_mutation_does_not_poison_loader(t01_loader):
    # frame() phải trả copy riêng — downstream annotate tại chỗ không được
    # làm bẩn JSON cache của loader (nền tảng dùng chung cho C1/C2)
    b = t01_loader.frame(299)
    n_targets = len(b.targets)
    b.targets.append({"fake": True})
    b.ego["speed_kmh"] = 9999.0
    b.gt["driver"]["state"] = "hacked"

    b2 = t01_loader.frame(299)
    assert len(b2.targets) == n_targets
    assert b2.ego["speed_kmh"] != 9999.0
    assert b2.gt["driver"]["state"] == "distracted"
    assert t01_loader.raw_frame(299)["driver"]["state"] == "distracted"


def test_bundle_fields(t01_loader):
    b = t01_loader.frame(300)
    assert b.trip_id == "T01-Sample"
    assert b.frame_id == 300
    assert b.timestamp == pytest.approx(15.0)
    assert isinstance(b.targets, list) and b.targets  # T01 frame 300 có target
    assert isinstance(b.events_active, list)
