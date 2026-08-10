from __future__ import annotations

import cv2
import numpy as np

from safeloop.c1.lane import LaneDetector, LaneSelfEvaluator


def test_lane_detector_finds_synthetic_lane() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(image, (120, 359), (290, 190), (255, 255, 255), 8)
    cv2.line(image, (520, 359), (350, 190), (0, 255, 255), 8)
    detector = LaneDetector()

    estimate = detector.detect(image)

    assert estimate.valid
    assert estimate.confidence >= 0.5
    assert abs(estimate.lane_center_offset) < 0.15
    assert 250 < estimate.lane_width_px < 500


def test_lane_blank_frame_is_invalid_and_proxy_is_named_honestly() -> None:
    detector = LaneDetector()
    estimate = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
    evaluator = LaneSelfEvaluator(640)
    evaluator.add(estimate)
    report = evaluator.report()

    assert not estimate.valid
    assert report.valid_fraction == 0.0
    assert "not_lane_accuracy" in report.metric_name


def test_lane_detector_infers_temporarily_missing_side() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(image, (120, 359), (290, 190), (255, 255, 255), 8)

    estimate = LaneDetector().detect(image)

    assert estimate.valid
    assert estimate.inferred_side == "right"
    assert estimate.right_points
