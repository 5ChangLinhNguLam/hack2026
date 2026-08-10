from __future__ import annotations

import math

import numpy as np
import pytest

from safeloop.c1.pipeline import MonocularC1Pipeline
from safeloop.c1.tracker import MonocularTTCTracker, TrackerConfig
from safeloop.c1.types import Detection


def detection_at_scale(scale: float) -> Detection:
    cx, cy = 320.0, 220.0
    width, height = 60.0 * scale, 40.0 * scale
    return Detection(
        class_id=2,
        label="car",
        confidence=0.9,
        bbox=(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
    )


def test_scale_rate_recovers_monocular_ttc() -> None:
    tracker = MonocularTTCTracker(TrackerConfig(history_size=12, min_history=4))
    risks = []
    # apparent scale grows as exp(t/2), so d(log(scale))/dt=0.5 and TTC=2s
    for timestamp in np.arange(0.0, 0.5, 0.05):
        scale = math.exp(timestamp / 2.0)
        risks = tracker.update(
            [detection_at_scale(scale)],
            timestamp=float(timestamp),
            image_shape=(360, 640),
        )

    assert len(risks) == 1
    assert risks[0].collision_relevant is True
    assert risks[0].predicted_ttc_s == pytest.approx(2.0, abs=0.05)


def test_pipeline_only_reads_left_camera() -> None:
    class StubDetector:
        def detect(self, image_bgr):
            assert image_bgr.shape == (12, 20, 3)
            return []

    class OneCameraBundle:
        frame_id = 7
        timestamp = 0.35
        ego = {"speed_kmh": 20.0}
        depth = object()  # pipeline must not inspect it

        def __init__(self):
            self.left_calls = 0

        def left(self):
            self.left_calls += 1
            return np.zeros((12, 20, 3), dtype=np.uint8)

        def right(self):
            raise AssertionError("C1 monocular must not read image_3")

        def driver(self):
            raise AssertionError("C1 must not read the driver camera")

    bundle = OneCameraBundle()
    pipeline = MonocularC1Pipeline(StubDetector(), MonocularTTCTracker())
    prediction, image = pipeline.predict(bundle)  # type: ignore[arg-type]

    assert bundle.left_calls == 1
    assert image.shape == (12, 20, 3)
    assert math.isinf(prediction.predicted_ttc_s)
    assert prediction.submission_row()["predicted_ttc"] == "inf"


def test_ego_fallback_is_optional() -> None:
    disabled = MonocularTTCTracker(TrackerConfig(enable_ego_fallback=False))
    enabled = MonocularTTCTracker(TrackerConfig(enable_ego_fallback=True))
    detection = detection_at_scale(1.0)

    no_fallback = disabled.update(
        [detection], timestamp=0.0, image_shape=(360, 640), ego_speed_kmh=40.0
    )[0]
    with_fallback = enabled.update(
        [detection], timestamp=0.0, image_shape=(360, 640), ego_speed_kmh=40.0
    )[0]

    assert math.isinf(no_fallback.predicted_ttc_s)
    assert math.isfinite(with_fallback.predicted_ttc_s)


def test_range_ttc_subtracts_safety_envelope() -> None:
    tracker = MonocularTTCTracker(
        TrackerConfig(
            history_size=6,
            min_history=3,
            enable_range_ttc=True,
            min_range_history=3,
            range_recent_observations=3,
            enable_fast_attack=False,
        )
    )
    risks = []
    # Effective car range falls from 20 m at 5 m/s.  TTC is measured to the
    # 4.5 m safety envelope: (19 - 4.5) / 5 = 2.9 s at the final sample.
    for timestamp, distance_m in ((0.0, 20.0), (0.1, 19.5), (0.2, 19.0)):
        height = 320.0 * 2.84 / distance_m
        detection = Detection(
            2,
            "car",
            0.9,
            (290.0, 220.0 - height, 350.0, 220.0),
        )
        risks = tracker.update(
            [detection],
            timestamp=timestamp,
            image_shape=(360, 640),
            ego_speed_kmh=30.0,
        )

    assert risks[0].range_ttc_s == pytest.approx(2.9, abs=0.05)
    assert risks[0].predicted_ttc_s == pytest.approx(2.9, abs=0.05)


def test_detector_stride_reuses_tracks_but_not_fake_observations() -> None:
    class CountingDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, image_bgr):
            self.calls += 1
            return [detection_at_scale(1.0 + self.calls * 0.05)]

    class Bundle:
        ego = {"speed_kmh": 20.0}
        depth = None

        def __init__(self, frame_id: int):
            self.frame_id = frame_id
            self.timestamp = frame_id / 20.0

        def left(self):
            return np.zeros((360, 640, 3), dtype=np.uint8)

    detector = CountingDetector()
    tracker = MonocularTTCTracker(TrackerConfig(min_history=2))
    pipeline = MonocularC1Pipeline(detector, tracker, detector_stride=3)

    predictions = [pipeline.predict(Bundle(i))[0] for i in range(7)]

    assert detector.calls == 3  # frames 0, 3, 6
    assert predictions[1].risks  # skipped frame still has the current track
    assert len(tracker._tracks[0].history) == 3  # only actual detector boxes


def test_cross_class_cut_in_keeps_track_and_produces_lateral_ttc() -> None:
    tracker = MonocularTTCTracker(
        TrackerConfig(min_history=4, enable_fast_attack=False)
    )
    sequence = [
        (0.00, "car", 2, (452.0, 189.0, 552.0, 265.0)),
        (0.30, "person", 0, (401.0, 179.0, 506.0, 340.0)),
        (0.60, "person", 0, (347.0, 175.0, 429.0, 332.0)),
    ]
    risks = []
    for timestamp, label, class_id, bbox in sequence:
        risks = tracker.update(
            [Detection(class_id, label, 0.8, bbox)],
            timestamp=timestamp,
            image_shape=(360, 640),
            ego_speed_kmh=29.0,
        )

    assert len(tracker._tracks) == 1
    assert risks[0].track_id == 1
    assert risks[0].collision_relevant is True
    assert risks[0].lateral_ttc_s < 1.0
    assert risks[0].predicted_ttc_s < 1.0


def test_fast_attack_and_short_coast_cover_detector_blink() -> None:
    tracker = MonocularTTCTracker(TrackerConfig())
    cut_in = Detection(2, "car", 0.8, (452.0, 189.0, 552.0, 265.0))

    first = tracker.update(
        [cut_in],
        timestamp=0.0,
        image_shape=(360, 640),
        ego_speed_kmh=29.0,
    )
    coast = tracker.update(
        [],
        timestamp=0.15,
        image_shape=(360, 640),
        ego_speed_kmh=29.0,
    )
    skipped = tracker.predict(
        timestamp=0.20,
        image_shape=(360, 640),
        ego_speed_kmh=29.0,
    )

    assert first[0].predicted_ttc_s < 2.0
    assert coast[0].track_id == first[0].track_id
    assert coast[0].predicted_ttc_s < first[0].predicted_ttc_s
    assert skipped[0].track_id == first[0].track_id
    assert skipped[0].predicted_ttc_s < coast[0].predicted_ttc_s
