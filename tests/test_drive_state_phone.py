from __future__ import annotations

import numpy as np
import pytest

from drive_state.phase_1.classifier import (
    ClassifierConfig,
    classify_window,
    compute_window_features,
)
from drive_state.phase_1.features import FEATURE_FIELDS, FrameFeatures
from drive_state.phase_1.phone import (
    PhoneDetector,
    PhoneObservation,
    create_phone_detector,
)


def make_frame(frame_id: int, **overrides: float) -> FrameFeatures:
    values: dict[str, object] = dict.fromkeys(FEATURE_FIELDS, 0.0)
    values.update(frame_id=frame_id, timestamp=frame_id / 20.0, face_found=1)
    values.update(overrides)
    return FrameFeatures(**values)  # type: ignore[arg-type]


class FakeDetector:
    """Stands in for the ONNX session; records how often it was asked."""

    def __init__(self, confidence: float = 0.9) -> None:
        self.calls = 0
        self.confidence = confidence

    def detect(self, packet: object) -> list[object]:
        self.calls += 1
        from drive_state.vendor.objects import ObjectObservation

        return [
            ObjectObservation(
                label="cell phone", confidence=self.confidence, bbox=(1, 2, 3, 4), provider="fake"
            )
        ]


def make_detector(stride: int, confidence: float = 0.9) -> tuple[PhoneDetector, FakeDetector]:
    detector = PhoneDetector.__new__(PhoneDetector)
    fake = FakeDetector(confidence)
    detector.stride = stride
    detector.confidence = 0.05
    detector._detector = fake  # type: ignore[assignment]
    detector._held = PhoneObservation(0.0, (0, 0, 0, 0), fresh=True)
    detector._seen = 0
    return detector, fake


def test_detector_runs_only_every_stride_frames() -> None:
    """The whole latency argument rests on this: at stride 5 the 123 ms model
    must run 2 times in 10 frames, not 10."""
    detector, fake = make_detector(stride=5)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(10):
        detector.detect(frame, i, i / 20.0)
    assert fake.calls == 2


def test_held_results_are_marked_stale() -> None:
    detector, _ = make_detector(stride=3)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    first = detector.detect(frame, 0, 0.0)
    second = detector.detect(frame, 1, 0.05)
    assert first.fresh
    assert not second.fresh
    # A hold still reports the confidence, or phone_frac would flicker at 1/stride.
    assert second.confidence == pytest.approx(first.confidence)


def test_stride_of_one_runs_every_frame() -> None:
    detector, fake = make_detector(stride=1)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(4):
        assert detector.detect(frame, i, i / 20.0).fresh
    assert fake.calls == 4


def test_rejects_a_stride_below_one() -> None:
    with pytest.raises(ValueError):
        PhoneDetector(stride=0)


def test_disabling_the_detector_is_explicit_but_a_bad_path_is_an_error() -> None:
    """Turning phone detection off costs real accuracy, so it must be a choice
    (`None`) and never the silent consequence of a mistyped path."""
    assert create_phone_detector(None) is None
    with pytest.raises(FileNotFoundError):
        create_phone_detector("models/does-not-exist.onnx")


def test_phone_frac_counts_frames_without_a_face() -> None:
    """Looking down at a handset is exactly when face tracking drops, so the
    phone rate must not be divided by the face-found count."""
    config = ClassifierConfig(window_frames=4, phone_confidence=0.1)
    blank = dict.fromkeys(FEATURE_FIELDS, 0.0)
    rows = [
        FrameFeatures(**{**blank, "frame_id": i, "phone_conf": 0.8})  # type: ignore[arg-type]
        for i in range(4)
    ]
    windows = compute_window_features(rows, config, trailing=True)
    assert windows[3].face_ratio == pytest.approx(0.0)
    assert windows[3].phone_frac == pytest.approx(1.0)


def test_phone_evidence_outranks_drowsiness() -> None:
    """A driver on a handset who is also blinking heavily is, first, on a
    handset -- and `distracted` is what the scorer expects for those frames."""
    config = ClassifierConfig(perclos_drowsy=0.1, phone_distracted=0.5)
    from drive_state.phase_1.classifier import WindowFeatures

    window = WindowFeatures(
        frame_id=0, perclos=0.4, mar_p75=0.0, mouth_open_frac=0.0, face_ratio=1.0, phone_frac=0.9
    )
    assert classify_window(window, config) == "distracted"


def test_mouth_fallback_still_fires_when_no_phone_is_seen() -> None:
    from drive_state.phase_1.classifier import WindowFeatures

    config = ClassifierConfig(mar_talking=0.14, phone_distracted=0.5)
    window = WindowFeatures(
        frame_id=0, perclos=0.0, mar_p75=0.2, mouth_open_frac=0.5, face_ratio=1.0, phone_frac=0.0
    )
    assert classify_window(window, config) == "distracted"


def test_features_default_phone_columns_when_reading_an_older_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Feature CSVs written before the detector existed must still load."""
    from drive_state.phase_1.features import read_features_csv

    legacy = [f for f in FEATURE_FIELDS if not f.startswith("phone_")]
    path = tmp_path / "old.csv"
    path.write_text(
        ",".join(legacy) + "\n" + ",".join("1" for _ in legacy) + "\n", encoding="utf-8"
    )
    rows = read_features_csv(path)
    assert rows[0].phone_conf == 0.0
    assert rows[0].phone_fresh == 0
