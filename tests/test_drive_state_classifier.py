from __future__ import annotations

import pytest

from drive_state.phase_1.classifier import (
    ClassifierConfig,
    WindowFeatures,
    classify_sequence,
    classify_window,
    compute_window_features,
)
from drive_state.phase_1.features import FEATURE_FIELDS, FrameFeatures


def make_frame(frame_id: int, **overrides: float) -> FrameFeatures:
    values: dict[str, object] = dict.fromkeys(FEATURE_FIELDS, 0.0)
    values.update(frame_id=frame_id, timestamp=frame_id / 20.0, face_found=1)
    values.update(overrides)
    return FrameFeatures(**values)  # type: ignore[arg-type]


def window(**overrides: float) -> WindowFeatures:
    base = {
        "frame_id": 0,
        "perclos": 0.0,
        "mar_p75": 0.0,
        "mouth_open_frac": 0.0,
        "face_ratio": 1.0,
    }
    base.update(overrides)
    return WindowFeatures(**base)  # type: ignore[arg-type]


def test_quiet_face_reads_as_alert() -> None:
    assert classify_window(window()) == "alert"


def test_sustained_closure_reads_as_microsleep() -> None:
    assert classify_window(window(perclos=0.95)) == "microsleep"


def test_partial_closure_reads_as_drowsy() -> None:
    assert classify_window(window(perclos=0.30)) == "drowsy"


def test_yawn_outranks_drowsy_despite_the_squint() -> None:
    """A yawn closes the eyes too -- PERCLOS is 0.36 across T03's yawning
    frames -- so the mouth has to be checked before PERCLOS or every yawn
    would be reported as drowsiness."""
    assert classify_window(window(perclos=0.36, mar_p75=0.63)) == "yawning"


def test_moderate_mouth_activity_reads_as_distracted() -> None:
    # Talking on the phone: mouth moving, eyes open, well short of a yawn.
    assert classify_window(window(perclos=0.03, mar_p75=0.20)) == "distracted"


def test_mouth_activity_below_the_talking_threshold_stays_alert() -> None:
    assert classify_window(window(mar_p75=0.05)) == "alert"


def test_lost_face_falls_back_rather_than_reading_as_closed_eyes() -> None:
    """No landmarks means no eye evidence. Treating that as closed eyes would
    turn every tracking dropout into a microsleep alarm."""
    assert classify_window(window(face_ratio=0.1, perclos=0.0)) == "distracted"


def test_window_features_average_over_the_window_not_the_frame() -> None:
    config = ClassifierConfig(window_frames=5, blink_closed=0.5)
    rows = [make_frame(i, blink=0.9 if i < 5 else 0.0) for i in range(10)]
    windows = compute_window_features(rows, config)

    assert windows[0].perclos == pytest.approx(1.0)  # truncated window, all closed
    assert windows[9].perclos == pytest.approx(0.0)
    # Frame 5 sits on the boundary: frames 3,4 closed and 5,6,7 open.
    assert windows[5].perclos == pytest.approx(2 / 5)


def test_window_features_ignore_frames_with_no_face() -> None:
    """A dropped frame must not count as an open eye and dilute PERCLOS."""
    config = ClassifierConfig(window_frames=3, blink_closed=0.5)
    rows = [
        make_frame(0, blink=0.9),
        FrameFeatures(**{**dict.fromkeys(FEATURE_FIELDS, 0.0), "frame_id": 1}),  # type: ignore[arg-type]
        make_frame(2, blink=0.9),
    ]
    windows = compute_window_features(rows, config)
    assert windows[1].perclos == pytest.approx(1.0)
    assert windows[1].face_ratio == pytest.approx(2 / 3)


def test_empty_input_produces_no_windows() -> None:
    assert compute_window_features([]) == []


def test_classify_sequence_returns_one_state_per_frame() -> None:
    rows = [make_frame(i) for i in range(30)]
    states = classify_sequence(rows, ClassifierConfig(window_frames=5))
    assert len(states) == len(rows)
    assert set(states) == {"alert"}
