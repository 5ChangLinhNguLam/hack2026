from __future__ import annotations

from collections import deque

import numpy as np

from c2.demo import render_hud
from c2.live_detector import (
    FaceSignals,
    LiveConfig,
    PhoneDetection,
    StateSnapshot,
    TemporalStateMachine,
    VALID_STATES,
)


def _machine(**overrides) -> TemporalStateMachine:
    values = {
        "calibration_seconds": 0.1,
        "calibration_min_frames": 2,
        "state_hold_seconds": 0.0,
        "microsleep_seconds": 0.5,
        "yawn_seconds": 0.4,
        "offroad_seconds": 0.4,
        "phone_hits_required": 2,
        "phone_window_seconds": 1.0,
        "drowsy_window_seconds": 3.0,
        "drowsy_min_history_seconds": 1.0,
        "drowsy_long_closure_seconds": 0.3,
        "drowsy_long_closures": 2,
    }
    values.update(overrides)
    machine = TemporalStateMachine(LiveConfig(**values))
    neutral = FaceSignals(face_found=True, yaw=5.0, pitch=-2.0)
    machine.update(0.0, neutral)
    result = machine.update(0.1, neutral)
    assert result.calibrated
    return machine


def test_state_vocabulary_matches_c2() -> None:
    assert set(VALID_STATES) == {
        "alert",
        "drowsy",
        "yawning",
        "distracted",
        "microsleep",
    }


def test_yawn_requires_sustained_mouth_opening() -> None:
    machine = _machine()
    opened = FaceSignals(face_found=True, jaw_open=0.8, yaw=5.0, pitch=-2.0)
    result = None
    for timestamp in (0.2, 0.3, 0.4, 0.5, 0.6):
        result = machine.update(timestamp, opened)
    assert result is not None
    assert result.state == "yawning"
    assert "Mouth open" in result.reason


def test_microsleep_has_priority_over_phone_when_mouth_is_neutral() -> None:
    machine = _machine()
    risky = FaceSignals(
        face_found=True,
        jaw_open=0.0,
        eye_blink=0.95,
        yaw=5.0,
        pitch=-2.0,
    )
    result = None
    for index, timestamp in enumerate((0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
        score = 0.8 if index < 2 else None
        result = machine.update(timestamp, risky, score)
    assert result is not None
    assert result.state == "microsleep"
    assert "Eyes closed" in result.reason


def test_strong_yawn_suppresses_eye_squeeze_false_microsleep() -> None:
    machine = _machine()
    yawning = FaceSignals(
        face_found=True,
        jaw_open=0.9,
        eye_blink=0.95,
        yaw=5.0,
        pitch=-2.0,
    )
    result = None
    for timestamp in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        result = machine.update(timestamp, yawning)
    assert result is not None
    assert result.state == "yawning"


def test_phone_needs_two_independent_detector_hits() -> None:
    machine = _machine()
    neutral = FaceSignals(face_found=True, yaw=5.0, pitch=-2.0)
    first = machine.update(0.2, neutral, 0.7)
    second = machine.update(0.3, neutral)
    third = machine.update(0.4, neutral, 0.6)
    assert first.state == "alert"
    assert second.state == "alert"
    assert third.state == "distracted"
    assert "Cell phone detected 2x" in third.reason


def test_head_pose_is_relative_to_neutral_calibration() -> None:
    machine = _machine(offroad_yaw_degrees=15.0)
    away = FaceSignals(face_found=True, yaw=30.0, pitch=-2.0)
    result = None
    for timestamp in (0.2, 0.3, 0.4, 0.5, 0.6):
        result = machine.update(timestamp, away)
    assert result is not None
    assert result.state == "distracted"
    assert "Head/gaze away" in result.reason


def test_eye_closure_in_progress_suppresses_offroad_flicker() -> None:
    machine = _machine(
        offroad_yaw_degrees=15.0,
        offroad_seconds=0.2,
        microsleep_seconds=0.8,
        drowsy_min_history_seconds=5.0,
    )
    closing_away = FaceSignals(
        face_found=True,
        eye_blink=0.7,
        yaw=30.0,
        pitch=-2.0,
    )
    result = None
    for timestamp in (0.2, 0.3, 0.4, 0.5):
        result = machine.update(timestamp, closing_away)
    assert result is not None
    assert result.state == "alert"


def test_repeated_long_eye_closures_create_drowsy_trend() -> None:
    machine = _machine(microsleep_seconds=1.0)
    neutral = FaceSignals(face_found=True, yaw=5.0, pitch=-2.0)
    closed = FaceSignals(face_found=True, eye_blink=0.4, yaw=5.0, pitch=-2.0)
    timeline = [
        (0.2, closed),
        (0.3, closed),
        (0.4, closed),
        (0.5, closed),
        (0.6, neutral),
        (0.7, neutral),
        (0.8, neutral),
        (0.9, neutral),
        (1.0, closed),
        (1.1, closed),
        (1.2, closed),
        (1.3, closed),
        (1.4, closed),
    ]
    result = None
    for timestamp, face in timeline:
        result = machine.update(timestamp, face)
    assert result is not None
    assert result.state == "drowsy"
    assert result.raw["long_closures"] >= 2


def test_reset_requires_new_calibration() -> None:
    machine = _machine()
    machine.reset(5.0)
    result = machine.update(5.0, FaceSignals())
    assert not result.calibrated
    assert result.calibration_progress == 0.0


def test_hud_renders_state_evidence_and_boxes_without_mutating_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    face = FaceSignals(
        face_found=True,
        face_box=(180, 80, 430, 390),
        inference_ms=9.5,
    )
    phone = PhoneDetection(score=0.72, box=(90, 220, 180, 360), inference_ms=24.0)
    snapshot = StateSnapshot(
        state="distracted",
        confidence=0.92,
        reason="Cell phone detected 2x",
        state_seconds=1.5,
        calibrated=True,
        calibration_progress=1.0,
        evidence={
            "eye": 0.1,
            "mouth": 0.2,
            "offroad": 0.4,
            "phone": 1.0,
            "fatigue": 0.0,
        },
    )
    rendered = render_hud(
        frame,
        face,
        phone,
        True,
        snapshot,
        display_fps=20.0,
        events=deque([(3.2, "distracted", snapshot.reason)]),
        source_name="test-camera",
        phone_enabled=True,
    )
    assert rendered.shape == (480, 1060, 3)
    assert int(rendered.sum()) > 0
    assert np.array_equal(frame, original)
