from types import SimpleNamespace

import cv2
import numpy as np

import safeloop.replay_video as replay_video
from safeloop.combined_replay import CombinedFramePrediction
from safeloop.replay_video import ReplayVideoWriter, render_combined_dashboard


class FakeBundle:
    trip_id = "T01-Sample"

    def __init__(self) -> None:
        self.road = np.full((180, 320, 3), 40, dtype=np.uint8)
        self.cabin = np.full((180, 320, 3), 80, dtype=np.uint8)

    def left(self) -> np.ndarray:
        return self.road

    def driver(self) -> np.ndarray:
        return self.cabin


class FakeDmsPrediction:
    frame_id = 2
    timestamp = 0.1
    state = "alert"
    confidence = 0.9

    def submission_row(self):
        return {"predicted_driver_state": self.state}

    def vss_signals(self):
        return SimpleNamespace(
            fatigue_level=5.0,
            distraction_level=3.0,
            is_warning=False,
        )


def prediction() -> CombinedFramePrediction:
    c1 = SimpleNamespace(
        frame_id=2,
        timestamp=0.1,
        collision_probability=0.25,
        predicted_ttc_s=4.0,
        display_ttc_s=4.2,
        is_warning=False,
        model_updated=True,
        model_frame_id=2,
        submission_row=lambda: {"predicted_ttc": 4.0},
    )
    return CombinedFramePrediction(
        frame_id=2,
        timestamp=0.1,
        c1=c1,
        c2=FakeDmsPrediction(),
        c3=SimpleNamespace(
            frame_id=2,
            timestamp=0.1,
            safe_score_estimate=81.0,
            grade="B",
            trip_complete=False,
            near_miss_frames=1,
            harsh_brake_frames=2,
            harsh_accel_frames=3,
            harsh_corner_frames=4,
            speeding_pct_time=5.0,
        ),
        drive_quality=SimpleNamespace(
            frame_id=2,
            timestamp=0.1,
            score_available=True,
            score_pct=91.0,
            grade="A",
            scope="PREFIX",
            total_penalty=9.0,
            event_counts_window={
                "near_miss": 0,
                "harsh_brake": 1,
                "harsh_accel": 1,
                "harsh_corner": 0,
            },
            speeding_pct_window=5.0,
        ),
        contextual_risk=SimpleNamespace(
            score_pct=72.0,
            level="HIGH",
            action="VISUAL_AUDIO_HAPTIC_WARNING",
        ),
        source_bundle=FakeBundle(),
    )


def test_dashboard_renders_synchronized_road_and_driver_panels() -> None:
    frame = render_combined_dashboard(prediction())
    assert frame.shape == (456, 1280, 3)
    assert frame.dtype == np.uint8


def test_dashboard_labels_official_product_and_instantaneous_scores(
    monkeypatch,
) -> None:
    labels: list[str] = []
    original = replay_video._text

    def capture(image, value, position, **kwargs):
        labels.append(value)
        return original(image, value, position, **kwargs)

    monkeypatch.setattr(replay_video, "_text", capture)
    render_combined_dashboard(prediction())

    assert any("C3 OFFICIAL 81.0/100 B PREFIX" in value for value in labels)
    assert any("DRIVE QUALITY 91.0/100 A PREFIX" in value for value in labels)
    assert any("CONTEXT RISK 72/100  HIGH" in value for value in labels)
    assert any("OFFICIAL frames:" in value for value in labels)
    assert any(
        "events near 0 | brake 1 | accel 1 | corner 0" in value
        for value in labels
    )


def test_dashboard_marks_unavailable_drive_quality(monkeypatch) -> None:
    labels: list[str] = []
    original = replay_video._text
    frame = prediction()
    frame.drive_quality.score_available = False

    def capture(image, value, position, **kwargs):
        labels.append(value)
        return original(image, value, position, **kwargs)

    monkeypatch.setattr(replay_video, "_text", capture)
    render_combined_dashboard(frame)

    assert any("DRIVE QUALITY N/A PREFIX" in value for value in labels)


def test_mp4_writer_records_rendered_frames(tmp_path) -> None:
    path = tmp_path / "combined.mp4"
    dashboard = render_combined_dashboard(prediction())
    with ReplayVideoWriter(path, fps=20.0) as writer:
        writer.write(dashboard)
        writer.write(dashboard)
        assert writer.frames == 2

    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1280
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 456
    finally:
        capture.release()
