from types import SimpleNamespace

import cv2
import numpy as np

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
        source_bundle=FakeBundle(),
    )


def test_dashboard_renders_synchronized_road_and_driver_panels() -> None:
    frame = render_combined_dashboard(prediction())
    assert frame.shape == (396, 1280, 3)
    assert frame.dtype == np.uint8


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
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 396
    finally:
        capture.release()
