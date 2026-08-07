"""OpenCV presentation helpers for the causal five-state runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from .data.five_state_labels import FiveState
from .face_detection import FaceDetection
from .runtime_mobilenet_lstm import RuntimeFiveStatePrediction
from .training.ocular_trainer import PHASE_NAMES


KeyboardCommand = Literal["quit", "reset", "screenshot", "pause"]


@dataclass(frozen=True)
class DashboardView:
    state_id: int
    state_name: str
    state_confidence: float
    probabilities: tuple[float, ...]
    eye_phase: str
    eye_phase_confidence: float
    ocular_phase_probabilities: tuple[float, ...]
    closed_probability: float
    ocular_reliability: float
    closure_duration_seconds: float
    perclos: tuple[float, float, float]
    slow_perclos: tuple[float, float, float]
    perclos_reliable: tuple[bool, bool, bool]
    nod_probability: float
    microsleep_active: bool


@dataclass(frozen=True)
class DashboardTelemetry:
    processing_fps: float
    latency_ms: float
    face_visible: bool
    face_confidence: float = 0.0
    paused: bool = False
    message: str = ""


def _normalized_probabilities(
    values: Sequence[float],
    *,
    expected: int,
    name: str,
) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    if len(probabilities) != expected:
        raise ValueError(f"{name} must contain {expected} probabilities")
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-4):
        raise ValueError(f"{name} probabilities must sum to one")
    return probabilities


def build_dashboard_view(
    prediction: RuntimeFiveStatePrediction,
) -> DashboardView:
    """Validate runtime output and reduce it to one exclusive UI state."""

    probabilities = _normalized_probabilities(
        prediction.probabilities,
        expected=len(FiveState),
        name="five-state",
    )
    state_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    if int(prediction.state_id) != state_id:
        raise ValueError("runtime exclusive state disagrees with softmax argmax")
    ocular = _normalized_probabilities(
        prediction.ocular_phase_probabilities,
        expected=len(PHASE_NAMES),
        name="ocular phase",
    )
    eye_phase_id = max(range(len(ocular)), key=ocular.__getitem__)
    return DashboardView(
        state_id=state_id,
        state_name=FiveState(state_id).name.lower(),
        state_confidence=probabilities[state_id],
        probabilities=probabilities,
        eye_phase=PHASE_NAMES[eye_phase_id],
        eye_phase_confidence=ocular[eye_phase_id],
        ocular_phase_probabilities=ocular,
        closed_probability=float(prediction.closed_probability),
        ocular_reliability=float(prediction.ocular_reliability),
        closure_duration_seconds=float(
            prediction.closure_duration_seconds
        ),
        perclos=tuple(float(value) for value in prediction.perclos),
        slow_perclos=tuple(float(value) for value in prediction.slow_perclos),
        perclos_reliable=tuple(bool(value) for value in prediction.perclos_reliable),
        nod_probability=float(prediction.nod_probability),
        microsleep_active=bool(prediction.microsleep_active),
    )


class FaceLossResetPolicy:
    """Request one temporal reset after each sustained face-loss episode."""

    def __init__(self, *, reset_after_frames: int) -> None:
        if reset_after_frames <= 0:
            raise ValueError("face-loss reset frames must be positive")
        self.reset_after_frames = int(reset_after_frames)
        self.missing_frames = 0
        self.reset_requested = False

    def reset(self) -> None:
        self.missing_frames = 0
        self.reset_requested = False

    def update(self, *, face_visible: bool) -> bool:
        if face_visible:
            self.reset()
            return False
        self.missing_frames += 1
        if (
            not self.reset_requested
            and self.missing_frames >= self.reset_after_frames
        ):
            self.reset_requested = True
            return True
        return False


def decode_key(key: int) -> KeyboardCommand | None:
    normalized = int(key) & 0xFF
    if normalized in (27, ord("q"), ord("Q")):
        return "quit"
    if normalized in (ord("r"), ord("R")):
        return "reset"
    if normalized in (ord("s"), ord("S")):
        return "screenshot"
    if normalized == ord(" "):
        return "pause"
    return None


class OpenCVDashboardRenderer:
    """Render a live camera frame and causal evidence into one OpenCV canvas."""

    STATE_COLORS = (
        (56, 189, 112),
        (30, 170, 255),
        (66, 66, 230),
        (80, 185, 255),
        (210, 120, 55),
    )

    def __init__(self, cv2_module, *, mirror: bool = True) -> None:
        self.cv2 = cv2_module
        self.mirror = bool(mirror)

    def _text(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        *,
        scale: float = 0.55,
        color: tuple[int, int, int] = (226, 232, 240),
        thickness: int = 1,
    ) -> None:
        self.cv2.putText(
            image,
            text,
            origin,
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            self.cv2.LINE_AA,
        )

    def _bar(
        self,
        canvas: np.ndarray,
        *,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        value = min(max(float(value), 0.0), 1.0)
        self._text(canvas, label, (x, y + 14), scale=0.45)
        bar_x = x + 105
        bar_width = max(20, width - 160)
        self.cv2.rectangle(
            canvas,
            (bar_x, y),
            (bar_x + bar_width, y + 15),
            (55, 65, 81),
            -1,
        )
        self.cv2.rectangle(
            canvas,
            (bar_x, y),
            (bar_x + round(bar_width * value), y + 15),
            color,
            -1,
        )
        self._text(
            canvas,
            f"{value:5.1%}",
            (bar_x + bar_width + 8, y + 14),
            scale=0.43,
        )

    def render(
        self,
        frame: np.ndarray,
        prediction: RuntimeFiveStatePrediction,
        *,
        telemetry: DashboardTelemetry,
        detection: FaceDetection | None = None,
    ) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("OpenCV frame must have shape [height, width, 3]")
        view = build_dashboard_view(prediction)
        video = self.cv2.flip(frame, 1) if self.mirror else frame.copy()
        frame_height, frame_width = video.shape[:2]
        panel_width = 430
        canvas_height = max(frame_height, 720)
        canvas = np.full(
            (canvas_height, frame_width + panel_width, 3),
            (20, 25, 34),
            dtype=np.uint8,
        )
        canvas[:frame_height, :frame_width] = video

        if detection is not None:
            box = detection.box
            x1, x2 = box.x1, box.x2
            if self.mirror:
                x1, x2 = frame_width - box.x2, frame_width - box.x1
            color = (72, 196, 125) if telemetry.face_visible else (80, 80, 230)
            self.cv2.rectangle(
                canvas,
                (x1, box.y1),
                (x2, box.y2),
                color,
                2,
            )
            self._text(
                canvas,
                f"face {telemetry.face_confidence:.0%}",
                (x1, max(20, box.y1 - 8)),
                color=color,
                thickness=2,
            )

        if not telemetry.face_visible:
            overlay = canvas[:frame_height, :frame_width].copy()
            self.cv2.rectangle(
                overlay,
                (0, 0),
                (frame_width, 70),
                (20, 20, 110),
                -1,
            )
            self.cv2.addWeighted(
                overlay,
                0.72,
                canvas[:frame_height, :frame_width],
                0.28,
                0.0,
                canvas[:frame_height, :frame_width],
            )
            self._text(
                canvas,
                "FACE NOT DETECTED - temporal history protected",
                (22, 44),
                scale=0.7,
                color=(235, 235, 255),
                thickness=2,
            )

        panel_x = frame_width
        state_color = self.STATE_COLORS[view.state_id]
        self.cv2.rectangle(
            canvas,
            (panel_x, 0),
            (panel_x + panel_width, 94),
            state_color,
            -1,
        )
        self._text(
            canvas,
            view.state_name.upper(),
            (panel_x + 22, 43),
            scale=1.05,
            color=(255, 255, 255),
            thickness=2,
        )
        self._text(
            canvas,
            f"exclusive confidence {view.state_confidence:.1%}",
            (panel_x + 23, 74),
            scale=0.52,
            color=(255, 255, 255),
        )

        x = panel_x + 22
        width = panel_width - 40
        self._text(canvas, "FIVE-STATE SOFTMAX", (x, 125), thickness=2)
        for index, (state, probability) in enumerate(
            zip(FiveState, view.probabilities, strict=True)
        ):
            self._bar(
                canvas,
                label=state.name.lower(),
                value=probability,
                x=x,
                y=142 + index * 26,
                width=width,
                color=self.STATE_COLORS[index],
            )

        self._text(canvas, "EYE PHASE", (x, 294), thickness=2)
        for index, (phase, probability) in enumerate(
            zip(PHASE_NAMES, view.ocular_phase_probabilities, strict=True)
        ):
            self._bar(
                canvas,
                label=phase,
                value=probability,
                x=x,
                y=311 + index * 25,
                width=width,
                color=(170, 120, 235),
            )

        y = 433
        self._text(canvas, "TEMPORAL EVIDENCE", (x, y), thickness=2)
        lines = (
            f"closed probability   {view.closed_probability:6.1%}",
            f"ocular reliability  {view.ocular_reliability:6.1%}",
            f"closure duration    {view.closure_duration_seconds:6.2f}s",
            f"head nod            {view.nod_probability:6.1%}",
            "microsleep gate      "
            + ("ACTIVE" if view.microsleep_active else "inactive"),
        )
        for index, line in enumerate(lines):
            color = (
                (80, 80, 255)
                if index == 4 and view.microsleep_active
                else (226, 232, 240)
            )
            self._text(canvas, line, (x, y + 28 + index * 23), color=color)

        y = 578
        self._text(canvas, "PERCLOS", (x, y), thickness=2)
        for index, seconds in enumerate((10, 30, 60)):
            reliable = view.perclos_reliable[index]
            suffix = "ready" if reliable else "warming"
            self._text(
                canvas,
                f"{seconds:>2}s {view.perclos[index]:6.1%}  {suffix}",
                (x, y + 27 + index * 22),
                color=(226, 232, 240) if reliable else (145, 155, 170),
            )

        footer_y = canvas_height - 46
        self.cv2.line(
            canvas,
            (panel_x, footer_y - 15),
            (panel_x + panel_width, footer_y - 15),
            (55, 65, 81),
            1,
        )
        status = (
            f"{telemetry.processing_fps:4.1f} FPS  "
            f"{telemetry.latency_ms:5.1f} ms"
        )
        if telemetry.paused:
            status += "  PAUSED"
        self._text(canvas, status, (x, footer_y), thickness=2)
        self._text(
            canvas,
            telemetry.message or "Q quit  R reset  Space pause  S capture",
            (x, footer_y + 25),
            scale=0.42,
            color=(145, 155, 170),
        )
        return canvas


__all__ = [
    "DashboardTelemetry",
    "DashboardView",
    "FaceLossResetPolicy",
    "OpenCVDashboardRenderer",
    "build_dashboard_view",
    "decode_key",
]
