"""Combined official-score and product-metric dashboard rendering."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .combined_replay import CombinedFramePrediction


PANEL_SIZE = (640, 360)
FOOTER_HEIGHT = 96
WINDOW_NAME = "SafeLoop C1 + C2 + C3"
WHITE = (245, 245, 245)
GREY = (165, 175, 185)
GREEN = (80, 220, 135)
AMBER = (0, 185, 255)
RED = (70, 70, 255)
CYAN = (235, 210, 65)
BACKGROUND = (12, 20, 25)


def _text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.62,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _ttc_text(value: float) -> str:
    return f"{value:.2f} s" if math.isfinite(value) else "INF"


def _safe_score_color(value: float) -> tuple[int, int, int]:
    if value >= 80.0:
        return GREEN
    if value >= 60.0:
        return AMBER
    return RED


def _event_count(counts: object, name: str) -> int:
    if not hasattr(counts, "get"):
        return 0
    return int(counts.get(name, 0))


def _road_panel(frame: CombinedFramePrediction) -> np.ndarray:
    if frame.source_bundle is None:
        raise ValueError("combined video requires the source FrameBundle")
    image = cv2.resize(frame.source_bundle.left(), PANEL_SIZE).copy()
    c1 = frame.c1
    color = RED if c1.is_warning else (AMBER if c1.predicted_ttc_s < 3.0 else GREEN)
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), BACKGROUND, -1)
    _text(image, "C1  COLLISION RISK", (16, 27), color=CYAN, scale=0.66)
    _text(
        image,
        f"TTC {_ttc_text(c1.display_ttc_s)}  p={c1.collision_probability:.2f}",
        (16, 56),
        color=color,
        scale=0.72,
    )
    update = "MODEL UPDATE" if c1.model_updated else f"HOLD FROM {c1.model_frame_id}"
    _text(image, update, (16, 77), color=GREY, scale=0.43, thickness=1)
    if c1.is_warning:
        cv2.rectangle(image, (image.shape[1] - 145, 15), (image.shape[1] - 16, 63), RED, -1)
        _text(image, "WARNING", (image.shape[1] - 132, 48), scale=0.7)
    return image


def _driver_panel(frame: CombinedFramePrediction) -> np.ndarray:
    if frame.source_bundle is None:
        raise ValueError("combined video requires the source FrameBundle")
    image = cv2.resize(frame.source_bundle.driver(), PANEL_SIZE).copy()
    c2 = frame.c2
    signals = c2.vss_signals()
    warning = bool(signals.is_warning)
    color = RED if warning else GREEN
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), BACKGROUND, -1)
    _text(image, "C2  DRIVER STATE", (16, 27), color=CYAN, scale=0.66)
    _text(
        image,
        f"{str(c2.state).upper()}  confidence={c2.confidence:.2f}",
        (16, 56),
        color=color,
        scale=0.68,
    )
    _text(
        image,
        f"fatigue={signals.fatigue_level:.0f}%  distraction={signals.distraction_level:.0f}%",
        (16, 77),
        color=GREY,
        scale=0.43,
        thickness=1,
    )
    if warning:
        cv2.rectangle(image, (image.shape[1] - 145, 15), (image.shape[1] - 16, 63), RED, -1)
        _text(image, "DMS WARN", (image.shape[1] - 137, 48), scale=0.63)
    return image


def render_combined_dashboard(frame: CombinedFramePrediction) -> np.ndarray:
    """Render one synchronized 1280x456 BGR dashboard frame.

    The footer deliberately keeps three different score contracts visible:
    official C3 and Drive Quality are higher-is-safer, while Context Risk is
    higher-is-more-dangerous.
    """

    content = np.hstack((_road_panel(frame), _driver_panel(frame)))
    footer = np.full(
        (FOOTER_HEIGHT, content.shape[1], 3), BACKGROUND, dtype=np.uint8
    )
    _text(
        footer,
        f"{frame.source_bundle.trip_id}  frame {frame.frame_id}  t={frame.timestamp:.2f}s",
        (16, 25),
        color=WHITE,
        scale=0.5,
        thickness=1,
    )
    c3 = frame.c3
    completion = "FULL_TRIP" if c3.trip_complete else "PREFIX"
    _text(
        footer,
        f"C3 OFFICIAL {c3.safe_score_estimate:.1f}/100 {c3.grade} {completion}",
        (285, 25),
        color=_safe_score_color(c3.safe_score_estimate),
        scale=0.5,
        thickness=1,
    )

    quality = frame.drive_quality
    quality_scope = str(quality.scope)
    if quality.score_available:
        quality_text = (
            f"DRIVE QUALITY {quality.score_pct:.1f}/100 "
            f"{quality.grade} {quality_scope}"
        )
        quality_color = _safe_score_color(float(quality.score_pct))
    else:
        quality_text = f"DRIVE QUALITY N/A {quality_scope}"
        quality_color = GREY
    _text(
        footer,
        quality_text,
        (650, 25),
        color=quality_color,
        scale=0.5,
        thickness=1,
    )

    risk = frame.contextual_risk
    risk_color = RED if risk.level in {"HIGH", "CRITICAL"} else (
        AMBER if risk.level == "CAUTION" else GREEN
    )
    _text(
        footer,
        f"CONTEXT RISK {risk.score_pct:.0f}/100  {risk.level}",
        (1030, 25),
        color=risk_color,
        scale=0.45,
        thickness=1,
    )
    _text(
        footer,
        (
            f"OFFICIAL frames: near {c3.near_miss_frames} | "
            f"brake {c3.harsh_brake_frames} | "
            f"accel {c3.harsh_accel_frames} | corner {c3.harsh_corner_frames} | "
            f"speeding {c3.speeding_pct_time:.1f}%"
        ),
        (16, 55),
        color=GREY,
        scale=0.44,
        thickness=1,
    )
    events = quality.event_counts_window
    _text(
        footer,
        (
            f"DRIVE QUALITY {quality_scope}: events near "
            f"{_event_count(events, 'near_miss')} | brake "
            f"{_event_count(events, 'harsh_brake')} | accel "
            f"{_event_count(events, 'harsh_accel')} | corner "
            f"{_event_count(events, 'harsh_corner')} | speeding "
            f"{quality.speeding_pct_window:.1f}% | penalty "
            f"{quality.total_penalty:.1f}  |  ACTION {risk.action}"
        ),
        (16, 83),
        color=GREY,
        scale=0.42,
        thickness=1,
    )
    return np.vstack((content, footer))


class ReplayVideoWriter:
    """Lazy MP4 writer so dimensions come from the first rendered frame."""

    def __init__(self, path: str | Path, *, fps: float, fourcc: str = "mp4v") -> None:
        if fps <= 0.0:
            raise ValueError("video fps must be positive")
        if len(fourcc) != 4:
            raise ValueError("video fourcc must contain four characters")
        self.path = Path(path)
        self.fps = float(fps)
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self.frames = 0

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*self.fourcc),
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f"cannot open replay video writer: {self.path}")
        self._writer.write(image)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "ReplayVideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "FOOTER_HEIGHT",
    "PANEL_SIZE",
    "ReplayVideoWriter",
    "WINDOW_NAME",
    "render_combined_dashboard",
]
