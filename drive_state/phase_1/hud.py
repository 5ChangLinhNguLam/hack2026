"""The demo overlay.

Drawn here rather than through `drive_state.vendor.overlay` because the two
want different things on screen. That overlay reports the realtime pipeline's
*signals* — eyes closed, drowsy, yawning, distracted, phone use — which are the
internal evidence channels. What Challenge 2 submits is one of five *states*,
and a demo of a classifier should show the classes it chooses between, not the
features underneath.
"""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from drive_state.phase_1.states import DRIVER_STATE_CLASSES

Array = npt.NDArray[np.uint8]

#: BGR. Green for the safe class, amber for degraded attention, red for the two
#: that would raise an alarm — so severity reads before the label does.
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "alert": (118, 200, 96),
    "drowsy": (56, 168, 240),
    "yawning": (72, 196, 228),
    "distracted": (58, 58, 240),
    "microsleep": (48, 48, 250),
}
DIM = (150, 156, 158)
PANEL = (18, 22, 23)


def draw_panel(frame: Array, x: int, y: int, w: int, h: int, *, alpha: float = 0.72) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _text(
    frame: Array,
    value: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def draw_state_panel(
    frame: Array,
    state: str,
    *,
    risk: float,
    latency_ms: float,
    truth: str | None,
    warming_up: bool,
    buffer_filled: int,
    window_frames: int,
) -> None:
    """Top-left: what the classifier decided, and how far it can be trusted."""
    color = STATE_COLORS.get(state, (220, 220, 220))
    if warming_up:
        # A decision made over a partial buffer should not read with the same
        # authority as one made over a full window.
        color = cast(Any, tuple(int(c * 0.5) for c in color))

    height = 116 if truth is None else 140
    draw_panel(frame, 12, 12, 300, height)
    _text(frame, state.upper(), (28, 56), 0.86, color, 2)

    if warming_up:
        _text(frame, f"warming up {buffer_filled}/{window_frames}", (28, 82), 0.44, DIM)
    else:
        _text(frame, f"Risk {risk:.2f}    {latency_ms:.1f} ms", (28, 84), 0.52, (206, 214, 214))

    if truth is not None:
        agree = truth == state
        _text(
            frame,
            f"truth  {truth}",
            (28, 114 if not warming_up else 110),
            0.5,
            (110, 210, 130) if agree else (58, 58, 240),
        )


TRACK = (44, 48, 50)  # empty part of a bar
UNSELECTED_FILL = (138, 144, 146)  # must stay well clear of TRACK, see below


def draw_confidence_panel(frame: Array, confidences: dict[str, float], selected: str) -> None:
    """Top-right: one bar per submitted class, with its value.

    The selected class is drawn in its own colour and the rest in grey, because
    several bars can be full at once — a yawn also closes the eyes, so `yawning`
    and `drowsy` both saturate on T03. The cascade picks by risk order, not by
    height, so colouring only the winner keeps that from reading as a bug.

    The numbers are printed as well as drawn. An earlier version used a grey
    fill close in value to the empty track, which made a full unselected bar
    indistinguishable from an empty one — `drowsy 1.00` looked like `drowsy
    0.00`. Contrast alone is a fragile way to carry a number that matters.
    """
    width, row_h, pad = 250, 30, 14
    height = pad * 2 + row_h * len(DRIVER_STATE_CLASSES)
    x = frame.shape[1] - width - 12
    y = 12
    draw_panel(frame, x, y, width, height, alpha=0.78)

    for index, state in enumerate(DRIVER_STATE_CLASSES):
        value = max(0.0, min(1.0, confidences.get(state, 0.0)))
        row_y = y + pad + index * row_h
        chosen = state == selected
        color = STATE_COLORS[state] if chosen else DIM
        _text(frame, state, (x + 12, row_y + 16), 0.42, color, 2 if chosen else 1)

        bar_x, bar_w = x + 108, 82
        cv2.rectangle(frame, (bar_x, row_y + 4), (bar_x + bar_w, row_y + 17), TRACK, -1)
        filled = int(bar_w * value)
        if filled:
            cv2.rectangle(
                frame,
                (bar_x, row_y + 4),
                (bar_x + filled, row_y + 17),
                STATE_COLORS[state] if chosen else UNSELECTED_FILL,
                -1,
            )
        _text(
            frame, f"{value:.2f}", (bar_x + bar_w + 10, row_y + 16), 0.42, color, 2 if chosen else 1
        )


def draw_face(
    frame: Array,
    state: str,
    bbox: tuple[int, int, int, int] | None,
    landmarks: list[tuple[float, float]],
) -> None:
    color = STATE_COLORS.get(state, (220, 220, 220))
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Every landmark would be a solid mask over the face; a fixed stride keeps
    # the mesh legible at any resolution.
    for point in landmarks[:: max(1, len(landmarks) // 40 or 1)]:
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (110, 224, 255), -1)


def draw_objects(frame: Array, objects: list[dict[str, Any]]) -> None:
    for obj in objects:
        x, y, w, h = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (58, 58, 240), 2)
        label = f"{obj['label']} {obj['confidence']:.2f}"
        _text(frame, label, (x, max(18, y - 8)), 0.44, (58, 58, 240), 1)


def draw_messages(frame: Array, messages: list[str]) -> None:
    """Bottom-left: which rule fired, in the rule's own words."""
    base = frame.shape[0] - 16 - 22 * (len(messages) - 1)
    for index, message in enumerate(messages):
        _text(frame, message, (24, base + index * 22), 0.48, (238, 240, 240))


def draw_footer(frame: Array, stamp: str) -> None:
    _text(
        frame,
        stamp,
        (frame.shape[1] - 236, frame.shape[0] - 16),
        0.46,
        (206, 212, 212),
    )
