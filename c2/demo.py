"""C2 realtime driver-state demo with an OpenCV HUD.

Examples
--------
Webcam (default):
    python c2/demo.py

Webcam without the optional phone detector:
    python c2/demo.py --no-phone

Recorded DMD/hackathon video:
    python c2/demo.py --video path/to/video.mp4

Automated smoke test without opening a window:
    python c2/demo.py --video path/to/video.mp4 --headless --max-frames 120

Controls: Q/Esc quit, C recalibrate, R start a fresh event timeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:  # Support both ``python c2/demo.py`` and ``python -m c2.demo``.
    from .live_detector import (
        CellPhoneDetector,
        FaceSignalExtractor,
        FaceSignals,
        LiveConfig,
        PhoneDetection,
        StateSnapshot,
        TemporalStateMachine,
    )
except ImportError:
    from live_detector import (
        CellPhoneDetector,
        FaceSignalExtractor,
        FaceSignals,
        LiveConfig,
        PhoneDetection,
        StateSnapshot,
        TemporalStateMachine,
    )


STATE_COLORS = {
    "alert": (76, 188, 76),
    "drowsy": (38, 155, 236),
    "yawning": (0, 205, 235),
    "distracted": (52, 66, 231),
    "microsleep": (179, 70, 190),
}
PANEL_BG = (24, 26, 31)
TEXT = (235, 238, 242)
MUTED = (150, 158, 170)


@dataclass(slots=True)
class DemoStats:
    frames: int = 0
    face_frames: int = 0
    face_ms_sum: float = 0.0
    phone_runs: int = 0
    phone_ms_sum: float = 0.0
    states: Counter | None = None

    def __post_init__(self) -> None:
        if self.states is None:
            self.states = Counter()


def _text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
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


def _fit_frame(frame: np.ndarray, max_width: int = 840, max_height: int = 600) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return frame
    return cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _scale_box(
    box: tuple[int, int, int, int] | None,
    source_shape: tuple[int, ...],
    destination_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    source_height, source_width = source_shape[:2]
    destination_height, destination_width = destination_shape[:2]
    sx = destination_width / source_width
    sy = destination_height / source_height
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def _bar(
    canvas: np.ndarray,
    label: str,
    value: float,
    origin: tuple[int, int],
    width: int,
    *,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    value = float(np.clip(value, 0.0, 1.0))
    _text(canvas, label.upper(), (x, y - 7), scale=0.42, color=MUTED)
    cv2.rectangle(canvas, (x, y), (x + width, y + 11), (55, 59, 68), -1)
    cv2.rectangle(canvas, (x, y), (x + int(width * value), y + 11), color, -1)
    _text(canvas, f"{value:>4.0%}", (x + width + 10, y + 10), scale=0.42)


def render_hud(
    frame: np.ndarray,
    face: FaceSignals,
    phone: PhoneDetection,
    phone_visible: bool,
    snapshot: StateSnapshot,
    *,
    display_fps: float,
    events: deque[tuple[float, str, str]],
    source_name: str,
    phone_enabled: bool,
) -> np.ndarray:
    view = _fit_frame(frame)
    face_box = _scale_box(face.face_box, frame.shape, view.shape)
    phone_box = _scale_box(phone.box, frame.shape, view.shape)
    height, width = view.shape[:2]
    panel_width = 420
    canvas = np.full((height, width + panel_width, 3), PANEL_BG, dtype=np.uint8)
    canvas[:, :width] = view

    calibrated = snapshot.calibrated
    state = snapshot.state
    color = STATE_COLORS.get(state, (190, 190, 190))
    if not calibrated:
        color = (210, 140, 35)

    overlay = canvas[:, :width].copy()
    cv2.rectangle(overlay, (0, 0), (width, 74), color, -1)
    cv2.addWeighted(overlay, 0.72, canvas[:, :width], 0.28, 0, canvas[:, :width])
    title = state.upper() if calibrated else "CALIBRATING"
    _text(canvas, title, (22, 43), scale=1.05, color=(255, 255, 255), thickness=2)
    if calibrated:
        _text(
            canvas,
            f"{snapshot.confidence:.0%}  |  {snapshot.state_seconds:.1f}s",
            (width - 175, 42),
            scale=0.54,
            color=(255, 255, 255),
        )
    else:
        bar_width = 150
        x0 = width - bar_width - 22
        cv2.rectangle(canvas, (x0, 31), (x0 + bar_width, 43), (255, 255, 255), 1)
        cv2.rectangle(
            canvas,
            (x0 + 2, 33),
            (
                x0 + 2 + int((bar_width - 3) * snapshot.calibration_progress),
                41,
            ),
            (255, 255, 255),
            -1,
        )

    if face_box is not None:
        x0, y0, x1, y1 = face_box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 230, 126), 2)
        _text(canvas, "FACE", (x0, max(91, y0 - 7)), scale=0.42, color=(92, 230, 126))
    if phone_visible and phone_box is not None:
        x0, y0, x1, y1 = phone_box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), STATE_COLORS["distracted"], 3)
        _text(
            canvas,
            f"PHONE {phone.score:.2f}",
            (x0, max(91, y0 - 8)),
            scale=0.52,
            color=STATE_COLORS["distracted"],
            thickness=2,
        )

    panel_x = width + 24
    _text(canvas, "C2 DRIVER MONITOR", (panel_x, 34), scale=0.72, thickness=2)
    _text(canvas, source_name, (panel_x, 57), scale=0.43, color=MUTED)
    cv2.line(canvas, (panel_x, 70), (width + panel_width - 24, 70), (58, 62, 71), 1)

    _text(canvas, "WHY", (panel_x, 99), scale=0.43, color=MUTED)
    reason = snapshot.reason
    words = reason.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = f"{current} {word}".strip()
        if len(proposal) > 44 and current:
            lines.append(current)
            current = word
        else:
            current = proposal
    if current:
        lines.append(current)
    for index, line in enumerate(lines[:2]):
        _text(canvas, line, (panel_x, 123 + index * 21), scale=0.48)

    _text(canvas, "LIVE EVIDENCE", (panel_x, 165), scale=0.43, color=MUTED)
    bar_width = 260
    evidence_colors = {
        "eye": (179, 70, 190),
        "mouth": (0, 205, 235),
        "offroad": (52, 120, 231),
        "phone": (52, 66, 231),
        "fatigue": (38, 155, 236),
    }
    for index, key in enumerate(("eye", "mouth", "offroad", "phone", "fatigue")):
        _bar(
            canvas,
            key,
            snapshot.evidence.get(key, 0.0),
            (panel_x, 188 + index * 32),
            bar_width,
            color=evidence_colors[key],
        )

    status_y = 348 if height < 520 else 378
    _text(canvas, "SYSTEM", (panel_x, status_y), scale=0.43, color=MUTED)
    _text(
        canvas,
        f"FPS {display_fps:4.1f}   Face {face.inference_ms:4.1f} ms",
        (panel_x, status_y + 25),
        scale=0.48,
    )
    phone_status = (
        f"{phone.inference_ms:4.1f} ms / {phone.score:.2f}"
        if phone_enabled
        else "disabled"
    )
    _text(
        canvas,
        f"Phone detector: {phone_status}",
        (panel_x, status_y + 47),
        scale=0.45,
        color=MUTED,
    )

    timeline_y = status_y + 79
    if timeline_y + 25 < height:
        _text(canvas, "RECENT EVENTS", (panel_x, timeline_y), scale=0.43, color=MUTED)
        max_events = max(1, min(3, (height - timeline_y - 36) // 20))
        for index, (event_time, event_state, _reason) in enumerate(
            list(events)[-max_events:]
        ):
            event_color = STATE_COLORS.get(event_state, TEXT)
            _text(
                canvas,
                f"{event_time:6.1f}s  {event_state.upper()}",
                (panel_x, timeline_y + 23 + index * 20),
                scale=0.43,
                color=event_color,
            )

    controls = "Q quit   C recalibrate   R clear events"
    _text(canvas, controls, (18, height - 14), scale=0.43, color=(232, 232, 232))
    return canvas


def _open_source(args: argparse.Namespace) -> tuple[cv2.VideoCapture, str, bool]:
    if args.video is not None:
        path = Path(args.video).expanduser().resolve()
        capture = cv2.VideoCapture(str(path))
        return capture, path.name, False
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return capture, f"webcam:{args.camera}", not args.no_mirror


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    source.add_argument("--video", help="video file instead of webcam")
    parser.add_argument("--width", type=int, default=640, help="requested webcam width")
    parser.add_argument("--height", type=int, default=480, help="requested webcam height")
    parser.add_argument(
        "--phone-every",
        type=int,
        default=5,
        help="run phone detection every N frames (default 5)",
    )
    parser.add_argument("--no-phone", action="store_true", help="disable phone detector")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror webcam")
    parser.add_argument("--record", help="optional output .mp4 with the rendered HUD")
    parser.add_argument("--headless", action="store_true", help="run without an OpenCV window")
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="seek to this position when using --video",
    )
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument("--max-seconds", type=float, help="stop after N source seconds")
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=2.0,
        help="neutral calibration duration (default 2.0)",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    capture, source_name, mirror = _open_source(args)
    if not capture.isOpened():
        print(f"ERROR: cannot open {source_name}", file=sys.stderr)
        return 2

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 1.0:
        source_fps = 20.0
    is_video = args.video is not None
    if is_video and args.start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)
        source_name = f"{source_name} @ {args.start_seconds:.1f}s"

    config = LiveConfig(calibration_seconds=max(0.1, args.calibration_seconds))
    state_machine = TemporalStateMachine(config)
    face_extractor = FaceSignalExtractor()
    phone_detector = None if args.no_phone else CellPhoneDetector()

    stats = DemoStats()
    events: deque[tuple[float, str, str]] = deque(maxlen=20)
    last_event_state: str | None = None
    latest_phone = PhoneDetection()
    latest_phone_time = -1e9
    frame_index = 0
    wall_started = time.perf_counter()
    fps_started = wall_started
    fps_frames = 0
    display_fps = 0.0
    writer: cv2.VideoWriter | None = None
    window_name = "C2 Realtime Driver Monitor"

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            if is_video:
                source_seconds = frame_index / source_fps
            else:
                source_seconds = frame_started - wall_started
            timestamp_ms = int(round(source_seconds * 1000.0))

            face = face_extractor.process(frame, timestamp_ms)
            phone_update: float | None = None
            phone_every = max(1, args.phone_every)
            if phone_detector is not None and frame_index % phone_every == 0:
                latest_phone = phone_detector.process(frame)
                latest_phone_time = source_seconds
                phone_update = latest_phone.score
                stats.phone_runs += 1
                stats.phone_ms_sum += latest_phone.inference_ms

            snapshot = state_machine.update(source_seconds, face, phone_update)
            if snapshot.calibrated and snapshot.state != last_event_state:
                events.append((source_seconds, snapshot.state, snapshot.reason))
                last_event_state = snapshot.state

            stats.frames += 1
            stats.face_frames += int(face.face_found)
            stats.face_ms_sum += face.inference_ms
            if snapshot.calibrated:
                stats.states[snapshot.state] += 1

            fps_frames += 1
            now = time.perf_counter()
            fps_elapsed = now - fps_started
            if fps_elapsed >= 0.5:
                display_fps = fps_frames / fps_elapsed
                fps_started = now
                fps_frames = 0

            phone_visible = (
                latest_phone.box is not None
                and source_seconds - latest_phone_time <= 0.8
            )
            hud = render_hud(
                frame,
                face,
                latest_phone,
                phone_visible,
                snapshot,
                display_fps=display_fps,
                events=events,
                source_name=source_name,
                phone_enabled=phone_detector is not None,
            )

            if args.record:
                if writer is None:
                    destination = Path(args.record).expanduser().resolve()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(destination),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (hud.shape[1], hud.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer: {destination}")
                writer.write(hud)

            key = -1
            if not args.headless:
                cv2.imshow(window_name, hud)
                if is_video:
                    spent_ms = (time.perf_counter() - frame_started) * 1000.0
                    delay_ms = max(1, int(round(1000.0 / source_fps - spent_ms)))
                else:
                    delay_ms = 1
                key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                state_machine.reset(source_seconds)
                events.clear()
                last_event_state = None
            if key == ord("r"):
                events.clear()
                last_event_state = None

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if args.max_seconds is not None and source_seconds >= args.max_seconds:
                break
    finally:
        capture.release()
        face_extractor.close()
        if phone_detector is not None:
            phone_detector.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    wall_elapsed = max(time.perf_counter() - wall_started, 1e-9)
    print(
        "C2 live demo summary:",
        {
            "frames": stats.frames,
            "wall_fps": round(stats.frames / wall_elapsed, 1),
            "face_found": (
                f"{stats.face_frames / stats.frames:.1%}" if stats.frames else "n/a"
            ),
            "face_mean_ms": (
                round(stats.face_ms_sum / stats.frames, 2) if stats.frames else None
            ),
            "phone_mean_ms": (
                round(stats.phone_ms_sum / stats.phone_runs, 2)
                if stats.phone_runs
                else None
            ),
            "states": dict(stats.states),
            "events": [
                (round(event_time, 1), event_state, event_reason)
                for event_time, event_state, event_reason in list(events)[-8:]
            ],
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
