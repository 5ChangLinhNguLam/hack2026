"""Interactive OpenCV webcam tester for the evidence-guided LSTM pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Sequence

from PIL import Image
import torch

from ..opencv_dashboard import (
    DashboardTelemetry,
    FaceLossResetPolicy,
    OpenCVDashboardRenderer,
    decode_key,
)
from .demo import LSTMDemoPipeline, create_lstm_pipeline


@dataclass(frozen=True)
class ModelArtifacts:
    visual: Path
    ocular: Path
    gate_ocular: Path
    temporal: Path


def resolve_model_artifacts(model_dir: Path | str) -> ModelArtifacts:
    model_dir = Path(model_dir)
    artifacts = ModelArtifacts(
        visual=model_dir / "visual" / "best.pt",
        ocular=model_dir / "ocular" / "best.pt",
        gate_ocular=model_dir / "gate_ocular" / "best.pt",
        temporal=model_dir / "temporal" / "best.pt",
    )
    missing = [
        path.relative_to(model_dir)
        for path in (
            artifacts.visual,
            artifacts.ocular,
            artifacts.gate_ocular,
            artifacts.temporal,
        )
        if not path.is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"webcam model directory {model_dir} is incomplete; missing: "
            f"{formatted}"
        )
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCV webcam dashboard for causal five-state inference"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ultralight-model", type=Path, required=True)
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--detector-interval", type=int, default=5)
    parser.add_argument("--face-loss-reset-seconds", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--screenshot-dir", type=Path, default=Path("webcam_captures"))
    return parser


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _video_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


class _RateMeter:
    def __init__(self, window: int = 40) -> None:
        self.timestamps: deque[float] = deque(maxlen=window)

    def update(self, timestamp: float) -> float:
        self.timestamps.append(float(timestamp))
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return 0.0 if elapsed <= 0.0 else (len(self.timestamps) - 1) / elapsed


def _reset_temporal(pipeline: LSTMDemoPipeline) -> None:
    pipeline.runtime.reset()


def _save_screenshot(cv2_module, image, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / datetime.now().strftime("driver_state_%Y%m%d_%H%M%S_%f.jpg")
    if not cv2_module.imwrite(str(path), image):
        raise RuntimeError(f"failed to save screenshot: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps <= 0.0:
        raise ValueError("webcam FPS must be positive")
    if args.detector_interval <= 0:
        raise ValueError("detector interval must be positive")
    if args.face_loss_reset_seconds <= 0.0:
        raise ValueError("face-loss reset duration must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("camera dimensions must be positive")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional UI dependency
        raise RuntimeError(
            "OpenCV GUI support is required; install the 'landmarks' extra"
        ) from error

    artifacts = resolve_model_artifacts(args.model_dir)
    device = _device(args.device)
    pipeline = create_lstm_pipeline(
        visual_checkpoint=artifacts.visual,
        ocular_checkpoint=artifacts.ocular,
        gate_ocular_checkpoint=artifacts.gate_ocular,
        temporal_checkpoint=artifacts.temporal,
        ultralight_model=args.ultralight_model,
        landmark_model=args.landmark_model,
        device=device,
        fps=float(args.fps),
        detector_interval=int(args.detector_interval),
    )
    capture = cv2.VideoCapture(_video_source(args.source))
    if not capture.isOpened():
        pipeline.close()
        raise SystemExit(f"cannot open video source: {args.source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    capture.set(cv2.CAP_PROP_FPS, float(args.fps))

    renderer = OpenCVDashboardRenderer(cv2, mirror=not args.no_mirror)
    face_loss = FaceLossResetPolicy(
        reset_after_frames=max(
            1,
            math.ceil(args.face_loss_reset_seconds * args.fps),
        )
    )
    rates = _RateMeter()
    window_name = "DMD five-state webcam"
    paused = False
    canvas = None
    session_started = time.monotonic()
    status_message = f"device {device.type} | temporal history started"
    target_period = 1.0 / float(args.fps)

    try:
        while True:
            loop_started = time.monotonic()
            if not paused:
                ok, bgr = capture.read()
                if not ok:
                    break
                inference_started = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                now = time.monotonic()
                prediction = pipeline.process(
                    Image.fromarray(rgb),
                    timestamp=now - session_started,
                )
                latency_ms = (time.perf_counter() - inference_started) * 1000.0
                detection = pipeline.region_detector.latest_detection
                face_visible = detection is not None
                face_confidence = (
                    0.0 if detection is None else float(detection.confidence)
                )
                if face_loss.update(face_visible=face_visible):
                    _reset_temporal(pipeline)
                    session_started = now
                    status_message = "temporal history reset after face loss"
                processing_fps = rates.update(now)
                canvas = renderer.render(
                    bgr,
                    prediction,
                    telemetry=DashboardTelemetry(
                        processing_fps=processing_fps,
                        latency_ms=latency_ms,
                        face_visible=face_visible,
                        face_confidence=face_confidence,
                        paused=False,
                        message=status_message,
                    ),
                    detection=detection,
                )
            if canvas is None:
                continue

            cv2.imshow(window_name, canvas)
            elapsed = time.monotonic() - loop_started
            delay_ms = 30 if paused else max(1, round((target_period - elapsed) * 1000))
            command = decode_key(cv2.waitKey(delay_ms))
            if command == "quit":
                break
            if command == "reset":
                _reset_temporal(pipeline)
                face_loss.reset()
                rates = _RateMeter()
                session_started = time.monotonic()
                status_message = "temporal history reset manually"
            elif command == "pause":
                paused = not paused
                status_message = "paused" if paused else "resumed with fresh history"
                if not paused:
                    _reset_temporal(pipeline)
                    face_loss.reset()
                    rates = _RateMeter()
                    session_started = time.monotonic()
            elif command == "screenshot":
                saved = _save_screenshot(cv2, canvas, args.screenshot_dir)
                status_message = f"saved {saved.name}"
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
