"""End-to-end monocular C1 pipeline and TripReplayer runner."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from tripkit.types import FrameBundle

from .detector import ObjectDetector
from .tracker import MonocularTTCTracker
from .types import C1FramePrediction, TrackRisk


@dataclass(frozen=True)
class ReplayStats:
    frame_count: int
    elapsed_s: float
    throughput_fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    finite_ttc_frames: int
    warning_frames: int
    output_csv: Path
    output_video: Optional[Path]


class MonocularC1Pipeline:
    """One-camera detector -> tracker -> TTC runtime."""

    def __init__(
        self,
        detector: ObjectDetector,
        tracker: MonocularTTCTracker,
        *,
        detector_stride: int = 1,
    ) -> None:
        if detector_stride < 1:
            raise ValueError("detector_stride phải >= 1")
        self.detector = detector
        self.tracker = tracker
        self.detector_stride = detector_stride
        self._frame_index = 0

    def reset(self) -> None:
        self.tracker.reset()
        self._frame_index = 0

    def predict(self, bundle: FrameBundle) -> tuple[C1FramePrediction, np.ndarray]:
        started = time.perf_counter()
        # The one-camera contract is enforced here: never call right(), driver(),
        # or consume bundle.depth in the inference path.
        image = bundle.left()
        ego = bundle.ego or {}
        tracker_args = {
            "timestamp": float(bundle.timestamp),
            "image_shape": image.shape[:2],
            "ego_speed_kmh": float(ego.get("speed_kmh") or 0.0),
        }
        if self._frame_index % self.detector_stride == 0:
            detections = self.detector.detect(image)
            risks = self.tracker.update(detections, **tracker_args)
        else:
            risks = self.tracker.predict(**tracker_args)
        self._frame_index += 1
        finite = [risk.predicted_ttc_s for risk in risks if math.isfinite(risk.predicted_ttc_s)]
        predicted_ttc = min(finite, default=float("inf"))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return C1FramePrediction(
            frame_id=bundle.frame_id,
            timestamp=float(bundle.timestamp),
            predicted_ttc_s=predicted_ttc,
            is_warning=predicted_ttc < 2.0,
            risks=tuple(risks),
            latency_ms=latency_ms,
        ), image


def render_hud(image: np.ndarray, prediction: C1FramePrediction) -> np.ndarray:
    panel = image.copy()
    for risk in prediction.risks:
        x1, y1, x2, y2 = (int(round(v)) for v in risk.bbox)
        if risk.predicted_ttc_s < 2.0:
            color = (0, 0, 255)
        elif math.isfinite(risk.predicted_ttc_s):
            color = (0, 165, 255)
        elif risk.collision_relevant:
            color = (0, 220, 220)
        else:
            color = (100, 200, 100)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        ttc_text = f"{risk.predicted_ttc_s:.2f}s" if math.isfinite(risk.predicted_ttc_s) else "inf"
        text = f"#{risk.track_id} {risk.label} {risk.confidence:.2f} TTC={ttc_text}"
        cv2.putText(
            panel,
            text,
            (max(0, x1), max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    overall = "inf"
    if math.isfinite(prediction.predicted_ttc_s):
        overall = f"{prediction.predicted_ttc_s:.2f}s"
    state = "WARNING" if prediction.is_warning else "MONITORING"
    state_color = (0, 0, 255) if prediction.is_warning else (0, 255, 255)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (12, 20, 25), -1)
    cv2.putText(
        panel,
        f"C1 MONOCULAR BASE | frame={prediction.frame_id} | TTC={overall} | "
        f"{state} | {prediction.latency_ms:.1f}ms",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        state_color,
        2,
        cv2.LINE_AA,
    )
    return panel


def run_replay(
    replayer: Iterable[FrameBundle],
    pipeline: MonocularC1Pipeline,
    *,
    output_csv: str | Path,
    fps: float,
    output_video: str | Path | None = None,
    show: bool = False,
    progress_every: int = 50,
) -> ReplayStats:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    video_path = Path(output_video) if output_video else None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    latencies: list[float] = []
    finite_count = 0
    warning_count = 0
    started = time.perf_counter()
    pipeline.reset()

    try:
        with output_csv.open("w", encoding="utf-8", newline="") as stream:
            csv_writer = csv.DictWriter(
                stream,
                fieldnames=["frame_id", "timestamp", "predicted_ttc"],
            )
            csv_writer.writeheader()
            for count, bundle in enumerate(replayer, start=1):
                prediction, image = pipeline.predict(bundle)
                csv_writer.writerow(prediction.submission_row())
                latencies.append(prediction.latency_ms)
                finite_count += int(math.isfinite(prediction.predicted_ttc_s))
                warning_count += int(prediction.is_warning)

                if show or video_path is not None:
                    panel = render_hud(image, prediction)
                    if video_path is not None:
                        if writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(
                                str(video_path),
                                fourcc,
                                fps,
                                (panel.shape[1], panel.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"Không mở được video output: {video_path}")
                        writer.write(panel)
                    if show:
                        cv2.imshow("SafeLoop C1 monocular baseline", panel)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break

                if progress_every > 0 and count % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"C1 progress: {count} frame, {count / max(elapsed, 1e-9):.1f} fps, "
                        f"latency={prediction.latency_ms:.1f}ms",
                        flush=True,
                    )
    finally:
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.perf_counter() - started
    count = len(latencies)
    latency_array = np.asarray(latencies, dtype=float)
    return ReplayStats(
        frame_count=count,
        elapsed_s=elapsed,
        throughput_fps=count / elapsed if elapsed > 0 else float("inf"),
        mean_latency_ms=float(latency_array.mean()) if count else 0.0,
        p95_latency_ms=float(np.percentile(latency_array, 95)) if count else 0.0,
        finite_ttc_frames=finite_count,
        warning_frames=warning_count,
        output_csv=output_csv,
        output_video=video_path,
    )
