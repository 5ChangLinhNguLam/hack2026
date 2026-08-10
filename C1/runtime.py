"""Stateful adapter from tripkit frames to the C1 StudentTTC checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from .features import FEATURES_USED, ScalarFeatureStream
from .model import StudentTTC, load_student_ttc


@dataclass(frozen=True)
class C1FramePrediction:
    frame_id: int
    timestamp: float
    collision_probability: float
    predicted_ttc_s: float
    display_ttc_s: float
    is_warning: bool
    model_updated: bool
    model_frame_id: int
    latency_ms: float

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.model_frame_id < 0:
            raise ValueError("C1 frame IDs must be non-negative")
        if self.model_frame_id > self.frame_id:
            raise ValueError("C1 model_frame_id cannot be ahead of source frame")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("C1 timestamp must be finite and non-negative")
        if (
            not math.isfinite(self.collision_probability)
            or not 0.0 <= self.collision_probability <= 1.0
        ):
            raise ValueError("C1 collision probability must be finite in [0, 1]")
        for name, value in (
            ("predicted_ttc_s", self.predicted_ttc_s),
            ("display_ttc_s", self.display_ttc_s),
        ):
            if math.isnan(value) or value == float("-inf") or value < 0.0:
                raise ValueError(f"C1 {name} must be non-negative or +inf")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("C1 latency must be finite and non-negative")

    def submission_row(self) -> dict[str, int | float | str]:
        value: float | str = (
            round(self.predicted_ttc_s, 6)
            if math.isfinite(self.predicted_ttc_s)
            else "inf"
        )
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "predicted_ttc": value,
        }


class InverseTtcSmoother:
    def __init__(
        self,
        *,
        ema: float = 0.9,
        warning_on: float = 0.85,
        warning_off: float = 0.5,
    ) -> None:
        if not 0.0 < ema <= 1.0:
            raise ValueError("C1 EMA must be in (0, 1]")
        if not 0.0 <= warning_off <= warning_on <= 1.0:
            raise ValueError("C1 warning thresholds must satisfy 0 <= off <= on <= 1")
        self.ema = float(ema)
        self.warning_on = float(warning_on)
        self.warning_off = float(warning_off)
        self.reset()

    def reset(self) -> None:
        self._inverse_ttc: float | None = None
        self._warning = False

    def step(self, probability: float, ttc_seconds: float) -> tuple[float, bool]:
        raw_inverse = (
            0.0
            if not math.isfinite(ttc_seconds)
            else 1.0 / max(ttc_seconds, 1e-6)
        )
        if self._inverse_ttc is None:
            self._inverse_ttc = raw_inverse
        else:
            self._inverse_ttc = (
                self.ema * raw_inverse
                + (1.0 - self.ema) * self._inverse_ttc
            )
        display_ttc = (
            1.0 / self._inverse_ttc
            if self._inverse_ttc > 0.1
            else float("inf")
        )
        self._warning = (
            probability >= self.warning_on
            if not self._warning
            else probability >= self.warning_off
        )
        return display_ttc, self._warning


class StudentTTCRuntime:
    """Run C1 at its trained 10-Hz cadence and forward-fill to source frames."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        metadata: Mapping[str, Any],
        source_fps: float = 20.0,
        stride: int = 2,
        device: str | torch.device = "auto",
        ema: float = 0.9,
        warning_on: float = 0.85,
        warning_off: float = 0.5,
    ) -> None:
        if source_fps <= 0.0:
            raise ValueError("source_fps must be positive")
        if stride <= 0:
            raise ValueError("C1 stride must be positive")
        if str(device) == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for C1 but is unavailable")

        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"C1 checkpoint not found: {checkpoint}")
        self.checkpoint_path = checkpoint
        self.device = resolved
        self.source_fps = float(source_fps)
        self.stride = int(stride)
        self.sample_hz = self.source_fps / self.stride
        if not math.isclose(self.sample_hz, 10.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "C1 checkpoint was trained at 10 Hz; "
                f"source_fps/stride is {self.sample_hz:g} Hz"
            )
        self.model: StudentTTC
        self.model, checkpoint_metadata = load_student_ttc(
            checkpoint, device=self.device
        )
        self.checkpoint_metadata = checkpoint_metadata
        self.scalar_stream = ScalarFeatureStream(
            metadata=metadata,
            sample_hz=self.sample_hz,
            feature_order=FEATURES_USED,
        )
        self.smoother = InverseTtcSmoother(
            ema=ema, warning_on=warning_on, warning_off=warning_off
        )
        self.reset()

    def reset(self) -> None:
        self.model.reset()
        self.scalar_stream.reset()
        self.smoother.reset()
        self._source_index = 0
        self._last_prediction: C1FramePrediction | None = None

    def close(self) -> None:
        self.reset()

    def __enter__(self) -> "StudentTTCRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        if bgr is None or getattr(bgr, "ndim", 0) != 3:
            raise ValueError("C1 road frame must be a BGR HxWxC image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        return (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )

    def _needs_model_update(self) -> bool:
        return (
            self._last_prediction is None
            or self._source_index % self.stride == 0
        )

    def _forward_fill(
        self, *, frame_id: int, timestamp: float
    ) -> C1FramePrediction:
        if self._last_prediction is None:
            raise RuntimeError("C1 cannot forward-fill before its first inference")
        self._source_index += 1
        return replace(
            self._last_prediction,
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            model_updated=False,
            latency_ms=0.0,
        )

    def process_bgr(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp: float,
        ego: Mapping[str, Any] | None,
        target_count: int,
    ) -> C1FramePrediction:
        if not self._needs_model_update():
            return self._forward_fill(frame_id=frame_id, timestamp=timestamp)
        self._source_index += 1

        scalar = self.scalar_stream.step(ego, target_count=target_count)
        frame = self._preprocess(image)
        scalar_tensor = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        probability, raw_ttc = self.model.predict(frame, scalar_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (perf_counter() - started) * 1000.0
        display_ttc, warning = self.smoother.step(probability, raw_ttc)
        prediction = C1FramePrediction(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            collision_probability=float(probability),
            predicted_ttc_s=float(raw_ttc),
            display_ttc_s=float(display_ttc),
            is_warning=bool(warning),
            model_updated=True,
            model_frame_id=int(frame_id),
            latency_ms=latency_ms,
        )
        self._last_prediction = prediction
        return prediction

    def process_bundle(self, bundle: Any) -> C1FramePrediction:
        # ``left`` is intentionally called only on model-update frames.  Avoid
        # decoding image_2 for the causal forward-filled source frames.
        if not self._needs_model_update():
            return self._forward_fill(
                frame_id=bundle.frame_id, timestamp=bundle.timestamp
            )
        return self.process_bgr(
            bundle.left(),
            frame_id=bundle.frame_id,
            timestamp=bundle.timestamp,
            ego=bundle.ego,
            target_count=len(bundle.targets),
        )


__all__ = [
    "C1FramePrediction",
    "InverseTtcSmoother",
    "StudentTTCRuntime",
]
