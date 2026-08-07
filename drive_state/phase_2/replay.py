"""Phase-2 DMD driver-state inference on the native tripkit replay stream.

The model is strictly causal: a result for frame ``t`` only uses frames up to
``t``.  ``GeneralDMS`` owns the recurrent state, so create one instance per
trip (or call :meth:`reset` before starting another independent stream).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

import cv2
from PIL import Image
import torch

from .cli.demo import LSTMDemoPipeline, create_lstm_pipeline


MODEL_STATES = ("alert", "drowsy", "microsleep", "yawning", "distraction")
PUBLIC_STATES = ("alert", "drowsy", "microsleep", "yawning", "distracted")


@dataclass(frozen=True)
class DMSBundle:
    """Paths making up one fingerprint-checked deployable checkpoint bundle."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "DMSBundle":
        return cls(Path(root))

    @property
    def visual(self) -> Path:
        return self.root / "visual" / "best.pt"

    @property
    def ocular(self) -> Path:
        return self.root / "ocular" / "best.pt"

    @property
    def gate_ocular(self) -> Path:
        return self.root / "gate_ocular" / "best.pt"

    @property
    def temporal(self) -> Path:
        return self.root / "temporal" / "best.pt"

    @property
    def face_detector(self) -> Path:
        return self.root / "version-RFB-320.onnx"

    @property
    def landmark_model(self) -> Path:
        return self.root / "face_landmarker.task"

    def validate(self) -> None:
        missing = [
            path
            for path in (
                self.visual,
                self.ocular,
                self.gate_ocular,
                self.temporal,
                self.face_detector,
                self.landmark_model,
            )
            if not path.is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"DMS checkpoint bundle is incomplete:\n{formatted}")


@dataclass(frozen=True)
class DMSVssSignals:
    """The five CarSky-facing values derived from the exclusive distribution.

    Percent fields are on the 0..100 scale used by the VSS signal catalogue.
    ``IsEyesOnRoad`` is an inference from distraction plus ocular visibility;
    it is not a dedicated gaze-zone measurement.
    """

    attentive_probability: float
    distraction_level: float
    fatigue_level: float
    is_eyes_on_road: bool
    is_warning: bool

    def as_vss_dict(self) -> dict[str, float | bool]:
        return {
            "Vehicle.Driver.AttentiveProbability": self.attentive_probability,
            "Vehicle.Driver.DistractionLevel": self.distraction_level,
            "Vehicle.Driver.FatigueLevel": self.fatigue_level,
            "Vehicle.Driver.IsEyesOnRoad": self.is_eyes_on_road,
            "Vehicle.ADAS.DMS.IsWarning": self.is_warning,
        }


@dataclass(frozen=True)
class DMSFramePrediction:
    frame_id: int
    timestamp: float
    state_id: int
    state: str
    confidence: float
    probabilities: tuple[float, float, float, float, float]
    closed_probability: float
    ocular_reliability: float
    closure_duration_seconds: float
    perclos: tuple[float, float, float]
    slow_perclos: tuple[float, float, float]
    perclos_reliable: tuple[bool, bool, bool]
    nod_probability: float
    microsleep_active: bool
    drowsy_episode_active: bool
    latency_ms: float

    def submission_row(self) -> dict[str, int | float | str]:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "predicted_driver_state": self.state,
        }

    def vss_signals(
        self,
        *,
        warning_threshold: float = 0.7,
        eye_visibility_threshold: float = 0.2,
    ) -> DMSVssSignals:
        if not 0.0 <= warning_threshold <= 1.0:
            raise ValueError("warning_threshold must be in [0, 1]")
        if not 0.0 <= eye_visibility_threshold <= 1.0:
            raise ValueError("eye_visibility_threshold must be in [0, 1]")
        alert, drowsy, microsleep, yawning, distraction = self.probabilities
        fatigue = max(drowsy, microsleep, yawning)
        return DMSVssSignals(
            attentive_probability=round(alert * 100.0, 3),
            distraction_level=round(distraction * 100.0, 3),
            fatigue_level=round(fatigue * 100.0, 3),
            is_eyes_on_road=(
                self.ocular_reliability >= eye_visibility_threshold
                and distraction < 0.5
            ),
            is_warning=(
                self.microsleep_active
                or fatigue >= warning_threshold
                or distraction >= warning_threshold
            ),
        )


def frame_prediction_from_runtime(
    *,
    frame_id: int,
    timestamp: float,
    prediction: Any,
    latency_ms: float,
) -> DMSFramePrediction:
    state_id = int(prediction.state_id)
    if not 0 <= state_id < len(PUBLIC_STATES):
        raise ValueError(f"invalid DMS state id: {state_id}")
    probabilities = tuple(float(value) for value in prediction.probabilities)
    if len(probabilities) != 5:
        raise ValueError("DMS runtime must emit exactly five probabilities")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("DMS probabilities must be in [0, 1]")
    if abs(sum(probabilities) - 1.0) > 1e-4:
        raise ValueError("DMS probabilities must sum to one")
    return DMSFramePrediction(
        frame_id=int(frame_id),
        timestamp=float(timestamp),
        state_id=state_id,
        state=PUBLIC_STATES[state_id],
        confidence=max(probabilities),
        probabilities=probabilities,  # type: ignore[arg-type]
        closed_probability=float(prediction.closed_probability),
        ocular_reliability=float(prediction.ocular_reliability),
        closure_duration_seconds=float(prediction.closure_duration_seconds),
        perclos=tuple(float(value) for value in prediction.perclos),  # type: ignore[arg-type]
        slow_perclos=tuple(float(value) for value in prediction.slow_perclos),  # type: ignore[arg-type]
        perclos_reliable=tuple(bool(value) for value in prediction.perclos_reliable),  # type: ignore[arg-type]
        nod_probability=float(prediction.nod_probability),
        microsleep_active=bool(prediction.microsleep_active),
        drowsy_episode_active=bool(prediction.drowsy_episode_active),
        latency_ms=float(latency_ms),
    )


class GeneralDMS:
    """Stateful adapter between BGR tripkit frames and the phase-2 DMD model."""

    def __init__(
        self,
        bundle: DMSBundle,
        *,
        device: str | torch.device = "auto",
        fps: float = 20.0,
        detector_interval: int = 5,
    ) -> None:
        bundle.validate()
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if detector_interval <= 0:
            raise ValueError("detector_interval must be positive")
        if str(device) == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.bundle = bundle
        self.device = resolved
        self.fps = float(fps)
        self.pipeline: LSTMDemoPipeline = create_lstm_pipeline(
            visual_checkpoint=bundle.visual,
            ocular_checkpoint=bundle.ocular,
            gate_ocular_checkpoint=bundle.gate_ocular,
            temporal_checkpoint=bundle.temporal,
            ultralight_model=bundle.face_detector,
            landmark_model=bundle.landmark_model,
            device=self.device,
            fps=self.fps,
            detector_interval=detector_interval,
        )

    def reset(self) -> None:
        self.pipeline.runtime.reset()

    def close(self) -> None:
        self.pipeline.close()

    def __enter__(self) -> "GeneralDMS":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def process_bgr(
        self,
        image: Any,
        *,
        frame_id: int,
        timestamp: float,
    ) -> DMSFramePrediction:
        if image is None or getattr(image, "ndim", 0) != 3:
            raise ValueError("driver frame must be a BGR HxWxC image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        started = perf_counter()
        prediction = self.pipeline.process(Image.fromarray(rgb), timestamp=timestamp)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (perf_counter() - started) * 1000.0
        return frame_prediction_from_runtime(
            frame_id=frame_id,
            timestamp=timestamp,
            prediction=prediction,
            latency_ms=latency_ms,
        )

    def replay_trip(
        self,
        trip_root: str | Path,
        *,
        mode: str = "fast",
        speed: float = 1.0,
        limit: int | None = None,
    ) -> Iterator[DMSFramePrediction]:
        """Yield one causal DMS result for each TripReplayer frame."""
        from tripkit import TripLoader, TripReplayer

        loader = TripLoader(trip_root)
        end = loader.n_frames if limit is None else min(loader.n_frames, limit)
        for bundle in TripReplayer(loader, mode=mode, speed=speed, end=end):
            yield self.process_bgr(
                bundle.driver(),
                frame_id=bundle.frame_id,
                timestamp=bundle.timestamp,
            )


__all__ = [
    "DMSBundle",
    "DMSFramePrediction",
    "DMSVssSignals",
    "GeneralDMS",
    "MODEL_STATES",
    "PUBLIC_STATES",
    "frame_prediction_from_runtime",
]
