"""Canonical causal physiological and behavioral evidence accumulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .perclos import OnlinePerclos


EVIDENCE_FEATURE_NAMES = (
    "closed_probability",
    "eye_visibility",
    "aperture_velocity",
    "closure_duration_norm",
    "slow_strength",
    "micro_strength",
    "blink_rate_10s",
    "mean_blink_duration_norm",
    "perclos_10s",
    "perclos_30s",
    "perclos_60s",
    "slow_perclos_10s",
    "slow_perclos_30s",
    "slow_perclos_60s",
    "perclos_reliable_10s",
    "perclos_reliable_30s",
    "perclos_reliable_60s",
    "pitch_norm",
    "yaw_norm",
    "pitch_velocity",
    "nod_probability",
    "yawn_probability",
    "yawn_persistence",
    "distraction_probability",
    "distraction_persistence",
    "face_visibility",
    "mouth_visibility",
)


@dataclass(frozen=True)
class PrimitiveFrameEvidence:
    closed_probability: float
    eye_visibility: float
    head_pose: tuple[float, float]
    yawn_probability: float = 0.0
    mouth_visibility: float = 0.0
    distraction_probability: float = 0.0
    face_visibility: float = 0.0
    timestamp: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "closed_probability",
            "eye_visibility",
            "yawn_probability",
            "mouth_visibility",
            "distraction_probability",
            "face_visibility",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if len(self.head_pose) != 2 or any(
            not math.isfinite(float(value))
            for value in self.head_pose
        ):
            raise ValueError("head pose must contain two finite values")
        if self.timestamp is not None and not math.isfinite(
            float(self.timestamp)
        ):
            raise ValueError("timestamp must be finite")


@dataclass(frozen=True)
class CausalEvidenceSnapshot:
    vector: np.ndarray
    closure_duration_seconds: float
    slow_strength: float
    micro_strength: float
    drowsy_active: bool
    microsleep_active: bool
    nod_probability: float
    perclos: tuple[float, float, float]
    slow_perclos: tuple[float, float, float]
    perclos_reliable: tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.shape != (len(EVIDENCE_FEATURE_NAMES),):
            raise ValueError("causal evidence vector has an incompatible shape")
        if not np.isfinite(vector).all():
            raise ValueError("causal evidence vector must be finite")
        object.__setattr__(self, "vector", vector.copy())


def _sigmoid_boundary(
    value: float,
    *,
    boundary: float,
    temperature: float,
) -> float:
    if value <= 0.0:
        return 0.0
    argument = max(-30.0, min(30.0, (value - boundary) / temperature))
    return 1.0 / (1.0 + math.exp(-argument))


class CausalEvidenceAccumulator:
    """Convert observable frame evidence into one stable causal feature vector."""

    def __init__(
        self,
        *,
        fps: float,
        perclos_windows: Sequence[float] = (10.0, 30.0, 60.0),
        closure_threshold: float = 0.6,
        closure_release_threshold: float = 0.3,
        visibility_threshold: float = 0.2,
        minimum_valid_fraction: float = 0.6,
        slow_closure_seconds: float = 0.5,
        microsleep_seconds: float = 2.0,
        uncertain_gap_seconds: float = 0.1,
        behavioral_gate_suppression: bool = False,
    ) -> None:
        windows = tuple(sorted({float(value) for value in perclos_windows}))
        if fps <= 0.0:
            raise ValueError("FPS must be positive")
        if len(windows) != 3 or any(value <= 0.0 for value in windows):
            raise ValueError("exactly three positive PERCLOS windows are required")
        self.fps = float(fps)
        self.windows = windows
        self.visibility_threshold = float(visibility_threshold)
        self.slow_closure_seconds = float(slow_closure_seconds)
        self.microsleep_seconds = float(microsleep_seconds)
        self.behavioral_gate_suppression = bool(
            behavioral_gate_suppression
        )
        self._perclos_config = {
            "fps": self.fps,
            "windows_seconds": self.windows,
            "closure_threshold": closure_threshold,
            "closure_release_threshold": closure_release_threshold,
            "visibility_threshold": visibility_threshold,
            "minimum_valid_fraction": minimum_valid_fraction,
            "slow_closure_seconds": slow_closure_seconds,
            "microsleep_seconds": microsleep_seconds,
            "uncertain_gap_seconds": uncertain_gap_seconds,
        }
        self.reset()

    def reset(self) -> None:
        self._perclos = OnlinePerclos(**self._perclos_config)
        self._step = 0
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._previous_closed_probability: float | None = None
        self._previous_closure_duration = 0.0
        self._blink_events: deque[tuple[float, float]] = deque()
        self._pitch_baseline: float | None = None
        self._previous_adjusted_pitch: float | None = None
        self._nod_values: deque[float] = deque(
            maxlen=max(6, round(1.5 * self.fps))
        )
        self._yawn_persistence = 0.0
        self._distraction_persistence = 0.0

    def _timestamp(self, supplied: float | None) -> float:
        timestamp = (
            self._step / self.fps
            if supplied is None
            else float(supplied)
        )
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        self._last_timestamp = timestamp
        self._step += 1
        return timestamp

    def _blink_features(
        self,
        *,
        timestamp: float,
        closure_duration: float,
    ) -> tuple[float, float]:
        if (
            self._previous_closure_duration > 0.0
            and closure_duration == 0.0
            and self._previous_closure_duration < self.slow_closure_seconds
        ):
            self._blink_events.append(
                (timestamp, self._previous_closure_duration)
            )
        cutoff = timestamp - 10.0
        while self._blink_events and self._blink_events[0][0] <= cutoff:
            self._blink_events.popleft()
        durations = tuple(value for _, value in self._blink_events)
        blink_rate = min(len(durations) / 10.0, 1.0)
        mean_duration = (
            min(
                sum(durations)
                / len(durations)
                / self.slow_closure_seconds,
                1.0,
            )
            if durations
            else 0.0
        )
        self._previous_closure_duration = closure_duration
        return blink_rate, mean_duration

    def _nod_features(
        self,
        evidence: PrimitiveFrameEvidence,
    ) -> tuple[float, float, float, float]:
        pitch, yaw = (float(value) for value in evidence.head_pose)
        visible = evidence.face_visibility >= self.visibility_threshold
        if not visible:
            self._nod_values.clear()
            self._previous_adjusted_pitch = None
            return 0.0, 0.0, 0.0, 0.0
        if self._pitch_baseline is None:
            self._pitch_baseline = pitch
        adjusted = pitch - self._pitch_baseline
        quiet = (
            evidence.closed_probability < 0.3
            and (
                evidence.yawn_probability * evidence.mouth_visibility
            ) < 0.3
            and evidence.distraction_probability < 0.3
            and abs(adjusted) < 5.0
        )
        if quiet:
            self._pitch_baseline = 0.99 * self._pitch_baseline + 0.01 * pitch
            adjusted = pitch - self._pitch_baseline
        velocity = (
            0.0
            if self._previous_adjusted_pitch is None
            else float(
                np.clip(
                    (adjusted - self._previous_adjusted_pitch) / 18.0,
                    -1.0,
                    1.0,
                )
            )
        )
        self._previous_adjusted_pitch = adjusted
        self._nod_values.append(adjusted)
        nod = 0.0
        if len(self._nod_values) >= 6:
            values = np.asarray(self._nod_values, dtype=np.float32)
            start = float(np.median(values[: min(3, len(values))]))
            end = float(np.median(values[-min(3, len(values)) :]))
            deviation = np.abs(values - start)
            peak_index = int(deviation.argmax())
            amplitude = float(deviation[peak_index])
            returned = abs(end - start) <= max(3.0, 0.35 * amplitude)
            has_internal_peak = 0 < peak_index < len(values) - 1
            if returned and has_internal_peak:
                nod = float(np.clip((amplitude - 6.0) / 12.0, 0.0, 1.0))
        return (
            float(np.clip(adjusted / 90.0, -1.0, 1.0)),
            float(np.clip(yaw / 90.0, -1.0, 1.0)),
            velocity,
            nod,
        )

    def update(
        self,
        evidence: PrimitiveFrameEvidence,
    ) -> CausalEvidenceSnapshot:
        timestamp = self._timestamp(evidence.timestamp)
        perclos = self._perclos.update(
            closed_probability=evidence.closed_probability,
            visibility=evidence.eye_visibility,
            timestamp=timestamp,
        )
        closure_duration = perclos.closure_duration_seconds
        slow_strength = _sigmoid_boundary(
            closure_duration,
            boundary=self.slow_closure_seconds,
            temperature=0.10,
        )
        micro_strength = _sigmoid_boundary(
            closure_duration,
            boundary=self.microsleep_seconds,
            temperature=0.10,
        )
        aperture_velocity = (
            0.0
            if self._previous_closed_probability is None
            else float(
                np.clip(
                    evidence.closed_probability
                    - self._previous_closed_probability,
                    -1.0,
                    1.0,
                )
            )
        )
        self._previous_closed_probability = evidence.closed_probability
        blink_rate, mean_blink_duration = self._blink_features(
            timestamp=timestamp,
            closure_duration=closure_duration,
        )
        pitch, yaw, pitch_velocity, nod = self._nod_features(evidence)
        decay = math.exp(-1.0 / self.fps)
        observed_yawn = (
            evidence.yawn_probability * evidence.mouth_visibility
        )
        self._yawn_persistence = max(
            observed_yawn,
            self._yawn_persistence * decay,
        )
        self._distraction_persistence = max(
            evidence.distraction_probability,
            self._distraction_persistence * decay,
        )
        elapsed = (
            1.0 / self.fps
            if self._first_timestamp is None
            else timestamp - self._first_timestamp + 1.0 / self.fps
        )
        values = tuple(
            0.0 if perclos.values[window] is None
            else float(perclos.values[window])
            for window in self.windows
        )
        slow_values = tuple(
            0.0 if perclos.slow_values[window] is None
            else float(perclos.slow_values[window])
            for window in self.windows
        )
        reliable = tuple(
            bool(
                perclos.reliable_by_window[window]
                and elapsed >= window
            )
            for window in self.windows
        )
        vector = np.asarray(
            (
                evidence.closed_probability,
                evidence.eye_visibility,
                aperture_velocity,
                min(closure_duration / self.microsleep_seconds, 1.0),
                slow_strength,
                micro_strength,
                blink_rate,
                mean_blink_duration,
                *values,
                *slow_values,
                *(float(value) for value in reliable),
                pitch,
                yaw,
                pitch_velocity,
                nod,
                observed_yawn,
                self._yawn_persistence,
                evidence.distraction_probability,
                self._distraction_persistence,
                evidence.face_visibility,
                evidence.mouth_visibility,
            ),
            dtype=np.float32,
        )
        physiology_clear = bool(
            not self.behavioral_gate_suppression
            or (
                observed_yawn < 0.5
                and evidence.distraction_probability < 0.3
            )
        )
        return CausalEvidenceSnapshot(
            vector=vector,
            closure_duration_seconds=closure_duration,
            slow_strength=slow_strength,
            micro_strength=micro_strength,
            drowsy_active=bool(
                closure_duration >= self.slow_closure_seconds
                and not perclos.microsleep
                and physiology_clear
            ),
            microsleep_active=bool(perclos.microsleep and physiology_clear),
            nod_probability=nod,
            perclos=values,
            slow_perclos=slow_values,
            perclos_reliable=reliable,
        )


__all__ = [
    "CausalEvidenceAccumulator",
    "CausalEvidenceSnapshot",
    "EVIDENCE_FEATURE_NAMES",
    "PrimitiveFrameEvidence",
]
