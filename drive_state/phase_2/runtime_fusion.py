"""Causal fusion from specialist probabilities to one public driver state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .state.arbiter import DriverState, EvidenceFrame, ExclusiveStateArbiter
from .state.perclos import OnlinePerclos


STATE_TO_ID = {
    DriverState.ALERT: 0,
    DriverState.DROWSY: 1,
    DriverState.MICROSLEEP: 2,
    DriverState.YAWNING: 3,
    DriverState.DISTRACTION: 4,
}


@dataclass(frozen=True)
class PrimitiveEvidence:
    closed_probability: float = 0.0
    eye_visibility: float = 0.0
    yawn_probability: float = 0.0
    mouth_visibility: float = 0.0
    distraction_probability: float = 0.0
    drowsy_probability: float = 0.0
    nod_probability: float = 0.0
    timestamp: float | None = None


@dataclass(frozen=True)
class RuntimeResult:
    state_id: int
    state: DriverState
    confidence: float
    visible: bool
    diagnostics: Mapping[str, float | bool]


class RuntimeEvidenceFusion:
    def __init__(
        self,
        *,
        fps: float,
        perclos_window_seconds: float | None = None,
        perclos_windows_seconds: Sequence[float] = (10.0, 30.0, 60.0),
        perclos_threshold: float = 0.25,
        microsleep_seconds: float = 2.0,
        confirmation_frames: int = 3,
        release_frames: int = 3,
    ) -> None:
        self.perclos_threshold = perclos_threshold
        windows = (
            (float(perclos_window_seconds),)
            if perclos_window_seconds is not None
            else tuple(float(value) for value in perclos_windows_seconds)
        )
        self.perclos = OnlinePerclos(
            fps=fps,
            windows_seconds=windows,
            microsleep_seconds=microsleep_seconds,
        )
        self.arbiter = ExclusiveStateArbiter(
            perclos_threshold=perclos_threshold,
            confirmation_frames=confirmation_frames,
            release_frames=release_frames,
        )

    def reset(self) -> None:
        self.perclos.reset()
        self.arbiter.reset()

    def update(self, evidence: PrimitiveEvidence) -> RuntimeResult:
        perclos = self.perclos.update(
            closed_probability=evidence.closed_probability,
            visibility=evidence.eye_visibility,
            timestamp=evidence.timestamp,
        )
        slow_candidates = [
            float(value)
            for window, value in perclos.slow_values.items()
            if value is not None and perclos.reliable_by_window[window]
        ]
        slow_perclos = max(slow_candidates, default=0.0)
        slow_reliable = bool(slow_candidates)
        decision = self.arbiter.update(
            EvidenceFrame(
                microsleep=perclos.microsleep,
                eye_visibility=evidence.eye_visibility,
                perclos=slow_perclos,
                perclos_reliable=slow_reliable,
                yawn_probability=evidence.yawn_probability,
                mouth_visibility=evidence.mouth_visibility,
                distraction_probability=evidence.distraction_probability,
                drowsy_probability=evidence.drowsy_probability,
                nod_probability=evidence.nod_probability,
            )
        )
        drowsy_score = max(
            evidence.drowsy_probability,
            evidence.nod_probability,
            min(slow_perclos / max(self.perclos_threshold, 1e-6), 1.0)
            if slow_reliable
            else 0.0,
        )
        scores = {
            DriverState.MICROSLEEP: evidence.closed_probability,
            DriverState.YAWNING: evidence.yawn_probability,
            DriverState.DISTRACTION: evidence.distraction_probability,
            DriverState.DROWSY: drowsy_score,
            DriverState.ALERT: max(
                0.0,
                1.0
                - max(
                    evidence.closed_probability,
                    evidence.yawn_probability,
                    evidence.distraction_probability,
                    drowsy_score,
                ),
            ),
        }
        diagnostics: dict[str, float | bool] = {
            "closed_probability": evidence.closed_probability,
            "eye_visibility": evidence.eye_visibility,
            "yawn_probability": evidence.yawn_probability,
            "mouth_visibility": evidence.mouth_visibility,
            "distraction_probability": evidence.distraction_probability,
            "perclos_value": slow_perclos,
            "perclos_reliable": slow_reliable,
            "closure_duration_seconds": perclos.closure_duration_seconds,
            "nod_probability": evidence.nod_probability,
        }
        for window, value in perclos.values.items():
            label = int(window) if float(window).is_integer() else window
            diagnostics[f"perclos_{label}s"] = 0.0 if value is None else value
            slow = perclos.slow_values[window]
            diagnostics[f"slow_perclos_{label}s"] = 0.0 if slow is None else slow
            diagnostics[f"perclos_{label}s_reliable"] = (
                perclos.reliable_by_window[window]
            )
        return RuntimeResult(
            state_id=STATE_TO_ID[decision.state],
            state=decision.state,
            confidence=max(0.0, min(1.0, scores[decision.state])),
            visible=evidence.eye_visibility >= self.perclos.visibility_threshold,
            diagnostics=diagnostics,
        )
