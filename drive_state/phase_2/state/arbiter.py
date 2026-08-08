"""Exclusive five-state arbitration over overlapping specialist evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriverState(str, Enum):
    ALERT = "alert"
    DROWSY = "drowsy"
    MICROSLEEP = "microsleep"
    YAWNING = "yawning"
    DISTRACTION = "distraction"


@dataclass(frozen=True)
class EvidenceFrame:
    microsleep: bool = False
    eye_visibility: float = 0.0
    perclos: float = 0.0
    perclos_reliable: bool = False
    yawn_probability: float = 0.0
    mouth_visibility: float = 0.0
    distraction_probability: float = 0.0
    drowsy_probability: float = 0.0
    nod_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "eye_visibility",
            "perclos",
            "yawn_probability",
            "mouth_visibility",
            "distraction_probability",
            "drowsy_probability",
            "nod_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class ArbiterDecision:
    """One externally visible state plus a single winning reason."""

    state: DriverState
    changed: bool
    reason: str


class ExclusiveStateArbiter:
    """Apply safety priority and causal hysteresis to specialist evidence."""

    def __init__(
        self,
        *,
        eye_visibility_threshold: float = 0.2,
        mouth_visibility_threshold: float = 0.2,
        yawn_threshold: float = 0.7,
        distraction_threshold: float = 0.7,
        drowsy_threshold: float = 0.7,
        nod_threshold: float = 0.7,
        perclos_threshold: float = 0.25,
        confirmation_frames: int = 3,
        release_frames: int = 3,
    ) -> None:
        thresholds = (
            eye_visibility_threshold,
            mouth_visibility_threshold,
            yawn_threshold,
            distraction_threshold,
            drowsy_threshold,
            nod_threshold,
            perclos_threshold,
        )
        if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError("all arbiter thresholds must be between 0 and 1")
        if confirmation_frames <= 0 or release_frames <= 0:
            raise ValueError("hysteresis frame counts must be positive")
        self.eye_visibility_threshold = eye_visibility_threshold
        self.mouth_visibility_threshold = mouth_visibility_threshold
        self.yawn_threshold = yawn_threshold
        self.distraction_threshold = distraction_threshold
        self.drowsy_threshold = drowsy_threshold
        self.nod_threshold = nod_threshold
        self.perclos_threshold = perclos_threshold
        self.confirmation_frames = confirmation_frames
        self.release_frames = release_frames
        self._state = DriverState.ALERT
        self._pending = DriverState.ALERT
        self._pending_frames = 0

    @property
    def state(self) -> DriverState:
        return self._state

    def reset(self) -> None:
        self._state = DriverState.ALERT
        self._pending = DriverState.ALERT
        self._pending_frames = 0

    def _candidate(self, evidence: EvidenceFrame) -> tuple[DriverState, str]:
        # Priority is explicit and produces exactly one winning candidate.
        if (
            evidence.microsleep
            and evidence.eye_visibility >= self.eye_visibility_threshold
        ):
            return DriverState.MICROSLEEP, "continuous_visible_eye_closure"
        if (
            evidence.yawn_probability >= self.yawn_threshold
            and evidence.mouth_visibility >= self.mouth_visibility_threshold
        ):
            return DriverState.YAWNING, "visible_yawn"
        if evidence.distraction_probability >= self.distraction_threshold:
            return DriverState.DISTRACTION, "cabin_or_gaze_distraction"
        if evidence.drowsy_probability >= self.drowsy_threshold:
            return DriverState.DROWSY, "drowsy_specialist"
        if evidence.nod_probability >= self.nod_threshold:
            return DriverState.DROWSY, "head_nod"
        if evidence.perclos_reliable and evidence.perclos >= self.perclos_threshold:
            return DriverState.DROWSY, "elevated_perclos"
        return DriverState.ALERT, "no_hazard_threshold_met"

    def update(self, evidence: EvidenceFrame) -> ArbiterDecision:
        candidate, reason = self._candidate(evidence)
        previous = self._state
        if candidate is self._state:
            self._pending = candidate
            self._pending_frames = 0
            return ArbiterDecision(self._state, False, reason)

        # Microsleep already contains a sustained two-second timer, so adding
        # another confirmation delay would be unsafe and harm boundary recall.
        if candidate is DriverState.MICROSLEEP:
            self._state = candidate
            self._pending = candidate
            self._pending_frames = 0
            return ArbiterDecision(self._state, self._state is not previous, reason)

        if candidate is self._pending:
            self._pending_frames += 1
        else:
            self._pending = candidate
            self._pending_frames = 1
        required = self.release_frames if candidate is DriverState.ALERT else self.confirmation_frames
        if self._pending_frames >= required:
            self._state = candidate
            self._pending_frames = 0
        return ArbiterDecision(self._state, self._state is not previous, reason)
