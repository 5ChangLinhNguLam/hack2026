"""Conservative causal five-state targets derived from partial DMD labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .primitive_labels import DISTRACTING_ACTIONS, IGNORE_INDEX


class FiveState(IntEnum):
    ALERT = 0
    DROWSY = 1
    MICROSLEEP = 2
    YAWNING = 3
    DISTRACTION = 4


@dataclass(frozen=True)
class WeakStateConfig:
    slow_closure_seconds: float = 0.5
    microsleep_seconds: float = 2.0
    microsleep_confirmation_seconds: float = 0.25
    alert_open_seconds: float = 2.0

    def __post_init__(self) -> None:
        durations = (
            self.slow_closure_seconds,
            self.microsleep_seconds,
            self.alert_open_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("weak-label durations must be positive")
        if (
            not math.isfinite(self.microsleep_confirmation_seconds)
            or self.microsleep_confirmation_seconds < 0.0
        ):
            raise ValueError(
                "microsleep confirmation duration must be finite and non-negative"
            )
        if self.slow_closure_seconds >= self.microsleep_seconds:
            raise ValueError("slow closure must be shorter than microsleep")


@dataclass(frozen=True)
class WeakStateTargets:
    targets: tuple[int, ...]
    confidence: tuple[float, ...]
    reasons: tuple[str, ...]
    event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.event_ids and self.targets:
            object.__setattr__(self, "event_ids", ("",) * len(self.targets))
        if not (len(self.targets) == len(self.confidence) == len(self.reasons)):
            raise ValueError("weak-state target arrays must have equal lengths")
        if len(self.event_ids) != len(self.targets):
            raise ValueError("weak-state event IDs must align with targets")


@dataclass(frozen=True)
class WeakStateSupport:
    frames: Mapping[FiveState, int]
    events: Mapping[FiveState, int]
    ignored: int


def _normalized(observation: Mapping[str, object], *names: str) -> str:
    for name in names:
        if name in observation:
            value = observation[name]
            return "" if value is None else str(value).strip().lower()
    return ""


def _yawning(observation: Mapping[str, object]) -> bool:
    if "yawn" in observation:
        value = observation["yawn"]
        if isinstance(value, bool):
            return value
        normalized = "" if value is None else str(value).strip().lower()
    else:
        normalized = _normalized(observation, "yawning")
    return normalized not in {"", "none", "false", "0"}


def build_weak_state_targets(
    protocol: str,
    observations: Sequence[Mapping[str, object]],
    *,
    fps: float,
    config: WeakStateConfig | None = None,
) -> WeakStateTargets:
    """Build one chronological target per observation without future lookahead."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if protocol not in {"s1", "s2", "s3", "s5", "s6"}:
        raise ValueError(f"unsupported DMD protocol: {protocol}")
    config = config or WeakStateConfig()
    action_protocol = protocol in {"s1", "s2", "s3"}
    eye_protocol = protocol == "s5"
    minimum_close_frames = max(1, math.ceil(config.slow_closure_seconds * fps))
    microsleep_frames = max(1, math.ceil(config.microsleep_seconds * fps))
    confirmed_microsleep_frames = max(
        microsleep_frames,
        math.ceil(
            (
                config.microsleep_seconds
                + config.microsleep_confirmation_seconds
            )
            * fps
        ),
    )
    minimum_open_frames = max(1, math.ceil(config.alert_open_seconds * fps))

    targets: list[int] = []
    confidence: list[float] = []
    reasons: list[str] = []
    open_frames = 0
    close_frames = 0

    for observation in observations:
        phase = _normalized(observation, "phase", "eyes_state")
        yawn = _yawning(observation)
        action = _normalized(observation, "action", "driver_actions")
        road_gaze = _normalized(observation, "road_gaze", "gaze_on_road")
        if eye_protocol:
            open_frames = open_frames + 1 if phase == "open" else 0
            close_frames = close_frames + 1 if phase == "close" else 0

        target = IGNORE_INDEX
        weight = 0.0
        reason = "ambiguous_or_unobserved"

        if eye_protocol and close_frames >= confirmed_microsleep_frames:
            target = FiveState.MICROSLEEP
            weight = 0.9
            reason = "continuous_eye_closure_confirmed"
        elif eye_protocol and yawn:
            target = FiveState.YAWNING
            weight = 1.0
            reason = "explicit_yawn"
        elif eye_protocol and close_frames >= microsleep_frames:
            reason = "microsleep_boundary_uncertain"
        elif action_protocol and action in DISTRACTING_ACTIONS:
            target = FiveState.DISTRACTION
            weight = 1.0
            reason = "explicit_distracting_action"
        elif eye_protocol and close_frames >= minimum_close_frames:
            target = FiveState.DROWSY
            weight = 0.7
            reason = "continuous_eye_closure"
        elif eye_protocol and phase == "open" and open_frames >= minimum_open_frames:
            target = FiveState.ALERT
            weight = 0.7
            reason = "sustained_open"
        elif action_protocol and action == "safe_drive" and road_gaze == "looking_road":
            target = FiveState.ALERT
            weight = 0.7
            reason = "explicit_safe_on_road"

        targets.append(target)
        confidence.append(weight)
        reasons.append(reason)

    event_ids: list[str] = []
    event_number = 0
    previous = IGNORE_INDEX
    for target in targets:
        if target == IGNORE_INDEX:
            event_ids.append("")
        else:
            if target != previous:
                event_number += 1
            event_ids.append(
                f"{protocol}:{FiveState(target).name.lower()}:{event_number:06d}"
            )
        previous = target

    return WeakStateTargets(
        tuple(targets),
        tuple(confidence),
        tuple(reasons),
        tuple(event_ids),
    )


def weak_state_support(results: Iterable[WeakStateTargets]) -> WeakStateSupport:
    frame_counts: Counter[FiveState] = Counter({state: 0 for state in FiveState})
    event_counts: Counter[FiveState] = Counter({state: 0 for state in FiveState})
    ignored = 0
    for result in results:
        previous = IGNORE_INDEX
        for target in result.targets:
            if target == IGNORE_INDEX:
                ignored += 1
            else:
                state = FiveState(target)
                frame_counts[state] += 1
                if target != previous:
                    event_counts[state] += 1
            previous = target
    return WeakStateSupport(
        frames=MappingProxyType(dict(frame_counts)),
        events=MappingProxyType(dict(event_counts)),
        ignored=ignored,
    )


__all__ = [
    "IGNORE_INDEX",
    "FiveState",
    "WeakStateConfig",
    "WeakStateSupport",
    "WeakStateTargets",
    "build_weak_state_targets",
    "weak_state_support",
]
