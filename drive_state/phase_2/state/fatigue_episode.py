"""Causal persistence for a behaviorally validated drowsiness event."""

from __future__ import annotations

import math


class CausalFatigueEpisode:
    """Hold a validated drowsiness event as a latent fatigue episode."""

    def __init__(self, *, fps: float, hold_seconds: float = 15.0) -> None:
        if (
            not math.isfinite(fps)
            or not math.isfinite(hold_seconds)
            or fps <= 0.0
            or hold_seconds <= 0.0
        ):
            raise ValueError("fatigue episode FPS and hold duration must be positive")
        self.fps = float(fps)
        self.hold_seconds = float(hold_seconds)
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._last_timestamp: float | None = None
        self._active_until: float | None = None

    def _timestamp(self, explicit: float | None) -> float:
        timestamp = self._step / self.fps if explicit is None else float(explicit)
        if not math.isfinite(timestamp):
            raise ValueError("fatigue episode timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("fatigue episode timestamps must be strictly increasing")
        self._step += 1
        self._last_timestamp = timestamp
        return timestamp

    def update(
        self,
        *,
        event_active: bool,
        blocked: bool,
        timestamp: float | None = None,
    ) -> bool:
        """Return whether the exclusive drowsy episode is active now."""

        now = self._timestamp(timestamp)
        if bool(event_active) and not bool(blocked):
            self._active_until = now + self.hold_seconds
        retained = self._active_until is not None and now < self._active_until
        return bool(retained and not blocked)


__all__ = ["CausalFatigueEpisode"]
