"""Visibility-weighted online PERCLOS and continuous-closure timing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass
class _WeightedEvent:
    timestamp: float
    probability: float
    quality: float
    slow_probability: float = 0.0


@dataclass(frozen=True)
class PerclosSnapshot:
    # Backward-compatible primary (largest-window) fields.
    perclos: float
    valid_fraction: float
    reliable: bool
    closure_duration_seconds: float
    microsleep: bool
    # Full multi-window evidence. None means no visible denominator.
    values: Mapping[float, float | None]
    slow_values: Mapping[float, float | None]
    valid_fractions: Mapping[float, float]
    reliable_by_window: Mapping[float, bool]


class OnlinePerclos:
    """Update raw and slow-closure PERCLOS causally at a fixed frame rate.

    Raw PERCLOS follows ``sum(visibility * p_closed) / sum(visibility)``.
    Slow PERCLOS uses the same denominator but retroactively commits a closure
    run only after it reaches ``slow_closure_seconds``. Thus ordinary short
    blinks never inflate the drowsiness signal. Missing eyes have quality zero
    and can preserve, but never increase, a closure for a short configured gap.
    """

    def __init__(
        self,
        *,
        fps: float,
        window_seconds: float = 60.0,
        windows_seconds: Sequence[float] | None = None,
        closure_threshold: float = 0.8,
        closure_release_threshold: float | None = None,
        visibility_threshold: float = 0.2,
        minimum_valid_fraction: float = 0.6,
        slow_closure_seconds: float = 0.5,
        microsleep_seconds: float = 2.0,
        uncertain_gap_seconds: float = 0.0,
    ) -> None:
        windows = (
            tuple(sorted({float(value) for value in windows_seconds}))
            if windows_seconds is not None
            else (float(window_seconds),)
        )
        if (
            fps <= 0.0
            or not windows
            or any(value <= 0.0 for value in windows)
            or slow_closure_seconds <= 0.0
            or microsleep_seconds <= 0.0
            or not math.isfinite(uncertain_gap_seconds)
            or uncertain_gap_seconds < 0.0
        ):
            raise ValueError(
                "fps, windows, slow_closure_seconds, and microsleep_seconds must be positive"
            )
        release_threshold = (
            closure_threshold
            if closure_release_threshold is None
            else closure_release_threshold
        )
        for name, value in (
            ("closure_threshold", closure_threshold),
            ("closure_release_threshold", release_threshold),
            ("visibility_threshold", visibility_threshold),
            ("minimum_valid_fraction", minimum_valid_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if release_threshold > closure_threshold:
            raise ValueError(
                "closure_release_threshold cannot exceed closure_threshold"
            )
        self.fps = float(fps)
        self.windows_seconds = windows
        self.window_seconds = max(windows)
        self.closure_threshold = closure_threshold
        self.closure_release_threshold = release_threshold
        self.visibility_threshold = visibility_threshold
        self.minimum_valid_fraction = minimum_valid_fraction
        self.slow_closure_seconds = slow_closure_seconds
        self.microsleep_seconds = microsleep_seconds
        self.uncertain_gap_seconds = float(uncertain_gap_seconds)
        self._events = {window: deque() for window in windows}
        self._numerators = {window: 0.0 for window in windows}
        self._slow_numerators = {window: 0.0 for window in windows}
        self._denominators = {window: 0.0 for window in windows}
        self._closure_run: list[_WeightedEvent] = []
        self._closure_duration_seconds = 0.0
        self._closed_state = False
        self._uncertain_gap_start: float | None = None
        self._microsleep_active = False
        self._last_timestamp: float | None = None

    @property
    def sample_count(self) -> int:
        return len(self._events[self.window_seconds])

    def reset(self) -> None:
        for events in self._events.values():
            events.clear()
        for totals in (
            self._numerators,
            self._slow_numerators,
            self._denominators,
        ):
            for window in totals:
                totals[window] = 0.0
        self._closure_run.clear()
        self._closure_duration_seconds = 0.0
        self._closed_state = False
        self._uncertain_gap_start = None
        self._microsleep_active = False
        self._last_timestamp = None

    def _reset_closure(self) -> None:
        self._closure_duration_seconds = 0.0
        self._closure_run.clear()
        self._closed_state = False
        self._uncertain_gap_start = None
        self._microsleep_active = False

    def _timestamp(self, timestamp: float | None) -> float:
        if timestamp is None:
            return (
                0.0
                if self._last_timestamp is None
                else self._last_timestamp + 1.0 / self.fps
            )
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        return timestamp

    def _append_and_evict(self, event: _WeightedEvent) -> None:
        for window, events in self._events.items():
            events.append(event)
            self._numerators[window] += event.quality * event.probability
            self._slow_numerators[window] += (
                event.quality * event.slow_probability
            )
            self._denominators[window] += event.quality
            cutoff = event.timestamp - window
            while events and events[0].timestamp <= cutoff:
                old = events.popleft()
                self._numerators[window] -= old.quality * old.probability
                self._slow_numerators[window] -= (
                    old.quality * old.slow_probability
                )
                self._denominators[window] -= old.quality

    def _commit_slow_run(self) -> None:
        for event in self._closure_run:
            if event.slow_probability != 0.0:
                continue
            event.slow_probability = event.probability
            for window, events in self._events.items():
                if any(candidate is event for candidate in events):
                    self._slow_numerators[window] += (
                        event.quality * event.probability
                    )

    def update(
        self,
        *,
        closed_probability: float,
        visibility: float,
        timestamp: float | None = None,
    ) -> PerclosSnapshot:
        if not 0.0 <= closed_probability <= 1.0 or not 0.0 <= visibility <= 1.0:
            raise ValueError(
                "closed_probability and visibility must be between 0 and 1"
            )
        previous_timestamp = self._last_timestamp
        timestamp = self._timestamp(timestamp)
        elapsed = (
            1.0 / self.fps
            if previous_timestamp is None
            else timestamp - previous_timestamp
        )
        self._last_timestamp = timestamp
        event = _WeightedEvent(timestamp, closed_probability, visibility)
        self._append_and_evict(event)

        visible = visibility >= self.visibility_threshold
        was_closed = self._closed_state
        had_uncertain_gap = self._uncertain_gap_start is not None
        reliable_closed = False
        if not visible:
            if was_closed and self.uncertain_gap_seconds > 0.0:
                if self._uncertain_gap_start is None:
                    self._uncertain_gap_start = timestamp
                uncertain_elapsed = (
                    timestamp
                    - self._uncertain_gap_start
                    + 1.0 / self.fps
                )
                if uncertain_elapsed > self.uncertain_gap_seconds + 1e-9:
                    self._reset_closure()
            else:
                self._reset_closure()
        else:
            closed = (
                closed_probability >= self.closure_release_threshold
                if was_closed
                else closed_probability >= self.closure_threshold
            )
            if closed:
                reliable_closed = True
                self._closed_state = True
                increment = (
                    1.0 / self.fps
                    if not was_closed or had_uncertain_gap
                    else elapsed
                )
                self._closure_duration_seconds = round(
                    self._closure_duration_seconds + increment,
                    9,
                )
                self._closure_run.append(event)
                self._uncertain_gap_start = None
                if (
                    self._closure_duration_seconds + 1e-9
                    >= self.slow_closure_seconds
                ):
                    self._commit_slow_run()
                if (
                    self._closure_duration_seconds + 1e-9
                    >= self.microsleep_seconds
                ):
                    self._microsleep_active = True
            else:
                self._reset_closure()

        values: dict[float, float | None] = {}
        slow_values: dict[float, float | None] = {}
        valid_fractions: dict[float, float] = {}
        reliability: dict[float, bool] = {}
        for window, events in self._events.items():
            denominator = max(self._denominators[window], 0.0)
            values[window] = (
                self._numerators[window] / denominator if denominator > 0.0 else None
            )
            slow_values[window] = (
                self._slow_numerators[window] / denominator
                if denominator > 0.0
                else None
            )
            fraction = denominator / len(events) if events else 0.0
            valid_fractions[window] = fraction
            reliability[window] = (
                denominator > 0.0
                and fraction >= self.minimum_valid_fraction
            )

        primary = self.window_seconds
        primary_value = values[primary]
        closure_duration = self._closure_duration_seconds
        return PerclosSnapshot(
            perclos=0.0 if primary_value is None else primary_value,
            valid_fraction=valid_fractions[primary],
            reliable=reliability[primary],
            closure_duration_seconds=closure_duration,
            microsleep=(
                self._microsleep_active
                and (reliable_closed or self._uncertain_gap_start is not None)
            ),
            values=MappingProxyType(values),
            slow_values=MappingProxyType(slow_values),
            valid_fractions=MappingProxyType(valid_fractions),
            reliable_by_window=MappingProxyType(reliability),
        )
