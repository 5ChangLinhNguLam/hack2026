"""Causal product-facing driving-quality score.

Challenge 3 deliberately counts every flagged frame, which makes its official
score saturate at zero on the supplied trips.  This module leaves that contract
untouched and derives a separate, easier-to-interpret product metric from the
already-computed :class:`~safeloop.c3.Challenge3FrameEstimate` flags.

The product metric groups nearby flagged frames into one driving event and
scores either the observed trip prefix or the latest 60 seconds.  It does not
read labels, ground truth, future frames, or Challenge 2 state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping


EVENT_KEYS = (
    "near_miss",
    "harsh_brake",
    "harsh_accel",
    "harsh_corner",
)
WINDOW_SECONDS = 60.0
EVENT_MERGE_GAP_SECONDS = 0.5
FORMULA_VERSION = "safeloop-drive-quality-v1"

NO_DATA = "NO_DATA"
PREFIX = "PREFIX"
FULL_TRIP = "FULL_TRIP"
ROLLING_60S = "ROLLING_60S"


def _grade(score_pct: float) -> str:
    if score_pct >= 90.0:
        return "A"
    if score_pct >= 80.0:
        return "B"
    if score_pct >= 70.0:
        return "C"
    if score_pct >= 60.0:
        return "D"
    return "E"


def _immutable_bool_map(values: Mapping[str, bool]) -> Mapping[str, bool]:
    return MappingProxyType({key: bool(values[key]) for key in EVENT_KEYS})


def _immutable_count_map(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({key: int(values[key]) for key in EVENT_KEYS})


@dataclass(frozen=True)
class DriveQualityFrameEstimate:
    """Driving-quality result after consuming one source frame.

    ``score_pct`` is higher-is-better and is intentionally distinct from both
    the official C3 safe score and the higher-is-dangerous contextual risk.
    """

    frame_id: int
    timestamp: float
    score_available: bool
    score_pct: float | None
    grade: str
    total_penalty: float | None
    scope: str
    observed_seconds: float
    window_ready: bool
    event_counts_window: Mapping[str, int]
    speeding_pct_window: float
    new_events: Mapping[str, bool]
    trip_complete: bool
    formula_version: str = FORMULA_VERSION

    def diagnostic_row(self) -> dict[str, object]:
        """Return flat diagnostics with an unambiguous product prefix."""

        counts = self.event_counts_window
        new = self.new_events
        return {
            "drive_quality_score_available": self.score_available,
            "drive_quality_score_pct": (
                round(self.score_pct, 3)
                if self.score_pct is not None
                else ""
            ),
            "drive_quality_grade": self.grade,
            "drive_quality_total_penalty": (
                round(self.total_penalty, 3)
                if self.total_penalty is not None
                else ""
            ),
            "drive_quality_scope": self.scope,
            "drive_quality_observed_seconds": round(
                self.observed_seconds, 3
            ),
            "drive_quality_window_ready": self.window_ready,
            "drive_quality_near_miss_events_window": counts["near_miss"],
            "drive_quality_harsh_brake_events_window": counts["harsh_brake"],
            "drive_quality_harsh_accel_events_window": counts["harsh_accel"],
            "drive_quality_harsh_corner_events_window": counts["harsh_corner"],
            "drive_quality_speeding_pct_window": round(
                self.speeding_pct_window, 6
            ),
            "drive_quality_new_near_miss_event": new["near_miss"],
            "drive_quality_new_harsh_brake_event": new["harsh_brake"],
            "drive_quality_new_harsh_accel_event": new["harsh_accel"],
            "drive_quality_new_harsh_corner_event": new["harsh_corner"],
            "drive_quality_trip_complete": self.trip_complete,
        }


class DriveQualityAccumulator:
    """Convert C3 frame flags into a causal event-based quality score.

    A flag starts a new event when no flag of the same kind was seen in the
    previous ``merge_gap_seconds``.  Thus a sustained flag, or a short gap in
    one, contributes one event instead of one penalty per video frame.
    """

    def __init__(
        self,
        *,
        window_seconds: float = WINDOW_SECONDS,
        merge_gap_seconds: float = EVENT_MERGE_GAP_SECONDS,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError("drive-quality window_seconds must be positive")
        if not math.isfinite(merge_gap_seconds) or merge_gap_seconds < 0.0:
            raise ValueError(
                "drive-quality merge_gap_seconds must be non-negative"
            )
        self.window_seconds = float(window_seconds)
        self.merge_gap_seconds = float(merge_gap_seconds)
        self.reset()

    def reset(self) -> None:
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._last_frame_id: int | None = None
        self._last_positive_timestamp: dict[str, float | None] = {
            key: None for key in EVENT_KEYS
        }
        # Each mutable pair is [start_timestamp, last_positive_timestamp].
        # Keeping the last positive time ensures a sustained event remains in
        # a rolling window even after its original onset has aged out.
        self._episodes: dict[str, deque[list[float]]] = {
            key: deque() for key in EVENT_KEYS
        }
        self._inactive_since: dict[str, float | None] = {
            key: None for key in EVENT_KEYS
        }
        self._speeding_samples: deque[tuple[float, bool]] = deque()
        self._complete = False
        self._latest: DriveQualityFrameEstimate | None = None

    @property
    def latest(self) -> DriveQualityFrameEstimate | None:
        return self._latest

    def _validate_position(self, frame_id: int, timestamp: float) -> None:
        if self._complete:
            raise ValueError("drive-quality trip is already complete")
        if frame_id < 0:
            raise ValueError("drive-quality frame_id must be non-negative")
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "drive-quality timestamp must be finite and non-negative"
            )
        if self._last_frame_id is None:
            if frame_id != 0:
                raise ValueError(
                    "drive-quality replay must start at frame 0 or be pre-rolled"
                )
        elif frame_id != self._last_frame_id + 1:
            raise ValueError(
                "drive-quality frames must be contiguous and ordered: "
                f"received {frame_id} after {self._last_frame_id}"
            )
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                "drive-quality timestamps must be strictly increasing"
            )

    @staticmethod
    def _flags(c3: Any) -> dict[str, bool]:
        return {
            "near_miss": bool(c3.is_near_miss),
            "harsh_brake": bool(c3.is_harsh_brake),
            "harsh_accel": bool(c3.is_harsh_accel),
            "harsh_corner": bool(c3.is_harsh_corner),
        }

    def _window_start(self, timestamp: float) -> float:
        if self._first_timestamp is None:
            return timestamp
        return max(self._first_timestamp, timestamp - self.window_seconds)

    def _prune(self, window_start: float) -> None:
        for episodes in self._episodes.values():
            while episodes and episodes[0][1] < window_start:
                episodes.popleft()
        while (
            self._speeding_samples
            and self._speeding_samples[0][0] < window_start
        ):
            self._speeding_samples.popleft()

    def _scope(self, timestamp: float, *, trip_complete: bool) -> str:
        if self._first_timestamp is None:
            return NO_DATA
        elapsed = timestamp - self._first_timestamp
        if elapsed >= self.window_seconds:
            return ROLLING_60S
        if trip_complete:
            return FULL_TRIP
        return PREFIX

    def update(self, c3: Any) -> DriveQualityFrameEstimate:
        frame_id = int(c3.frame_id)
        timestamp = float(c3.timestamp)
        self._validate_position(frame_id, timestamp)
        if self._first_timestamp is None:
            self._first_timestamp = timestamp

        flags = self._flags(c3)
        new_events: dict[str, bool] = {}
        for key in EVENT_KEYS:
            is_new = False
            if flags[key]:
                previous = self._last_positive_timestamp[key]
                is_new = (
                    previous is None
                    or (
                        self._inactive_since[key] is not None
                        and timestamp - self._inactive_since[key]
                        > self.merge_gap_seconds
                    )
                )
                if is_new:
                    self._episodes[key].append([timestamp, timestamp])
                else:
                    if self._episodes[key]:
                        self._episodes[key][-1][1] = timestamp
                    else:
                        # A custom scoring window may be shorter than the merge
                        # gap.  Reintroduce this carry-in episode to the window
                        # without incorrectly announcing a new event.
                        self._episodes[key].append([timestamp, timestamp])
                self._last_positive_timestamp[key] = timestamp
                self._inactive_since[key] = None
            elif (
                self._last_positive_timestamp[key] is not None
                and self._inactive_since[key] is None
            ):
                self._inactive_since[key] = timestamp
            new_events[key] = is_new

        self._speeding_samples.append((timestamp, bool(c3.is_speeding)))
        window_start = self._window_start(timestamp)
        self._prune(window_start)
        counts = {
            key: len(self._episodes[key]) for key in EVENT_KEYS
        }
        speeding_pct = (
            100.0
            * sum(int(flag) for _, flag in self._speeding_samples)
            / len(self._speeding_samples)
        )
        total_penalty = (
            counts["near_miss"] * 5.0
            + counts["harsh_brake"] * 3.0
            + counts["harsh_accel"] * 2.0
            + counts["harsh_corner"] * 2.0
            + speeding_pct * 0.15
        )
        score = max(0.0, min(100.0, 100.0 - total_penalty))
        elapsed = timestamp - self._first_timestamp
        observed_seconds = min(elapsed, self.window_seconds)
        trip_complete = bool(c3.trip_complete)
        estimate = DriveQualityFrameEstimate(
            frame_id=frame_id,
            timestamp=timestamp,
            score_available=True,
            score_pct=score,
            grade=_grade(score),
            total_penalty=total_penalty,
            scope=self._scope(timestamp, trip_complete=trip_complete),
            observed_seconds=observed_seconds,
            window_ready=elapsed >= self.window_seconds,
            event_counts_window=_immutable_count_map(counts),
            speeding_pct_window=speeding_pct,
            new_events=_immutable_bool_map(new_events),
            trip_complete=trip_complete,
        )
        self._last_frame_id = frame_id
        self._last_timestamp = timestamp
        self._complete = trip_complete
        self._latest = estimate
        return estimate

    def summary(self) -> dict[str, object]:
        if self._latest is None:
            return {
                "semantics": "event-based product driving-quality score",
                "score_direction": "higher_is_better",
                "formula_version": FORMULA_VERSION,
                "score_available": False,
                "score_pct": None,
                "grade": "N/A",
                "total_penalty": None,
                "scope": NO_DATA,
                "observed_seconds": 0.0,
                "window_ready": False,
                "event_counts_window": {key: 0 for key in EVENT_KEYS},
                "speeding_pct_window": 0.0,
                "trip_complete": False,
            }
        latest = self._latest
        return {
            "semantics": "event-based product driving-quality score",
            "score_direction": "higher_is_better",
            "formula_version": FORMULA_VERSION,
            "score_available": latest.score_available,
            "score_pct": round(float(latest.score_pct), 3),
            "grade": latest.grade,
            "total_penalty": round(float(latest.total_penalty), 3),
            "scope": latest.scope,
            "observed_seconds": round(latest.observed_seconds, 3),
            "window_ready": latest.window_ready,
            "event_counts_window": dict(latest.event_counts_window),
            "speeding_pct_window": round(latest.speeding_pct_window, 6),
            "trip_complete": latest.trip_complete,
        }


__all__ = [
    "DriveQualityAccumulator",
    "DriveQualityFrameEstimate",
    "EVENT_KEYS",
    "EVENT_MERGE_GAP_SECONDS",
    "FORMULA_VERSION",
    "FULL_TRIP",
    "NO_DATA",
    "PREFIX",
    "ROLLING_60S",
    "WINDOW_SECONDS",
]
