"""Causal Challenge 3 safe-driving score for the unified replay.

The hackathon evaluator reconstructs the trip score from unredacted ego
kinematics plus the team's raw C1 TTC.  This module mirrors that contract
without reading ground truth, labels, targets, depth, or event annotations.

The organizer formula also contains a tailgating percentage.  The submission
contract has no predicted headway field, so both the provided evaluator and
this runtime deliberately omit that term and expose the omission in every
report.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


G_MS2 = 9.81
HARSH_BRAKE_THRESHOLD_MPS2 = -0.40 * G_MS2
HARSH_ACCEL_THRESHOLD_MPS2 = 0.35 * G_MS2
HARSH_CORNER_THRESHOLD_MPS2 = 0.30 * G_MS2
SPEEDING_TOLERANCE_KMH = 5.0
NEAR_MISS_TTC_SECONDS = 1.5
FORMULA_VERSION = "hackathon-evaluator-v1-no-tailgating"


def _grade(safe_score: float) -> str:
    if safe_score >= 90.0:
        return "A"
    if safe_score >= 80.0:
        return "B"
    if safe_score >= 70.0:
        return "C"
    if safe_score >= 60.0:
        return "D"
    return "E"


def _finite_number(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"C3 requires numeric {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"C3 requires finite {field}")
    return number


@dataclass(frozen=True)
class Challenge3FrameEstimate:
    """Evaluator-compatible prefix score after one source frame.

    ``safe_score_estimate`` is higher-is-safer.  ``risk_score_pct`` is its
    higher-is-more-dangerous complement and is kept separate to avoid mixing
    the two score directions.
    """

    frame_id: int
    timestamp: float
    safe_score_estimate: float
    risk_score_pct: float
    grade: str
    total_penalty: float
    processed_frames: int
    near_miss_frames: int
    harsh_brake_frames: int
    harsh_accel_frames: int
    harsh_corner_frames: int
    speeding_frames: int
    speeding_pct_time: float
    is_near_miss: bool
    is_harsh_brake: bool
    is_harsh_accel: bool
    is_harsh_corner: bool
    is_speeding: bool
    trip_complete: bool
    formula_version: str = FORMULA_VERSION
    tailgating_penalty_omitted: bool = True

    def diagnostic_row(self) -> dict[str, object]:
        return {
            "c3_safe_score_estimate": round(self.safe_score_estimate, 3),
            "c3_risk_score_pct": round(self.risk_score_pct, 3),
            "c3_grade": self.grade,
            "c3_total_penalty": round(self.total_penalty, 3),
            "c3_processed_frames": self.processed_frames,
            "c3_near_miss_frames": self.near_miss_frames,
            "c3_harsh_brake_frames": self.harsh_brake_frames,
            "c3_harsh_accel_frames": self.harsh_accel_frames,
            "c3_harsh_corner_frames": self.harsh_corner_frames,
            "c3_speeding_frames": self.speeding_frames,
            "c3_speeding_pct_time": round(self.speeding_pct_time, 6),
            "c3_is_near_miss": self.is_near_miss,
            "c3_is_harsh_brake": self.is_harsh_brake,
            "c3_is_harsh_accel": self.is_harsh_accel,
            "c3_is_harsh_corner": self.is_harsh_corner,
            "c3_is_speeding": self.is_speeding,
            "c3_trip_complete": self.trip_complete,
            "c3_tailgating_penalty_omitted": self.tailgating_penalty_omitted,
        }


class Challenge3Accumulator:
    """Accumulate the Challenge 3 formula over an ordered source stream."""

    def __init__(
        self,
        *,
        speed_limit_kmh: float,
        expected_frames: int | None = None,
    ) -> None:
        self.speed_limit_kmh = _finite_number(
            speed_limit_kmh, field="metadata.speed_limit_kmh"
        )
        if self.speed_limit_kmh < 0.0:
            raise ValueError("C3 speed_limit_kmh must be non-negative")
        if expected_frames is not None and expected_frames < 0:
            raise ValueError("C3 expected_frames must be non-negative")
        self.expected_frames = expected_frames
        self.reset()

    def reset(self) -> None:
        self.processed_frames = 0
        self.near_miss_frames = 0
        self.harsh_brake_frames = 0
        self.harsh_accel_frames = 0
        self.harsh_corner_frames = 0
        self.speeding_frames = 0
        self._last_frame_id: int | None = None
        self._last_timestamp: float | None = None
        self._latest: Challenge3FrameEstimate | None = None

    @staticmethod
    def _near_miss(predicted_ttc_s: object) -> bool:
        try:
            value = float(predicted_ttc_s)
        except (TypeError, ValueError):
            return False
        # Match team_kit.evaluation exactly.  The C1 runtime separately rejects
        # negative outputs, but the scorer itself classifies every finite value
        # below the threshold as a near miss.
        return math.isfinite(value) and value < NEAR_MISS_TTC_SECONDS

    def _validate_position(self, frame_id: int, timestamp: float) -> None:
        if frame_id < 0:
            raise ValueError("C3 frame_id must be non-negative")
        if self._last_frame_id is None:
            if frame_id != 0:
                raise ValueError("C3 replay must start at frame 0 or be pre-rolled")
        elif frame_id != self._last_frame_id + 1:
            raise ValueError(
                "C3 frames must be contiguous and ordered: "
                f"received {frame_id} after {self._last_frame_id}"
            )
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("C3 timestamp must be finite and non-negative")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("C3 timestamps must be strictly increasing")
        if (
            self.expected_frames is not None
            and self.processed_frames >= self.expected_frames
        ):
            raise ValueError("C3 received more frames than expected")

    def update(
        self,
        bundle: Any,
        *,
        predicted_ttc_s: object,
    ) -> Challenge3FrameEstimate:
        frame_id = int(bundle.frame_id)
        timestamp = float(bundle.timestamp)
        self._validate_position(frame_id, timestamp)
        ego = getattr(bundle, "ego", None)
        if not isinstance(ego, Mapping):
            raise ValueError("C3 requires bundle.ego telemetry")

        longitudinal = _finite_number(
            ego.get("longitudinal_accel"), field="ego.longitudinal_accel"
        )
        lateral = _finite_number(
            ego.get("lateral_accel"), field="ego.lateral_accel"
        )
        speed = _finite_number(ego.get("speed_kmh"), field="ego.speed_kmh")
        is_harsh_brake = longitudinal < HARSH_BRAKE_THRESHOLD_MPS2
        is_harsh_accel = longitudinal > HARSH_ACCEL_THRESHOLD_MPS2
        is_harsh_corner = abs(lateral) > HARSH_CORNER_THRESHOLD_MPS2
        is_speeding = speed > self.speed_limit_kmh + SPEEDING_TOLERANCE_KMH
        is_near_miss = self._near_miss(predicted_ttc_s)

        self.processed_frames += 1
        self.near_miss_frames += int(is_near_miss)
        self.harsh_brake_frames += int(is_harsh_brake)
        self.harsh_accel_frames += int(is_harsh_accel)
        self.harsh_corner_frames += int(is_harsh_corner)
        self.speeding_frames += int(is_speeding)
        speeding_pct = 100.0 * self.speeding_frames / self.processed_frames
        total_penalty = (
            self.harsh_brake_frames * 3.0
            + self.harsh_accel_frames * 2.0
            + self.harsh_corner_frames * 2.0
            + self.near_miss_frames * 5.0
            + (speeding_pct / 100.0) * 15.0
        )
        safe_score = max(0.0, min(100.0, 100.0 - total_penalty))
        trip_complete = (
            self.expected_frames is not None
            and self.processed_frames == self.expected_frames
        )
        estimate = Challenge3FrameEstimate(
            frame_id=frame_id,
            timestamp=timestamp,
            safe_score_estimate=safe_score,
            risk_score_pct=100.0 - safe_score,
            grade=_grade(safe_score),
            total_penalty=total_penalty,
            processed_frames=self.processed_frames,
            near_miss_frames=self.near_miss_frames,
            harsh_brake_frames=self.harsh_brake_frames,
            harsh_accel_frames=self.harsh_accel_frames,
            harsh_corner_frames=self.harsh_corner_frames,
            speeding_frames=self.speeding_frames,
            speeding_pct_time=speeding_pct,
            is_near_miss=is_near_miss,
            is_harsh_brake=is_harsh_brake,
            is_harsh_accel=is_harsh_accel,
            is_harsh_corner=is_harsh_corner,
            is_speeding=is_speeding,
            trip_complete=trip_complete,
        )
        self._last_frame_id = frame_id
        self._last_timestamp = timestamp
        self._latest = estimate
        return estimate

    def summary(self) -> dict[str, object]:
        if self._latest is None:
            safe_score = 100.0
            total_penalty = 0.0
            speeding_pct = 0.0
            trip_complete = self.expected_frames == 0
        else:
            safe_score = self._latest.safe_score_estimate
            total_penalty = self._latest.total_penalty
            speeding_pct = self._latest.speeding_pct_time
            trip_complete = self._latest.trip_complete
        return {
            "semantics": "evaluator-compatible trip safe-score estimate",
            "score_direction": "higher_is_safer",
            "formula_version": FORMULA_VERSION,
            "trip_complete": trip_complete,
            "processed_frames": self.processed_frames,
            "expected_frames": self.expected_frames,
            "final_safe_score_estimate": round(safe_score, 3),
            "final_risk_score_pct": round(100.0 - safe_score, 3),
            "grade": _grade(safe_score),
            "total_penalty": round(total_penalty, 3),
            "counts": {
                "near_miss_frames": self.near_miss_frames,
                "harsh_brake_frames": self.harsh_brake_frames,
                "harsh_accel_frames": self.harsh_accel_frames,
                "harsh_corner_frames": self.harsh_corner_frames,
                "speeding_frames": self.speeding_frames,
            },
            "penalties": {
                "near_miss": round(self.near_miss_frames * 5.0, 3),
                "harsh_brake": round(self.harsh_brake_frames * 3.0, 3),
                "harsh_accel": round(self.harsh_accel_frames * 2.0, 3),
                "harsh_corner": round(self.harsh_corner_frames * 2.0, 3),
                "speeding": round((speeding_pct / 100.0) * 15.0, 3),
            },
            "speeding_pct_time": round(speeding_pct, 6),
            "tailgating_status": "unavailable",
            "tailgating_penalty_omitted": True,
        }


__all__ = [
    "Challenge3Accumulator",
    "Challenge3FrameEstimate",
    "FORMULA_VERSION",
    "G_MS2",
    "HARSH_ACCEL_THRESHOLD_MPS2",
    "HARSH_BRAKE_THRESHOLD_MPS2",
    "HARSH_CORNER_THRESHOLD_MPS2",
    "NEAR_MISS_TTC_SECONDS",
    "SPEEDING_TOLERANCE_KMH",
]
