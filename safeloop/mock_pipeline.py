"""Deterministic C1/C2/Risk mock used to integrate SafeLoop before models land.

The mock only consumes the operational telemetry contract.  It never reads
practice ground truth, event labels, target positions, or driver labels.  The
``mock`` marker is intentionally carried on every decision so demo output
cannot be mistaken for inference from a trained model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol, TextIO

from tripkit.types import FrameBundle

from .telemetry import (
    CarSkyApiError,
    CarSkyRestClient,
    CarSkyScalar,
    PublishStats,
    TelemetryMessage,
)

SCHEMA_VERSION = "safeloop.decision.v1"
SCENARIO_PERIOD_MS = 16_000


def _lerp(start: float, end: float, ratio: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, ratio))


@dataclass(frozen=True)
class C1Prediction:
    predicted_ttc_s: float
    obstacle_distance_m: float
    confidence: float
    is_warning: bool


@dataclass(frozen=True)
class C2Prediction:
    driver_state: str
    confidence: float
    attentive_probability_pct: float
    distraction_level_pct: float
    fatigue_level_pct: float
    is_eyes_on_road: bool
    is_warning: bool


@dataclass(frozen=True)
class RiskDecision:
    score: float
    level: str
    action: str
    brake_request_pct: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class C3Score:
    score: float
    grade: str
    collision_score: float
    driver_score: float
    comfort_score: float
    total_penalty: float


class C3Accumulator:
    """Stateful mock trip score; penalties accumulate and never use GT."""

    def __init__(self):
        self.previous_timestamp_ms: int | None = None
        self.previous_phase_ms: int | None = None
        self.collision_penalty = 0.0
        self.driver_penalty = 0.0
        self.comfort_penalty = 0.0

    def _reset(self) -> None:
        self.collision_penalty = 0.0
        self.driver_penalty = 0.0
        self.comfort_penalty = 0.0

    def update(
        self,
        telemetry: TelemetryMessage,
        c1: C1Prediction,
        c2: C2Prediction,
    ) -> C3Score:
        phase_ms = telemetry.timestamp_ms % SCENARIO_PERIOD_MS
        if self.previous_phase_ms is not None and phase_ms < self.previous_phase_ms:
            self._reset()

        if self.previous_timestamp_ms is None:
            dt_s = 0.0
        else:
            dt_s = min(0.25, max(0.0, (telemetry.timestamp_ms - self.previous_timestamp_ms) / 1_000))

        collision_exposure = min(1.0, max(0.0, (3.0 - c1.predicted_ttc_s) / 2.2))
        driver_exposure = max(
            100.0 - c2.attentive_probability_pct,
            c2.distraction_level_pct,
            c2.fatigue_level_pct,
        ) / 100.0
        longitudinal = abs(telemetry.ego.longitudinal_accel_mps2)
        lateral = abs(telemetry.ego.lateral_accel_mps2)
        comfort_exposure = min(
            1.0,
            max(0.0, longitudinal - 2.0) / 6.0
            + max(0.0, lateral - 1.5) / 4.0,
        )

        self.collision_penalty += dt_s * 2.8 * collision_exposure
        self.driver_penalty += dt_s * 1.5 * driver_exposure
        self.comfort_penalty += dt_s * 0.7 * comfort_exposure
        total_penalty = self.collision_penalty + self.driver_penalty + self.comfort_penalty
        score = max(0.0, 100.0 - total_penalty)
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "E"

        self.previous_timestamp_ms = telemetry.timestamp_ms
        self.previous_phase_ms = phase_ms
        return C3Score(
            score=round(score, 1),
            grade=grade,
            collision_score=round(max(0.0, 100.0 - self.collision_penalty), 1),
            driver_score=round(max(0.0, 100.0 - self.driver_penalty), 1),
            comfort_score=round(max(0.0, 100.0 - self.comfort_penalty), 1),
            total_penalty=round(total_penalty, 1),
        )


@dataclass(frozen=True)
class MockDecisionMessage:
    schema_version: str
    mock: bool
    mock_scenario: str
    message_id: str
    trip_id: str
    frame_id: int
    timestamp_ms: int
    telemetry: dict
    c1: C1Prediction
    c2: C2Prediction
    c3: C3Score
    risk: RiskDecision

    @classmethod
    def from_telemetry(
        cls,
        telemetry: TelemetryMessage,
        *,
        c3_accumulator: C3Accumulator | None = None,
    ) -> "MockDecisionMessage":
        c1 = predict_c1(telemetry)
        c2 = predict_c2(telemetry)
        accumulator = c3_accumulator if c3_accumulator is not None else C3Accumulator()
        return cls(
            schema_version=SCHEMA_VERSION,
            mock=True,
            mock_scenario="normal-to-critical-to-recovery-v1",
            message_id=telemetry.message_id,
            trip_id=telemetry.trip_id,
            frame_id=telemetry.frame_id,
            timestamp_ms=telemetry.timestamp_ms,
            telemetry=telemetry.to_dict()["ego"],
            c1=c1,
            c2=c2,
            c3=accumulator.update(telemetry, c1, c2),
            risk=fuse_risk(c1, c2),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )


def predict_c1(telemetry: TelemetryMessage) -> C1Prediction:
    """Generate a visible, repeatable TTC scenario without using labels."""

    phase_ms = telemetry.timestamp_ms % SCENARIO_PERIOD_MS
    if phase_ms < 4_000:
        ttc_s = _lerp(8.0, 5.0, phase_ms / 4_000)
    elif phase_ms < 8_000:
        ttc_s = _lerp(5.0, 2.2, (phase_ms - 4_000) / 4_000)
    elif phase_ms < 12_000:
        ttc_s = _lerp(2.2, 0.8, (phase_ms - 8_000) / 4_000)
    else:
        ttc_s = _lerp(0.8, 8.0, (phase_ms - 12_000) / 4_000)

    # Keep distance meaningful while the sample trip is accelerating from rest.
    closing_speed_mps = max(4.0, telemetry.ego.speed_kmh / 3.6)
    distance_m = max(0.5, closing_speed_mps * ttc_s)
    confidence = 0.88 + 0.04 * math.sin(phase_ms / 900.0)
    return C1Prediction(
        predicted_ttc_s=round(ttc_s, 3),
        obstacle_distance_m=round(distance_m, 3),
        confidence=round(confidence, 3),
        is_warning=ttc_s < 2.0,
    )


_DRIVER_PROFILES = {
    "alert": (0.96, 95.0, 5.0, 5.0, True),
    "distracted": (0.90, 32.0, 92.0, 20.0, False),
    "drowsy": (0.87, 43.0, 12.0, 78.0, True),
    "microsleep": (0.94, 5.0, 8.0, 98.0, False),
    "yawning": (0.84, 55.0, 10.0, 68.0, True),
}


def predict_c2(telemetry: TelemetryMessage) -> C2Prediction:
    """Cycle through the five challenge states on a deterministic timeline."""

    phase_ms = telemetry.timestamp_ms % SCENARIO_PERIOD_MS
    if phase_ms < 4_000:
        state = "alert"
    elif phase_ms < 8_000:
        state = "distracted"
    elif phase_ms < 10_000:
        state = "drowsy"
    elif phase_ms < 12_000:
        state = "microsleep"
    elif phase_ms < 14_000:
        state = "yawning"
    else:
        state = "alert"

    confidence, attentive, distraction, fatigue, eyes_on_road = _DRIVER_PROFILES[state]
    return C2Prediction(
        driver_state=state,
        confidence=confidence,
        attentive_probability_pct=attentive,
        distraction_level_pct=distraction,
        fatigue_level_pct=fatigue,
        is_eyes_on_road=eyes_on_road,
        is_warning=state != "alert",
    )


def _collision_risk(ttc_s: float) -> float:
    if ttc_s <= 1.0:
        return 1.0
    if ttc_s <= 2.0:
        return _lerp(1.0, 0.80, ttc_s - 1.0)
    if ttc_s <= 3.0:
        return _lerp(0.80, 0.55, ttc_s - 2.0)
    if ttc_s <= 5.0:
        return _lerp(0.55, 0.20, (ttc_s - 3.0) / 2.0)
    return _lerp(0.20, 0.05, (ttc_s - 5.0) / 3.0)


def fuse_risk(c1: C1Prediction, c2: C2Prediction) -> RiskDecision:
    driver_risk = max(
        100.0 - c2.attentive_probability_pct,
        c2.distraction_level_pct,
        c2.fatigue_level_pct,
    ) / 100.0
    collision_risk = _collision_risk(c1.predicted_ttc_s)
    score = min(1.0, 0.75 * collision_risk + 0.45 * driver_risk)

    reasons = []
    if c1.predicted_ttc_s < 2.5:
        reasons.append("LOW_TTC")
    if c2.driver_state != "alert":
        reasons.append(c2.driver_state.upper())
    if not reasons:
        reasons.append("NORMAL")

    if c1.predicted_ttc_s <= 1.2 or (
        c1.predicted_ttc_s < 2.0 and driver_risk >= 0.70
    ):
        level, action, brake = "CRITICAL", "EMERGENCY_BRAKE_REQUEST", 70.0
    elif c1.predicted_ttc_s < 2.5 or score >= 0.75:
        level, action, brake = "HIGH", "VISUAL_AUDIO_HAPTIC_WARNING", 0.0
    elif score >= 0.45:
        level, action, brake = "CAUTION", "VISUAL_WARNING", 0.0
    else:
        level, action, brake = "SAFE", "MONITOR", 0.0

    return RiskDecision(
        score=round(score, 3),
        level=level,
        action=action,
        brake_request_pct=brake,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class MockCarSkySignalPaths:
    """Standard VSS mapping available in the starter CarSky artifact."""

    speed_kmh: str = "Vehicle.Speed"
    longitudinal_accel_mps2: str = "Vehicle.Acceleration.Longitudinal"
    lateral_accel_mps2: str = "Vehicle.Acceleration.Lateral"
    c1_ttc_ms: str = "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap"
    c1_distance_m: str = "Vehicle.ADAS.ObstacleDetection.Front.Center.Distance"
    c1_warning: str = "Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning"
    c2_attentive_pct: str = "Vehicle.Driver.AttentiveProbability"
    c2_distraction_pct: str = "Vehicle.Driver.DistractionLevel"
    c2_fatigue_pct: str = "Vehicle.Driver.FatigueLevel"
    c2_eyes_on_road: str = "Vehicle.Driver.IsEyesOnRoad"
    c2_warning: str = "Vehicle.ADAS.DMS.IsWarning"

    def required(self) -> tuple[str, ...]:
        paths = tuple(asdict(self).values())
        if any(not path.strip() for path in paths):
            raise ValueError("CarSky mock signal path không được rỗng")
        if len(paths) != len(set(paths)):
            raise ValueError("Các CarSky mock signal path phải khác nhau")
        return paths

    def values(self, message: MockDecisionMessage) -> list[tuple[str, CarSkyScalar]]:
        telemetry = message.telemetry
        return [
            (self.speed_kmh, telemetry["speed_kmh"]),
            (self.longitudinal_accel_mps2, telemetry["longitudinal_accel_mps2"]),
            (self.lateral_accel_mps2, telemetry["lateral_accel_mps2"]),
            (self.c1_ttc_ms, round(message.c1.predicted_ttc_s * 1_000)),
            (self.c1_distance_m, message.c1.obstacle_distance_m),
            (self.c1_warning, message.c1.is_warning),
            (self.c2_attentive_pct, message.c2.attentive_probability_pct),
            (self.c2_distraction_pct, message.c2.distraction_level_pct),
            (self.c2_fatigue_pct, message.c2.fatigue_level_pct),
            (self.c2_eyes_on_road, message.c2.is_eyes_on_road),
            (self.c2_warning, message.c2.is_warning),
        ]


class MockDecisionSink(Protocol):
    def publish(self, message: MockDecisionMessage) -> None: ...

    def close(self) -> None: ...


class MockJsonLinesSink:
    def __init__(self, stream: TextIO, *, flush: bool = True):
        self.stream = stream
        self.flush = flush

    def publish(self, message: MockDecisionMessage) -> None:
        self.stream.write(message.to_json() + "\n")
        if self.flush:
            self.stream.flush()

    def close(self) -> None:
        if self.flush:
            self.stream.flush()


class MockCarSkyRestSink:
    """Publish ego + mock C1/C2 outputs as one bounded REST batch per frame."""

    def __init__(
        self,
        client: CarSkyRestClient,
        paths: MockCarSkySignalPaths | None = None,
        *,
        validate_signals: bool = True,
    ):
        self.client = client
        self.paths = paths if paths is not None else MockCarSkySignalPaths()
        required = set(self.paths.required())
        if validate_signals:
            available = {
                signal.get("path")
                for signal in client.list_signals()
                if isinstance(signal, dict) and isinstance(signal.get("path"), str)
            }
            missing = sorted(required - available)
            if missing:
                raise CarSkyApiError(
                    "Signal path cho mock chưa có trên CarSky node: " + ", ".join(missing)
                )

    def publish(self, message: MockDecisionMessage) -> None:
        self.client.publish_signals(self.paths.values(message))

    def close(self) -> None:
        pass


def publish_mock_replay(
    bundles: Iterable[FrameBundle],
    sink: MockDecisionSink,
) -> PublishStats:
    """Run the replaceable mock inference/fusion stage for every replay frame."""

    import time

    started_ns = time.monotonic_ns()
    count = 0
    first_frame_id = None
    last_frame_id = None
    c3_accumulator = C3Accumulator()
    for bundle in bundles:
        telemetry = TelemetryMessage.from_bundle(bundle)
        message = MockDecisionMessage.from_telemetry(
            telemetry, c3_accumulator=c3_accumulator
        )
        sink.publish(message)
        if first_frame_id is None:
            first_frame_id = message.frame_id
        last_frame_id = message.frame_id
        count += 1

    elapsed_s = max(0, time.monotonic_ns() - started_ns) / 1_000_000_000
    return PublishStats(count, first_frame_id, last_frame_id, elapsed_s)
