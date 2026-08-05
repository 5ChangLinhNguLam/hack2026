"""Deterministic SafeLoop mock and its CarSky VSS mapping."""

from __future__ import annotations

import io
import json

from safeloop.mock_pipeline import (
    C3Accumulator,
    MockCarSkyRestSink,
    MockDecisionMessage,
    MockJsonLinesSink,
    fuse_risk,
    publish_mock_replay,
)
from safeloop.telemetry import TelemetryMessage
from tripkit import TripLoader, TripReplayer


def _at_ms(bundle, timestamp_ms):
    bundle.timestamp = timestamp_ms / 1000
    return MockDecisionMessage.from_telemetry(
        TelemetryMessage.from_bundle(bundle, clock_ns=lambda: 123)
    )


def test_mock_scenario_covers_all_c2_states_and_critical_action(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    bundle = loader.frame(0)

    states = {_at_ms(bundle, t).c2.driver_state for t in (0, 4_000, 8_000, 10_000, 12_000)}
    critical = _at_ms(bundle, 11_500)

    assert states == {"alert", "distracted", "drowsy", "microsleep", "yawning"}
    assert critical.mock is True
    assert critical.c1.predicted_ttc_s < 1.0
    assert critical.risk.level == "CRITICAL"
    assert critical.risk.action == "EMERGENCY_BRAKE_REQUEST"


def test_mock_does_not_copy_ground_truth_into_output(t01_loader):
    message = MockDecisionMessage.from_telemetry(
        TelemetryMessage.from_bundle(t01_loader.frame(320), clock_ns=lambda: 123)
    )
    payload = message.to_json()

    assert '"mock":true' in payload
    for forbidden in ("min_ttc", "events_active", "behavior_flags", "ground_truth"):
        assert forbidden not in payload


def test_mock_jsonl_replay_is_ordered(redacted_trip_dir):
    stream = io.StringIO()
    loader = TripLoader(redacted_trip_dir)
    stats = publish_mock_replay(TripReplayer(loader), MockJsonLinesSink(stream))
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert stats.count == 7
    assert [row["frame_id"] for row in rows] == list(range(7))
    assert all(row["mock"] is True for row in rows)
    assert all("c3" in row for row in rows)


def test_c3_score_accumulates_penalty_and_resets_each_mock_trip(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    bundle = loader.frame(0)
    accumulator = C3Accumulator()

    scores = []
    for timestamp_ms in (8_000, 9_000, 10_000, 11_000, 12_000):
        bundle.timestamp = timestamp_ms / 1_000
        message = MockDecisionMessage.from_telemetry(
            TelemetryMessage.from_bundle(bundle), c3_accumulator=accumulator
        )
        scores.append(message.c3.score)

    bundle.timestamp = 16.0
    reset = MockDecisionMessage.from_telemetry(
        TelemetryMessage.from_bundle(bundle), c3_accumulator=accumulator
    )

    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < scores[0]
    assert reset.c3.score == 100.0
    assert reset.c3.grade == "A"


def test_mock_carsky_sink_validates_and_publishes_standard_vss(redacted_trip_dir):
    published = []

    class Client:
        def list_signals(self):
            from safeloop.mock_pipeline import MockCarSkySignalPaths

            return [{"path": path} for path in MockCarSkySignalPaths().required()]

        def publish_signals(self, signals):
            published.extend(signals)

    loader = TripLoader(redacted_trip_dir)
    decision = _at_ms(loader.frame(0), 11_500)
    sink = MockCarSkyRestSink(Client())
    sink.publish(decision)
    values = dict(published)

    assert len(values) == 11
    assert values["Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap"] == 975
    assert values["Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning"] is True
    assert values["Vehicle.Driver.FatigueLevel"] == 98.0
    assert values["Vehicle.ADAS.DMS.IsWarning"] is True
