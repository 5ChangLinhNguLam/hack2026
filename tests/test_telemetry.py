"""SafeLoop telemetry contract, replay ordering, and CarSky REST sink."""

from __future__ import annotations

import io
import json

import pytest

from conftest import make_redacted_frame, make_trip
from safeloop.telemetry import (
    CarSkyApiError,
    CarSkyRestClient,
    CarSkyRestSink,
    CarSkySignalPaths,
    JsonLinesSink,
    SCHEMA_VERSION,
    TelemetryContractError,
    TelemetryMessage,
    publish_replay,
)
from tripkit import TripLoader, TripReplayer


def _clock(values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_message_contract_contains_only_allowed_telemetry(t01_loader):
    # Practice frames contain GT, events, world position and driver state.
    # None of them may leak into the operational telemetry envelope.
    message = TelemetryMessage.from_bundle(t01_loader.frame(320), clock_ns=lambda: 123)
    payload = message.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["message_id"] == "T01-Sample:320"
    assert payload["frame_id"] == 320
    assert payload["timestamp_ms"] == 16_000
    assert payload["emitted_monotonic_ns"] == 123
    assert set(payload["ego"]) == {
        "speed_kmh",
        "longitudinal_accel_mps2",
        "lateral_accel_mps2",
    }
    serialized = message.to_json()
    for forbidden in ("driver", "min_ttc", "events_active", "location", "geolocation"):
        assert forbidden not in serialized


def test_redacted_trip_emits_ordered_jsonl(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    stream = io.StringIO()
    # publish_replay clock: start, one per 7 message, end
    clock_ns = _clock(range(9))
    stats = publish_replay(
        TripReplayer(loader), JsonLinesSink(stream), clock_ns=clock_ns
    )
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert stats.count == 7
    assert (stats.first_frame_id, stats.last_frame_id) == (0, 6)
    assert [row["frame_id"] for row in rows] == list(range(7))
    assert [row["timestamp_ms"] for row in rows] == [i * 50 for i in range(7)]
    assert rows[2]["ego"]["speed_kmh"] == 32.0


def test_contract_rejects_missing_signal(redacted_trip_dir):
    loader = TripLoader(redacted_trip_dir)
    bundle = loader.frame(0)
    del bundle.ego["lateral_accel"]
    with pytest.raises(TelemetryContractError, match="ego.lateral_accel"):
        TelemetryMessage.from_bundle(bundle)


def test_publish_rejects_decreasing_timestamp(tmp_path):
    frames = [make_redacted_frame(0), make_redacted_frame(1)]
    frames[1]["timestamp"] = -0.1
    loader = TripLoader(make_trip(tmp_path, "T95d", frames))
    with pytest.raises(TelemetryContractError, match="timestamp phải >= 0"):
        publish_replay(TripReplayer(loader), JsonLinesSink(io.StringIO()))


class _FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_carsky_rest_sink_validates_then_posts_one_sensor_batch(t01_loader):
    requests = []
    paths = [
        "Vehicle.Speed",
        "Vehicle.Acceleration.Longitudinal",
        "Vehicle.Acceleration.Lateral",
    ]

    def fake_urlopen(http_request, timeout):
        requests.append((http_request, timeout))
        if http_request.method == "GET":
            return _FakeResponse({"nodeKey": "ego", "signals": [{"path": p} for p in paths]})
        return _FakeResponse({"ok": True, "sent": 3})

    client = CarSkyRestClient(
        "https://carsky.example/",
        "secret-test-key",
        "room/01",
        "ego signals",
        timeout_s=4.5,
        urlopen=fake_urlopen,
    )
    sink = CarSkyRestSink(client)
    message = TelemetryMessage.from_bundle(t01_loader.frame(0), clock_ns=lambda: 99)

    sink.publish(message)

    assert len(requests) == 2
    get_request, get_timeout = requests[0]
    assert get_request.full_url.endswith("/signals/room%2F01/ego%20signals")
    assert get_timeout == 4.5
    post_request, _ = requests[1]
    assert post_request.method == "POST"
    assert post_request.full_url.endswith("/signals/room%2F01/ego%20signals/actuate")
    assert post_request.get_header("X-api-key") == "secret-test-key"
    payload = json.loads(post_request.data)
    assert payload == {
        "signals": [
            {"path": "Vehicle.Speed", "value": message.ego.speed_kmh},
            {
                "path": "Vehicle.Acceleration.Longitudinal",
                "value": message.ego.longitudinal_accel_mps2,
            },
            {
                "path": "Vehicle.Acceleration.Lateral",
                "value": message.ego.lateral_accel_mps2,
            },
        ]
    }
    assert all("actuate" not in signal for signal in payload["signals"])


def test_carsky_rest_sink_can_publish_optional_trace_signals(t01_loader):
    published = []

    class Client:
        def list_signals(self):
            return []

        def publish_signals(self, signals):
            published.extend(signals)

    paths = CarSkySignalPaths(
        trip_id="SafeLoop.TripId",
        frame_id="SafeLoop.FrameId",
        timestamp_ms="SafeLoop.TimestampMs",
    )
    sink = CarSkyRestSink(Client(), paths, validate_signals=False)
    message = TelemetryMessage.from_bundle(t01_loader.frame(2), clock_ns=lambda: 99)

    sink.publish(message)

    assert published[-3:] == [
        ("SafeLoop.TripId", "T01-Sample"),
        ("SafeLoop.FrameId", 2),
        ("SafeLoop.TimestampMs", 100),
    ]


def test_carsky_rest_sink_fails_fast_when_signal_path_is_missing():
    class Client:
        def list_signals(self):
            return [{"path": "Vehicle.Speed"}]

    with pytest.raises(CarSkyApiError, match="Acceleration.Lateral"):
        CarSkyRestSink(Client())


def test_carsky_rest_client_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="http"):
        CarSkyRestClient("carsky.io", "key", "room", "node")
    with pytest.raises(ValueError, match="A8_API_KEY"):
        CarSkyRestClient("https://carsky.io", "", "room", "node")


def test_carsky_rest_client_checks_confirmed_sent_count():
    def fake_urlopen(_request, timeout):
        assert timeout == 10.0
        return _FakeResponse({"ok": True, "sent": 2})

    client = CarSkyRestClient(
        "https://carsky.io", "key", "room", "node", urlopen=fake_urlopen
    )
    with pytest.raises(CarSkyApiError, match="sent=2"):
        client.publish_signals([("a", 1), ("b", 2), ("c", 3)])
