from __future__ import annotations

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from safeloop.carsky_decision import DecisionEnvelopeBuilder, DecisionValidity
from safeloop.carsky_kuksa import (
    KuksaGrpcBackend,
    KuksaPublishError,
    KuksaSignalPublisher,
    NO_FINITE_TTC_MS,
    STANDARD_SIGNAL_TYPES,
    standard_signal_values,
)


class FakeBackend:
    def __init__(self, *, fail_writes: int = 0) -> None:
        self.expected = None
        self.batches = []
        self.closed = False
        self.fail_writes = fail_writes

    def connect(self, expected_types):
        self.expected = dict(expected_types)

    def write(self, values):
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise KuksaPublishError("transient broker failure")
        self.batches.append(values)

    def close(self):
        self.closed = True


def _frame(*, frame_id: int = 0, ttc: float = 2.5):
    signals = SimpleNamespace(
        attentive_probability=91.0,
        distraction_level=5.0,
        fatigue_level=7.0,
        is_eyes_on_road=True,
        is_warning=False,
    )
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp=frame_id * 0.05,
        source_bundle=SimpleNamespace(
            ego={
                "speed_kmh": 40.0,
                "longitudinal_accel": -0.2,
                "lateral_accel": 0.1,
            }
        ),
        c1=SimpleNamespace(
            predicted_ttc_s=ttc,
            collision_probability=0.5,
            is_warning=False,
            model_updated=True,
            model_frame_id=frame_id,
        ),
        c2=SimpleNamespace(
            state="alert", confidence=0.91, vss_signals=lambda: signals
        ),
        c3=SimpleNamespace(
            safe_score_estimate=80.0,
            grade="B",
            trip_complete=False,
            formula_version="hackathon-evaluator-v1-no-tailgating",
            tailgating_penalty_omitted=True,
        ),
        drive_quality=SimpleNamespace(
            score_available=True,
            score_pct=90.0,
            grade="A",
            scope="PREFIX",
            window_ready=False,
            formula_version="safeloop-drive-quality-v1",
        ),
        contextual_risk=SimpleNamespace(
            score_pct=20.0,
            level="SAFE",
            action="MONITOR",
            brake_request_pct=0.0,
            reasons=("NORMAL",),
        ),
    )


def _validity():
    return DecisionValidity(
        ego=True,
        front_camera=True,
        driver_camera=True,
        c1=True,
        c2=True,
        c3=True,
        drive_quality=True,
        contextual_risk=True,
    )


def _builder():
    return DecisionEnvelopeBuilder(
        session_id="T01:run",
        source_mode="replay",
        clock_ms=lambda: 10_000,
    )


def test_standard_mirror_has_proved_paths_and_no_distance() -> None:
    frame = _frame()
    envelope = _builder().build(frame, validity=_validity())
    values = standard_signal_values(frame, envelope)
    mapped = {item.path: item.value for item in values}

    assert set(mapped) == set(STANDARD_SIGNAL_TYPES)
    assert mapped["Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap"] == 2500
    assert mapped["Vehicle.Driver.AttentiveProbability"] == 91.0
    assert mapped["Vehicle.Acceleration.Longitudinal"] == -0.2
    assert all("Distance" not in path for path in mapped)


def test_infinite_ttc_refreshes_timegap_with_documented_uint32_sentinel() -> None:
    frame = _frame(ttc=float("inf"))
    envelope = _builder().build(frame, validity=_validity())
    mapped = {item.path: item.value for item in standard_signal_values(frame, envelope)}

    assert (
        mapped["Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap"]
        == NO_FINITE_TTC_MS
    )
    assert mapped["Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning"] is False


def test_publisher_preflights_schema_and_writes_optional_atomic_envelope() -> None:
    backend = FakeBackend()
    frame = _frame()
    envelope = _builder().build(frame, validity=_validity())
    publisher = KuksaSignalPublisher(
        backend,
        decision_envelope_path="Vehicle.ADAS.SafeLoop.Decision.Envelope",
    )

    publisher.connect()
    receipt = publisher.publish(frame, envelope)
    publisher.close()

    assert backend.expected["Vehicle.Speed"] == "FLOAT"
    assert backend.expected["Vehicle.ADAS.SafeLoop.Decision.Envelope"] == "STRING"
    assert receipt.signal_count == 11
    assert backend.batches[0][-1].value.startswith("{")
    assert backend.closed


def test_publisher_allows_late_join_but_rejects_duplicate_and_retired() -> None:
    backend = FakeBackend()
    frame = _frame()
    envelope = _builder().build(frame, validity=_validity())
    publisher = KuksaSignalPublisher(backend)
    publisher.connect()
    publisher.publish(frame, envelope)

    with pytest.raises(KuksaPublishError, match="trùng hoặc lùi"):
        publisher.publish(frame, envelope)

    late_join = replace(envelope, session_id="other", sequence=2)
    assert publisher.publish(frame, late_join).sequence == 2
    with pytest.raises(KuksaPublishError, match="kết thúc"):
        publisher.publish(frame, envelope)


def test_publisher_first_write_failure_does_not_poison_later_sequence() -> None:
    backend = FakeBackend(fail_writes=1)
    first_frame = _frame()
    second_frame = _frame(frame_id=1)
    build = _builder()
    first = build.build(first_frame, validity=_validity())
    second = build.build(second_frame, validity=_validity())
    publisher = KuksaSignalPublisher(backend)
    publisher.connect()

    with pytest.raises(KuksaPublishError, match="transient"):
        publisher.publish(first_frame, first)
    assert publisher.publish(second_frame, second).sequence == 1
    assert len(backend.batches) == 1


def test_publisher_retired_session_lru_is_bounded() -> None:
    backend = FakeBackend()
    frame = _frame()
    envelope = _builder().build(frame, validity=_validity())
    publisher = KuksaSignalPublisher(backend)
    publisher.connect()

    for index in range(70):
        packet = replace(envelope, session_id=f"run-{index}", sequence=5)
        assert publisher.publish(frame, packet).sequence == 5

    with pytest.raises(KuksaPublishError, match="kết thúc"):
        publisher.publish(
            frame, replace(envelope, session_id="run-68", sequence=0)
        )
    assert publisher.publish(
        frame, replace(envelope, session_id="run-0", sequence=9)
    ).sequence == 9


def test_standard_mirror_rejects_missing_or_nonfinite_ego() -> None:
    frame = _frame()
    envelope = _builder().build(frame, validity=_validity())
    frame.source_bundle.ego["speed_kmh"] = float("nan")
    with pytest.raises(KuksaPublishError, match="hữu hạn"):
        standard_signal_values(frame, envelope)


def test_grpc_backend_uses_bounded_metadata_as_its_startup_probe(
    monkeypatch,
) -> None:
    calls = {}

    class FakeClient:
        def __init__(self, host, port, **kwargs):
            calls["init"] = (host, port, kwargs)

        def connect(self):
            calls["connected"] = True

        def get_metadata(self, paths, **kwargs):
            calls["metadata"] = (tuple(paths), kwargs)
            return {
                path: SimpleNamespace(
                    data_type=SimpleNamespace(name=data_type)
                )
                for path, data_type in STANDARD_SIGNAL_TYPES.items()
            }

        def disconnect(self):
            calls["disconnected"] = True

    fake_grpc = SimpleNamespace(VSSClient=FakeClient)
    monkeypatch.setitem(
        sys.modules, "kuksa_client", SimpleNamespace(grpc=fake_grpc)
    )
    backend = KuksaGrpcBackend("127.0.0.10", 55_555, timeout_s=1.25)

    backend.connect(STANDARD_SIGNAL_TYPES)
    backend.close()

    assert calls["init"] == (
        "127.0.0.10",
        55_555,
        {"ensure_startup_connection": False},
    )
    assert calls["metadata"][1] == {"timeout": 1.25}
    assert calls["connected"] is True
    assert calls["disconnected"] is True
