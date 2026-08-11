from __future__ import annotations

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from safeloop.carsky_decision import (
    DecisionContractError,
    DecisionEnvelope,
    DecisionEnvelopeBuilder,
    DecisionValidity,
    SCHEMA_VERSION,
)
from safeloop.carsky_hmi import (
    DEFAULT_MAX_DATAGRAM_BYTES,
    HmiDecisionGuard,
    HmiDecisionRejected,
    HmiTransportError,
    MAX_SAFE_UDP_PAYLOAD_BYTES,
    MAX_UDP_PAYLOAD_BYTES,
    UdpDecisionPublisher,
)


class FakeSocket:
    def __init__(
        self, *, short_write: bool = False, fail_writes: int = 0
    ) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.short_write = short_write
        self.fail_writes = fail_writes
        self.closed = False

    def sendto(self, data: bytes, address: tuple[str, int]) -> int:
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise OSError("transient send failure")
        self.sent.append((data, address))
        return len(data) - 1 if self.short_write else len(data)

    def close(self) -> None:
        self.closed = True


def combined_frame(
    *,
    frame_id: int = 4,
    timestamp: float = 0.2,
    ttc: float = 2.25,
    action: str = "VISUAL_WARNING",
    level: str = "CAUTION",
    brake_request_pct: float = 0.0,
):
    signals = SimpleNamespace(
        attentive_probability=91.0,
        distraction_level=6.0,
        fatigue_level=8.0,
        is_eyes_on_road=True,
        is_warning=False,
    )
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp=timestamp,
        c1=SimpleNamespace(
            predicted_ttc_s=ttc,
            collision_probability=0.42,
            is_warning=False,
            model_updated=True,
            model_frame_id=frame_id,
        ),
        c2=SimpleNamespace(
            state="alert",
            confidence=0.91,
            vss_signals=lambda: signals,
        ),
        c3=SimpleNamespace(
            safe_score_estimate=72.0,
            grade="C",
            trip_complete=False,
            formula_version="hackathon-evaluator-v1-no-tailgating",
            tailgating_penalty_omitted=True,
        ),
        drive_quality=SimpleNamespace(
            score_available=True,
            score_pct=88.0,
            grade="B",
            scope="PREFIX",
            window_ready=False,
            formula_version="safeloop-drive-quality-v1",
        ),
        contextual_risk=SimpleNamespace(
            score_pct=48.0,
            level=level,
            action=action,
            brake_request_pct=brake_request_pct,
            reasons=("LOW_TTC",),
        ),
    )


def all_valid(**overrides: bool) -> DecisionValidity:
    values = {
        "ego": True,
        "front_camera": True,
        "driver_camera": True,
        "c1": True,
        "c2": True,
        "c3": True,
        "drive_quality": True,
        "contextual_risk": True,
    }
    values.update(overrides)
    return DecisionValidity(**values)


def builder(*, session_id: str = "trip:T01:run-1") -> DecisionEnvelopeBuilder:
    return DecisionEnvelopeBuilder(
        session_id=session_id,
        source_mode="replay",
        source_fps=20.0,
        ttl_ms=200,
        model_versions={"c1": "student-ttc", "c2": "phase2-v13"},
        clock_ms=lambda: 10_000,
    )


def test_envelope_is_strict_atomic_json_without_fake_distance() -> None:
    envelope = builder().build(combined_frame(), validity=all_valid())
    payload = envelope.to_json_bytes()
    decoded = json.loads(payload)

    assert decoded["schema_version"] == SCHEMA_VERSION
    assert decoded["source_mode"] == "replay"
    assert decoded["session_id"] == "trip:T01:run-1"
    assert decoded["sequence"] == 0
    assert decoded["frame_id"] == 4
    assert decoded["source_timestamp_ms"] == 200
    assert decoded["decision_timestamp_ms"] == 10_000
    assert decoded["expires_at_ms"] == 10_200
    assert decoded["c1"]["ttc_ms"] == 2_250
    assert decoded["c1"]["collision_probability_pct"] == 42.0
    assert decoded["contextual_risk"]["actuation_authorized"] is False
    assert decoded["health"]["mode"] == "NOMINAL"
    assert "distance" not in payload.decode("utf-8").lower()
    assert DecisionEnvelope.from_json_bytes(payload) == envelope


def test_infinite_ttc_is_null_not_nonstandard_json() -> None:
    envelope = builder().build(
        combined_frame(ttc=math.inf), validity=all_valid()
    )
    payload = envelope.to_json_bytes()
    c1 = json.loads(payload)["c1"]

    assert c1["ttc_ms"] is None
    assert c1["ttc_valid"] is False
    assert b"Infinity" not in payload
    assert b"NaN" not in payload

    with pytest.raises(DecisionContractError, match=r"non-negative or \+inf"):
        builder().build(
            combined_frame(ttc=float("nan")),
            validity=all_valid(),
        )


def test_invalid_context_fails_safe_and_reports_degraded_health() -> None:
    validity = all_valid(
        driver_camera=False,
        c2=False,
        contextual_risk=False,
    )
    envelope = builder().build(
        combined_frame(
            action="EMERGENCY_BRAKE_REQUEST",
            level="CRITICAL",
            brake_request_pct=70.0,
        ),
        validity=validity,
    )

    assert envelope.contextual_risk == {
        "score_pct": None,
        "level": "UNAVAILABLE",
        "action": "MONITOR",
        "brake_request_pct": 0.0,
        "reasons": ["STALE_OR_INVALID_INPUT"],
        "actuation_authorized": False,
    }
    assert envelope.health["mode"] == "DEGRADED"
    assert envelope.health["decision_valid"] is False
    assert set(envelope.health["stale_or_invalid_components"]) == {
        "driver_camera",
        "c2",
        "contextual_risk",
    }


def test_emergency_recommendation_requires_fresh_collision_inputs() -> None:
    with pytest.raises(
        DecisionContractError, match="emergency recommendation requires"
    ):
        builder().build(
            combined_frame(
                action="EMERGENCY_BRAKE_REQUEST",
                level="CRITICAL",
                brake_request_pct=70.0,
            ),
            validity=all_valid(front_camera=False),
        )


def test_builder_normalizes_legacy_emergency_risk_to_warning_only() -> None:
    envelope = builder().build(
        combined_frame(
            action="EMERGENCY_BRAKE_REQUEST",
            level="CRITICAL",
            brake_request_pct=70.0,
        ),
        validity=all_valid(),
    )

    assert envelope.contextual_risk["level"] == "CRITICAL"
    assert envelope.contextual_risk["action"] == "VISUAL_AUDIO_HAPTIC_WARNING"
    assert envelope.contextual_risk["brake_request_pct"] == 0.0
    assert b"EMERGENCY_BRAKE_REQUEST" not in envelope.to_json_bytes()


@pytest.mark.parametrize(
    "legacy_action",
    ["EMERGENCY_BRAKE_REQUEST", "VISUAL_WARNING"],
)
def test_parser_accepts_legacy_brake_wire_but_returns_warning_only(
    legacy_action: str,
) -> None:
    payload = builder().build(combined_frame(), validity=all_valid()).to_dict()
    payload["contextual_risk"].update(
        {
            "level": "CRITICAL",
            "action": legacy_action,
            "brake_request_pct": 70.0,
        }
    )

    envelope = DecisionEnvelope.from_json_bytes(json.dumps(payload).encode("utf-8"))

    assert (
        envelope.contextual_risk["action"]
        == "VISUAL_AUDIO_HAPTIC_WARNING"
    )
    assert envelope.contextual_risk["brake_request_pct"] == 0.0
    assert b"EMERGENCY_BRAKE_REQUEST" not in envelope.to_json_bytes()


def test_builder_sequence_is_monotonic_and_c1_forward_fill_has_age() -> None:
    build = builder()
    first = build.build(combined_frame(frame_id=4), validity=all_valid())
    held = combined_frame(frame_id=5, timestamp=0.25)
    held.c1.model_frame_id = 4
    held.c1.model_updated = False
    second = build.build(held, validity=all_valid())

    assert (first.sequence, second.sequence) == (0, 1)
    assert second.c1["age_ms"] == 50


def test_builder_rejects_non_monotonic_source_without_consuming_sequence() -> None:
    build = builder()
    with pytest.raises(DecisionContractError, match="invalid contextual_risk.action"):
        build.build(
            combined_frame(action="NOT_AN_ACTION"),
            validity=all_valid(),
        )

    first = build.build(combined_frame(), validity=all_valid())
    assert first.sequence == 0
    with pytest.raises(DecisionContractError, match="frame_id must increase"):
        build.build(combined_frame(), validity=all_valid())
    with pytest.raises(DecisionContractError, match="timestamp cannot move backwards"):
        build.build(
            combined_frame(frame_id=5, timestamp=0.1),
            validity=all_valid(),
        )
    with pytest.raises(
        DecisionContractError, match="decision timestamp cannot move backwards"
    ):
        build.build(
            combined_frame(frame_id=5, timestamp=0.25),
            validity=all_valid(),
            decision_timestamp_ms=9_999,
        )


def test_contract_rejects_nonfinite_nested_data_and_fake_distance() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    payload = valid.to_dict()
    payload["health"]["component_age_ms"] = {
        **payload["health"]["component_age_ms"],
        "c1": float("nan"),
    }
    with pytest.raises(DecisionContractError, match="non-negative integers"):
        DecisionEnvelope(**payload)

    payload = valid.to_dict()
    payload["c1"] = {**payload["c1"], "estimated_distance_m": 10.0}
    with pytest.raises(DecisionContractError, match="no calibrated distance"):
        DecisionEnvelope(**payload)


def test_contract_rejects_bad_expiry_and_unknown_top_level_key() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    with pytest.raises(DecisionContractError, match="expires_at_ms"):
        replace(valid, expires_at_ms=valid.expires_at_ms + 1)

    decoded = valid.to_dict()
    decoded["unknown"] = True
    with pytest.raises(DecisionContractError, match="keys must be exactly"):
        DecisionEnvelope.from_json_bytes(json.dumps(decoded).encode())


def test_wire_encoder_revalidates_mutated_nested_mappings() -> None:
    envelope = builder().build(combined_frame(), validity=all_valid())
    envelope.c1["estimated_distance_m"] = 10.0
    with pytest.raises(DecisionContractError, match="no calibrated distance"):
        envelope.to_json_bytes()


def test_udp_publisher_sends_one_envelope_per_datagram() -> None:
    fake = FakeSocket()
    publisher = UdpDecisionPublisher(
        "10.99.0.2",
        48100,
        socket_factory=lambda: fake,
        clock_ms=lambda: 10_000,
    )
    build = builder()
    first = build.build(combined_frame(), validity=all_valid())
    second = build.build(
        combined_frame(frame_id=5, timestamp=0.25), validity=all_valid()
    )

    receipt = publisher.publish(first)
    publisher.publish(second)
    publisher.close()

    assert receipt.sequence == 0
    assert receipt.destination == ("10.99.0.2", 48100)
    assert len(fake.sent) == 2
    assert fake.sent[0][1] == ("10.99.0.2", 48100)
    assert DecisionEnvelope.from_json_bytes(fake.sent[1][0]).sequence == 1
    assert fake.closed


def test_udp_first_send_failure_does_not_poison_later_sequence() -> None:
    fake = FakeSocket(fail_writes=1)
    publisher = UdpDecisionPublisher(
        "10.99.0.2",
        48100,
        socket_factory=lambda: fake,
        clock_ms=lambda: 10_000,
    )
    build = builder()
    first = build.build(combined_frame(), validity=all_valid())
    second = build.build(
        combined_frame(frame_id=5, timestamp=0.25), validity=all_valid()
    )

    with pytest.raises(HmiTransportError, match="cannot send"):
        publisher.publish(first)
    receipt = publisher.publish(second)

    assert receipt.sequence == 1
    assert len(fake.sent) == 1
    assert DecisionEnvelope.from_json_bytes(fake.sent[0][0]).sequence == 1


def test_default_udp_boundary_rejects_fragmenting_packet_before_send() -> None:
    assert DEFAULT_MAX_DATAGRAM_BYTES == MAX_SAFE_UDP_PAYLOAD_BYTES == 1_472
    assert MAX_SAFE_UDP_PAYLOAD_BYTES < MAX_UDP_PAYLOAD_BYTES

    valid = builder().build(combined_frame(), validity=all_valid())
    oversized = replace(
        valid,
        health={
            **valid.health,
            "model_versions": {"c1": "x" * MAX_SAFE_UDP_PAYLOAD_BYTES},
        },
    )
    payload = oversized.to_json_bytes()
    assert len(payload) > MAX_SAFE_UDP_PAYLOAD_BYTES

    fake = FakeSocket()
    publisher = UdpDecisionPublisher(
        "127.0.0.1",
        48100,
        socket_factory=lambda: fake,
        clock_ms=lambda: 10_000,
    )
    with pytest.raises(HmiTransportError, match=r"limit is 1472"):
        publisher.publish(oversized)
    assert fake.sent == []

    guard = HmiDecisionGuard(monotonic_ms=lambda: 100)
    with pytest.raises(HmiDecisionRejected, match=r"exceeds 1472 bytes"):
        guard.accept(payload)


def test_udp_publisher_rejects_old_partial_and_oversized_snapshots() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    fake = FakeSocket()
    publisher = UdpDecisionPublisher(
        "127.0.0.1",
        48100,
        socket_factory=lambda: fake,
        clock_ms=lambda: 10_000,
    )
    publisher.publish(valid)
    with pytest.raises(HmiTransportError, match="duplicate/out-of-order"):
        publisher.publish(valid)

    short = FakeSocket(short_write=True)
    with pytest.raises(HmiTransportError, match="partial UDP"):
        UdpDecisionPublisher(
            "127.0.0.1",
            48100,
            socket_factory=lambda: short,
            clock_ms=lambda: 10_000,
        ).publish(valid)

    tiny = UdpDecisionPublisher(
        "127.0.0.1",
        48100,
        max_datagram_bytes=10,
        socket_factory=FakeSocket,
        clock_ms=lambda: 10_000,
    )
    with pytest.raises(HmiTransportError, match="limit is 10"):
        tiny.publish(valid)


def test_udp_publisher_rejects_expired_future_and_retired_sessions() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    fake = FakeSocket()
    expired = UdpDecisionPublisher(
        "127.0.0.1",
        48100,
        socket_factory=lambda: fake,
        clock_ms=lambda: 10_200,
    )
    with pytest.raises(HmiTransportError, match="expired"):
        expired.publish(valid)
    assert fake.sent == []

    future = UdpDecisionPublisher(
        "127.0.0.1",
        48100,
        max_future_skew_ms=50,
        socket_factory=lambda: fake,
        clock_ms=lambda: 9_900,
    )
    with pytest.raises(HmiTransportError, match="future"):
        future.publish(valid)

    publisher = UdpDecisionPublisher(
        "127.0.0.1",
        48100,
        socket_factory=lambda: fake,
        clock_ms=lambda: 10_000,
    )
    publisher.publish(valid)
    publisher.publish(replace(valid, session_id="session-2", sequence=0))
    with pytest.raises(HmiTransportError, match="retired"):
        publisher.publish(replace(valid, sequence=0))


def test_hmi_guard_accepts_only_fresh_ordered_snapshots() -> None:
    build = builder()
    first = build.build(combined_frame(), validity=all_valid())
    second = build.build(
        combined_frame(frame_id=5, timestamp=0.25), validity=all_valid()
    )
    guard = HmiDecisionGuard(monotonic_ms=lambda: 500)

    assert guard.accept(first.to_json_bytes()).sequence == 0
    assert guard.accept(second.to_json_bytes()).sequence == 1
    assert guard.current(now_monotonic_ms=699) == second
    assert guard.current(now_monotonic_ms=700) is None

    with pytest.raises(HmiDecisionRejected, match="duplicate/out-of-order"):
        guard.accept(second.to_json_bytes(), received_monotonic_ms=500)

    # Expiry is a receiver-local TTL. A later ordered packet gets a fresh
    # monotonic deadline even if its producer wall timestamp is unchanged.
    accepted = guard.accept(
        replace(second, sequence=2).to_json_bytes(),
        received_monotonic_ms=700,
    )
    assert accepted.sequence == 2
    assert guard.current(now_monotonic_ms=899) == accepted
    assert guard.current(now_monotonic_ms=900) is None


def test_hmi_guard_ignores_wall_clock_skew_and_late_joins_new_session() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    skewed = replace(
        valid,
        decision_timestamp_ms=9_000_000_000_000,
        expires_at_ms=9_000_000_000_200,
    )
    guard = HmiDecisionGuard(monotonic_ms=lambda: 100)

    # Producer and receiver wall clocks may differ arbitrarily. The internal
    # decision+TTL expiry arithmetic is still checked by envelope parsing.
    assert guard.accept(skewed.to_json_bytes()).sequence == 0
    assert guard.current(now_monotonic_ms=299) is not None
    assert guard.current(now_monotonic_ms=300) is None
    with pytest.raises(HmiDecisionRejected, match="invalid UTF-8"):
        guard.accept(b"not-json", received_monotonic_ms=100)

    other = replace(valid, session_id="new-session", sequence=3)
    assert guard.accept(other.to_json_bytes(), received_monotonic_ms=301) == other


def test_hmi_guard_late_join_baseline_and_subsequent_gaps() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    guard = HmiDecisionGuard()

    first_seen = replace(valid, sequence=5)
    assert guard.accept(first_seen.to_json_bytes(), received_monotonic_ms=100) == first_seen
    assert guard.dropped_sequences == 0

    later = replace(valid, sequence=8)
    assert guard.accept(later.to_json_bytes(), received_monotonic_ms=110) == later
    assert guard.dropped_sequences == 2

    late_new = replace(valid, session_id="late-new", sequence=12)
    assert guard.accept(late_new.to_json_bytes(), received_monotonic_ms=120) == late_new
    assert guard.dropped_sequences == 2


def test_hmi_guard_does_not_allow_a_retired_session_to_return() -> None:
    first = builder(session_id="session-1").build(
        combined_frame(), validity=all_valid()
    )
    second = replace(first, session_id="session-2", sequence=0)
    guard = HmiDecisionGuard(monotonic_ms=lambda: 100)

    guard.accept(first.to_json_bytes())
    guard.accept(second.to_json_bytes())
    with pytest.raises(HmiDecisionRejected, match="retired"):
        guard.accept(first.to_json_bytes())


def test_hmi_guard_allows_retired_sequence_zero_only_after_stale_dwell() -> None:
    first = builder(session_id="session-1").build(
        combined_frame(), validity=all_valid()
    )
    second = replace(first, session_id="session-2", sequence=0)
    guard = HmiDecisionGuard()

    guard.accept(first.to_json_bytes(), received_monotonic_ms=100)
    guard.accept(second.to_json_bytes(), received_monotonic_ms=110)
    with pytest.raises(HmiDecisionRejected, match="retired"):
        guard.accept(first.to_json_bytes(), received_monotonic_ms=309)

    assert guard.accept(first.to_json_bytes(), received_monotonic_ms=310) == first
    assert guard.session_restarts == 2


def test_hmi_guard_allows_same_session_sequence_zero_restart_only_when_stale() -> None:
    first = builder(session_id="same-process-id").build(
        combined_frame(), validity=all_valid()
    )
    guard = HmiDecisionGuard()

    guard.accept(replace(first, sequence=7).to_json_bytes(), received_monotonic_ms=100)
    with pytest.raises(HmiDecisionRejected, match="duplicate"):
        guard.accept(first.to_json_bytes(), received_monotonic_ms=299)
    assert guard.accept(first.to_json_bytes(), received_monotonic_ms=300) == first
    assert guard.session_restarts == 1


def test_hmi_guard_retired_session_lru_is_bounded_and_keeps_streaming() -> None:
    valid = builder().build(combined_frame(), validity=all_valid())
    guard = HmiDecisionGuard()

    for index in range(70):
        packet = replace(valid, session_id=f"run-{index}", sequence=5)
        assert guard.accept(
            packet.to_json_bytes(), received_monotonic_ms=100 + index
        ) == packet

    assert guard.retired_session_count == 64
    with pytest.raises(HmiDecisionRejected, match="retired"):
        guard.accept(
            replace(valid, session_id="run-68", sequence=0).to_json_bytes(),
            received_monotonic_ms=170,
        )

    evicted = replace(valid, session_id="run-0", sequence=9)
    assert guard.accept(evicted.to_json_bytes(), received_monotonic_ms=171) == evicted
    assert guard.retired_session_count == 64


def test_invalid_configuration_is_rejected_before_network_use() -> None:
    with pytest.raises(DecisionContractError, match="source_mode"):
        DecisionEnvelopeBuilder(session_id="x", source_mode="mock")
    with pytest.raises(ValueError, match="host"):
        UdpDecisionPublisher("", 48100)
    with pytest.raises(ValueError, match="port"):
        UdpDecisionPublisher("127.0.0.1", 0)
    with pytest.raises(DecisionContractError, match="validity"):
        DecisionValidity(
            ego=1,  # type: ignore[arg-type]
            front_camera=True,
            driver_camera=True,
            c1=True,
            c2=True,
            c3=True,
            drive_quality=True,
            contextual_risk=True,
        )
