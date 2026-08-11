from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from safeloop.carsky_decision_v2 import (
    MAX_MQTT_PAYLOAD_BYTES,
    MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS,
    SCHEMA,
    DecisionEnvelopeV2,
    DecisionV2ContractError,
)


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "carsky_decision_v2_golden.json"


@pytest.fixture()
def golden_payload() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def create_kwargs(payload: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": payload["session_id"],
        "generation": payload["generation"],
        "sequence": payload["sequence"],
        "source_sequence": payload["source_sequence"],
        "video_source": payload["video_source"],
        "capture_timestamp_ms": payload["capture_timestamp_ms"],
        "source_media_timestamp_ms": payload.get("source_media_timestamp_ms"),
        "server_receive_timestamp_ms": payload["server_receive_timestamp_ms"],
        "decision_timestamp_ms": payload["decision_timestamp_ms"],
        "clock_status": payload["clock"]["status"],  # type: ignore[index]
        "clock_offset_uncertainty_ms": payload["clock"][  # type: ignore[index]
            "offset_uncertainty_ms"
        ],
        "validity": deepcopy(payload["validity"]),
        "c1": deepcopy(payload["c1"]),
        "c2": deepcopy(payload["c2"]),
        "c3": deepcopy(payload["c3"]),
        "drive_quality": deepcopy(payload["drive_quality"]),
        "contextual_risk": deepcopy(payload["contextual_risk"]),
        "models": deepcopy(payload["models"]),
        "ttl_ms": payload["ttl_ms"],
    }


def test_golden_payload_has_exact_schema_and_canonical_round_trip(
    golden_payload: dict[str, object],
) -> None:
    envelope = DecisionEnvelopeV2.from_dict(golden_payload)

    assert envelope.schema == SCHEMA
    assert "schema" in envelope.to_dict()
    assert "schema_version" not in envelope.to_dict()
    assert envelope.to_dict() == golden_payload

    canonical = json.dumps(
        golden_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    assert envelope.to_json_bytes() == canonical
    assert DecisionEnvelopeV2.from_json_bytes(canonical).to_dict() == golden_payload


def test_optional_source_media_timestamp_can_be_absent(
    golden_payload: dict[str, object],
) -> None:
    golden_payload.pop("source_media_timestamp_ms")
    envelope = DecisionEnvelopeV2.from_dict(golden_payload)

    assert envelope.source_media_timestamp_ms is None
    assert "source_media_timestamp_ms" not in envelope.to_dict()
    assert DecisionEnvelopeV2.from_json_bytes(envelope.to_json_bytes()) == envelope


def test_optional_source_media_timestamp_rejects_explicit_null(
    golden_payload: dict[str, object],
) -> None:
    golden_payload["source_media_timestamp_ms"] = None
    with pytest.raises(DecisionV2ContractError, match="omitted rather than null"):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_root_rejects_alias_missing_and_extra_keys(
    golden_payload: dict[str, object],
) -> None:
    alias = deepcopy(golden_payload)
    alias["schema_version"] = alias.pop("schema")
    with pytest.raises(DecisionV2ContractError):
        DecisionEnvelopeV2.from_dict(alias)

    missing = deepcopy(golden_payload)
    missing.pop("generation")
    with pytest.raises(DecisionV2ContractError):
        DecisionEnvelopeV2.from_dict(missing)

    extra = deepcopy(golden_payload)
    extra["source_mode"] = "live"
    with pytest.raises(DecisionV2ContractError):
        DecisionEnvelopeV2.from_dict(extra)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("schema",), "safeloop.decision.v1"),
        (("generation",), True),
        (("video_source",), "RECORDED_SOURCE"),
        (("inference_mode",), "MODEL_OUTPUT_REPLAY"),
        (("source_age_ms",), 49),
        (("expires_at_ms",), 1700000000300),
        (("clock", "offset_uncertainty_ms"), -1),
        (("models", "c1", "digest_sha256"), "a" * 12),
    ],
)
def test_malformed_fields_are_rejected(
    golden_payload: dict[str, object], path: tuple[str, ...], bad_value: object
) -> None:
    cursor: dict[str, object] = golden_payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[path[-1]] = bad_value

    with pytest.raises(DecisionV2ContractError):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_cross_field_timestamp_order_is_strict(
    golden_payload: dict[str, object],
) -> None:
    golden_payload["server_receive_timestamp_ms"] = 1699999999999
    with pytest.raises(DecisionV2ContractError, match="cannot precede capture"):
        DecisionEnvelopeV2.from_dict(golden_payload)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_numbers_are_rejected(
    golden_payload: dict[str, object], constant: str
) -> None:
    wire = json.dumps(golden_payload, separators=(",", ":")).encode("utf-8")
    if constant == "1e999":
        wire = wire.replace(b'"score_pct":84.0', b'"score_pct":1e999')
    else:
        wire = wire.replace(b'"score_pct":84.0', f'"score_pct":{constant}'.encode())

    with pytest.raises(DecisionV2ContractError):
        DecisionEnvelopeV2.from_json_bytes(wire)


def test_duplicate_json_keys_are_rejected(golden_payload: dict[str, object]) -> None:
    wire = json.dumps(golden_payload, separators=(",", ":")).encode("utf-8")
    wire = wire.replace(
        b'"schema":"safeloop.decision.v2",',
        b'"schema":"safeloop.decision.v2","schema":"safeloop.decision.v2",',
        1,
    )
    with pytest.raises(DecisionV2ContractError, match="duplicate JSON key"):
        DecisionEnvelopeV2.from_json_bytes(wire)


@pytest.mark.parametrize("ttc", [float("inf"), float("-inf"), float("nan")])
def test_create_maps_nonfinite_ttc_to_null_and_invalid(
    golden_payload: dict[str, object], ttc: float
) -> None:
    kwargs = create_kwargs(golden_payload)
    kwargs["c1"]["ttc_ms"] = ttc  # type: ignore[index]

    envelope = DecisionEnvelopeV2.create(**kwargs)  # type: ignore[arg-type]

    assert envelope.c1["ttc_ms"] is None
    assert envelope.c1["collision_probability_pct"] is None
    assert envelope.c1["warning"] is False
    assert envelope.validity["c1"] is False
    assert envelope.validity["c3"] is False
    assert envelope.validity["drive_quality"] is False
    assert envelope.validity["contextual_risk"] is False
    assert envelope.health["status"] == "DEGRADED"
    wire = envelope.to_json_bytes()
    assert b"Infinity" not in wire and b"NaN" not in wire


def test_expiry_is_derived_from_capture_and_stale_suppresses_outputs(
    golden_payload: dict[str, object],
) -> None:
    kwargs = create_kwargs(golden_payload)
    capture = kwargs["capture_timestamp_ms"]
    ttl = kwargs["ttl_ms"]
    kwargs["decision_timestamp_ms"] = capture + ttl  # type: ignore[operator]
    kwargs["server_receive_timestamp_ms"] = capture + 20  # type: ignore[operator]

    envelope = DecisionEnvelopeV2.create(**kwargs)  # type: ignore[arg-type]

    assert envelope.expires_at_ms == envelope.capture_timestamp_ms + envelope.ttl_ms
    assert envelope.expires_at_ms != envelope.decision_timestamp_ms + envelope.ttl_ms
    assert envelope.health == {
        "status": "STALE",
        "reasons": ["SOURCE_EXPIRED", "INVALID_COMPONENTS"],
    }
    assert not any(envelope.validity.values())
    assert envelope.c1["warning"] is False
    assert envelope.c2["warning"] is False
    assert envelope.contextual_risk["action"] == "MONITOR"
    assert envelope.is_expired(envelope.expires_at_ms)
    assert not envelope.is_expired(envelope.expires_at_ms - 1)


@pytest.mark.parametrize(
    ("status", "uncertainty"),
    [
        ("UNHEALTHY", 0),
        ("UNKNOWN", 0),
        ("SYNCHRONIZED", MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS + 1),
    ],
)
def test_untrusted_or_uncertain_clock_forces_stale(
    golden_payload: dict[str, object], status: str, uncertainty: int
) -> None:
    kwargs = create_kwargs(golden_payload)
    kwargs["clock_status"] = status
    kwargs["clock_offset_uncertainty_ms"] = uncertainty

    envelope = DecisionEnvelopeV2.create(**kwargs)  # type: ignore[arg-type]

    assert envelope.health["status"] == "STALE"
    assert "CLOCK_UNTRUSTED" in envelope.health["reasons"]
    assert not any(envelope.validity.values())


def test_direct_payload_cannot_claim_nominal_with_untrusted_clock(
    golden_payload: dict[str, object],
) -> None:
    golden_payload["clock"]["status"] = "UNHEALTHY"  # type: ignore[index]
    with pytest.raises(DecisionV2ContractError, match="health.status must be STALE"):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_direct_payload_cannot_keep_valid_outputs_after_expiry(
    golden_payload: dict[str, object],
) -> None:
    capture = golden_payload["capture_timestamp_ms"]
    ttl = golden_payload["ttl_ms"]
    golden_payload["decision_timestamp_ms"] = capture + ttl  # type: ignore[operator]
    golden_payload["source_age_ms"] = ttl
    with pytest.raises(DecisionV2ContractError, match="health.status must be STALE"):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_clock_uncertainty_boundary_is_nominal(
    golden_payload: dict[str, object],
) -> None:
    kwargs = create_kwargs(golden_payload)
    kwargs["clock_offset_uncertainty_ms"] = MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS
    envelope = DecisionEnvelopeV2.create(**kwargs)  # type: ignore[arg-type]
    assert envelope.health == {"status": "NOMINAL", "reasons": []}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "EMERGENCY_BRAKE_REQUEST"),
        ("brake_request_pct", 0.01),
        ("reasons", ["REQUEST_BRAKE_NOW"]),
    ],
)
def test_emergency_brake_semantics_are_rejected(
    golden_payload: dict[str, object], field: str, value: object
) -> None:
    risk = golden_payload["contextual_risk"]
    risk[field] = value  # type: ignore[index]
    with pytest.raises(DecisionV2ContractError):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_warning_only_action_is_allowed_but_actuation_never_is(
    golden_payload: dict[str, object],
) -> None:
    assert DecisionEnvelopeV2.from_dict(golden_payload).actuation_authorized is False
    golden_payload["actuation_authorized"] = True
    with pytest.raises(DecisionV2ContractError, match="never authorize"):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_model_provenance_keys_are_exact(golden_payload: dict[str, object]) -> None:
    missing = deepcopy(golden_payload)
    missing["models"].pop("c2")  # type: ignore[union-attr]
    with pytest.raises(DecisionV2ContractError, match="models keys must be exactly"):
        DecisionEnvelopeV2.from_dict(missing)

    extra = deepcopy(golden_payload)
    extra["models"]["decoder"] = {  # type: ignore[index]
        "version": "decoder-1",
        "digest_sha256": "f" * 64,
    }
    with pytest.raises(DecisionV2ContractError, match="models keys must be exactly"):
        DecisionEnvelopeV2.from_dict(extra)


def test_health_reason_allowlist_is_strict(golden_payload: dict[str, object]) -> None:
    golden_payload["health"]["reasons"] = ["UNRECOGNIZED"]  # type: ignore[index]
    with pytest.raises(DecisionV2ContractError, match="known health reason"):
        DecisionEnvelopeV2.from_dict(golden_payload)


@pytest.mark.parametrize(
    "reasons",
    [
        [f"RISK_{index}" for index in range(17)],
        ["X" * 161],
    ],
)
def test_contextual_reason_count_and_length_are_bounded(
    golden_payload: dict[str, object], reasons: list[str]
) -> None:
    golden_payload["contextual_risk"]["reasons"] = reasons  # type: ignore[index]
    with pytest.raises(DecisionV2ContractError, match="at most 16"):
        DecisionEnvelopeV2.from_dict(golden_payload)


def test_mqtt_payload_limit_is_independent_of_legacy_udp_size(
    golden_payload: dict[str, object],
) -> None:
    envelope = DecisionEnvelopeV2.from_dict(golden_payload)
    risk = golden_payload["contextual_risk"]
    risk["reasons"] = [f"RISK_{index:02d}_" + "X" * 140 for index in range(16)]  # type: ignore[index]
    largest_fixture = DecisionEnvelopeV2.from_dict(golden_payload)
    encoded_fixtures = [envelope.to_mqtt_json_bytes(), largest_fixture.to_mqtt_json_bytes()]
    max_fixture_size = max(map(len, encoded_fixtures))

    assert MAX_MQTT_PAYLOAD_BYTES == 16_384
    assert max_fixture_size > 1_472
    assert max_fixture_size <= MAX_MQTT_PAYLOAD_BYTES
    with pytest.raises(DecisionV2ContractError, match="maximum"):
        largest_fixture.to_mqtt_json_bytes(max_payload_bytes=max_fixture_size - 1)
    with pytest.raises(DecisionV2ContractError, match="maximum"):
        DecisionEnvelopeV2.from_mqtt_json_bytes(
            encoded_fixtures[-1], max_payload_bytes=max_fixture_size - 1
        )


def test_constructor_breaks_caller_mapping_aliases(
    golden_payload: dict[str, object],
) -> None:
    envelope = DecisionEnvelopeV2.from_dict(golden_payload)
    golden_payload["c1"]["warning"] = False  # type: ignore[index]
    golden_payload["models"]["c1"]["digest_sha256"] = "f" * 64  # type: ignore[index]

    assert envelope.c1["warning"] is True
    assert envelope.models["c1"]["digest_sha256"] == "a" * 64


def test_wire_revalidates_exposed_mutable_mappings(
    golden_payload: dict[str, object],
) -> None:
    envelope = DecisionEnvelopeV2.from_dict(golden_payload)
    envelope.contextual_risk["brake_request_pct"] = 5.0  # type: ignore[index]
    with pytest.raises(DecisionV2ContractError):
        envelope.to_json_bytes()
