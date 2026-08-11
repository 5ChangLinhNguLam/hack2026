"""Strict cloud decision contract for live SafeLoop inference.

``safeloop.decision.v2`` is deliberately separate from the legacy UDP v1
contract.  It carries no image bytes, ground truth, labels, depth, or
precomputed predictions.  Freshness starts at source capture time, not at
publish or receiver time, and the contract is warning-only: vehicle actuation
can never be authorized by a valid v2 envelope.

The constructor/parser is intentionally strict.  ``create`` is the live
producer boundary: it derives freshness/health, suppresses invalid component
outputs, and converts a non-finite C1 TTC to ``null`` plus invalid C1 state.
The MQTT encoder has its own application payload ceiling; the legacy 1,472
byte UDP ceiling is not applicable to this cloud contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA = "safeloop.decision.v2"
VIDEO_SOURCES = frozenset({"LIVE_CAMERA", "RECORDED_STREAM"})
INFERENCE_MODE = "LIVE_MODEL"

CLOCK_STATUSES = frozenset({"SYNCHRONIZED", "UNHEALTHY", "UNKNOWN"})
HEALTH_STATUSES = frozenset({"NOMINAL", "DEGRADED", "STALE"})
MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS = 25

DEFAULT_TTL_MS = 250
MAX_TTL_MS = 1_000
MAX_MQTT_PAYLOAD_BYTES = 16 * 1_024
MAX_SIGNED_64 = (1 << 63) - 1

VALIDITY_KEYS = frozenset(
    {
        "ego",
        "front_camera",
        "cabin_camera",
        "c1",
        "c2",
        "c3",
        "drive_quality",
        "contextual_risk",
    }
)
MODEL_KEYS = frozenset(
    {"c1", "c2", "c3", "drive_quality", "contextual_risk"}
)

DRIVER_STATES = frozenset(
    {"alert", "drowsy", "microsleep", "yawning", "distracted", "unavailable"}
)
GRADES = frozenset({"A", "B", "C", "D", "E", "N/A"})
C3_SCOPES = frozenset({"PREFIX", "FULL_TRIP", "NO_DATA"})
DRIVE_QUALITY_SCOPES = frozenset(
    {"NO_DATA", "PREFIX", "FULL_TRIP", "ROLLING_60S"}
)
RISK_LEVELS = frozenset({"SAFE", "CAUTION", "HIGH", "CRITICAL", "UNAVAILABLE"})
WARNING_ACTIONS = frozenset(
    {"MONITOR", "VISUAL_WARNING", "VISUAL_AUDIO_WARNING", "VISUAL_AUDIO_HAPTIC_WARNING"}
)
HEALTH_REASONS = frozenset(
    {
        "INVALID_COMPONENTS",
        "CLOCK_UNTRUSTED",
        "SOURCE_EXPIRED",
        "STREAM_GAP",
        "MODEL_ERROR",
        "CUDA_OOM",
        "WORKER_RESTART",
        "INITIALIZING",
    }
)

_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN_ACTUATION_TERMS = ("emergency", "brake", "braking", "aeb", "actuat")

_ROOT_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "session_id",
        "generation",
        "sequence",
        "source_sequence",
        "video_source",
        "inference_mode",
        "capture_timestamp_ms",
        "server_receive_timestamp_ms",
        "decision_timestamp_ms",
        "source_age_ms",
        "ttl_ms",
        "expires_at_ms",
        "clock",
        "validity",
        "c1",
        "c2",
        "c3",
        "drive_quality",
        "contextual_risk",
        "models",
        "health",
        "actuation_authorized",
    }
)
_ROOT_OPTIONAL_KEYS = frozenset({"source_media_timestamp_ms"})


class DecisionV2ContractError(ValueError):
    """A payload cannot be represented safely as a decision-v2 snapshot."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = MAX_SIGNED_64,
) -> int:
    if not _is_int(value) or not minimum <= value <= maximum:
        raise DecisionV2ContractError(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise DecisionV2ContractError(f"{field} must be boolean")
    return value


def _require_string(
    value: object, *, field: str, maximum_length: int = 128
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum_length
        or any(ord(char) < 0x20 for char in value)
    ):
        raise DecisionV2ContractError(
            f"{field} must be a non-empty string of at most {maximum_length} characters"
        )
    return value


def _require_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionV2ContractError(f"{field} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise DecisionV2ContractError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise DecisionV2ContractError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise DecisionV2ContractError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise DecisionV2ContractError(f"{field} must be <= {maximum}")
    return number


def _require_percentage(value: object, *, field: str) -> float:
    return _require_number(value, field=field, minimum=0.0, maximum=100.0)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionV2ContractError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DecisionV2ContractError(f"{field} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, field: str, expected: set[str] | frozenset[str]
) -> None:
    if set(value) != set(expected):
        raise DecisionV2ContractError(
            f"{field} keys must be exactly " + ", ".join(sorted(expected))
        )


def _copy_mapping(value: object, *, field: str) -> dict[str, Any]:
    return deepcopy(dict(_require_mapping(value, field=field)))


def _contains_actuation_vocabulary(value: str) -> bool:
    lowered = value.casefold()
    return any(term in lowered for term in _FORBIDDEN_ACTUATION_TERMS)


def _parse_json_object(payload: bytes | bytearray | memoryview) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise DecisionV2ContractError("decision JSON must be bytes-like")

    def reject_constant(value: str) -> None:
        raise DecisionV2ContractError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise DecisionV2ContractError(f"duplicate JSON key: {key}")
            decoded[key] = value
        return decoded

    try:
        decoded = json.loads(
            bytes(payload).decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except DecisionV2ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecisionV2ContractError("invalid UTF-8 decision JSON") from exc
    if not isinstance(decoded, dict):
        raise DecisionV2ContractError("decision JSON root must be an object")
    return decoded


def _neutral_c1(c1: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ttc_ms": None,
        "collision_probability_pct": None,
        "warning": False,
        "model_updated": (
            c1.get("model_updated")
            if isinstance(c1.get("model_updated"), bool)
            else False
        ),
    }


def _neutral_c2() -> dict[str, Any]:
    return {
        "state": "unavailable",
        "confidence_pct": None,
        "attentive_probability_pct": None,
        "distraction_level_pct": None,
        "fatigue_level_pct": None,
        "eyes_on_road": None,
        "warning": False,
    }


def _neutral_c3() -> dict[str, Any]:
    return {"safe_score_estimate_pct": None, "grade": "N/A", "scope": "NO_DATA"}


def _neutral_drive_quality() -> dict[str, Any]:
    return {"score_pct": None, "grade": "N/A", "scope": "NO_DATA"}


def _neutral_contextual_risk() -> dict[str, Any]:
    return {
        "score_pct": None,
        "level": "UNAVAILABLE",
        "action": "MONITOR",
        "brake_request_pct": 0.0,
        "reasons": ["STALE_OR_INVALID_INPUT"],
    }


@dataclass(frozen=True)
class DecisionEnvelopeV2:
    """One atomic, latest-value cloud snapshot for an Android consumer."""

    schema: str
    session_id: str
    generation: int
    sequence: int
    source_sequence: int
    video_source: str
    inference_mode: str
    capture_timestamp_ms: int
    source_media_timestamp_ms: int | None
    server_receive_timestamp_ms: int
    decision_timestamp_ms: int
    source_age_ms: int
    ttl_ms: int
    expires_at_ms: int
    clock: Mapping[str, Any]
    validity: Mapping[str, bool]
    c1: Mapping[str, Any]
    c2: Mapping[str, Any]
    c3: Mapping[str, Any]
    drive_quality: Mapping[str, Any]
    contextual_risk: Mapping[str, Any]
    models: Mapping[str, Mapping[str, str]]
    health: Mapping[str, Any]
    actuation_authorized: bool

    def __post_init__(self) -> None:
        # Break all producer-owned mutable aliases.  Nested mappings are still
        # revalidated at every wire encode in case a caller mutates an exposed
        # mapping after construction.
        for field in (
            "clock",
            "validity",
            "c1",
            "c2",
            "c3",
            "drive_quality",
            "contextual_risk",
            "models",
            "health",
        ):
            object.__setattr__(
                self,
                field,
                _copy_mapping(getattr(self, field), field=field),
            )
        self._validate()

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        generation: int,
        sequence: int,
        source_sequence: int,
        video_source: str,
        capture_timestamp_ms: int,
        server_receive_timestamp_ms: int,
        decision_timestamp_ms: int,
        clock_status: str,
        clock_offset_uncertainty_ms: int,
        validity: Mapping[str, bool],
        c1: Mapping[str, Any],
        c2: Mapping[str, Any],
        c3: Mapping[str, Any],
        drive_quality: Mapping[str, Any],
        contextual_risk: Mapping[str, Any],
        models: Mapping[str, Mapping[str, str]],
        source_media_timestamp_ms: int | None = None,
        ttl_ms: int = DEFAULT_TTL_MS,
        health_reasons: Sequence[str] = (),
    ) -> "DecisionEnvelopeV2":
        """Build and fail-safe a live-model envelope.

        This is the only API that accepts a non-finite TTC from model space.
        Such a TTC is not serialized: C1 becomes invalid and all C1-derived
        warning outputs are neutralized.  Stale or clock-untrusted snapshots
        suppress every component so old alerts cannot survive transport.
        """

        validity_payload = _copy_mapping(validity, field="validity")
        _require_exact_keys(validity_payload, field="validity", expected=VALIDITY_KEYS)
        for name, available in validity_payload.items():
            _require_bool(available, field=f"validity.{name}")

        c1_payload = _copy_mapping(c1, field="c1")
        c2_payload = _copy_mapping(c2, field="c2")
        c3_payload = _copy_mapping(c3, field="c3")
        quality_payload = _copy_mapping(drive_quality, field="drive_quality")
        risk_payload = _copy_mapping(contextual_risk, field="contextual_risk")

        ttc = c1_payload.get("ttc_ms")
        if isinstance(ttc, float) and not math.isfinite(ttc):
            validity_payload["c1"] = False

        # A derived component cannot be fresher than its required inputs.
        if not (validity_payload["ego"] and validity_payload["front_camera"]):
            validity_payload["c1"] = False
        if not validity_payload["cabin_camera"]:
            validity_payload["c2"] = False
        if not (validity_payload["ego"] and validity_payload["c1"]):
            validity_payload["c3"] = False
        if not (validity_payload["ego"] and validity_payload["c3"]):
            validity_payload["drive_quality"] = False
        if not (validity_payload["c1"] and validity_payload["c2"]):
            validity_payload["contextual_risk"] = False

        capture = _require_int(capture_timestamp_ms, field="capture_timestamp_ms")
        decided = _require_int(decision_timestamp_ms, field="decision_timestamp_ms")
        ttl = _require_int(ttl_ms, field="ttl_ms", minimum=1, maximum=MAX_TTL_MS)
        uncertainty = _require_int(
            clock_offset_uncertainty_ms,
            field="clock.offset_uncertainty_ms",
            maximum=60_000,
        )
        clock_trusted = (
            clock_status == "SYNCHRONIZED"
            and uncertainty <= MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS
        )
        expired = decided - capture >= ttl

        reasons: list[str] = []
        for reason in health_reasons:
            if not isinstance(reason, str):
                raise DecisionV2ContractError("health_reasons must contain strings")
            if reason not in reasons:
                reasons.append(reason)

        if not clock_trusted:
            reasons.append("CLOCK_UNTRUSTED")
        if expired:
            reasons.append("SOURCE_EXPIRED")
        if not clock_trusted or expired:
            for name in validity_payload:
                validity_payload[name] = False

        if not all(validity_payload.values()):
            reasons.append("INVALID_COMPONENTS")

        reasons = list(dict.fromkeys(reasons))
        if not clock_trusted or expired:
            health_status = "STALE"
        elif not all(validity_payload.values()):
            health_status = "DEGRADED"
        else:
            health_status = "NOMINAL"

        if not validity_payload["c1"]:
            c1_payload = _neutral_c1(c1_payload)
        if not validity_payload["c2"]:
            c2_payload = _neutral_c2()
        if not validity_payload["c3"]:
            c3_payload = _neutral_c3()
        if not validity_payload["drive_quality"]:
            quality_payload = _neutral_drive_quality()
        if not validity_payload["contextual_risk"]:
            risk_payload = _neutral_contextual_risk()

        return cls(
            schema=SCHEMA,
            session_id=session_id,
            generation=generation,
            sequence=sequence,
            source_sequence=source_sequence,
            video_source=video_source,
            inference_mode=INFERENCE_MODE,
            capture_timestamp_ms=capture,
            source_media_timestamp_ms=source_media_timestamp_ms,
            server_receive_timestamp_ms=server_receive_timestamp_ms,
            decision_timestamp_ms=decided,
            source_age_ms=decided - capture,
            ttl_ms=ttl,
            expires_at_ms=capture + ttl,
            clock={
                "status": clock_status,
                "offset_uncertainty_ms": uncertainty,
            },
            validity=validity_payload,
            c1=c1_payload,
            c2=c2_payload,
            c3=c3_payload,
            drive_quality=quality_payload,
            contextual_risk=risk_payload,
            models=models,
            health={"status": health_status, "reasons": reasons},
            actuation_authorized=False,
        )

    def _validate(self) -> None:
        if self.schema != SCHEMA:
            raise DecisionV2ContractError(f"unsupported decision schema: {self.schema!r}")
        _require_string(self.session_id, field="session_id")
        for name in ("generation", "sequence", "source_sequence"):
            _require_int(getattr(self, name), field=name)
        if self.video_source not in VIDEO_SOURCES:
            raise DecisionV2ContractError(
                f"video_source must be one of {sorted(VIDEO_SOURCES)}"
            )
        if self.inference_mode != INFERENCE_MODE:
            raise DecisionV2ContractError("inference_mode must be LIVE_MODEL")

        for name in (
            "capture_timestamp_ms",
            "server_receive_timestamp_ms",
            "decision_timestamp_ms",
            "source_age_ms",
            "expires_at_ms",
        ):
            _require_int(getattr(self, name), field=name)
        if self.source_media_timestamp_ms is not None:
            _require_int(
                self.source_media_timestamp_ms,
                field="source_media_timestamp_ms",
            )
        _require_int(self.ttl_ms, field="ttl_ms", minimum=1, maximum=MAX_TTL_MS)
        if self.server_receive_timestamp_ms < self.capture_timestamp_ms:
            raise DecisionV2ContractError(
                "server_receive_timestamp_ms cannot precede capture_timestamp_ms"
            )
        if self.decision_timestamp_ms < self.server_receive_timestamp_ms:
            raise DecisionV2ContractError(
                "decision_timestamp_ms cannot precede server_receive_timestamp_ms"
            )
        if self.source_age_ms != self.decision_timestamp_ms - self.capture_timestamp_ms:
            raise DecisionV2ContractError(
                "source_age_ms must equal decision_timestamp_ms - capture_timestamp_ms"
            )
        if self.expires_at_ms != self.capture_timestamp_ms + self.ttl_ms:
            raise DecisionV2ContractError(
                "expires_at_ms must equal capture_timestamp_ms + ttl_ms"
            )

        clock = _require_mapping(self.clock, field="clock")
        _require_exact_keys(
            clock, field="clock", expected={"status", "offset_uncertainty_ms"}
        )
        if clock.get("status") not in CLOCK_STATUSES:
            raise DecisionV2ContractError(
                f"clock.status must be one of {sorted(CLOCK_STATUSES)}"
            )
        uncertainty = _require_int(
            clock.get("offset_uncertainty_ms"),
            field="clock.offset_uncertainty_ms",
            maximum=60_000,
        )
        clock_trusted = (
            clock["status"] == "SYNCHRONIZED"
            and uncertainty <= MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS
        )

        validity = _require_mapping(self.validity, field="validity")
        _require_exact_keys(validity, field="validity", expected=VALIDITY_KEYS)
        for name, available in validity.items():
            _require_bool(available, field=f"validity.{name}")
        if validity["c1"] and not (validity["ego"] and validity["front_camera"]):
            raise DecisionV2ContractError("valid C1 requires ego and front_camera")
        if validity["c2"] and not validity["cabin_camera"]:
            raise DecisionV2ContractError("valid C2 requires cabin_camera")
        if validity["c3"] and not (validity["ego"] and validity["c1"]):
            raise DecisionV2ContractError("valid C3 requires ego and C1")
        if validity["drive_quality"] and not (validity["ego"] and validity["c3"]):
            raise DecisionV2ContractError("valid drive_quality requires ego and C3")
        if validity["contextual_risk"] and not (validity["c1"] and validity["c2"]):
            raise DecisionV2ContractError("valid contextual_risk requires C1 and C2")

        self._validate_c1(validity["c1"])
        self._validate_c2(validity["c2"])
        self._validate_c3(validity["c3"])
        self._validate_drive_quality(validity["drive_quality"])
        self._validate_contextual_risk(validity["contextual_risk"])
        self._validate_models()

        health = _require_mapping(self.health, field="health")
        _require_exact_keys(health, field="health", expected={"status", "reasons"})
        status = health.get("status")
        if status not in HEALTH_STATUSES:
            raise DecisionV2ContractError(
                f"health.status must be one of {sorted(HEALTH_STATUSES)}"
            )
        reasons = health.get("reasons")
        if (
            not isinstance(reasons, list)
            or len(reasons) > 16
            or any(reason not in HEALTH_REASONS for reason in reasons)
            or len(set(reasons)) != len(reasons)
        ):
            raise DecisionV2ContractError(
                "health.reasons must be a unique list of known health reason strings"
            )

        expired_at_decision = self.source_age_ms >= self.ttl_ms
        expected_status = (
            "STALE"
            if expired_at_decision or not clock_trusted
            else "DEGRADED"
            if not all(validity.values())
            else "NOMINAL"
        )
        if status != expected_status:
            raise DecisionV2ContractError(
                f"health.status must be {expected_status} for envelope state"
            )
        required_reasons: set[str] = set()
        if expired_at_decision:
            required_reasons.add("SOURCE_EXPIRED")
        if not clock_trusted:
            required_reasons.add("CLOCK_UNTRUSTED")
        if not all(validity.values()):
            required_reasons.add("INVALID_COMPONENTS")
        if not required_reasons.issubset(reasons):
            raise DecisionV2ContractError(
                "health.reasons does not describe every degraded/stale condition"
            )
        if "SOURCE_EXPIRED" in reasons and not expired_at_decision:
            raise DecisionV2ContractError("SOURCE_EXPIRED reason requires expired source data")
        if "CLOCK_UNTRUSTED" in reasons and clock_trusted:
            raise DecisionV2ContractError("CLOCK_UNTRUSTED reason requires an untrusted clock")
        if status == "NOMINAL" and reasons:
            raise DecisionV2ContractError("NOMINAL health cannot contain reasons")
        if status == "STALE" and any(validity.values()):
            raise DecisionV2ContractError("STALE health must invalidate every component")

        if self.actuation_authorized is not False:
            raise DecisionV2ContractError(
                "decision v2 must never authorize vehicle actuation"
            )

        try:
            json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DecisionV2ContractError(
                "decision envelope contains a non-JSON or non-finite value"
            ) from exc

    def _validate_c1(self, valid: bool) -> None:
        c1 = _require_mapping(self.c1, field="c1")
        _require_exact_keys(
            c1,
            field="c1",
            expected={"ttc_ms", "collision_probability_pct", "warning", "model_updated"},
        )
        ttc = c1.get("ttc_ms")
        if ttc is not None:
            _require_int(ttc, field="c1.ttc_ms")
        probability = c1.get("collision_probability_pct")
        if probability is not None:
            _require_percentage(probability, field="c1.collision_probability_pct")
        _require_bool(c1.get("warning"), field="c1.warning")
        _require_bool(c1.get("model_updated"), field="c1.model_updated")
        if valid and (ttc is None or probability is None):
            raise DecisionV2ContractError("valid C1 requires finite TTC and probability")
        if not valid and (ttc is not None or probability is not None or c1["warning"]):
            raise DecisionV2ContractError(
                "invalid C1 must contain null estimates and warning=false"
            )

    def _validate_c2(self, valid: bool) -> None:
        c2 = _require_mapping(self.c2, field="c2")
        optional_fields = {
            "confidence_pct",
            "attentive_probability_pct",
            "distraction_level_pct",
            "fatigue_level_pct",
        }
        _require_exact_keys(
            c2,
            field="c2",
            expected={"state", *optional_fields, "eyes_on_road", "warning"},
        )
        if c2.get("state") not in DRIVER_STATES:
            raise DecisionV2ContractError(f"invalid c2.state: {c2.get('state')!r}")
        for field in optional_fields:
            value = c2.get(field)
            if value is not None:
                _require_percentage(value, field=f"c2.{field}")
        eyes = c2.get("eyes_on_road")
        if eyes is not None:
            _require_bool(eyes, field="c2.eyes_on_road")
        _require_bool(c2.get("warning"), field="c2.warning")
        values = [c2.get(field) for field in optional_fields] + [eyes]
        if valid and (c2["state"] == "unavailable" or any(v is None for v in values)):
            raise DecisionV2ContractError("valid C2 requires all driver-state values")
        if not valid and (
            c2["state"] != "unavailable"
            or any(v is not None for v in values)
            or c2["warning"]
        ):
            raise DecisionV2ContractError("invalid C2 must expose only unavailable/null values")

    def _validate_c3(self, valid: bool) -> None:
        c3 = _require_mapping(self.c3, field="c3")
        _require_exact_keys(
            c3,
            field="c3",
            expected={"safe_score_estimate_pct", "grade", "scope"},
        )
        score = c3.get("safe_score_estimate_pct")
        if score is not None:
            _require_percentage(score, field="c3.safe_score_estimate_pct")
        if c3.get("grade") not in GRADES:
            raise DecisionV2ContractError(f"invalid c3.grade: {c3.get('grade')!r}")
        if c3.get("scope") not in C3_SCOPES:
            raise DecisionV2ContractError(f"invalid c3.scope: {c3.get('scope')!r}")
        if valid and (score is None or c3["grade"] == "N/A" or c3["scope"] == "NO_DATA"):
            raise DecisionV2ContractError("valid C3 requires score, grade, and data scope")
        if not valid and c3 != _neutral_c3():
            raise DecisionV2ContractError("invalid C3 must expose null/N/A/NO_DATA")

    def _validate_drive_quality(self, valid: bool) -> None:
        quality = _require_mapping(self.drive_quality, field="drive_quality")
        _require_exact_keys(
            quality, field="drive_quality", expected={"score_pct", "grade", "scope"}
        )
        score = quality.get("score_pct")
        if score is not None:
            _require_percentage(score, field="drive_quality.score_pct")
        if quality.get("grade") not in GRADES:
            raise DecisionV2ContractError(
                f"invalid drive_quality.grade: {quality.get('grade')!r}"
            )
        if quality.get("scope") not in DRIVE_QUALITY_SCOPES:
            raise DecisionV2ContractError(
                f"invalid drive_quality.scope: {quality.get('scope')!r}"
            )
        if valid and (
            score is None or quality["grade"] == "N/A" or quality["scope"] == "NO_DATA"
        ):
            raise DecisionV2ContractError(
                "valid drive_quality requires score, grade, and data scope"
            )
        if not valid and quality != _neutral_drive_quality():
            raise DecisionV2ContractError(
                "invalid drive_quality must expose null/N/A/NO_DATA"
            )

    def _validate_contextual_risk(self, valid: bool) -> None:
        risk = _require_mapping(self.contextual_risk, field="contextual_risk")
        _require_exact_keys(
            risk,
            field="contextual_risk",
            expected={"score_pct", "level", "action", "brake_request_pct", "reasons"},
        )
        score = risk.get("score_pct")
        if score is not None:
            _require_percentage(score, field="contextual_risk.score_pct")
        if risk.get("level") not in RISK_LEVELS:
            raise DecisionV2ContractError(
                f"invalid contextual_risk.level: {risk.get('level')!r}"
            )
        action = risk.get("action")
        if action not in WARNING_ACTIONS:
            raise DecisionV2ContractError(
                f"contextual_risk.action must be warning-only: {action!r}"
            )
        brake = _require_percentage(
            risk.get("brake_request_pct"), field="contextual_risk.brake_request_pct"
        )
        if brake != 0.0:
            raise DecisionV2ContractError("contextual_risk.brake_request_pct must be zero")
        reasons = risk.get("reasons")
        if (
            not isinstance(reasons, list)
            or len(reasons) > 16
            or any(
                not isinstance(reason, str)
                or not reason
                or len(reason) > 160
                or any(ord(char) < 0x20 for char in reason)
                for reason in reasons
            )
        ):
            raise DecisionV2ContractError(
                "contextual_risk.reasons must contain at most 16 safe non-empty strings"
            )
        unsafe_strings = [value for value in [action, *reasons] if _contains_actuation_vocabulary(value)]
        if unsafe_strings:
            raise DecisionV2ContractError(
                "emergency/brake/actuation vocabulary is forbidden in decision v2"
            )
        if valid and (score is None or risk["level"] == "UNAVAILABLE"):
            raise DecisionV2ContractError("valid contextual risk requires score and level")
        if not valid and risk != _neutral_contextual_risk():
            raise DecisionV2ContractError(
                "invalid contextual risk must expose the neutral unavailable state"
            )

    def _validate_models(self) -> None:
        models = _require_mapping(self.models, field="models")
        _require_exact_keys(models, field="models", expected=MODEL_KEYS)
        for name, raw_model in models.items():
            model = _require_mapping(raw_model, field=f"models.{name}")
            _require_exact_keys(
                model,
                field=f"models.{name}",
                expected={"version", "digest_sha256"},
            )
            _require_string(model.get("version"), field=f"models.{name}.version")
            digest = model.get("digest_sha256")
            if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
                raise DecisionV2ContractError(
                    f"models.{name}.digest_sha256 must be a full 64-hex SHA-256 digest"
                )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "session_id": self.session_id,
            "generation": self.generation,
            "sequence": self.sequence,
            "source_sequence": self.source_sequence,
            "video_source": self.video_source,
            "inference_mode": self.inference_mode,
            "capture_timestamp_ms": self.capture_timestamp_ms,
            "server_receive_timestamp_ms": self.server_receive_timestamp_ms,
            "decision_timestamp_ms": self.decision_timestamp_ms,
            "source_age_ms": self.source_age_ms,
            "ttl_ms": self.ttl_ms,
            "expires_at_ms": self.expires_at_ms,
            "clock": deepcopy(dict(self.clock)),
            "validity": deepcopy(dict(self.validity)),
            "c1": deepcopy(dict(self.c1)),
            "c2": deepcopy(dict(self.c2)),
            "c3": deepcopy(dict(self.c3)),
            "drive_quality": deepcopy(dict(self.drive_quality)),
            "contextual_risk": deepcopy(dict(self.contextual_risk)),
            "models": deepcopy(dict(self.models)),
            "health": deepcopy(dict(self.health)),
            "actuation_authorized": self.actuation_authorized,
        }
        if self.source_media_timestamp_ms is not None:
            payload["source_media_timestamp_ms"] = self.source_media_timestamp_ms
        return payload

    def to_json_bytes(self) -> bytes:
        self._validate()
        try:
            encoded = json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise DecisionV2ContractError(
                "decision envelope cannot be encoded as strict JSON"
            ) from exc
        return encoded.encode("utf-8")

    def to_mqtt_json_bytes(
        self, *, max_payload_bytes: int = MAX_MQTT_PAYLOAD_BYTES
    ) -> bytes:
        maximum = _require_int(
            max_payload_bytes,
            field="max_payload_bytes",
            minimum=1,
            maximum=MAX_SIGNED_64,
        )
        payload = self.to_json_bytes()
        if len(payload) > maximum:
            raise DecisionV2ContractError(
                f"decision MQTT payload is {len(payload)} bytes; maximum is {maximum}"
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionEnvelopeV2":
        raw = _copy_mapping(payload, field="decision")
        keys = set(raw)
        if not _ROOT_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
            _ROOT_REQUIRED_KEYS | _ROOT_OPTIONAL_KEYS
        ):
            allowed = sorted(_ROOT_REQUIRED_KEYS | _ROOT_OPTIONAL_KEYS)
            raise DecisionV2ContractError(
                "decision root keys must be the exact v2 keys (source media timestamp optional): "
                + ", ".join(allowed)
            )
        if (
            "source_media_timestamp_ms" in raw
            and raw["source_media_timestamp_ms"] is None
        ):
            raise DecisionV2ContractError(
                "source_media_timestamp_ms must be omitted rather than null"
            )
        return cls(
            schema=raw["schema"],
            session_id=raw["session_id"],
            generation=raw["generation"],
            sequence=raw["sequence"],
            source_sequence=raw["source_sequence"],
            video_source=raw["video_source"],
            inference_mode=raw["inference_mode"],
            capture_timestamp_ms=raw["capture_timestamp_ms"],
            source_media_timestamp_ms=raw.get("source_media_timestamp_ms"),
            server_receive_timestamp_ms=raw["server_receive_timestamp_ms"],
            decision_timestamp_ms=raw["decision_timestamp_ms"],
            source_age_ms=raw["source_age_ms"],
            ttl_ms=raw["ttl_ms"],
            expires_at_ms=raw["expires_at_ms"],
            clock=raw["clock"],
            validity=raw["validity"],
            c1=raw["c1"],
            c2=raw["c2"],
            c3=raw["c3"],
            drive_quality=raw["drive_quality"],
            contextual_risk=raw["contextual_risk"],
            models=raw["models"],
            health=raw["health"],
            actuation_authorized=raw["actuation_authorized"],
        )

    @classmethod
    def from_json_bytes(
        cls, payload: bytes | bytearray | memoryview
    ) -> "DecisionEnvelopeV2":
        return cls.from_dict(_parse_json_object(payload))

    @classmethod
    def from_mqtt_json_bytes(
        cls,
        payload: bytes | bytearray | memoryview,
        *,
        max_payload_bytes: int = MAX_MQTT_PAYLOAD_BYTES,
    ) -> "DecisionEnvelopeV2":
        maximum = _require_int(
            max_payload_bytes,
            field="max_payload_bytes",
            minimum=1,
            maximum=MAX_SIGNED_64,
        )
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise DecisionV2ContractError("decision MQTT payload must be bytes-like")
        if len(payload) > maximum:
            raise DecisionV2ContractError(
                f"decision MQTT payload is {len(payload)} bytes; maximum is {maximum}"
            )
        return cls.from_json_bytes(payload)

    def is_expired(self, now_ms: int) -> bool:
        now = _require_int(now_ms, field="now_ms")
        return now >= self.expires_at_ms


__all__ = [
    "CLOCK_STATUSES",
    "C3_SCOPES",
    "DEFAULT_TTL_MS",
    "DRIVER_STATES",
    "DRIVE_QUALITY_SCOPES",
    "DecisionEnvelopeV2",
    "DecisionV2ContractError",
    "GRADES",
    "HEALTH_REASONS",
    "HEALTH_STATUSES",
    "INFERENCE_MODE",
    "MAX_MQTT_PAYLOAD_BYTES",
    "MAX_TRUSTED_CLOCK_OFFSET_UNCERTAINTY_MS",
    "MAX_TTL_MS",
    "MODEL_KEYS",
    "RISK_LEVELS",
    "SCHEMA",
    "VALIDITY_KEYS",
    "VIDEO_SOURCES",
    "WARNING_ACTIONS",
]
