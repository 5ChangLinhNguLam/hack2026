"""Versioned SafeLoop decision envelope for the CarSky/AAOS data plane.

This is a product transport contract, not the hackathon submission schema.
It deliberately carries no image bytes and no inferred obstacle distance.  A
finite TTC is encoded in milliseconds; ``+inf`` (no finite collision horizon)
is encoded as ``null`` so JSON never contains NaN/Infinity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import threading
import time
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "safeloop.decision.v1"
SOURCE_MODES = frozenset({"live", "replay", "simulation"})
HEALTH_MODES = frozenset({"INITIALIZING", "NOMINAL", "DEGRADED"})
RISK_LEVELS = frozenset({"SAFE", "CAUTION", "HIGH", "CRITICAL", "UNAVAILABLE"})
RISK_ACTIONS = frozenset(
    {
        "MONITOR",
        "VISUAL_WARNING",
        "VISUAL_AUDIO_HAPTIC_WARNING",
        "EMERGENCY_BRAKE_REQUEST",
    }
)
DRIVER_STATES = frozenset(
    {"alert", "drowsy", "microsleep", "yawning", "distracted", "unavailable"}
)
GRADES = frozenset({"A", "B", "C", "D", "E", "N/A"})
C3_SCOPES = frozenset({"PREFIX", "FULL_TRIP"})
DRIVE_QUALITY_SCOPES = frozenset(
    {"NO_DATA", "PREFIX", "FULL_TRIP", "ROLLING_60S"}
)
DEFAULT_TTL_MS = 200
MAX_TTL_MS = 1_000


class DecisionContractError(ValueError):
    """A decision cannot be represented safely on the product wire."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise DecisionContractError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionContractError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise DecisionContractError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise DecisionContractError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise DecisionContractError(f"{field} must be <= {maximum}")
    return number


def _percentage(value: object, *, field: str) -> float:
    return round(
        _finite_number(value, field=field, minimum=0.0, maximum=100.0),
        3,
    )


def _probability_percentage(value: object, *, field: str) -> float:
    probability = _finite_number(
        value, field=field, minimum=0.0, maximum=1.0
    )
    return round(probability * 100.0, 3)


def _optional_percentage(value: object, *, valid: bool, field: str) -> float | None:
    return _percentage(value, field=field) if valid else None


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionContractError(f"{field} must be an object")
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise DecisionContractError(f"{field} must be boolean")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, field: str, expected: set[str]
) -> None:
    if set(value) != expected:
        raise DecisionContractError(
            f"{field} keys must be exactly " + ", ".join(sorted(expected))
        )


@dataclass(frozen=True)
class DecisionValidity:
    """Freshness/availability flags for one synchronized source snapshot."""

    ego: bool
    front_camera: bool
    driver_camera: bool
    c1: bool
    c2: bool
    c3: bool
    drive_quality: bool
    contextual_risk: bool

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_bool(value, field=f"validity.{name}")

    def invalid_components(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if not value)


@dataclass(frozen=True)
class DecisionEnvelope:
    """One atomic, latest-value SafeLoop snapshot for the Android HMI."""

    schema_version: str
    source_mode: str
    session_id: str
    sequence: int
    frame_id: int
    source_timestamp_ms: int
    decision_timestamp_ms: int
    ttl_ms: int
    expires_at_ms: int
    validity: Mapping[str, bool]
    c1: Mapping[str, Any]
    c2: Mapping[str, Any]
    c3: Mapping[str, Any]
    drive_quality: Mapping[str, Any]
    contextual_risk: Mapping[str, Any]
    health: Mapping[str, Any]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DecisionContractError(
                f"unsupported decision schema: {self.schema_version!r}"
            )
        if self.source_mode not in SOURCE_MODES:
            raise DecisionContractError(
                f"source_mode must be one of {sorted(SOURCE_MODES)}"
            )
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise DecisionContractError("session_id must be a non-empty string")
        if len(self.session_id) > 128:
            raise DecisionContractError("session_id is too long")
        for name, value in (
            ("sequence", self.sequence),
            ("frame_id", self.frame_id),
            ("source_timestamp_ms", self.source_timestamp_ms),
            ("decision_timestamp_ms", self.decision_timestamp_ms),
        ):
            if not _is_int(value) or value < 0:
                raise DecisionContractError(f"{name} must be a non-negative integer")
        if not _is_int(self.ttl_ms) or not 1 <= self.ttl_ms <= MAX_TTL_MS:
            raise DecisionContractError(
                f"ttl_ms must be an integer in [1, {MAX_TTL_MS}]"
            )
        if (
            not _is_int(self.expires_at_ms)
            or self.expires_at_ms != self.decision_timestamp_ms + self.ttl_ms
        ):
            raise DecisionContractError(
                "expires_at_ms must equal decision_timestamp_ms + ttl_ms"
            )

        validity = _require_mapping(self.validity, field="validity")
        expected_validity = set(DecisionValidity.__dataclass_fields__)
        if set(validity) != expected_validity:
            raise DecisionContractError(
                "validity keys must be exactly " + ", ".join(sorted(expected_validity))
            )
        for name, value in validity.items():
            _require_bool(value, field=f"validity.{name}")

        c1 = _require_mapping(self.c1, field="c1")
        forbidden_distance = {
            key for key in c1 if "distance" in str(key).lower()
        }
        if forbidden_distance:
            raise DecisionContractError(
                "C1 v1 has no calibrated distance output: "
                + ", ".join(sorted(forbidden_distance))
            )
        _require_exact_keys(
            c1,
            field="c1",
            expected={
                "ttc_ms",
                "ttc_valid",
                "collision_probability_pct",
                "warning",
                "model_updated",
                "model_frame_id",
                "age_ms",
            },
        )
        ttc_ms = c1.get("ttc_ms")
        if ttc_ms is not None and (not _is_int(ttc_ms) or ttc_ms < 0):
            raise DecisionContractError("c1.ttc_ms must be null or a non-negative integer")
        ttc_valid = _require_bool(c1.get("ttc_valid"), field="c1.ttc_valid")
        if ttc_valid != (ttc_ms is not None):
            raise DecisionContractError("c1.ttc_valid must match ttc_ms availability")
        _require_bool(c1.get("warning"), field="c1.warning")
        _require_bool(c1.get("model_updated"), field="c1.model_updated")
        for name in ("model_frame_id", "age_ms"):
            value = c1.get(name)
            if not _is_int(value) or value < 0:
                raise DecisionContractError(f"c1.{name} must be a non-negative integer")
        if c1["model_frame_id"] > self.frame_id:
            raise DecisionContractError("c1.model_frame_id cannot exceed frame_id")
        probability = c1.get("collision_probability_pct")
        if probability is not None:
            _percentage(probability, field="c1.collision_probability_pct")
        if validity["c1"] != (probability is not None):
            raise DecisionContractError(
                "c1 collision probability availability must match validity.c1"
            )
        if not validity["c1"] and (c1["ttc_valid"] or c1["warning"]):
            raise DecisionContractError(
                "invalid C1 must suppress TTC validity and collision warning"
            )

        c2 = _require_mapping(self.c2, field="c2")
        _require_exact_keys(
            c2,
            field="c2",
            expected={
                "state",
                "confidence_pct",
                "attentive_probability_pct",
                "distraction_level_pct",
                "fatigue_level_pct",
                "eyes_on_road",
                "warning",
            },
        )
        if c2.get("state") not in DRIVER_STATES:
            raise DecisionContractError(f"invalid c2.state: {c2.get('state')!r}")
        for name in (
            "confidence_pct",
            "attentive_probability_pct",
            "distraction_level_pct",
            "fatigue_level_pct",
        ):
            value = c2.get(name)
            if value is not None:
                _percentage(value, field=f"c2.{name}")
        _require_bool(c2.get("warning"), field="c2.warning")
        if c2.get("eyes_on_road") is not None:
            _require_bool(c2.get("eyes_on_road"), field="c2.eyes_on_road")
        c2_optional = (
            c2.get("confidence_pct"),
            c2.get("attentive_probability_pct"),
            c2.get("distraction_level_pct"),
            c2.get("fatigue_level_pct"),
            c2.get("eyes_on_road"),
        )
        if validity["c2"]:
            if c2["state"] == "unavailable" or any(
                value is None for value in c2_optional
            ):
                raise DecisionContractError(
                    "valid C2 requires state and all driver signal fields"
                )
        elif (
            c2["state"] != "unavailable"
            or any(value is not None for value in c2_optional)
            or c2["warning"]
        ):
            raise DecisionContractError(
                "invalid C2 must expose only the unavailable state"
            )

        c3 = _require_mapping(self.c3, field="c3")
        _require_exact_keys(
            c3,
            field="c3",
            expected={
                "safe_score_estimate_pct",
                "grade",
                "scope",
                "formula_version",
                "tailgating_penalty_omitted",
            },
        )
        c3_score = c3.get("safe_score_estimate_pct")
        if c3_score is not None:
            _percentage(c3_score, field="c3.safe_score_estimate_pct")
        if validity["c3"] != (c3_score is not None):
            raise DecisionContractError(
                "C3 score availability must match validity.c3"
            )
        if c3.get("grade") not in GRADES:
            raise DecisionContractError(f"invalid c3.grade: {c3.get('grade')!r}")
        if validity["c3"] == (c3.get("grade") == "N/A"):
            raise DecisionContractError("C3 grade availability must match validity.c3")
        if c3.get("scope") not in C3_SCOPES:
            raise DecisionContractError(f"invalid c3.scope: {c3.get('scope')!r}")
        if not isinstance(c3.get("formula_version"), str) or not c3["formula_version"]:
            raise DecisionContractError("c3.formula_version must be a non-empty string")
        _require_bool(
            c3.get("tailgating_penalty_omitted"),
            field="c3.tailgating_penalty_omitted",
        )

        quality = _require_mapping(self.drive_quality, field="drive_quality")
        _require_exact_keys(
            quality,
            field="drive_quality",
            expected={
                "score_available",
                "score_pct",
                "grade",
                "scope",
                "window_ready",
                "formula_version",
            },
        )
        quality_available = _require_bool(
            quality.get("score_available"), field="drive_quality.score_available"
        )
        quality_score = quality.get("score_pct")
        if quality_score is not None:
            _percentage(quality_score, field="drive_quality.score_pct")
        if quality_available != (quality_score is not None):
            raise DecisionContractError(
                "drive_quality score_available must match score_pct availability"
            )
        if quality.get("grade") not in GRADES:
            raise DecisionContractError(
                f"invalid drive_quality.grade: {quality.get('grade')!r}"
            )
        if quality.get("scope") not in DRIVE_QUALITY_SCOPES:
            raise DecisionContractError(
                f"invalid drive_quality.scope: {quality.get('scope')!r}"
            )
        _require_bool(
            quality.get("window_ready"), field="drive_quality.window_ready"
        )
        if not validity["drive_quality"] and (
            quality_available
            or quality.get("grade") != "N/A"
            or quality.get("scope") != "NO_DATA"
            or quality.get("window_ready")
        ):
            raise DecisionContractError(
                "invalid drive quality must expose an explicit NO_DATA state"
            )
        if (
            not isinstance(quality.get("formula_version"), str)
            or not quality["formula_version"]
        ):
            raise DecisionContractError(
                "drive_quality.formula_version must be a non-empty string"
            )

        risk = _require_mapping(self.contextual_risk, field="contextual_risk")
        _require_exact_keys(
            risk,
            field="contextual_risk",
            expected={
                "score_pct",
                "level",
                "action",
                "brake_request_pct",
                "reasons",
                "actuation_authorized",
            },
        )
        score = risk.get("score_pct")
        if score is not None:
            _percentage(score, field="contextual_risk.score_pct")
        if validity["contextual_risk"] != (score is not None):
            raise DecisionContractError(
                "contextual_risk score availability must match its validity flag"
            )
        level = risk.get("level")
        action = risk.get("action")
        if level not in RISK_LEVELS:
            raise DecisionContractError(f"invalid contextual_risk.level: {level!r}")
        if action not in RISK_ACTIONS:
            raise DecisionContractError(f"invalid contextual_risk.action: {action!r}")
        brake = _percentage(
            risk.get("brake_request_pct"),
            field="contextual_risk.brake_request_pct",
        )
        if risk.get("actuation_authorized") is not False:
            raise DecisionContractError(
                "Android envelope must never authorize vehicle actuation"
            )
        if not validity["contextual_risk"] and (
            level != "UNAVAILABLE" or action != "MONITOR" or brake != 0.0
        ):
            raise DecisionContractError(
                "invalid contextual risk must be UNAVAILABLE/MONITOR with zero brake"
            )
        if validity["contextual_risk"] and level == "UNAVAILABLE":
            raise DecisionContractError(
                "valid contextual risk cannot have UNAVAILABLE level"
            )
        if action == "EMERGENCY_BRAKE_REQUEST" and not all(
            validity[name] for name in ("ego", "front_camera", "c1")
        ):
            raise DecisionContractError(
                "emergency recommendation requires fresh ego/front/C1 inputs"
            )
        reasons = risk.get("reasons")
        if (
            not isinstance(reasons, list)
            or len(reasons) > 16
            or any(not isinstance(item, str) or not item for item in reasons)
        ):
            raise DecisionContractError(
                "contextual_risk.reasons must be a list of <=16 non-empty strings"
            )

        health = _require_mapping(self.health, field="health")
        _require_exact_keys(
            health,
            field="health",
            expected={
                "mode",
                "decision_valid",
                "stale_or_invalid_components",
                "component_age_ms",
                "ttl_ms",
                "model_versions",
            },
        )
        if health.get("mode") not in HEALTH_MODES:
            raise DecisionContractError(f"invalid health.mode: {health.get('mode')!r}")
        decision_valid = _require_bool(
            health.get("decision_valid"), field="health.decision_valid"
        )
        if decision_valid != validity["contextual_risk"]:
            raise DecisionContractError(
                "health.decision_valid must match validity.contextual_risk"
            )
        invalid_components = [
            name
            for name in DecisionValidity.__dataclass_fields__
            if not validity[name]
        ]
        if health.get("stale_or_invalid_components") != invalid_components:
            raise DecisionContractError(
                "health stale component list must match validity flags"
            )
        if health.get("mode") == "NOMINAL" and invalid_components:
            raise DecisionContractError(
                "NOMINAL health cannot contain invalid components"
            )
        if health.get("ttl_ms") != self.ttl_ms:
            raise DecisionContractError("health.ttl_ms must match envelope ttl_ms")
        ages = _require_mapping(
            health.get("component_age_ms"), field="health.component_age_ms"
        )
        for name, age in ages.items():
            if not isinstance(name, str) or not name or not _is_int(age) or age < 0:
                raise DecisionContractError(
                    "health.component_age_ms must map names to non-negative integers"
                )
        versions = _require_mapping(
            health.get("model_versions"), field="health.model_versions"
        )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in versions.items()
        ):
            raise DecisionContractError(
                "health.model_versions must map non-empty strings"
            )

        # Last defense against nested non-finite values and unsupported JSON
        # objects.  The wire encoder below repeats this with the final payload.
        try:
            json.dumps(self.to_dict(), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DecisionContractError(
                "decision envelope contains a non-JSON or non-finite value"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_mode": self.source_mode,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "frame_id": self.frame_id,
            "source_timestamp_ms": self.source_timestamp_ms,
            "decision_timestamp_ms": self.decision_timestamp_ms,
            "ttl_ms": self.ttl_ms,
            "expires_at_ms": self.expires_at_ms,
            "validity": dict(self.validity),
            "c1": dict(self.c1),
            "c2": dict(self.c2),
            "c3": dict(self.c3),
            "drive_quality": dict(self.drive_quality),
            "contextual_risk": dict(self.contextual_risk),
            "health": dict(self.health),
        }

    def to_json_bytes(self) -> bytes:
        # Mapping fields remain interoperable with plain dicts. Revalidate at
        # the wire boundary so post-construction mutation cannot bypass the
        # safety contract of this frozen envelope.
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
            raise DecisionContractError(
                "decision envelope cannot be encoded as strict JSON"
            ) from exc
        return encoded.encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes | bytearray | memoryview) -> "DecisionEnvelope":
        try:
            decoded = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecisionContractError("invalid UTF-8 decision JSON") from exc
        if not isinstance(decoded, dict):
            raise DecisionContractError("decision JSON root must be an object")
        required = set(cls.__dataclass_fields__)
        if set(decoded) != required:
            raise DecisionContractError(
                "decision JSON keys must be exactly " + ", ".join(sorted(required))
            )
        try:
            return cls(**decoded)
        except TypeError as exc:
            raise DecisionContractError("invalid decision JSON structure") from exc

    def is_expired(self, now_ms: int) -> bool:
        if not _is_int(now_ms) or now_ms < 0:
            raise DecisionContractError("now_ms must be a non-negative integer")
        return now_ms >= self.expires_at_ms


class DecisionEnvelopeBuilder:
    """Build ordered envelopes from synchronized unified-replay predictions."""

    def __init__(
        self,
        *,
        session_id: str,
        source_mode: str,
        source_fps: float = 20.0,
        ttl_ms: int = DEFAULT_TTL_MS,
        model_versions: Mapping[str, str] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        # Validate immutable configuration through a small dummy-independent
        # path so errors occur before the first live frame.
        if not isinstance(session_id, str) or not session_id.strip():
            raise DecisionContractError("session_id must be a non-empty string")
        if len(session_id) > 128:
            raise DecisionContractError("session_id is too long")
        if source_mode not in SOURCE_MODES:
            raise DecisionContractError(
                f"source_mode must be one of {sorted(SOURCE_MODES)}"
            )
        self.source_fps = _finite_number(
            source_fps, field="source_fps", minimum=0.001
        )
        if not _is_int(ttl_ms) or not 1 <= ttl_ms <= MAX_TTL_MS:
            raise DecisionContractError(
                f"ttl_ms must be an integer in [1, {MAX_TTL_MS}]"
            )
        versions = dict(model_versions or {})
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in versions.items()
        ):
            raise DecisionContractError(
                "model_versions must map non-empty strings to non-empty strings"
            )
        self.session_id = session_id
        self.source_mode = source_mode
        self.ttl_ms = ttl_ms
        self.model_versions = versions
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._sequence = 0
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ms: int | None = None
        self._last_decision_timestamp_ms: int | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _finite_ttc_ms(value: object, *, c1_valid: bool) -> tuple[int | None, bool]:
        if not c1_valid:
            return None, False
        if isinstance(value, bool):
            raise DecisionContractError("C1 TTC must be numeric")
        try:
            ttc = float(value)
        except (TypeError, ValueError) as exc:
            raise DecisionContractError("C1 TTC must be numeric") from exc
        if math.isnan(ttc) or ttc == float("-inf"):
            raise DecisionContractError("C1 TTC must be non-negative or +inf")
        if ttc == float("inf"):
            return None, False
        if ttc < 0.0:
            raise DecisionContractError("C1 TTC must be non-negative or +inf")
        return round(ttc * 1_000.0), True

    def build(
        self,
        frame: Any,
        *,
        validity: DecisionValidity,
        decision_timestamp_ms: int | None = None,
        runtime_mode: str | None = None,
        component_age_ms: Mapping[str, int] | None = None,
    ) -> DecisionEnvelope:
        if not isinstance(validity, DecisionValidity):
            raise DecisionContractError("validity must be DecisionValidity")
        frame_id = frame.frame_id
        if not _is_int(frame_id) or frame_id < 0:
            raise DecisionContractError("frame_id must be a non-negative integer")
        source_timestamp = _finite_number(
            frame.timestamp, field="frame.timestamp", minimum=0.0
        )
        source_timestamp_ms = round(source_timestamp * 1_000.0)
        decided_ms = (
            self._clock_ms()
            if decision_timestamp_ms is None
            else decision_timestamp_ms
        )
        if not _is_int(decided_ms) or decided_ms < 0:
            raise DecisionContractError(
                "decision_timestamp_ms must be a non-negative integer"
            )

        c1 = frame.c1
        model_frame_id = c1.model_frame_id
        if not _is_int(model_frame_id) or model_frame_id < 0:
            raise DecisionContractError(
                "c1.model_frame_id must be a non-negative integer"
            )
        if model_frame_id > frame_id:
            raise DecisionContractError("c1.model_frame_id cannot exceed frame_id")
        ttc_ms, ttc_valid = self._finite_ttc_ms(
            c1.predicted_ttc_s, c1_valid=validity.c1
        )
        c1_age_ms = max(
            0,
            round(
                (frame_id - model_frame_id)
                * 1_000.0
                / self.source_fps
            ),
        )
        c1_payload = {
            "ttc_ms": ttc_ms,
            "ttc_valid": ttc_valid,
            "collision_probability_pct": (
                _probability_percentage(
                    c1.collision_probability,
                    field="c1.collision_probability",
                )
                if validity.c1
                else None
            ),
            "warning": bool(c1.is_warning) if validity.c1 else False,
            "model_updated": bool(c1.model_updated),
            "model_frame_id": model_frame_id,
            "age_ms": c1_age_ms,
        }

        c2 = frame.c2
        signals = c2.vss_signals()
        c2_payload = {
            "state": str(c2.state) if validity.c2 else "unavailable",
            "confidence_pct": (
                _probability_percentage(c2.confidence, field="c2.confidence")
                if validity.c2
                else None
            ),
            "attentive_probability_pct": _optional_percentage(
                signals.attentive_probability,
                valid=validity.c2,
                field="c2.attentive_probability",
            ),
            "distraction_level_pct": _optional_percentage(
                signals.distraction_level,
                valid=validity.c2,
                field="c2.distraction_level",
            ),
            "fatigue_level_pct": _optional_percentage(
                signals.fatigue_level,
                valid=validity.c2,
                field="c2.fatigue_level",
            ),
            "eyes_on_road": bool(signals.is_eyes_on_road) if validity.c2 else None,
            "warning": bool(signals.is_warning) if validity.c2 else False,
        }

        c3 = frame.c3
        c3_payload = {
            "safe_score_estimate_pct": (
                _percentage(
                    c3.safe_score_estimate,
                    field="c3.safe_score_estimate",
                )
                if validity.c3
                else None
            ),
            "grade": str(c3.grade) if validity.c3 else "N/A",
            "scope": (
                "FULL_TRIP" if bool(c3.trip_complete) else "PREFIX"
            ),
            "formula_version": str(c3.formula_version),
            "tailgating_penalty_omitted": bool(
                c3.tailgating_penalty_omitted
            ),
        }

        quality = frame.drive_quality
        quality_available = bool(quality.score_available) and validity.drive_quality
        quality_payload = {
            "score_available": quality_available,
            "score_pct": (
                _percentage(
                    quality.score_pct,
                    field="drive_quality.score_pct",
                )
                if quality_available
                else None
            ),
            "grade": str(quality.grade) if quality_available else "N/A",
            "scope": str(quality.scope) if validity.drive_quality else "NO_DATA",
            "window_ready": bool(quality.window_ready) if validity.drive_quality else False,
            "formula_version": str(quality.formula_version),
        }

        risk = frame.contextual_risk
        if validity.contextual_risk:
            risk_payload = {
                "score_pct": _percentage(
                    risk.score_pct, field="contextual_risk.score_pct"
                ),
                "level": str(risk.level),
                "action": str(risk.action),
                "brake_request_pct": _percentage(
                    risk.brake_request_pct,
                    field="contextual_risk.brake_request_pct",
                ),
                "reasons": [str(reason) for reason in risk.reasons],
                "actuation_authorized": False,
            }
        else:
            risk_payload = {
                "score_pct": None,
                "level": "UNAVAILABLE",
                "action": "MONITOR",
                "brake_request_pct": 0.0,
                "reasons": ["STALE_OR_INVALID_INPUT"],
                "actuation_authorized": False,
            }

        invalid = list(validity.invalid_components())
        if runtime_mode is None:
            mode = "NOMINAL" if not invalid else "DEGRADED"
        else:
            mode = str(runtime_mode).upper()
            if mode not in HEALTH_MODES:
                raise DecisionContractError(
                    f"runtime_mode must be one of {sorted(HEALTH_MODES)}"
                )
        ages = {"c1": c1_age_ms, "c2": 0, "ego": 0}
        for name, age in dict(component_age_ms or {}).items():
            if not isinstance(name, str) or not name:
                raise DecisionContractError(
                    "component_age_ms keys must be non-empty strings"
                )
            if not _is_int(age) or age < 0:
                raise DecisionContractError(
                    f"component_age_ms.{name} must be a non-negative integer"
                )
            ages[name] = age
        health_payload = {
            "mode": mode,
            "decision_valid": validity.contextual_risk,
            "stale_or_invalid_components": invalid,
            "component_age_ms": ages,
            "ttl_ms": self.ttl_ms,
            "model_versions": dict(self.model_versions),
        }

        with self._lock:
            if self._last_frame_id is not None and frame_id <= self._last_frame_id:
                raise DecisionContractError(
                    "frame_id must increase strictly within a decision session"
                )
            if (
                self._last_source_timestamp_ms is not None
                and source_timestamp_ms < self._last_source_timestamp_ms
            ):
                raise DecisionContractError(
                    "source timestamp cannot move backwards within a decision session"
                )
            if (
                self._last_decision_timestamp_ms is not None
                and decided_ms < self._last_decision_timestamp_ms
            ):
                raise DecisionContractError(
                    "decision timestamp cannot move backwards within a decision session"
                )
            sequence = self._sequence
            envelope = DecisionEnvelope(
                schema_version=SCHEMA_VERSION,
                source_mode=self.source_mode,
                session_id=self.session_id,
                sequence=sequence,
                frame_id=frame_id,
                source_timestamp_ms=source_timestamp_ms,
                decision_timestamp_ms=decided_ms,
                ttl_ms=self.ttl_ms,
                expires_at_ms=decided_ms + self.ttl_ms,
                validity=asdict(validity),
                c1=c1_payload,
                c2=c2_payload,
                c3=c3_payload,
                drive_quality=quality_payload,
                contextual_risk=risk_payload,
                health=health_payload,
            )
            # Failed envelope construction must not consume a wire sequence.
            self._sequence += 1
            self._last_frame_id = frame_id
            self._last_source_timestamp_ms = source_timestamp_ms
            self._last_decision_timestamp_ms = decided_ms
            return envelope


__all__ = [
    "C3_SCOPES",
    "DEFAULT_TTL_MS",
    "DRIVER_STATES",
    "DRIVE_QUALITY_SCOPES",
    "DecisionContractError",
    "DecisionEnvelope",
    "DecisionEnvelopeBuilder",
    "DecisionValidity",
    "GRADES",
    "HEALTH_MODES",
    "MAX_TTL_MS",
    "RISK_ACTIONS",
    "RISK_LEVELS",
    "SCHEMA_VERSION",
    "SOURCE_MODES",
]
