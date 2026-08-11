"""Translate live-model snapshots into strict decision-v2 envelopes.

This module is the product boundary between the synchronized inference core
and an asynchronous MQTT/KUKSA publisher.  It never reads media, datasets,
labels, depth, events, targets, or prediction files.  It preserves source
capture age and generation ordering while projecting model-native objects into
the exact warning-only :mod:`safeloop.carsky_decision_v2` contract.
"""

from __future__ import annotations

from copy import deepcopy
import math
from threading import Lock
from typing import Any, Mapping

from .carsky_decision_v2 import (
    DEFAULT_TTL_MS,
    INFERENCE_MODE,
    VIDEO_SOURCES,
    DecisionEnvelopeV2,
    DecisionV2ContractError,
)
from .carsky_live import LiveRuntimeSnapshot


_HEARTBEAT_REASON_MAP = {
    "COHERENCE_DWELL": "INITIALIZING",
    "NO_SESSION": "INITIALIZING",
    "RESET_FAILED": "MODEL_ERROR",
    "MODEL_EXCEPTION": "MODEL_ERROR",
    "CUDA_OOM": "CUDA_OOM",
    "RESTART": "WORKER_RESTART",
    "STATE_LOSS": "WORKER_RESTART",
    "ENDED": "WORKER_RESTART",
    "SOURCE_GAP": "STREAM_GAP",
    "BACKLOG_OVERWRITE": "STREAM_GAP",
    "INFERENCE_BACKLOG": "STREAM_GAP",
    "CAPTURE_SKEW": "STREAM_GAP",
    "CAPTURE_REORDER": "STREAM_GAP",
}


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecisionV2ContractError(f"{field} must be a non-negative integer")
    return value


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent_from_probability(value: object) -> float | None:
    number = _finite(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return round(number * 100.0, 3)


def _percent(value: object) -> float | None:
    number = _finite(value)
    if number is None or not 0.0 <= number <= 100.0:
        return None
    return round(number, 3)


def _neutral_payloads() -> dict[str, dict[str, Any]]:
    return {
        "c1": {
            "ttc_ms": None,
            "collision_probability_pct": None,
            "warning": False,
            "model_updated": False,
        },
        "c2": {
            "state": "unavailable",
            "confidence_pct": None,
            "attentive_probability_pct": None,
            "distraction_level_pct": None,
            "fatigue_level_pct": None,
            "eyes_on_road": None,
            "warning": False,
        },
        "c3": {
            "safe_score_estimate_pct": None,
            "grade": "N/A",
            "scope": "NO_DATA",
        },
        "drive_quality": {
            "score_pct": None,
            "grade": "N/A",
            "scope": "NO_DATA",
        },
        "contextual_risk": {
            "score_pct": None,
            "level": "UNAVAILABLE",
            "action": "MONITOR",
            "brake_request_pct": 0.0,
            "reasons": ["STALE_OR_INVALID_INPUT"],
        },
    }


def _prediction_payloads(
    prediction: Any,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    payloads = _neutral_payloads()
    validity = {
        "ego": True,
        "front_camera": True,
        "cabin_camera": True,
        "c1": True,
        "c2": True,
        "c3": True,
        "drive_quality": True,
        "contextual_risk": True,
    }

    c1 = prediction.c1
    ttc_seconds = _finite(getattr(c1, "predicted_ttc_s", None))
    collision_pct = _percent_from_probability(
        getattr(c1, "collision_probability", None)
    )
    validity["c1"] = (
        ttc_seconds is not None
        and ttc_seconds >= 0.0
        and collision_pct is not None
    )
    payloads["c1"] = {
        # Passing a non-finite sentinel through create() deliberately invokes
        # its null+invalid fail-safe; no NaN/Infinity reaches JSON.
        "ttc_ms": (
            round(ttc_seconds * 1_000.0)
            if ttc_seconds is not None and ttc_seconds >= 0.0
            else float("inf")
        ),
        "collision_probability_pct": collision_pct,
        "warning": bool(getattr(c1, "is_warning", False)),
        "model_updated": bool(getattr(c1, "model_updated", False)),
    }

    c2 = prediction.c2
    try:
        signals = c2.vss_signals()
    except Exception:
        signals = None
    c2_values = {
        "state": str(getattr(c2, "state", "unavailable")),
        "confidence_pct": _percent_from_probability(
            getattr(c2, "confidence", None)
        ),
        "attentive_probability_pct": _percent(
            getattr(signals, "attentive_probability", None)
        ),
        "distraction_level_pct": _percent(
            getattr(signals, "distraction_level", None)
        ),
        "fatigue_level_pct": _percent(
            getattr(signals, "fatigue_level", None)
        ),
        "eyes_on_road": (
            getattr(signals, "is_eyes_on_road", None)
            if isinstance(getattr(signals, "is_eyes_on_road", None), bool)
            else None
        ),
        "warning": bool(getattr(signals, "is_warning", False)),
    }
    validity["c2"] = (
        c2_values["state"] != "unavailable"
        and all(
            c2_values[name] is not None
            for name in (
                "confidence_pct",
                "attentive_probability_pct",
                "distraction_level_pct",
                "fatigue_level_pct",
                "eyes_on_road",
            )
        )
    )
    payloads["c2"] = c2_values

    c3 = prediction.c3
    c3_score = _percent(getattr(c3, "safe_score_estimate", None))
    c3_grade = str(getattr(c3, "grade", "N/A"))
    c3_scope = "FULL_TRIP" if bool(getattr(c3, "trip_complete", False)) else "PREFIX"
    validity["c3"] = c3_score is not None and c3_grade != "N/A"
    payloads["c3"] = {
        "safe_score_estimate_pct": c3_score,
        "grade": c3_grade,
        "scope": c3_scope,
    }

    quality = prediction.drive_quality
    quality_score = _percent(getattr(quality, "score_pct", None))
    quality_grade = str(getattr(quality, "grade", "N/A"))
    quality_scope = str(getattr(quality, "scope", "NO_DATA"))
    validity["drive_quality"] = (
        bool(getattr(quality, "score_available", False))
        and quality_score is not None
        and quality_grade != "N/A"
        and quality_scope != "NO_DATA"
    )
    payloads["drive_quality"] = {
        "score_pct": quality_score,
        "grade": quality_grade,
        "scope": quality_scope,
    }

    risk = prediction.contextual_risk
    risk_score = _percent(getattr(risk, "score_pct", None))
    raw_reasons = getattr(risk, "reasons", ())
    try:
        reasons = [str(reason) for reason in raw_reasons]
    except TypeError:
        reasons = []
    payloads["contextual_risk"] = {
        "score_pct": risk_score,
        "level": str(getattr(risk, "level", "UNAVAILABLE")),
        "action": str(getattr(risk, "action", "MONITOR")),
        "brake_request_pct": getattr(risk, "brake_request_pct", 0.0),
        "reasons": reasons,
    }
    validity["contextual_risk"] = (
        risk_score is not None
        and payloads["contextual_risk"]["level"] != "UNAVAILABLE"
    )
    return validity, payloads


def _health_reason(value: str) -> str:
    normalized = value.split(":", 1)[0].strip().upper()
    if normalized in _HEARTBEAT_REASON_MAP:
        return _HEARTBEAT_REASON_MAP[normalized]
    if "GAP" in normalized or "BACKLOG" in normalized or "SKEW" in normalized:
        return "STREAM_GAP"
    if "OOM" in normalized:
        return "CUDA_OOM"
    if "MODEL" in normalized or "RESET_FAILED" in value.upper():
        return "MODEL_ERROR"
    return "WORKER_RESTART"


class LiveDecisionV2Builder:
    """Stateful generation/order guard for live decision-v2 production."""

    def __init__(
        self,
        *,
        video_source: str,
        models: Mapping[str, Mapping[str, str]],
        ttl_ms: int = DEFAULT_TTL_MS,
        clock_status: str = "UNKNOWN",
        clock_offset_uncertainty_ms: int = 60_000,
    ) -> None:
        if video_source not in VIDEO_SOURCES:
            raise DecisionV2ContractError(
                f"video_source must be one of {sorted(VIDEO_SOURCES)}"
            )
        self.video_source = video_source
        self.models = deepcopy(dict(models))
        self.ttl_ms = _non_negative_int(ttl_ms, field="ttl_ms")
        if not 1 <= self.ttl_ms <= 1_000:
            raise DecisionV2ContractError("ttl_ms must be in [1, 1000]")
        self._lock = Lock()
        self._session_id: str | None = None
        self._generation: int | None = None
        self._last_runtime_sequence: int | None = None
        self._last_source_sequence: int | None = None
        self._last_capture_timestamp_ms: int | None = None
        self._last_source_media_timestamp_ms: int | None = None
        self.set_clock_health(
            clock_status,
            offset_uncertainty_ms=clock_offset_uncertainty_ms,
        )

    def set_clock_health(
        self, status: str, *, offset_uncertainty_ms: int
    ) -> None:
        if status not in {"SYNCHRONIZED", "UNHEALTHY", "UNKNOWN"}:
            raise DecisionV2ContractError("unsupported clock status")
        uncertainty = _non_negative_int(
            offset_uncertainty_ms, field="clock_offset_uncertainty_ms"
        )
        if uncertainty > 60_000:
            raise DecisionV2ContractError(
                "clock_offset_uncertainty_ms must be <= 60000"
            )
        with self._lock:
            self._clock_status = status
            self._clock_offset_uncertainty_ms = uncertainty

    def start_generation(self, session_id: str, generation: int) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise DecisionV2ContractError("session_id must be non-empty")
        generation_value = _non_negative_int(generation, field="generation")
        with self._lock:
            self._session_id = session_id
            self._generation = generation_value
            self._last_runtime_sequence = None
            self._last_source_sequence = None
            self._last_capture_timestamp_ms = None
            self._last_source_media_timestamp_ms = None

    def build(
        self,
        snapshot: LiveRuntimeSnapshot,
        *,
        server_receive_timestamp_ms: int,
        decision_timestamp_ms: int,
        source_media_timestamp_ms: int | None = None,
    ) -> DecisionEnvelopeV2:
        if not isinstance(snapshot, LiveRuntimeSnapshot):
            raise TypeError("snapshot must be a LiveRuntimeSnapshot")
        received = _non_negative_int(
            server_receive_timestamp_ms, field="server_receive_timestamp_ms"
        )
        decided = _non_negative_int(
            decision_timestamp_ms, field="decision_timestamp_ms"
        )
        if source_media_timestamp_ms is not None:
            source_media_timestamp_ms = _non_negative_int(
                source_media_timestamp_ms,
                field="source_media_timestamp_ms",
            )

        with self._lock:
            if snapshot.session_id != self._session_id:
                raise DecisionV2ContractError(
                    "runtime snapshot belongs to a stale or foreign session"
                )
            if snapshot.generation != self._generation:
                raise DecisionV2ContractError(
                    "runtime snapshot belongs to a stale or future generation"
                )
            if (
                self._last_runtime_sequence is not None
                and snapshot.sequence <= self._last_runtime_sequence
            ):
                raise DecisionV2ContractError(
                    "runtime sequence must increase within a generation"
                )

            if snapshot.source_sequence is None:
                source_sequence = self._last_source_sequence or 0
            else:
                source_sequence = snapshot.source_sequence
                if (
                    self._last_source_sequence is not None
                    and source_sequence <= self._last_source_sequence
                ):
                    raise DecisionV2ContractError(
                        "source_sequence must increase for live decisions"
                    )

            if snapshot.capture_timestamp_ms is None:
                capture = self._last_capture_timestamp_ms or received
                media_timestamp = self._last_source_media_timestamp_ms
            else:
                capture = snapshot.capture_timestamp_ms
                if (
                    self._last_capture_timestamp_ms is not None
                    and capture <= self._last_capture_timestamp_ms
                ):
                    raise DecisionV2ContractError(
                        "capture timestamp must increase for live decisions"
                    )
                media_timestamp = source_media_timestamp_ms

            if snapshot.prediction is None:
                validity = {name: False for name in (
                    "ego",
                    "front_camera",
                    "cabin_camera",
                    "c1",
                    "c2",
                    "c3",
                    "drive_quality",
                    "contextual_risk",
                )}
                payloads = _neutral_payloads()
            else:
                validity, payloads = _prediction_payloads(snapshot.prediction)
                if snapshot.health != "LIVE":
                    # Temporal state is still warming. Component diagnostics
                    # may be carried, but contextual warning validity is held
                    # false until the coherent dwell completes.
                    validity["contextual_risk"] = False

            health_reasons = (
                []
                if snapshot.health == "LIVE"
                else [_health_reason(snapshot.reason)]
            )
            envelope = DecisionEnvelopeV2.create(
                session_id=snapshot.session_id,
                generation=snapshot.generation,
                sequence=snapshot.sequence,
                source_sequence=source_sequence,
                video_source=self.video_source,
                capture_timestamp_ms=capture,
                source_media_timestamp_ms=media_timestamp,
                server_receive_timestamp_ms=received,
                decision_timestamp_ms=decided,
                clock_status=self._clock_status,
                clock_offset_uncertainty_ms=self._clock_offset_uncertainty_ms,
                validity=validity,
                c1=payloads["c1"],
                c2=payloads["c2"],
                c3=payloads["c3"],
                drive_quality=payloads["drive_quality"],
                contextual_risk=payloads["contextual_risk"],
                models=self.models,
                ttl_ms=self.ttl_ms,
                health_reasons=health_reasons,
            )

            self._last_runtime_sequence = snapshot.sequence
            if snapshot.source_sequence is not None:
                self._last_source_sequence = source_sequence
            if snapshot.capture_timestamp_ms is not None:
                self._last_capture_timestamp_ms = capture
                self._last_source_media_timestamp_ms = media_timestamp
            return envelope


__all__ = ["INFERENCE_MODE", "LiveDecisionV2Builder"]
