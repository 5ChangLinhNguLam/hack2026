from __future__ import annotations

from types import SimpleNamespace

import pytest

from safeloop.carsky_decision_v2 import DecisionV2ContractError
from safeloop.carsky_live import LiveRuntimeSnapshot
from safeloop.live_decision import LiveDecisionV2Builder


MODELS = {
    "c1": {"version": "c1", "digest_sha256": "a" * 64},
    "c2": {"version": "c2", "digest_sha256": "b" * 64},
    "c3": {"version": "c3", "digest_sha256": "c" * 64},
    "drive_quality": {"version": "dq", "digest_sha256": "d" * 64},
    "contextual_risk": {"version": "risk", "digest_sha256": "e" * 64},
}


def prediction(*, ttc: float = 1.8, emergency: bool = False):
    signals = SimpleNamespace(
        attentive_probability=20.0,
        distraction_level=82.0,
        fatigue_level=10.0,
        is_eyes_on_road=False,
        is_warning=True,
    )
    return SimpleNamespace(
        c1=SimpleNamespace(
            predicted_ttc_s=ttc,
            collision_probability=0.83,
            is_warning=True,
            model_updated=True,
        ),
        c2=SimpleNamespace(
            state="distracted",
            confidence=0.91,
            vss_signals=lambda: signals,
        ),
        c3=SimpleNamespace(
            safe_score_estimate=88.0,
            grade="B",
            trip_complete=False,
        ),
        drive_quality=SimpleNamespace(
            score_available=True,
            score_pct=91.0,
            grade="A",
            scope="PREFIX",
        ),
        contextual_risk=SimpleNamespace(
            score_pct=84.0,
            level="CRITICAL" if emergency else "HIGH",
            action=(
                "EMERGENCY_BRAKE_REQUEST"
                if emergency
                else "VISUAL_AUDIO_HAPTIC_WARNING"
            ),
            brake_request_pct=70.0 if emergency else 0.0,
            reasons=("LOW_TTC",),
        ),
    )


def builder(**kwargs) -> LiveDecisionV2Builder:
    return LiveDecisionV2Builder(
        video_source="RECORDED_STREAM",
        models=MODELS,
        clock_status="SYNCHRONIZED",
        clock_offset_uncertainty_ms=2,
        **kwargs,
    )


def runtime_snapshot(
    *,
    sequence: int = 1,
    source_sequence: int | None = 7,
    capture_timestamp_ms: int | None = 1_700_000_000_000,
    health: str = "LIVE",
    reason: str = "",
    value=...,
) -> LiveRuntimeSnapshot:
    selected_prediction = prediction() if value is ... else value
    return LiveRuntimeSnapshot(
        session_id="drive-a",
        generation=3,
        sequence=sequence,
        health=health,
        reason=reason,
        source_sequence=source_sequence,
        capture_timestamp_ms=capture_timestamp_ms,
        prediction=selected_prediction,
        warning_action=(
            "VISUAL_AUDIO_HAPTIC_WARNING" if health == "LIVE" else "MONITOR"
        ),
    )


def test_live_prediction_maps_to_strict_capture_aged_v2() -> None:
    build = builder()
    build.start_generation("drive-a", 3)
    envelope = build.build(
        runtime_snapshot(),
        source_media_timestamp_ms=350,
        server_receive_timestamp_ms=1_700_000_000_008,
        decision_timestamp_ms=1_700_000_000_061,
    )

    assert envelope.video_source == "RECORDED_STREAM"
    assert envelope.inference_mode == "LIVE_MODEL"
    assert envelope.source_age_ms == 61
    assert envelope.expires_at_ms == envelope.capture_timestamp_ms + envelope.ttl_ms
    assert envelope.source_media_timestamp_ms == 350
    assert envelope.c1["ttc_ms"] == 1_800
    assert envelope.c1["collision_probability_pct"] == 83.0
    assert envelope.c2["distraction_level_pct"] == 82.0
    assert envelope.contextual_risk["action"] == "VISUAL_AUDIO_HAPTIC_WARNING"
    assert envelope.contextual_risk["brake_request_pct"] == 0.0
    assert envelope.health == {"status": "NOMINAL", "reasons": []}


def test_coherence_dwell_is_degraded_and_suppresses_context_warning() -> None:
    build = builder()
    build.start_generation("drive-a", 3)
    envelope = build.build(
        runtime_snapshot(
            sequence=0,
            source_sequence=0,
            health="DEGRADED",
            reason="COHERENCE_DWELL",
        ),
        source_media_timestamp_ms=0,
        server_receive_timestamp_ms=1_700_000_000_010,
        decision_timestamp_ms=1_700_000_000_050,
    )

    assert envelope.health["status"] == "DEGRADED"
    assert "INITIALIZING" in envelope.health["reasons"]
    assert envelope.validity["c1"] is True
    assert envelope.validity["contextual_risk"] is False
    assert envelope.contextual_risk["action"] == "MONITOR"


def test_nonfinite_ttc_is_null_invalid_and_invalidates_dependencies() -> None:
    build = builder()
    build.start_generation("drive-a", 3)
    snapshot = runtime_snapshot()
    snapshot.prediction.c1.predicted_ttc_s = float("inf")
    envelope = build.build(
        snapshot,
        server_receive_timestamp_ms=1_700_000_000_010,
        decision_timestamp_ms=1_700_000_000_050,
    )

    assert envelope.c1["ttc_ms"] is None
    assert envelope.validity["c1"] is False
    assert envelope.validity["c3"] is False
    assert envelope.validity["drive_quality"] is False
    assert envelope.validity["contextual_risk"] is False
    assert envelope.health["status"] == "DEGRADED"


def test_degraded_heartbeat_reuses_last_source_position_without_warning() -> None:
    build = builder()
    build.start_generation("drive-a", 3)
    build.build(
        runtime_snapshot(sequence=0, source_sequence=5),
        source_media_timestamp_ms=250,
        server_receive_timestamp_ms=1_700_000_000_005,
        decision_timestamp_ms=1_700_000_000_020,
    )
    heartbeat = build.build(
        runtime_snapshot(
            sequence=1,
            source_sequence=None,
            capture_timestamp_ms=None,
            health="DEGRADED",
            reason="SOURCE_GAP",
            value=None,
        ),
        server_receive_timestamp_ms=1_700_000_000_030,
        decision_timestamp_ms=1_700_000_000_040,
    )

    assert heartbeat.source_sequence == 5
    assert heartbeat.capture_timestamp_ms == 1_700_000_000_000
    assert heartbeat.source_media_timestamp_ms == 250
    assert not any(heartbeat.validity.values())
    assert heartbeat.health["status"] == "DEGRADED"
    assert "STREAM_GAP" in heartbeat.health["reasons"]
    assert heartbeat.contextual_risk["action"] == "MONITOR"


def test_builder_fences_generation_and_runtime_order() -> None:
    build = builder()
    build.start_generation("drive-a", 3)
    build.build(
        runtime_snapshot(sequence=2),
        server_receive_timestamp_ms=1_700_000_000_001,
        decision_timestamp_ms=1_700_000_000_010,
    )
    with pytest.raises(DecisionV2ContractError, match="runtime sequence"):
        build.build(
            runtime_snapshot(sequence=2, source_sequence=8, capture_timestamp_ms=1_700_000_000_050),
            server_receive_timestamp_ms=1_700_000_000_051,
            decision_timestamp_ms=1_700_000_000_060,
        )

    build.start_generation("drive-a", 4)
    stale = runtime_snapshot(sequence=0)
    with pytest.raises(DecisionV2ContractError, match="generation"):
        build.build(
            stale,
            server_receive_timestamp_ms=1_700_000_000_001,
            decision_timestamp_ms=1_700_000_000_010,
        )


def test_emergency_vocabulary_is_rejected_at_v2_boundary() -> None:
    build = builder()
    build.start_generation("drive-a", 3)
    snapshot = runtime_snapshot(value=prediction(emergency=True))
    with pytest.raises(DecisionV2ContractError, match="warning-only|vocabulary"):
        build.build(
            snapshot,
            server_receive_timestamp_ms=1_700_000_000_010,
            decision_timestamp_ms=1_700_000_000_050,
        )
