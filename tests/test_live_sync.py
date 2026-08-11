from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from safeloop.carsky_live import (
    CabinFrameEnvelope,
    EgoTelemetryEnvelope,
    LatestRuntimeSnapshotSlot,
    LiveFrameInput,
    LiveInferenceController,
    LiveInferenceSession,
    LiveInputSynchronizer,
    RoadFrameEnvelope,
    RtpFrameIdentity,
    SourceTickKey,
)


EGO = {
    "speed_kmh": 42.0,
    "longitudinal_accel": 0.2,
    "lateral_accel": -0.1,
}


def pixels(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value % 256, dtype=np.uint8)


@dataclass(frozen=True)
class Prediction:
    frame_id: int
    timestamp: float
    value: object
    challenge: str
    model_updated: bool = False

    @property
    def predicted_ttc_s(self) -> float:
        return float(self.value)

    def submission_row(self) -> dict[str, object]:
        if self.challenge == "c1":
            return {
                "frame_id": self.frame_id,
                "timestamp": self.timestamp,
                "predicted_ttc": self.value,
            }
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "predicted_driver_state": self.value,
        }


class Resettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class C1(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[np.ndarray, int]] = []

    def process_bgr(self, image, *, frame_id, timestamp, ego, target_count):
        self.calls.append((image, frame_id))
        assert target_count == 0
        assert dict(ego) == EGO
        return Prediction(
            frame_id,
            timestamp,
            1.0,
            "c1",
            model_updated=frame_id % 2 == 0,
        )


class C2(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[np.ndarray, int]] = []

    def process_bgr(self, image, *, frame_id, timestamp):
        self.calls.append((image, frame_id))
        return Prediction(frame_id, timestamp, "alert", "c2")


class C3(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update(self, bundle, *, predicted_ttc_s):
        self.calls += 1
        assert predicted_ttc_s == 1.0
        return Prediction(bundle.frame_id, bundle.timestamp, 90.0, "c3")


class DriveQuality(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update(self, c3):
        self.calls += 1
        return Prediction(c3.frame_id, c3.timestamp, 88.0, "dq")


class Risk(Resettable):
    def __init__(self, *, emergency: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self.emergency = emergency

    def evaluate(self, _c1, _c2):
        self.calls += 1
        return type(
            "RiskDecision",
            (),
            {
                "score_pct": 95.0 if self.emergency else 20.0,
                "level": "CRITICAL" if self.emergency else "SAFE",
                "action": (
                    "EMERGENCY_BRAKE_REQUEST"
                    if self.emergency
                    else "MONITOR"
                ),
                "brake_request_pct": 70.0 if self.emergency else 0.0,
                "reasons": ("LOW_TTC",) if self.emergency else ("NORMAL",),
            },
        )()


def parts(*, emergency: bool = False):
    return C1(), C2(), C3(), DriveQuality(), Risk(emergency=emergency)


def controller_from(parts_, *, dwell: int = 2) -> LiveInferenceController:
    c1, c2, c3, dq, risk = parts_
    return LiveInferenceController(
        LiveInferenceSession(
            c1=c1,
            c2=c2,
            c3=c3,
            drive_quality=dq,
            contextual_risk=risk,
        ),
        coherent_dwell_ticks=dwell,
    )


def source_envelopes(
    sequence: int,
    *,
    generation: int = 0,
    session_id: str = "live",
    capture_ms: int | None = None,
    skew: tuple[int, int, int] = (0, 0, 0),
    road: np.ndarray | None = None,
    cabin: np.ndarray | None = None,
):
    key = SourceTickKey(session_id, generation, sequence)
    captured = sequence * 50 + 1_000 if capture_ms is None else capture_ms
    road_frame = RoadFrameEnvelope(
        RtpFrameIdentity((90_000 + sequence * 4_500) & 0xFFFFFFFF, key),
        captured + skew[0],
        pixels(sequence) if road is None else road,
    )
    cabin_frame = CabinFrameEnvelope(
        # A different RTP clock value still maps explicitly to the same key.
        RtpFrameIdentity((12_345 + sequence * 4_500) & 0xFFFFFFFF, key),
        captured + skew[1],
        pixels(sequence + 100) if cabin is None else cabin,
    )
    ego = EgoTelemetryEnvelope(key, captured + skew[2], EGO)
    return road_frame, cabin_frame, ego


def offer_tick(controller, sequence: int, *, generation: int, **kwargs):
    road, cabin, ego = source_envelopes(
        sequence, generation=generation, **kwargs
    )
    assert controller.offer_cabin(cabin).disposition == "PENDING"
    assert controller.offer_ego(ego).disposition == "PENDING"
    result = controller.offer_road(road)
    assert result.disposition == "READY"
    return road, cabin, ego


def test_decoder_boundary_copies_once_and_inference_receives_same_references() -> None:
    raw_road = pixels(7)
    raw_cabin = pixels(8)
    road, cabin, ego = source_envelopes(
        10,
        generation=4,
        road=raw_road,
        cabin=raw_cabin,
        skew=(0, 12, 20),
    )
    raw_road.fill(99)
    raw_cabin.fill(99)

    runtime_parts = parts()
    controller = controller_from(runtime_parts, dwell=1)
    controller.start_session("live", generation=4)
    controller.offer_ego(ego)
    controller.offer_road(road)
    ready = controller.offer_cabin(cabin)
    assert ready.disposition == "READY"
    prediction = controller.process_latest()

    assert prediction is not None
    assert np.array_equal(road.bgr, pixels(7))
    assert np.array_equal(cabin.bgr, pixels(8))
    assert road.bgr.flags.c_contiguous and not road.bgr.flags.writeable
    assert cabin.bgr.flags.c_contiguous and not cabin.bgr.flags.writeable
    assert runtime_parts[0].calls[0][0] is road.bgr
    assert runtime_parts[1].calls[0][0] is cabin.bgr
    assert prediction.source_bundle.road_bgr is road.bgr
    assert prediction.source_bundle.cabin_bgr is cabin.bgr
    assert prediction.source_bundle.source_sequence == 10
    assert prediction.source_bundle.generation == 4


def test_synchronizer_drops_duplicate_reorder_and_future_generation() -> None:
    sync = LiveInputSynchronizer("live", 3)
    road10, cabin10, ego10 = source_envelopes(10, generation=3)
    assert sync.offer_road(road10).accepted
    assert sync.offer_road(road10).disposition == "DUPLICATE"

    old_road, _, _ = source_envelopes(9, generation=3)
    assert sync.offer_road(old_road).disposition == "REORDERED"
    future_road, _, _ = source_envelopes(11, generation=4)
    assert sync.offer_road(future_road).disposition == "FUTURE_GENERATION"
    foreign_road, _, _ = source_envelopes(
        11, generation=3, session_id="foreign"
    )
    assert sync.offer_road(foreign_road).disposition == "STALE_SESSION"

    assert sync.offer_cabin(cabin10).accepted
    assert sync.offer_ego(ego10).disposition == "READY"
    stats = sync.stats
    assert stats.dropped_duplicate == 1
    assert stats.dropped_reordered == 1
    assert stats.dropped_future == 1
    assert stats.dropped_stale == 1
    assert stats.paired_ticks == 1

    road12, cabin12, _ = source_envelopes(12, generation=3)
    gap = sync.offer_road(road12)
    assert gap.disposition == "SOURCE_GAP" and gap.reset_required
    assert sync.offer_cabin(cabin12).disposition == "RESET_REQUIRED"
    assert sync.stats.faulted


@pytest.mark.parametrize(
    ("fault", "expected"),
    [("skew", "CAPTURE_SKEW"), ("partial_future", "BACKLOG_OVERWRITE")],
)
def test_skew_or_partial_backlog_bumps_generation_and_emits_heartbeat(
    fault, expected
) -> None:
    runtime_parts = parts()
    controller = controller_from(runtime_parts)
    controller.start_session("live")
    first_heartbeat = controller.handoff.pop_latest()
    assert first_heartbeat is not None
    assert first_heartbeat.health == "DEGRADED"

    if fault == "skew":
        road, cabin, ego = source_envelopes(0, skew=(0, 10, 26))
        controller.offer_road(road)
        controller.offer_cabin(cabin)
        result = controller.offer_ego(ego)
    else:
        road0, _, _ = source_envelopes(0)
        road1, _, _ = source_envelopes(1)
        controller.offer_road(road0)
        result = controller.offer_road(road1)

    assert result.disposition == expected
    assert result.reset_required
    assert controller.generation == 1
    assert all(component.reset_count == 2 for component in runtime_parts)
    heartbeat = controller.handoff.pop_latest()
    assert heartbeat is not None
    assert heartbeat.generation == 1
    assert heartbeat.health == "DEGRADED"
    assert heartbeat.reason == expected
    assert heartbeat.warning_action == "MONITOR"

    stale_road, _, _ = source_envelopes(2, generation=0)
    assert controller.offer_road(stale_road).disposition == "STALE_GENERATION"


def test_source_gap_never_continues_temporal_state() -> None:
    runtime_parts = parts()
    controller = controller_from(runtime_parts, dwell=1)
    controller.start_session("live")
    offer_tick(controller, 10, generation=0)
    first = controller.process_latest()
    assert first is not None and first.frame_id == 0

    road12, _, _ = source_envelopes(12, generation=0)
    result = controller.offer_road(road12)
    assert result.disposition == "SOURCE_GAP"
    assert controller.generation == 1
    assert controller.session.snapshot.processed_frames == 0
    assert all(component.reset_count == 2 for component in runtime_parts)

    offer_tick(controller, 12, generation=1)
    restarted = controller.process_latest()
    assert restarted is not None
    assert restarted.frame_id == 0
    assert restarted.source_bundle.generation == 1
    assert restarted.c1.model_updated


def test_ready_tick_is_capacity_one_and_new_input_drops_backlog() -> None:
    controller = controller_from(parts())
    controller.start_session("live")
    offer_tick(controller, 0, generation=0)
    road1, _, _ = source_envelopes(1, generation=0)

    result = controller.offer_road(road1)

    assert result.disposition == "INFERENCE_BACKLOG"
    assert result.reset_required
    assert controller.generation == 1
    assert controller.process_latest() is None
    assert not controller.controller_snapshot.pending_input


def test_restart_and_reported_state_loss_each_advance_generation() -> None:
    runtime_parts = parts()
    controller = controller_from(runtime_parts)
    controller.start_session("live", generation=7)
    with pytest.raises(RuntimeError, match="already active"):
        controller.start_session("other")
    assert controller.generation == 7
    controller.reset_session("live")
    assert controller.generation == 8
    restart = controller.handoff.pop_latest()
    assert restart is not None and restart.reason == "RESTART"

    controller.report_state_loss("CUDA_OOM")
    assert controller.generation == 9
    state_loss = controller.handoff.pop_latest()
    assert state_loss is not None
    assert state_loss.reason == "CUDA_OOM"
    assert state_loss.warning_action == "MONITOR"
    assert all(component.reset_count == 3 for component in runtime_parts)


def test_coherence_dwell_and_warning_only_runtime_snapshots() -> None:
    controller = controller_from(parts(emergency=True), dwell=2)
    controller.start_session("live")
    offer_tick(controller, 0, generation=0)
    first = controller.process_latest()
    first_snapshot = controller.handoff.pop_latest()
    assert first is not None and first_snapshot is not None
    assert first.contextual_risk.action == "VISUAL_AUDIO_HAPTIC_WARNING"
    assert first.contextual_risk.brake_request_pct == 0.0
    assert first_snapshot.health == "DEGRADED"
    assert first_snapshot.warning_action == "MONITOR"
    assert first_snapshot.prediction is None

    offer_tick(controller, 1, generation=0)
    second = controller.process_latest()
    live_snapshot = controller.handoff.pop_latest()
    assert second is not None and live_snapshot is not None
    assert live_snapshot.health == "LIVE"
    assert live_snapshot.warning_action == "VISUAL_AUDIO_HAPTIC_WARNING"
    assert live_snapshot.brake_request_pct == 0.0
    assert live_snapshot.actuation_authorized is False


def test_model_exception_resets_faulted_session_into_new_generation() -> None:
    class FailOnceC2(C2):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def process_bgr(self, image, *, frame_id, timestamp):
            if not self.failed:
                self.failed = True
                raise RuntimeError("model failed")
            return super().process_bgr(
                image, frame_id=frame_id, timestamp=timestamp
            )

    runtime_parts = (C1(), FailOnceC2(), C3(), DriveQuality(), Risk())
    controller = controller_from(runtime_parts, dwell=1)
    controller.start_session("live")
    offer_tick(controller, 0, generation=0)
    with pytest.raises(RuntimeError, match="model failed"):
        controller.process_latest()

    assert controller.generation == 1
    assert not controller.session.snapshot.faulted
    assert controller.session.snapshot.processed_frames == 0
    heartbeat = controller.handoff.pop_latest()
    assert heartbeat is not None
    assert heartbeat.reason == "MODEL_EXCEPTION"
    offer_tick(controller, 1, generation=1)
    assert controller.process_latest() is not None


def test_reset_failure_records_requested_session_and_fault_state() -> None:
    class ResetFailsOnceC1(C1):
        def reset(self) -> None:
            super().reset()
            if self.reset_count == 1:
                raise RuntimeError("reset failed")

    c1 = ResetFailsOnceC1()
    c2, c3, dq, risk = C2(), C3(), DriveQuality(), Risk()
    session = LiveInferenceSession(
        c1=c1,
        c2=c2,
        c3=c3,
        drive_quality=dq,
        contextual_risk=risk,
    )
    with pytest.raises(RuntimeError, match="reset failed"):
        session.start_session("requested", generation=5)

    failed = session.snapshot
    assert failed.session_id == "requested"
    assert failed.generation == 5
    assert failed.faulted and not failed.active
    with pytest.raises(RuntimeError, match="faulted"):
        session.process(
            LiveFrameInput(
                session_id="requested",
                generation=5,
                frame_id=0,
                source_sequence=0,
                source_timestamp=1.0,
                road_bgr=pixels(1),
                cabin_bgr=pixels(2),
                ego=EGO,
            )
        )

    session.reset_session("requested", generation=6)
    recovered = session.snapshot
    assert recovered.active and not recovered.faulted
    assert recovered.generation == 6


def test_600_ticks_have_exact_cadence_and_blocked_publisher_stays_capacity_one() -> None:
    runtime_parts = parts()
    handoff = LatestRuntimeSnapshotSlot()
    c1, c2, c3, dq, risk = runtime_parts
    controller = LiveInferenceController(
        LiveInferenceSession(
            c1=c1,
            c2=c2,
            c3=c3,
            drive_quality=dq,
            contextual_risk=risk,
        ),
        handoff=handoff,
        coherent_dwell_ticks=1,
    )
    controller.start_session("soak")

    for sequence in range(600):
        offer_tick(
            controller,
            sequence,
            generation=0,
            session_id="soak",
        )
        assert controller.process_latest() is not None

    snapshot = controller.session.snapshot
    assert snapshot.processed_frames == 600
    assert snapshot.c1_calls == 600
    assert snapshot.c1_model_updates == 300
    assert snapshot.c2_calls == 600
    assert snapshot.c3_calls == 600
    assert snapshot.drive_quality_calls == 600
    assert snapshot.contextual_risk_calls == 600
    assert len(c1.calls) == len(c2.calls) == c3.calls == dq.calls == risk.calls == 600
    # No publisher drained the handoff. Inference still completed and memory
    # remained one pending snapshot rather than a 600-item retry queue.
    assert handoff.stats.offered == 601
    assert handoff.stats.overwritten == 600
    assert handoff.stats.pending


def test_handoff_outage_is_recorded_without_faulting_or_blocking_inference() -> None:
    class FailedHandoff:
        def offer(self, _snapshot):
            raise OSError("publisher unavailable")

    runtime_parts = parts()
    c1, c2, c3, dq, risk = runtime_parts
    controller = LiveInferenceController(
        LiveInferenceSession(
            c1=c1,
            c2=c2,
            c3=c3,
            drive_quality=dq,
            contextual_risk=risk,
        ),
        handoff=FailedHandoff(),
        coherent_dwell_ticks=1,
    )
    controller.start_session("outage")
    for sequence in range(2):
        offer_tick(
            controller,
            sequence,
            generation=0,
            session_id="outage",
        )
        assert controller.process_latest() is not None

    assert controller.session.snapshot.processed_frames == 2
    assert not controller.session.snapshot.faulted
    health = controller.controller_snapshot
    assert health.handoff_errors == 3  # startup heartbeat + two decisions
    assert "publisher unavailable" in (health.latest_handoff_error or "")
