from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
import threading
import time

import numpy as np

from safeloop.aws_live_contract import InputTick
from safeloop.aws_live_runtime import FastBridgeController
from safeloop.combined_replay import CombinedFramePrediction
from safeloop.contextual_risk import ContextualRiskDecision


class FakeC2(SimpleNamespace):
    def vss_signals(self):
        return SimpleNamespace(
            attentive_probability=90.0,
            distraction_level=5.0,
            fatigue_level=5.0,
            is_eyes_on_road=True,
            is_warning=False,
        )


class FakeModels:
    model_versions = MappingProxyType({"c1": "x", "c2": "x", "c3": "x"})
    model_hashes = MappingProxyType({"c1": "a" * 64, "c2": "b" * 64})
    load_count = 1

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.sessions: list[str] = []
        self.processed: list[tuple[str, int]] = []

    def start_session(self, session_id, metadata, *, expected_frames):
        self.sessions.append(session_id)

    def process(self, tick, *, frame_id):
        if self.delay:
            time.sleep(self.delay)
        self.processed.append((self.sessions[-1], tick.source_sequence))
        timestamp = tick.capture_timestamp_ms / 1000.0
        return CombinedFramePrediction(
            frame_id=frame_id,
            timestamp=timestamp,
            c1=SimpleNamespace(
                frame_id=frame_id,
                timestamp=timestamp,
                predicted_ttc_s=1.0,
                collision_probability=0.9,
                is_warning=True,
                model_updated=frame_id % 2 == 0,
                model_frame_id=frame_id - frame_id % 2,
            ),
            c2=FakeC2(
                frame_id=frame_id,
                timestamp=timestamp,
                state="alert",
                confidence=0.9,
            ),
            c3=SimpleNamespace(
                frame_id=frame_id,
                timestamp=timestamp,
                safe_score_estimate=80.0,
                grade="B",
                trip_complete=False,
                formula_version="x",
                tailgating_penalty_omitted=True,
            ),
            drive_quality=SimpleNamespace(
                frame_id=frame_id,
                timestamp=timestamp,
                score_available=True,
                score_pct=90.0,
                grade="A",
                scope="PREFIX",
                window_ready=False,
                formula_version="x",
            ),
            contextual_risk=ContextualRiskDecision(
                score_pct=100.0,
                level="CRITICAL",
                action="EMERGENCY_BRAKE_REQUEST",
                brake_request_pct=70.0,
                reasons=("LOW_TTC",),
            ),
        )

    def close(self):
        pass


def tick(sequence: int, *, session: str = "source-a") -> InputTick:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image.setflags(write=False)
    return InputTick(
        session_id=session,
        source_sequence=sequence,
        capture_timestamp_ms=1_700_000_000_000 + sequence * 50,
        source_kind="LIVE_CAMERA",
        telemetry_source="THIRD_PARTY",
        metadata=MappingProxyType({"speed_limit_kmh": 60.0, "weather": MappingProxyType({})}),
        ego=MappingProxyType({"speed_kmh": 0.0, "longitudinal_accel": 0.0, "lateral_accel": 0.0}),
        road_bgr=image,
        cabin_bgr=image,
    )


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition timed out")


def test_real_time_mapper_is_strict_v1_warning_only_and_cadence_labels() -> None:
    published = []
    models = FakeModels()
    controller = FastBridgeController(models, on_decision=published.append)
    try:
        controller.start_live(expected_revision=0)
        token = controller.live_connect()
        for sequence in range(6):
            assert controller.ingest(tick(sequence))
            wait_for(lambda: len(published) == sequence + 1)
        last = published[-1]
        wire = last.payload.decode("utf-8")
        assert len(last.payload) <= 1472
        assert "NaN" not in wire and "Infinity" not in wire
        assert last.envelope.schema_version == "safeloop.decision.v1"
        assert last.envelope.contextual_risk["action"] == "VISUAL_AUDIO_HAPTIC_WARNING"
        assert last.envelope.contextual_risk["brake_request_pct"] == 0.0
        assert last.envelope.contextual_risk["actuation_authorized"] is False
        assert [value[1] % 2 == 0 for value in models.processed] == [True, False, True, False, True, False]
        assert controller.status()["state"] == "RUNNING"
        controller.live_disconnect(token)
    finally:
        controller.close()


def test_gap_and_cross_session_reset_all_state() -> None:
    published = []
    models = FakeModels()
    controller = FastBridgeController(models, on_decision=published.append)
    try:
        controller.start_live(expected_revision=0)
        controller.live_connect()
        controller.ingest(tick(0))
        wait_for(lambda: len(published) == 1)
        first_session = published[-1].envelope.session_id
        controller.ingest(tick(2))
        wait_for(lambda: len(published) == 2)
        second_session = published[-1].envelope.session_id
        controller.ingest(tick(0, session="source-b"))
        wait_for(lambda: len(published) == 3)
        third_session = published[-1].envelope.session_id
        assert len({first_session, second_session, third_session}) == 3
        assert [item.envelope.sequence for item in published] == [0, 0, 0]
        assert controller.status()["counts"]["generation_resets"] >= 2
    finally:
        controller.close()


def test_capacity_one_overflow_resets_generation_and_discards_old_result() -> None:
    published = []
    models = FakeModels(delay=0.08)
    controller = FastBridgeController(models, on_decision=published.append)
    try:
        controller.start_live(expected_revision=0)
        controller.live_connect()
        controller.ingest(tick(0))
        time.sleep(0.01)
        controller.ingest(tick(1))
        controller.ingest(tick(2))
        wait_for(lambda: bool(published), timeout=3)
        assert published[-1].source_sequence == 2
        assert controller.status()["counts"]["overflow_drops"] == 1
        assert controller.status()["counts"]["discarded_generation_results"] >= 1
    finally:
        controller.close()
