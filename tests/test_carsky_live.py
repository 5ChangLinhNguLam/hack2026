from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from safeloop.carsky_live import (
    LatestPairedFrameSlot,
    LiveFrameInput,
    LiveInferenceController,
    LiveInferenceSession,
    PairedCameraSample,
)


EGO = {
    "speed_kmh": 36.0,
    "longitudinal_accel": 0.1,
    "lateral_accel": -0.2,
}


def image(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


@dataclass(frozen=True)
class FakePrediction:
    frame_id: int
    timestamp: float
    value: object
    challenge: str

    @property
    def model_updated(self) -> bool:
        return self.challenge == "c1" and self.frame_id % 2 == 0

    @property
    def predicted_ttc_s(self) -> float:
        if self.challenge != "c1":
            raise AttributeError("only C1 has TTC")
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


class FakeC1(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, int, float, object, int]] = []

    def process_bgr(
        self,
        image,
        *,
        frame_id: int,
        timestamp: float,
        ego,
        target_count: int,
    ) -> FakePrediction:
        self.calls.append((image, frame_id, timestamp, ego, target_count))
        return FakePrediction(frame_id, timestamp, 1.25, "c1")


class FakeC2(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, int, float]] = []

    def process_bgr(
        self, image, *, frame_id: int, timestamp: float
    ) -> FakePrediction:
        self.calls.append((image, frame_id, timestamp))
        return FakePrediction(frame_id, timestamp, "alert", "c2")


class FakeC3(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.frames: list[LiveFrameInput] = []

    def update(self, bundle, *, predicted_ttc_s):
        self.frames.append(bundle)
        assert predicted_ttc_s == 1.25
        # A live bundle exposes neither labels nor ground truth.
        for forbidden in ("gt", "depth", "labels", "targets", "events_active"):
            with pytest.raises(AttributeError):
                getattr(bundle, forbidden)
        return FakePrediction(
            bundle.frame_id, bundle.timestamp, 99.0, "c3"
        )


class FakeDriveQuality(Resettable):
    def update(self, c3):
        return FakePrediction(
            c3.frame_id, c3.timestamp, 98.0, "drive_quality"
        )


class FakeRisk(Resettable):
    def evaluate(self, c1, c2):
        return type("Risk", (), {"score_pct": 12.0})()


def processors():
    return FakeC1(), FakeC2(), FakeC3(), FakeDriveQuality(), FakeRisk()


def session_from(parts) -> LiveInferenceSession:
    c1, c2, c3, drive_quality, risk = parts
    return LiveInferenceSession(
        c1=c1,
        c2=c2,
        c3=c3,
        drive_quality=drive_quality,
        contextual_risk=risk,
    )


def frame(
    frame_id: int,
    timestamp: float,
    *,
    source_sequence: int | None = None,
    session_id: str = "test",
) -> LiveFrameInput:
    return LiveFrameInput(
        session_id=session_id,
        frame_id=frame_id,
        source_sequence=source_sequence,
        source_timestamp=timestamp,
        road_bgr=image(frame_id),
        cabin_bgr=image(frame_id + 100),
        ego=EGO,
    )


def test_live_session_runs_all_processors_without_privileged_inputs() -> None:
    parts = processors()
    c1, c2, c3, drive_quality, risk = parts
    live = session_from(parts)
    live.start_session("carsky:boot-1")

    first = live.process(
        frame(0, 1000.0, source_sequence=40, session_id="carsky:boot-1")
    )
    second = live.process(
        frame(1, 1000.05, source_sequence=41, session_id="carsky:boot-1")
    )

    assert first.frame_id == 0
    assert second.submission_row() == {
        "frame_id": 1,
        "timestamp": 1000.05,
        "predicted_ttc": 1.25,
        "predicted_driver_state": "alert",
        "predicted_risk_score": 12.0,
    }
    # The orchestration never adds a second stride.  C1 is called every tick
    # and remains responsible for its own model-update/forward-fill cadence.
    assert [call[1] for call in c1.calls] == [0, 1]
    assert [call[1] for call in c2.calls] == [0, 1]
    assert all(call[4] == 0 for call in c1.calls)
    assert dict(c1.calls[0][3]) == EGO
    assert [item.source_sequence for item in c3.frames] == [40, 41]
    assert np.array_equal(first.source_bundle.left(), image(0))
    assert np.array_equal(first.source_bundle.driver(), image(100))
    assert all(
        component.reset_count == 1
        for component in (c1, c2, c3, drive_quality, risk)
    )
    assert live.snapshot.processed_frames == 2
    assert live.snapshot.next_frame_id == 2
    assert live.snapshot.last_source_sequence == 41


def test_input_whitelists_ego_and_detaches_caller_mapping() -> None:
    caller_ego = {**EGO, "gt_ttc": 0.01, "label": "danger"}
    live_frame = LiveFrameInput(
        session_id="test",
        frame_id=0,
        source_timestamp=0.0,
        road_bgr=image(1),
        cabin_bgr=image(2),
        ego=caller_ego,
    )
    caller_ego["speed_kmh"] = 999.0

    assert dict(live_frame.ego) == EGO
    assert "gt_ttc" not in live_frame.ego
    with pytest.raises(TypeError):
        live_frame.ego["speed_kmh"] = 10.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"longitudinal_accel": 0.0, "lateral_accel": 0.0}, "speed_kmh"),
        ({**EGO, "speed_kmh": float("nan")}, "finite"),
    ],
)
def test_input_requires_finite_minimal_ego(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        LiveFrameInput(
            session_id="test",
            frame_id=0,
            source_timestamp=0.0,
            road_bgr=image(1),
            cabin_bgr=image(2),
            ego=values,
        )


def test_live_input_rejects_boolean_timestamp() -> None:
    with pytest.raises(ValueError, match="source_timestamp must be numeric"):
        LiveFrameInput(
            session_id="test",
            frame_id=0,
            source_timestamp=True,
            road_bgr=image(1),
            cabin_bgr=image(2),
            ego=EGO,
        )


@pytest.mark.parametrize(
    ("road", "message"),
    [
        ("not-an-image", "numpy array"),
        (np.zeros((2, 3, 3), dtype=np.float32), "uint8"),
        (np.zeros((2, 3), dtype=np.uint8), "HxWx3"),
        (np.zeros((2, 3, 4), dtype=np.uint8), "HxWx3"),
        (np.zeros((2, 3, 3), dtype=np.uint8)[:, ::-1], "C-contiguous"),
    ],
)
def test_input_rejects_invalid_live_pixel_contract(road, message) -> None:
    with pytest.raises(ValueError, match=message):
        LiveFrameInput(
            session_id="test",
            frame_id=0,
            source_timestamp=0.0,
            road_bgr=road,
            cabin_bgr=image(2),
            ego=EGO,
        )


def test_live_input_owns_read_only_pixels_from_reusable_transport_buffer() -> None:
    road = image(7)
    cabin = image(8)
    live_frame = LiveFrameInput(
        session_id="test",
        frame_id=0,
        source_timestamp=0.0,
        road_bgr=road,
        cabin_bgr=cabin,
        ego=EGO,
    )

    road.fill(99)
    cabin.fill(99)

    assert np.array_equal(live_frame.road_bgr, image(7))
    assert np.array_equal(live_frame.cabin_bgr, image(8))
    assert live_frame.road_bgr.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        live_frame.road_bgr[0, 0, 0] = 1


def test_live_ordering_is_strict_but_validation_error_does_not_fault() -> None:
    live = session_from(processors())
    live.start_session("strict")

    with pytest.raises(ValueError, match="expected 0"):
        live.process(frame(1, 0.05, session_id="strict"))
    assert not live.snapshot.faulted

    live.process(frame(0, 10.0, source_sequence=100, session_id="strict"))
    with pytest.raises(ValueError, match="expected 1"):
        live.process(frame(2, 10.1, source_sequence=102, session_id="strict"))
    with pytest.raises(ValueError, match="source_sequence"):
        live.process(frame(1, 10.05, source_sequence=100, session_id="strict"))
    with pytest.raises(ValueError, match="fresh session"):
        live.process(frame(1, 10.05, source_sequence=103, session_id="strict"))
    with pytest.raises(ValueError, match="timestamps"):
        live.process(frame(1, 10.0, source_sequence=101, session_id="strict"))

    accepted = live.process(
        frame(1, 10.05, source_sequence=101, session_id="strict")
    )
    assert accepted.frame_id == 1


def test_session_lifecycle_resets_state_and_returns_to_frame_zero() -> None:
    parts = processors()
    live = session_from(parts)
    with pytest.raises(RuntimeError, match="no live inference session"):
        live.process(frame(0, 0.0))

    live.start_session("one")
    live.process(frame(0, 1.0, session_id="one"))
    with pytest.raises(RuntimeError, match="already active"):
        live.start_session("two")

    ended = live.end_session()
    assert not ended.active
    assert ended.processed_frames == 1
    assert not live.snapshot.active
    with pytest.raises(RuntimeError, match="no live inference session"):
        live.process(frame(1, 1.05))

    live.start_session("two")
    restarted = live.process(
        frame(0, 50.0, source_sequence=900, session_id="two")
    )
    assert restarted.frame_id == 0
    assert all(part.reset_count == 2 for part in parts)


def test_inference_exception_faults_session_until_explicit_reset() -> None:
    class FailingC2(FakeC2):
        def process_bgr(self, image, *, frame_id, timestamp):
            if self.reset_count == 1:
                raise RuntimeError("camera model failed")
            return super().process_bgr(
                image, frame_id=frame_id, timestamp=timestamp
            )

    parts = (FakeC1(), FailingC2(), FakeC3(), FakeDriveQuality(), FakeRisk())
    live = session_from(parts)
    live.start_session("broken")
    with pytest.raises(RuntimeError, match="camera model failed"):
        live.process(frame(0, 0.0, session_id="broken"))
    assert live.snapshot.faulted
    assert live.snapshot.processed_frames == 0
    with pytest.raises(RuntimeError, match="faulted"):
        live.process(frame(0, 0.0, session_id="broken"))

    live.reset_session("recovered")
    result = live.process(frame(0, 1.0, session_id="recovered"))
    assert result.frame_id == 0
    assert not live.snapshot.faulted


def sample(
    sequence: int,
    timestamp: float,
    *,
    session_id: str = "stream",
) -> PairedCameraSample:
    return PairedCameraSample(
        session_id=session_id,
        source_sequence=sequence,
        source_timestamp=timestamp,
        road_bgr=image(sequence),
        cabin_bgr=image(sequence + 100),
        ego=EGO,
    )


def test_latest_slot_drops_stale_pair_and_reindexes_emitted_frames() -> None:
    slot = LatestPairedFrameSlot("stream")
    slot.offer(sample(10, 5.0))
    slot.offer(sample(11, 5.05))

    first = slot.pop_latest()
    assert first is not None
    assert first.frame_id == 0
    assert first.source_sequence == 11
    assert np.array_equal(first.road_bgr, image(11))
    assert slot.pop_latest() is None

    slot.offer(sample(15, 5.25))
    second = slot.pop_latest()
    assert second is not None
    assert second.frame_id == 1
    assert second.source_sequence == 15
    assert slot.stats.offered == 3
    assert slot.stats.emitted == 2
    assert slot.stats.overwritten == 1
    assert slot.stats.source_gaps == 3
    assert not slot.stats.pending


def test_latest_slot_rejects_out_of_order_source_and_reset_restarts_ids() -> None:
    slot = LatestPairedFrameSlot("stream")
    slot.offer(sample(3, 2.0))
    with pytest.raises(ValueError, match="source_sequence"):
        slot.offer(sample(3, 2.1))
    with pytest.raises(ValueError, match="timestamps"):
        slot.offer(sample(4, 2.0))

    slot.reset("stream")
    slot.offer(sample(1, 0.5))
    emitted = slot.pop_latest()
    assert emitted is not None
    assert emitted.frame_id == 0
    assert slot.stats.overwritten == 0
    assert slot.stats.source_gaps == 0


def test_controller_fences_stale_callbacks_and_popped_old_frames() -> None:
    live = session_from(processors())
    controller = LiveInferenceController(live)
    controller.start_session("old")
    controller.offer(sample(0, 1.0, session_id="old"))
    old_frame = controller.slot.pop_latest()
    assert old_frame is not None

    controller.reset_session("new")
    with pytest.raises(ValueError, match="stale|foreign"):
        controller.offer(sample(1, 1.05, session_id="old"))
    with pytest.raises(ValueError, match="stale|foreign"):
        live.process(old_frame)

    controller.offer(sample(50, 5.0, session_id="new"))
    prediction = controller.process_latest()
    assert prediction is not None
    assert prediction.frame_id == 0
    assert prediction.source_bundle.session_id == "new"
