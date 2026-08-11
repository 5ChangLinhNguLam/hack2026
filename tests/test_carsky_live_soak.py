from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.carsky_live_soak import (
    DecodedTick,
    InferenceBackpressure,
    InferenceWorker,
    LocalMockDecisionSink,
    RecordedSoakAdapter,
    RuntimeStack,
    SoakMetrics,
    _validate_c2_bundle_manifest,
    run_soak,
)
from tools.carsky_recorded_stream_sender import (
    RecordedStreamSender,
    TruthFreeRecordedBundle,
)
from tools.prepare_carsky_demo import prepare_bundle
from safeloop.carsky_live import (
    CabinFrameEnvelope,
    EgoTelemetryEnvelope,
    RoadFrameEnvelope,
    RtpFrameIdentity,
    SourceTickKey,
)
from safeloop.live_publish import LatestOnlyKuksaMirror


MODELS = {
    "c1": {"version": "fake-c1", "digest_sha256": "a" * 64},
    "c2": {"version": "fake-c2", "digest_sha256": "b" * 64},
    "c3": {"version": "fake-c3", "digest_sha256": "c" * 64},
    "drive_quality": {"version": "fake-dq", "digest_sha256": "d" * 64},
    "contextual_risk": {"version": "fake-risk", "digest_sha256": "e" * 64},
}


def _jpeg(value: int) -> bytes:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _prepared_bundle(tmp_path: Path, *, trip_id: str = "T01-Sample", count: int = 4) -> Path:
    source = tmp_path / "source" / trip_id
    (source / "driver").mkdir(parents=True)
    (source / "kitti" / "image_2").mkdir(parents=True)
    frames = []
    for sequence in range(count):
        (source / "driver" / f"frame_{sequence:06d}.jpg").write_bytes(
            _jpeg(100 + sequence)
        )
        (source / "kitti" / "image_2" / f"{sequence:06d}.jpg").write_bytes(
            _jpeg(10 + sequence)
        )
        frames.append(
            {
                "frame_id": sequence,
                "timestamp": sequence / 20.0,
                "ego": {
                    "speed_kmh": 30.0 + sequence,
                    "longitudinal_accel": -0.1,
                    "lateral_accel": 0.2,
                    "privileged": "must be removed",
                },
                "targets": [{"ttc": 0.01}],
                "events_active": ["ground-truth"],
            }
        )
    payload = {
        "trip_id": trip_id,
        "metadata": {
            "trip_id": trip_id,
            "fps": 20,
            "duration_sec": count / 20.0,
            "description": "pedestrian jaywalk only",
            "speed_limit_kmh": 77,
            "weather": {"cloudiness": 12},
            "secret": "removed",
        },
        "frames": frames,
        "events_log": ["ground-truth"],
    }
    (source / f"{trip_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    result = prepare_bundle(source, tmp_path / "prepared")
    return Path(result["destination"])


class FakeClock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.monotonic = 0
        self.wall = 1_800_000_000_000_000_000

    def monotonic_ns(self) -> int:
        with self._lock:
            return self.monotonic

    def time_ns(self) -> int:
        with self._lock:
            return self.wall

    def sleep(self, seconds: float) -> None:
        amount = round(seconds * 1_000_000_000)
        with self._lock:
            self.monotonic += amount
            self.wall += amount


class Resettable:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class FakeC1(Resettable):
    def __init__(self, block_started=None, release=None) -> None:
        super().__init__()
        self.metadata: list[dict[str, object]] = []
        self.block_started = block_started
        self.release = release
        self.calls = 0

    def configure_session(self, metadata, *, expected_frames: int) -> None:
        self.metadata.append({**dict(metadata), "expected_frames": expected_frames})

    def process_bgr(self, image, *, frame_id, timestamp, ego, target_count):
        assert image.dtype == np.uint8
        assert set(ego) == {"speed_kmh", "longitudinal_accel", "lateral_accel"}
        assert target_count == 0
        if self.calls == 0 and self.block_started is not None:
            self.block_started.set()
            assert self.release.wait(2.0)
        self.calls += 1
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=timestamp,
            predicted_ttc_s=3.0,
            collision_probability=0.2,
            is_warning=False,
            model_updated=frame_id % 2 == 0,
            latency_ms=0.1 if frame_id % 2 == 0 else 0.0,
        )


class FakeC2(Resettable):
    def process_bgr(self, image, *, frame_id, timestamp):
        assert image.dtype == np.uint8
        signals = SimpleNamespace(
            attentive_probability=95.0,
            distraction_level=3.0,
            fatigue_level=2.0,
            is_eyes_on_road=True,
            is_warning=False,
        )
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=timestamp,
            state="alert",
            confidence=0.95,
            latency_ms=0.2,
            vss_signals=lambda: signals,
        )


class FakeC3(Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.speed_limits: list[float] = []
        self.expected = 0

    def configure_session(self, metadata, *, expected_frames: int) -> None:
        self.speed_limits.append(float(metadata["speed_limit_kmh"]))
        self.expected = expected_frames

    def update(self, bundle, *, predicted_ttc_s):
        assert predicted_ttc_s == 3.0
        return SimpleNamespace(
            frame_id=bundle.frame_id,
            timestamp=bundle.timestamp,
            safe_score_estimate=99.0,
            grade="A",
            trip_complete=bundle.frame_id + 1 == self.expected,
            is_near_miss=False,
            is_harsh_brake=False,
            is_harsh_accel=False,
            is_harsh_corner=False,
            is_speeding=False,
        )


class FakeDriveQuality(Resettable):
    def update(self, c3):
        return SimpleNamespace(
            frame_id=c3.frame_id,
            timestamp=c3.timestamp,
            score_available=True,
            score_pct=99.0,
            grade="A",
            scope="FULL_TRIP" if c3.trip_complete else "PREFIX",
        )


class FakeRisk(Resettable):
    def evaluate(self, c1, c2):
        return SimpleNamespace(
            score_pct=5.0,
            level="SAFE",
            action="MONITOR",
            brake_request_pct=0.0,
            reasons=("NORMAL",),
        )


def _stack(*, c1=None) -> RuntimeStack:
    return RuntimeStack(
        c1=c1 or FakeC1(),
        c2=FakeC2(),
        c3=FakeC3(),
        drive_quality=FakeDriveQuality(),
        contextual_risk=FakeRisk(),
        models=MODELS,
    )


def _tick(sequence: int, *, capture_ms: int = 1_800_000_000_000) -> DecodedTick:
    key = SourceTickKey("capacity-test", 0, sequence)
    image = np.full((3, 4, 3), sequence, dtype=np.uint8)
    road = RoadFrameEnvelope(RtpFrameIdentity(sequence * 4500, key), capture_ms + sequence * 50, image)
    cabin = CabinFrameEnvelope(RtpFrameIdentity(sequence * 4500, key), capture_ms + sequence * 50, image)
    ego = EgoTelemetryEnvelope(
        key,
        capture_ms + sequence * 50,
        {"speed_kmh": 20.0, "longitudinal_accel": 0.0, "lateral_accel": 0.0},
    )
    return DecodedTick(
        session_id=key.session_id,
        generation=key.generation,
        source_sequence=sequence,
        source_media_timestamp_ms=sequence * 50,
        server_receive_timestamp_ms=capture_ms + sequence * 50,
        road=road,
        cabin=cabin,
        ego=ego,
    )


def test_short_fake_soak_drives_controller_v2_and_strict_report(tmp_path: Path) -> None:
    bundle = _prepared_bundle(tmp_path, count=4)
    clock = FakeClock()
    stack = _stack()

    result = run_soak(
        [bundle],
        duration_seconds=0.2,
        output_dir=tmp_path / ".carsky-build" / "soak",
        clock=clock,
        runtime_stack=stack,
        drain_each_frame_for_test=True,
        device="cpu",
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "safeloop.carsky.live-soak-report.v1"
    assert report["outcome"] == "PASS"
    boundary = report["claim_boundary"]
    assert boundary["video_source"] == "RECORDED_STREAM"
    assert boundary["inference_mode"] == "LIVE_MODEL"
    assert boundary["media_codec"] == "IMAGE_FILE_DECODE"
    assert boundary["h264_included"] is False
    assert boundary["aws_connectivity"] is False
    assert boundary["synthetic_frames"] is False
    assert boundary["synthetic_ego"] is False
    assert boundary["raw_or_mixed_dataset_access"] is False
    assert boundary["independent_webrtc_decode_arrival_skew_measured"] is False
    assert report["throughput"]["source_ticks_attempted"] == 4
    assert "description" not in report["bundles"][0]["metadata"]
    assert report["throughput"]["inference_outputs"] == 4
    assert report["cadence"]["c1_calls"] == 4
    assert report["cadence"]["c1_model_updates"] == 2
    assert report["cadence"]["exact"] is True
    assert report["acceptance"]["passed"] is True
    assert report["timing"]["source_clock_elapsed_seconds"] >= 0.2
    assert report["queues"]["max_ingress_waiting"] <= 1
    assert report["publisher"]["max_serialized_published_envelope_bytes"] > 0
    assert len(report["resources"]["samples"]) >= 3
    assert stack.c1.metadata[0]["speed_limit_kmh"] == 77.0
    assert stack.c3.speed_limits == [77.0]


class CollectingWorker:
    def __init__(self) -> None:
        self.ticks: list[DecodedTick] = []
        self.mirror = SimpleNamespace(wait_until_idle=lambda timeout: True)

    def start_session(self, session, metadata, *, expected_frames):
        self.session = session
        self.metadata = metadata
        self.expected_frames = expected_frames

    def offer(self, tick):
        self.ticks.append(tick)

    def wait_until_idle(self, timeout):
        return True

    def current_error(self):
        return None

    def mark_clock_unhealthy(self, *, timeout_s):
        self.clock_unhealthy = True

    def finish_session(self, *, timeout_s):
        return {"error": None, "processed_frames": len(self.ticks)}


def test_adapter_decodes_real_bytes_and_maps_only_explicit_rtp_identity(tmp_path: Path) -> None:
    path = _prepared_bundle(tmp_path, count=2)
    bundle = TruthFreeRecordedBundle.load(path)
    worker = CollectingWorker()
    clock = FakeClock()
    adapter = RecordedSoakAdapter(
        worker,
        bundle,
        SoakMetrics(),
        expected_frames=2,
        epoch_ms=lambda: clock.time_ns() // 1_000_000,
    )
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="mapping-test",
        generation=0,
        clock=clock,
    )

    sender.run(limit=2)

    assert len(worker.ticks) == 2
    for sequence, tick in enumerate(worker.ticks):
        expected = SourceTickKey("mapping-test", 0, sequence)
        assert tick.road.identity.metadata_key == expected
        assert tick.cabin.identity.metadata_key == expected
        assert tick.ego.key == expected
        assert tick.road.identity.rtp_timestamp == sequence * 4500
        assert tick.cabin.identity.rtp_timestamp == sequence * 4500
        assert tick.road.bgr.shape == (12, 16, 3)
        assert tick.cabin.bgr.shape == (12, 16, 3)
        assert tick.source_media_timestamp_ms == sequence * 50


def test_inference_worker_has_one_waiting_slot_and_no_growing_queue() -> None:
    entered = threading.Event()
    release = threading.Event()
    metrics = SoakMetrics()
    sink = LocalMockDecisionSink(metrics)
    mirror = LatestOnlyKuksaMirror(sink)
    worker = InferenceWorker(
        _stack(c1=FakeC1(entered, release)),
        mirror,
        metrics,
        device="cpu",
        epoch_ms=lambda: 1_800_000_000_500,
    )
    session = SimpleNamespace(
        session_id="capacity-test",
        generation=0,
        source_hz=20,
        video_source="RECORDED_STREAM",
        inference_mode="LIVE_MODEL",
        road_media_id="capacity-test:g0:road",
        cabin_media_id="capacity-test:g0:cabin",
    )
    try:
        worker.start_session(session, {"speed_limit_kmh": 80}, expected_frames=3)
        worker.offer(_tick(0))
        assert entered.wait(1.0)
        worker.offer(_tick(1))
        with pytest.raises(InferenceBackpressure, match="capacity-one"):
            worker.offer(_tick(2))
        assert metrics.count("ingress_queue_drops") == 1
        release.set()
        assert worker.wait_until_idle(2.0)
        summary = worker.finish_session(timeout_s=2.0)
        assert summary["processed_frames"] == 2
    finally:
        release.set()
        worker.close(timeout_s=2.0)
        mirror.close(timeout_s=2.0)


def test_adapter_backpressure_emits_state_loss_reset_before_session_end(
    tmp_path: Path,
) -> None:
    bundle = TruthFreeRecordedBundle.load(_prepared_bundle(tmp_path, count=3))
    entered = threading.Event()
    release = threading.Event()
    source_clock = FakeClock()
    metrics = SoakMetrics()
    sink = LocalMockDecisionSink(metrics)
    mirror = LatestOnlyKuksaMirror(sink)
    worker = InferenceWorker(
        _stack(c1=FakeC1(entered, release)),
        mirror,
        metrics,
        device="cpu",
        epoch_ms=lambda: 1_800_000_001_000,
    )
    adapter = RecordedSoakAdapter(
        worker,
        bundle,
        metrics,
        expected_frames=3,
        epoch_ms=lambda: source_clock.time_ns() // 1_000_000,
        close_timeout_s=2.0,
    )
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="backpressure-test",
        generation=0,
        clock=source_clock,
    )
    errors: list[BaseException] = []

    def run_sender() -> None:
        try:
            sender.run(limit=3)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_sender)
    thread.start()
    try:
        assert entered.wait(1.0)
        deadline = time.monotonic() + 1.0
        while metrics.count("ingress_queue_drops") == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert metrics.count("ingress_queue_drops") == 1
        release.set()
        thread.join(2.0)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], InferenceBackpressure)
        assert metrics.count("backpressure_state_loss_resets") == 1
        assert metrics.count("controller_resets") == 1
        assert adapter.session_summary["generation_ended"] == 1
    finally:
        release.set()
        thread.join(2.0)
        worker.close(timeout_s=2.0)
        mirror.close(timeout_s=2.0)


def test_clock_regression_marks_affected_decision_untrusted_before_publish(
    tmp_path: Path,
) -> None:
    bundle = TruthFreeRecordedBundle.load(_prepared_bundle(tmp_path, count=1))
    metrics = SoakMetrics()
    sink = LocalMockDecisionSink(metrics)
    mirror = LatestOnlyKuksaMirror(sink)
    worker = InferenceWorker(
        _stack(),
        mirror,
        metrics,
        device="cpu",
        epoch_ms=lambda: 1_800_000_000_100,
    )
    source_clock = FakeClock()
    adapter = RecordedSoakAdapter(
        worker,
        bundle,
        metrics,
        expected_frames=1,
        # One millisecond behind the sender's capture epoch.
        epoch_ms=lambda: source_clock.time_ns() // 1_000_000 - 1,
        drain_each_frame_for_test=True,
        close_timeout_s=2.0,
    )
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="clock-regression-test",
        generation=0,
        clock=source_clock,
    )
    try:
        sender.run(limit=1)
        assert mirror.wait_until_idle(2.0)
        assert metrics.count("clock_regressions") == 1
        statuses = sink.summary()
        assert statuses["clock_status_counts"]["UNHEALTHY"] == 1
        assert statuses["health_status_counts"]["STALE"] == 1
    finally:
        worker.close(timeout_s=2.0)
        mirror.close(timeout_s=2.0)


def test_c2_manifest_verification_rejects_one_tampered_loaded_artifact(tmp_path: Path) -> None:
    root = tmp_path / "dms"
    names = (
        "face_landmarker.task",
        "version-RFB-320.onnx",
        "visual/best.pt",
        "ocular/best.pt",
        "gate_ocular/best.pt",
        "temporal/best.pt",
    )
    hashes = {}
    for index, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"model-{index}".encode()
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    (root / "bundle_manifest.json").write_text(
        json.dumps({"bundle": "test", "sha256": hashes}), encoding="utf-8"
    )
    assert _validate_c2_bundle_manifest(root)["bundle"] == "test"

    (root / "visual" / "best.pt").write_bytes(b"tampered")
    with pytest.raises(Exception, match="checksum mismatch"):
        _validate_c2_bundle_manifest(root)

    (root / "bundle_manifest.json").write_text(
        '{"sha256":{},"sha256":{}}', encoding="utf-8"
    )
    with pytest.raises(Exception, match="duplicate JSON key"):
        _validate_c2_bundle_manifest(root)
