from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.carsky_recorded_stream_sender import (
    BundleValidationError,
    FRAME_SCHEMA,
    RecordedStreamSender,
    TransformingTransportAdapter,
    TruthFreeRecordedBundle,
)
from tools.prepare_carsky_demo import prepare_bundle


def _source_trip(root: Path, *, count: int = 3, fps: int = 20) -> Path:
    trip = root / "T02-Sample"
    (trip / "driver").mkdir(parents=True)
    (trip / "kitti" / "image_2").mkdir(parents=True)
    (trip / "kitti" / "image_3").mkdir(parents=True)
    (trip / "kitti" / "depth").mkdir(parents=True)
    (trip / "kitti" / "label_2").mkdir(parents=True)
    frames = []
    for sequence in range(count):
        (trip / "driver" / f"frame_{sequence:06d}.jpg").write_bytes(
            f"cabin-{sequence}".encode()
        )
        (trip / "kitti" / "image_2" / f"{sequence:06d}.jpg").write_bytes(
            f"road-{sequence}".encode()
        )
        (trip / "kitti" / "image_3" / f"{sequence:06d}.jpg").write_bytes(
            b"forbidden-right-camera"
        )
        (trip / "kitti" / "depth" / f"{sequence:06d}.npy").write_bytes(
            b"forbidden-depth"
        )
        (trip / "kitti" / "label_2" / f"{sequence:06d}.txt").write_text(
            "forbidden-label", encoding="utf-8"
        )
        frames.append(
            {
                "frame_id": sequence,
                "timestamp": sequence / fps,
                "ego": {
                    "speed_kmh": 30.0 + sequence,
                    "longitudinal_accel": -0.1,
                    "lateral_accel": 0.2,
                    "location": {"x": 123.0},
                },
                "targets": [{"ttc": 0.01}],
                "driver": {"state": "ground-truth"},
                "events_active": ["ground-truth"],
                "min_ttc": 0.01,
            }
        )
    payload = {
        "trip_id": trip.name,
        "metadata": {
            "trip_id": trip.name,
            "description": "pedestrian jaywalk ground-truth scenario",
            "fps": fps,
            "duration_sec": count / fps,
            "speed_limit_kmh": 80,
            "weather": {"cloudiness": 10},
            "random_seed": 999,
        },
        "frames": frames,
        "driver_summary": {"state": "ground-truth"},
        "trip_aggregate": {"ttc": 0.01},
        "events_log": [{"event": "ground-truth"}],
    }
    (trip / f"{trip.name}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return trip


def _bundle(tmp_path: Path, *, count: int = 3, fps: int = 20) -> Path:
    source = _source_trip(tmp_path / "mixed-source", count=count, fps=fps)
    result = prepare_bundle(source, tmp_path / "prepared")
    return Path(result["destination"])


def _resign_file(bundle: Path, relative_name: str) -> None:
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (bundle / relative_name).read_bytes()
    manifest["files"][relative_name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_trip_payload(bundle: Path, mutate) -> None:
    relative_name = f"{bundle.name}.json"
    path = bundle / relative_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _resign_file(bundle, relative_name)


def test_loader_accepts_only_prepared_truth_free_contract(tmp_path: Path) -> None:
    bundle_path = _bundle(tmp_path)
    bundle = TruthFreeRecordedBundle.load(bundle_path)

    assert bundle.trip_id == "T02-Sample"
    assert len(bundle.frames) == 3
    assert set(bundle.metadata) == {
        "trip_id",
        "fps",
        "duration_sec",
        "speed_limit_kmh",
        "weather",
    }
    assert bundle.metadata["speed_limit_kmh"] == 80.0
    assert "random_seed" not in bundle.metadata
    assert "description" not in bundle.metadata
    with pytest.raises(TypeError):
        bundle.metadata["speed_limit_kmh"] = 1.0
    weather = bundle.metadata["weather"]
    with pytest.raises(TypeError):
        weather["cloudiness"] = 99
    assert [frame.source_sequence for frame in bundle.frames] == [0, 1, 2]
    assert [frame.source_media_timestamp_ms for frame in bundle.frames] == [
        0,
        50,
        100,
    ]
    assert dict(bundle.frames[1].ego) == {
        "speed_kmh": 31.0,
        "longitudinal_accel": -0.1,
        "lateral_accel": 0.2,
    }
    assert bundle.frames[1].road.read_verified() == b"road-1"
    assert bundle.frames[1].cabin.read_verified() == b"cabin-1"
    assert not (bundle_path / "kitti" / "depth").exists()
    assert not (bundle_path / "kitti" / "label_2").exists()


def test_loader_rejects_raw_mixed_trip_without_opening_any_source_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_trip(tmp_path / "source")
    opened: list[Path] = []
    original_open = Path.open

    def tracked_open(path: Path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    with pytest.raises(BundleValidationError, match="raw/mixed dataset"):
        TruthFreeRecordedBundle.load(source)
    assert opened == []


def test_loader_rejects_unmanifested_prediction_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    prediction = bundle / "prediction.csv"
    prediction.write_text("forbidden,prediction\n", encoding="utf-8")
    original_open = Path.open

    def refuse_prediction(path: Path, *args, **kwargs):
        if path == prediction:
            raise AssertionError("forbidden prediction file was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_prediction)
    with pytest.raises(BundleValidationError, match="forbidden bundle file"):
        TruthFreeRecordedBundle.load(bundle)


@pytest.mark.parametrize(
    "forbidden",
    ("targets", "events", "labels", "depth", "prediction_csv"),
)
def test_loader_rejects_forbidden_json_even_if_manifest_is_rewritten(
    tmp_path: Path, forbidden: str
) -> None:
    bundle = _bundle(tmp_path)

    def add_forbidden(payload: dict[str, object]) -> None:
        payload[forbidden] = {"ground_truth": True}

    _rewrite_trip_payload(bundle, add_forbidden)
    with pytest.raises(BundleValidationError, match="forbidden fields"):
        TruthFreeRecordedBundle.load(bundle)


def test_loader_rejects_noncausal_ego_even_if_manifest_is_rewritten(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    def add_label(payload: dict[str, object]) -> None:
        frames = payload["frames"]
        assert isinstance(frames, list)
        ego = frames[0]["ego"]
        ego["driver_state_label"] = "drowsy"

    _rewrite_trip_payload(bundle, add_label)
    with pytest.raises(BundleValidationError, match="only causal fields"):
        TruthFreeRecordedBundle.load(bundle)


def test_loader_never_synthesizes_a_missing_camera_frame(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "driver" / "frame_000001.jpg").unlink()

    with pytest.raises(BundleValidationError, match="inventory mismatch"):
        TruthFreeRecordedBundle.load(bundle)


def test_loader_rechecks_manifest_checksum_and_runtime_asset(tmp_path: Path) -> None:
    bundle_path = _bundle(tmp_path)
    road = bundle_path / "kitti" / "image_2" / "000000.jpg"
    road.write_bytes(b"tampered-before-load")
    with pytest.raises(BundleValidationError, match="size mismatch|checksum mismatch"):
        TruthFreeRecordedBundle.load(bundle_path)

    bundle_path = _bundle(tmp_path / "second")
    bundle = TruthFreeRecordedBundle.load(bundle_path)
    bundle.frames[0].road.path.write_bytes(b"tampered-after-load")
    with pytest.raises(BundleValidationError, match="changed after validation"):
        bundle.frames[0].road.read_verified()


def test_loader_requires_a_20hz_source_bundle(tmp_path: Path) -> None:
    with pytest.raises(BundleValidationError, match="requires a 20 Hz bundle"):
        TruthFreeRecordedBundle.load(_bundle(tmp_path, fps=10))


class FakeClock:
    def __init__(self) -> None:
        self.monotonic = 0
        self.wall = 1_800_000_000_000_000_000
        self.sleeps_ns: list[int] = []

    def monotonic_ns(self) -> int:
        return self.monotonic

    def time_ns(self) -> int:
        return self.wall

    def sleep(self, seconds: float) -> None:
        duration = round(seconds * 1_000_000_000)
        self.sleeps_ns.append(duration)
        self.advance(duration)

    def advance(self, duration_ns: int) -> None:
        self.monotonic += duration_ns
        self.wall += duration_ns


class CapturingAdapter:
    def __init__(self, clock: FakeClock, *, send_time_ns: int = 0) -> None:
        self.clock = clock
        self.send_time_ns = send_time_ns
        self.session = None
        self.frames = []
        self.emission_monotonic_ns: list[int] = []
        self.emission_wall_ms: list[int] = []
        self.closed = False

    def open(self, session) -> None:
        self.session = session

    def emit(self, frame) -> None:
        self.frames.append(frame)
        self.emission_monotonic_ns.append(self.clock.monotonic_ns())
        self.emission_wall_ms.append(self.clock.time_ns() // 1_000_000)
        self.clock.advance(self.send_time_ns)

    def close(self) -> None:
        self.closed = True


def test_sender_uses_absolute_monotonic_20hz_pacing_without_drift(
    tmp_path: Path,
) -> None:
    bundle = TruthFreeRecordedBundle.load(_bundle(tmp_path, count=5))
    clock = FakeClock()
    adapter = CapturingAdapter(clock, send_time_ns=10_000_000)
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="p2-t02",
        generation=7,
        clock=clock,
    )

    stats = sender.run()

    assert adapter.emission_monotonic_ns == [
        0,
        50_000_000,
        100_000_000,
        150_000_000,
        200_000_000,
    ]
    # Send work consumes 10 ms, so absolute deadlines sleep only the
    # remaining 40 ms.  A relative sleep(50 ms) loop would drift to 60 ms.
    assert clock.sleeps_ns == [40_000_000] * 4
    assert [frame.capture_timestamp_ms for frame in adapter.frames] == [
        1_800_000_000_000,
        1_800_000_000_050,
        1_800_000_000_100,
        1_800_000_000_150,
        1_800_000_000_200,
    ]
    assert [frame.capture_timestamp_ms for frame in adapter.frames] == (
        adapter.emission_wall_ms
    )
    assert stats.frames_emitted == 5
    assert stats.late_frames == 0
    assert adapter.closed


def test_sender_preserves_exact_mapping_labels_timestamps_and_payloads(
    tmp_path: Path,
) -> None:
    bundle = TruthFreeRecordedBundle.load(_bundle(tmp_path))
    clock = FakeClock()
    adapter = CapturingAdapter(clock)
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="vehicle-42:boot-a",
        generation=3,
        clock=clock,
    )

    sender.run(limit=3)

    assert adapter.session.metadata() == {
        "session_id": "vehicle-42:boot-a",
        "generation": 3,
        "source_hz": 20,
        "video_source": "RECORDED_STREAM",
        "inference_mode": "LIVE_MODEL",
        "media_ids": {
            "road": "vehicle-42:boot-a:g3:road",
            "cabin": "vehicle-42:boot-a:g3:cabin",
        },
    }
    assert [frame.mapping_key for frame in adapter.frames] == [
        ("vehicle-42:boot-a", 3, 0),
        ("vehicle-42:boot-a", 3, 1),
        ("vehicle-42:boot-a", 3, 2),
    ]
    assert [frame.source_media_timestamp_ms for frame in adapter.frames] == [
        0,
        50,
        100,
    ]
    assert [frame.road.rtp_timestamp for frame in adapter.frames] == [0, 4500, 9000]
    assert [frame.cabin.rtp_timestamp for frame in adapter.frames] == [
        0,
        4500,
        9000,
    ]

    second = adapter.frames[1]
    assert second.schema == FRAME_SCHEMA
    assert second.video_source == "RECORDED_STREAM"
    assert second.inference_mode == "LIVE_MODEL"
    assert second.road.media_id == "vehicle-42:boot-a:g3:road"
    assert second.cabin.media_id == "vehicle-42:boot-a:g3:cabin"
    assert second.road.source_sequence == second.cabin.source_sequence == 1
    assert second.road.rtp_clock_rate_hz == second.cabin.rtp_clock_rate_hz == 90_000
    assert second.road.source_asset_id == "kitti/image_2/000001.jpg"
    assert second.cabin.source_asset_id == "driver/frame_000001.jpg"
    assert second.road.payload == b"road-1"
    assert second.cabin.payload == b"cabin-1"
    assert dict(second.ego) == {
        "speed_kmh": 31.0,
        "longitudinal_accel": -0.1,
        "lateral_accel": 0.2,
    }
    with pytest.raises(TypeError):
        second.ego["speed_kmh"] = 99.0


def test_sender_closes_adapter_if_transport_emit_fails(tmp_path: Path) -> None:
    bundle = TruthFreeRecordedBundle.load(_bundle(tmp_path))
    clock = FakeClock()

    class FailingAdapter(CapturingAdapter):
        def emit(self, frame) -> None:
            super().emit(frame)
            raise RuntimeError("transport queue unavailable")

    adapter = FailingAdapter(clock)
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="failure-test",
        generation=0,
        clock=clock,
    )
    with pytest.raises(RuntimeError, match="queue unavailable"):
        sender.run()
    assert adapter.closed


class PersistentH264TestTransform:
    """Test double for one session-scoped pair of persistent codec pipes."""

    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.transformed = []

    def open(self, session) -> None:
        self.open_count += 1

    def transform(self, frame):
        self.transformed.append(frame)
        payload = b"annex-b-h264:" + frame.payload
        return replace(
            frame,
            content_type="video/h264",
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def close(self) -> None:
        self.close_count += 1


def test_persistent_codec_hook_opens_once_and_preserves_explicit_mapping(
    tmp_path: Path,
) -> None:
    bundle = TruthFreeRecordedBundle.load(_bundle(tmp_path))
    clock = FakeClock()
    downstream = CapturingAdapter(clock)
    transform = PersistentH264TestTransform()
    adapter = TransformingTransportAdapter(transform, downstream)
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="persistent-h264-test",
        generation=5,
        clock=clock,
    )

    sender.run(limit=3)

    assert transform.open_count == transform.close_count == 1
    assert len(transform.transformed) == 6
    assert [frame.mapping_key for frame in downstream.frames] == [
        ("persistent-h264-test", 5, 0),
        ("persistent-h264-test", 5, 1),
        ("persistent-h264-test", 5, 2),
    ]
    assert [frame.road.rtp_timestamp for frame in downstream.frames] == [
        0,
        4500,
        9000,
    ]
    assert all(
        media.content_type == "video/h264"
        for frame in downstream.frames
        for media in (frame.road, frame.cabin)
    )


def test_codec_hook_rejects_callback_order_as_frame_identity(tmp_path: Path) -> None:
    bundle = TruthFreeRecordedBundle.load(_bundle(tmp_path))
    clock = FakeClock()
    downstream = CapturingAdapter(clock)

    class OrderOnlyTransform(PersistentH264TestTransform):
        def transform(self, frame):
            encoded = super().transform(frame)
            return replace(encoded, source_sequence=frame.source_sequence + 1)

    transform = OrderOnlyTransform()
    adapter = TransformingTransportAdapter(transform, downstream)
    sender = RecordedStreamSender(
        bundle,
        adapter,
        session_id="bad-codec-mapping",
        generation=0,
        clock=clock,
    )

    with pytest.raises(RuntimeError, match="changed mapping identity"):
        sender.run(limit=1)
    assert downstream.frames == []
    assert transform.open_count == transform.close_count == 1
    assert downstream.closed
