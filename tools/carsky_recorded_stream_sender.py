#!/usr/bin/env python3
"""Pace a truth-free CarSky demo bundle as a recorded realtime source.

This module is deliberately source-side and transport agnostic.  It accepts
only the exact ``safeloop.carsky.demo-bundle.v1`` output produced by
``prepare_carsky_demo.py``.  There is no fallback to a practice-dataset
directory, prediction file, label, target, event, or depth modality.

The default command-line adapter prints payload metadata for local inspection;
it makes no AWS connection.  A future WebRTC/Kinesis adapter implements
``TransportAdapter`` and receives the same immutable road, cabin, ego, timing,
and mapping contract.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TextIO


BUNDLE_SCHEMA = "safeloop.carsky.demo-bundle.v1"
FRAME_SCHEMA = "safeloop.carsky.recorded-stream-frame.v1"
VIDEO_SOURCE = "RECORDED_STREAM"
INFERENCE_MODE = "LIVE_MODEL"
SOURCE_HZ = 20
PERIOD_NS = 1_000_000_000 // SOURCE_HZ
RTP_CLOCK_RATE_HZ = 90_000
RTP_MODULUS = 2**32

METADATA_FIELDS = frozenset(
    {
        "trip_id",
        "description",
        "duration_sec",
        "fps",
        "weather",
        "speed_limit_kmh",
    }
)
EGO_FIELDS = (
    "speed_kmh",
    "longitudinal_accel",
    "lateral_accel",
)
MODALITIES = (
    "road_camera_left",
    "driver_camera",
    "ego_kinematics",
)
FORBIDDEN_MODALITIES = (
    "depth",
    "image_3",
    "label_2",
    "calib",
)
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "trip_id",
        "frames",
        "source_fps",
        "modalities",
        "truth_free",
        "forbidden_modalities_absent",
        "files",
    }
)
ALLOWED_DIRECTORIES = frozenset({"driver", "kitti", "kitti/image_2"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
TRIP_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)
SESSION_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", re.ASCII
)
ROAD_NAME_PATTERN = re.compile(r"[0-9]{6,}\.(?:jpg|jpeg|png)", re.ASCII)
CABIN_NAME_PATTERN = re.compile(
    r"frame_[0-9]{6,}\.(?:jpg|jpeg|png)", re.ASCII
)


class BundleValidationError(ValueError):
    """The supplied path is not an intact truth-free runtime bundle."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise BundleValidationError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
    except BundleValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"cannot read {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleValidationError(f"{field} must be a JSON object")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BundleValidationError(f"{field} must be finite")
    return number


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BundleValidationError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    number = _non_negative_int(value, field=field)
    if number == 0:
        raise BundleValidationError(f"{field} must be positive")
    return number


def _safe_trip_id(value: object, *, directory_name: str) -> str:
    if not isinstance(value, str) or TRIP_ID_PATTERN.fullmatch(value) is None:
        raise BundleValidationError("manifest.trip_id is unsafe")
    if value != directory_name:
        raise BundleValidationError(
            "manifest.trip_id must match the bundle directory name"
        )
    return value


def _safe_session_id(value: object) -> str:
    if not isinstance(value, str) or SESSION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "session_id must use 1-128 ASCII letters, digits, '.', '_', ':', or '-'"
        )
    return value


def _manifest_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BundleValidationError("manifest file path is not canonical")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BundleValidationError(f"unsafe manifest file path: {value!r}")
    return value


def _is_allowed_image(path: PurePosixPath) -> bool:
    if len(path.parts) == 2 and path.parts[0] == "driver":
        return CABIN_NAME_PATTERN.fullmatch(path.name) is not None
    if len(path.parts) == 3 and path.parts[:2] == ("kitti", "image_2"):
        return ROAD_NAME_PATTERN.fullmatch(path.name) is not None
    return False


def _is_allowed_file(path: PurePosixPath, *, trip_id: str) -> bool:
    if len(path.parts) == 1:
        return path.name in {"BUNDLE_MANIFEST.json", f"{trip_id}.json"}
    return _is_allowed_image(path)


def _scan_allowed_tree(root: Path, *, trip_id: str) -> dict[str, Path]:
    """List names/types without following or opening any unapproved entry."""

    found: dict[str, Path] = {}
    pending: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise BundleValidationError(f"cannot scan bundle: {exc}") from exc
        for entry in entries:
            parts = (*prefix, entry.name)
            relative = PurePosixPath(*parts)
            relative_name = relative.as_posix()
            if entry.is_symlink():
                raise BundleValidationError(
                    f"bundle symlinks are forbidden: {relative_name}"
                )
            if entry.is_dir(follow_symlinks=False):
                if relative_name not in ALLOWED_DIRECTORIES:
                    raise BundleValidationError(
                        f"unexpected/forbidden bundle directory: {relative_name}"
                    )
                pending.append((Path(entry.path), parts))
                continue
            if not entry.is_file(follow_symlinks=False):
                raise BundleValidationError(
                    f"non-regular bundle entry is forbidden: {relative_name}"
                )
            if not _is_allowed_file(relative, trip_id=trip_id):
                raise BundleValidationError(
                    f"unexpected/forbidden bundle file: {relative_name}"
                )
            found[relative_name] = Path(entry.path)
    return found


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BundleValidationError(f"cannot hash bundle file {path.name}: {exc}") from exc
    return digest.hexdigest()


def _validate_manifest_entry(
    relative_name: str,
    raw: object,
    *,
    path: Path,
) -> tuple[int, str]:
    if not isinstance(raw, dict) or set(raw) != {"bytes", "sha256"}:
        raise BundleValidationError(
            f"manifest.files[{relative_name!r}] must contain bytes and sha256"
        )
    size = _non_negative_int(
        raw["bytes"], field=f"manifest.files[{relative_name!r}].bytes"
    )
    digest = raw["sha256"]
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest, re.ASCII) is None
    ):
        raise BundleValidationError(
            f"manifest.files[{relative_name!r}].sha256 is invalid"
        )
    try:
        current_size = path.stat().st_size
    except OSError as exc:
        raise BundleValidationError(
            f"cannot stat bundle file {relative_name}: {exc}"
        ) from exc
    if current_size != size:
        raise BundleValidationError(f"bundle size mismatch: {relative_name}")
    current_digest = _sha256_file(path)
    if not hmac.compare_digest(current_digest, digest):
        raise BundleValidationError(f"bundle checksum mismatch: {relative_name}")
    return size, digest


def _milliseconds_from_source_timestamp(value: object, *, field: str) -> int:
    seconds = _finite_number(value, field=field)
    if seconds < 0.0:
        raise BundleValidationError(f"{field} must be non-negative")
    milliseconds = seconds * 1000.0
    rounded = int(round(milliseconds))
    if not math.isclose(milliseconds, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise BundleValidationError(
            f"{field} cannot be represented exactly in integer milliseconds"
        )
    return rounded


def _content_type(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def _freeze_json_value(value: Any) -> Any:
    """Recursively detach and freeze already-sanitized metadata values."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


@dataclass(frozen=True)
class BundleAsset:
    """One allow-listed image whose identity is bound by the manifest."""

    path: Path
    relative_id: str
    content_type: str
    size_bytes: int
    sha256: str

    def read_verified(self) -> bytes:
        """Read only this allow-listed file and recheck it against the manifest."""

        try:
            metadata = self.path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise BundleValidationError(
                    f"bundle asset is no longer a regular file: {self.relative_id}"
                )
            payload = self.path.read_bytes()
        except BundleValidationError:
            raise
        except OSError as exc:
            raise BundleValidationError(
                f"cannot read bundle asset {self.relative_id}: {exc}"
            ) from exc
        if len(payload) != self.size_bytes or not hmac.compare_digest(
            _sha256_bytes(payload), self.sha256
        ):
            raise BundleValidationError(
                f"bundle asset changed after validation: {self.relative_id}"
            )
        return payload


@dataclass(frozen=True)
class RecordedBundleFrame:
    """One source tick copied verbatim from the sanitized bundle contract."""

    source_sequence: int
    source_media_timestamp_ms: int
    ego: Mapping[str, float]
    road: BundleAsset
    cabin: BundleAsset


@dataclass(frozen=True)
class TruthFreeRecordedBundle:
    """Validated, immutable index over ``prepare_carsky_demo.py`` output."""

    root: Path
    trip_id: str
    metadata: Mapping[str, Any]
    frames: tuple[RecordedBundleFrame, ...]
    manifest_path: Path

    @classmethod
    def load(cls, bundle_root: Path | str) -> "TruthFreeRecordedBundle":
        supplied_root = Path(bundle_root)
        if supplied_root.is_symlink():
            raise BundleValidationError("bundle root symlink is forbidden")
        root = supplied_root.resolve()
        if not root.is_dir():
            raise BundleValidationError(f"bundle directory does not exist: {root}")

        # A raw Txx-Sample directory stops here.  We never probe its trip JSON,
        # images, labels, depth, targets, events, or prediction artifacts.
        manifest_path = root / "BUNDLE_MANIFEST.json"
        try:
            manifest_metadata = manifest_path.stat(follow_symlinks=False)
        except OSError as exc:
            raise BundleValidationError(
                "BUNDLE_MANIFEST.json is required; raw/mixed dataset directories "
                "are forbidden"
            ) from exc
        if not stat.S_ISREG(manifest_metadata.st_mode):
            raise BundleValidationError("BUNDLE_MANIFEST.json must be a regular file")

        manifest = _read_json(manifest_path, field="bundle manifest")
        if set(manifest) != MANIFEST_FIELDS:
            raise BundleValidationError("bundle manifest fields do not match v1")
        if manifest["schema"] != BUNDLE_SCHEMA:
            raise BundleValidationError("unsupported bundle manifest schema")
        if manifest["truth_free"] is not True:
            raise BundleValidationError("bundle manifest must assert truth_free=true")
        trip_id = _safe_trip_id(manifest["trip_id"], directory_name=root.name)
        frame_count = _positive_int(manifest["frames"], field="manifest.frames")
        source_fps = _finite_number(
            manifest["source_fps"], field="manifest.source_fps"
        )
        if source_fps != float(SOURCE_HZ):
            raise BundleValidationError("recorded sender requires a 20 Hz bundle")
        if manifest["modalities"] != list(MODALITIES):
            raise BundleValidationError("bundle modalities do not match the truth-free contract")
        if manifest["forbidden_modalities_absent"] != list(FORBIDDEN_MODALITIES):
            raise BundleValidationError(
                "bundle forbidden-modality declaration does not match v1"
            )

        actual_files = _scan_allowed_tree(root, trip_id=trip_id)
        manifest_files_raw = manifest["files"]
        if not isinstance(manifest_files_raw, dict):
            raise BundleValidationError("manifest.files must be an object")
        manifest_names: set[str] = set()
        for raw_name in manifest_files_raw:
            manifest_names.add(_manifest_relative_path(raw_name))
        expected_inventory = set(actual_files) - {"BUNDLE_MANIFEST.json"}
        if manifest_names != expected_inventory:
            missing = sorted(manifest_names - expected_inventory)
            extra = sorted(expected_inventory - manifest_names)
            raise BundleValidationError(
                f"bundle inventory mismatch (missing={missing[:3]}, extra={extra[:3]})"
            )

        manifest_entries: dict[str, tuple[int, str]] = {}
        for relative_name in sorted(manifest_names):
            manifest_entries[relative_name] = _validate_manifest_entry(
                relative_name,
                manifest_files_raw[relative_name],
                path=actual_files[relative_name],
            )

        payload_name = f"{trip_id}.json"
        if payload_name not in manifest_entries:
            raise BundleValidationError("sanitized trip JSON is absent from manifest")
        payload = _read_json(actual_files[payload_name], field="sanitized trip JSON")
        if set(payload) != {"trip_id", "metadata", "frames"}:
            raise BundleValidationError("sanitized trip JSON contains forbidden fields")
        if payload["trip_id"] != trip_id:
            raise BundleValidationError("trip JSON ID does not match manifest")

        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise BundleValidationError("trip metadata must be an object")
        if not {"trip_id", "fps", "speed_limit_kmh"}.issubset(metadata):
            raise BundleValidationError("trip metadata lacks required bundle fields")
        if not set(metadata).issubset(METADATA_FIELDS):
            raise BundleValidationError("trip metadata contains non-runtime fields")
        if metadata["trip_id"] != trip_id:
            raise BundleValidationError("metadata.trip_id does not match manifest")
        if _finite_number(metadata["fps"], field="metadata.fps") != float(SOURCE_HZ):
            raise BundleValidationError("metadata.fps must be 20")
        if _finite_number(
            metadata["speed_limit_kmh"], field="metadata.speed_limit_kmh"
        ) < 0.0:
            raise BundleValidationError("metadata.speed_limit_kmh must be non-negative")

        raw_frames = payload["frames"]
        if not isinstance(raw_frames, list) or len(raw_frames) != frame_count:
            raise BundleValidationError("trip frame count does not match manifest")

        indexed_frames: list[RecordedBundleFrame] = []
        expected_names = {payload_name}
        previous_timestamp_ms: int | None = None
        for expected_sequence, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, dict) or set(raw_frame) != {
                "frame_id",
                "timestamp",
                "ego",
            }:
                raise BundleValidationError(
                    f"frames[{expected_sequence}] contains forbidden/missing fields"
                )
            source_sequence = _non_negative_int(
                raw_frame["frame_id"], field=f"frames[{expected_sequence}].frame_id"
            )
            if source_sequence != expected_sequence:
                raise BundleValidationError(
                    "bundle source_sequence must be contiguous and start at zero"
                )
            source_timestamp_ms = _milliseconds_from_source_timestamp(
                raw_frame["timestamp"], field=f"frames[{expected_sequence}].timestamp"
            )
            if (
                previous_timestamp_ms is not None
                and source_timestamp_ms <= previous_timestamp_ms
            ):
                raise BundleValidationError(
                    "bundle source media timestamps must be strictly increasing"
                )
            previous_timestamp_ms = source_timestamp_ms

            raw_ego = raw_frame["ego"]
            if not isinstance(raw_ego, dict) or set(raw_ego) != set(EGO_FIELDS):
                raise BundleValidationError(
                    f"frames[{expected_sequence}].ego must contain only causal fields"
                )
            ego = MappingProxyType(
                {
                    field: _finite_number(
                        raw_ego[field],
                        field=f"frames[{expected_sequence}].ego.{field}",
                    )
                    for field in EGO_FIELDS
                }
            )

            road_prefix = f"kitti/image_2/{source_sequence:06d}."
            cabin_prefix = f"driver/frame_{source_sequence:06d}."
            road_names = sorted(
                name for name in manifest_names if name.startswith(road_prefix)
            )
            cabin_names = sorted(
                name for name in manifest_names if name.startswith(cabin_prefix)
            )
            if len(road_names) != 1 or len(cabin_names) != 1:
                raise BundleValidationError(
                    f"source tick {source_sequence} must have exactly one road and cabin frame"
                )

            def asset(relative_name: str) -> BundleAsset:
                size, digest = manifest_entries[relative_name]
                path = actual_files[relative_name]
                return BundleAsset(
                    path=path,
                    relative_id=relative_name,
                    content_type=_content_type(path),
                    size_bytes=size,
                    sha256=digest,
                )

            expected_names.update((road_names[0], cabin_names[0]))
            indexed_frames.append(
                RecordedBundleFrame(
                    source_sequence=source_sequence,
                    source_media_timestamp_ms=source_timestamp_ms,
                    ego=ego,
                    road=asset(road_names[0]),
                    cabin=asset(cabin_names[0]),
                )
            )

        if manifest_names != expected_names:
            raise BundleValidationError("bundle contains unpaired or surplus media frames")
        return cls(
            root=root,
            trip_id=trip_id,
            metadata=_freeze_json_value(metadata),
            frames=tuple(indexed_frames),
            manifest_path=manifest_path,
        )


@dataclass(frozen=True)
class TransportSession:
    """Stable stream identity negotiated once by a production adapter."""

    session_id: str
    generation: int
    source_hz: int
    video_source: str
    inference_mode: str
    road_media_id: str
    cabin_media_id: str

    def metadata(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "generation": self.generation,
            "source_hz": self.source_hz,
            "video_source": self.video_source,
            "inference_mode": self.inference_mode,
            "media_ids": {
                "road": self.road_media_id,
                "cabin": self.cabin_media_id,
            },
        }


@dataclass(frozen=True)
class TransportMediaFrame:
    """One camera payload and the RTP identity an adapter must preserve."""

    stream_id: str
    media_id: str
    source_sequence: int
    rtp_timestamp: int
    rtp_clock_rate_hz: int
    source_asset_id: str
    content_type: str
    payload_sha256: str
    payload: bytes

    def metadata(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "media_id": self.media_id,
            "source_sequence": self.source_sequence,
            "rtp_timestamp": self.rtp_timestamp,
            "rtp_clock_rate_hz": self.rtp_clock_rate_hz,
            "source_asset_id": self.source_asset_id,
            "content_type": self.content_type,
            "payload_bytes": len(self.payload),
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class TransportFrame:
    """Atomic road+cabin+ego input presented to the transport adapter.

    ``(session_id, generation, source_sequence)`` is the canonical mapping
    key.  Each camera payload additionally carries its stable media ID and RTP
    timestamp so receiver-side decoded frames can be joined to the same key.
    """

    schema: str
    session_id: str
    generation: int
    source_sequence: int
    video_source: str
    inference_mode: str
    capture_timestamp_ms: int
    source_media_timestamp_ms: int
    ego: Mapping[str, float]
    road: TransportMediaFrame
    cabin: TransportMediaFrame

    @property
    def mapping_key(self) -> tuple[str, int, int]:
        return (self.session_id, self.generation, self.source_sequence)

    def metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "generation": self.generation,
            "source_sequence": self.source_sequence,
            "video_source": self.video_source,
            "inference_mode": self.inference_mode,
            "capture_timestamp_ms": self.capture_timestamp_ms,
            "source_media_timestamp_ms": self.source_media_timestamp_ms,
            "ego": dict(self.ego),
            "road": self.road.metadata(),
            "cabin": self.cabin.metadata(),
        }


class TransportAdapter(Protocol):
    """Boundary implemented by local, WebRTC, SRT, or Kinesis transports.

    ``emit`` receives a complete immutable source tick.  Production adapters
    must preserve its mapping key in telemetry/data-channel metadata and bind
    the road/cabin payloads through their respective media IDs and 90-kHz RTP
    timestamps.  The method should enqueue without waiting for cloud delivery.
    """

    def open(self, session: TransportSession) -> None: ...

    def emit(self, frame: TransportFrame) -> None: ...

    def close(self) -> None: ...


class PersistentMediaTransform(Protocol):
    """Optional long-lived codec/packetizer boundary.

    ``open`` and ``close`` are session-scoped: implementations may own two
    persistent FFmpeg/GStreamer/WebRTC pipelines, but must never spawn a codec
    process per frame.  ``transform`` must derive output identity from packet
    PTS/RTP/application metadata and return it unchanged.  Decoder callback
    order is not a mapping signal.

    A raw FFmpeg pipe alone cannot prove this identity contract, so this module
    intentionally provides the guarded hook rather than an order-based H.264
    implementation.
    """

    def open(self, session: TransportSession) -> None: ...

    def transform(self, frame: TransportMediaFrame) -> TransportMediaFrame: ...

    def close(self) -> None: ...


def _validate_transformed_media(
    source: TransportMediaFrame,
    transformed: object,
) -> TransportMediaFrame:
    if not isinstance(transformed, TransportMediaFrame):
        raise TypeError("media transform must return TransportMediaFrame")
    identity_fields = (
        "stream_id",
        "media_id",
        "source_sequence",
        "rtp_timestamp",
        "rtp_clock_rate_hz",
        "source_asset_id",
    )
    changed = [
        field
        for field in identity_fields
        if getattr(source, field) != getattr(transformed, field)
    ]
    if changed:
        raise RuntimeError(
            "media transform changed mapping identity: " + ", ".join(changed)
        )
    if not isinstance(transformed.payload, bytes):
        raise TypeError("transformed media payload must be immutable bytes")
    if not hmac.compare_digest(
        _sha256_bytes(transformed.payload), transformed.payload_sha256
    ):
        raise RuntimeError("transformed media payload digest is invalid")
    return transformed


class TransformingTransportAdapter:
    """Apply one persistent media transform before a downstream transport.

    The wrapper is the enforcement point for H.264 encode/decode hooks.  It
    rejects any transform that silently associates output by callback order
    and therefore changes the explicit media/RTP mapping identity.
    """

    def __init__(
        self,
        transform: PersistentMediaTransform,
        downstream: TransportAdapter,
    ) -> None:
        self.transform = transform
        self.downstream = downstream
        self._opened = False

    def open(self, session: TransportSession) -> None:
        if self._opened:
            raise RuntimeError("transforming adapter is already open")
        self.transform.open(session)
        try:
            self.downstream.open(session)
        except BaseException:
            self.transform.close()
            raise
        self._opened = True

    def emit(self, frame: TransportFrame) -> None:
        if not self._opened:
            raise RuntimeError("transforming adapter is not open")
        road = _validate_transformed_media(
            frame.road, self.transform.transform(frame.road)
        )
        cabin = _validate_transformed_media(
            frame.cabin, self.transform.transform(frame.cabin)
        )
        self.downstream.emit(replace(frame, road=road, cabin=cabin))

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self.downstream.close()
        finally:
            self._opened = False
            self.transform.close()


class SenderClock(Protocol):
    def monotonic_ns(self) -> int: ...

    def time_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def time_ns(self) -> int:
        return time.time_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class SenderStats:
    frames_emitted: int
    first_capture_timestamp_ms: int | None
    last_capture_timestamp_ms: int | None
    late_frames: int
    max_lateness_ms: float

    def metadata(self) -> dict[str, object]:
        return {
            "frames_emitted": self.frames_emitted,
            "first_capture_timestamp_ms": self.first_capture_timestamp_ms,
            "last_capture_timestamp_ms": self.last_capture_timestamp_ms,
            "late_frames": self.late_frames,
            "max_lateness_ms": self.max_lateness_ms,
        }


class RecordedStreamSender:
    """Emit every real bundle tick against absolute monotonic deadlines."""

    def __init__(
        self,
        bundle: TruthFreeRecordedBundle,
        adapter: TransportAdapter,
        *,
        session_id: str,
        generation: int,
        clock: SenderClock | None = None,
    ) -> None:
        if not isinstance(bundle, TruthFreeRecordedBundle):
            raise TypeError("bundle must be a TruthFreeRecordedBundle")
        self.bundle = bundle
        self.adapter = adapter
        self.session_id = _safe_session_id(session_id)
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        self.generation = generation
        self.clock = clock or SystemClock()
        media_prefix = f"{self.session_id}:g{self.generation}"
        self.session = TransportSession(
            session_id=self.session_id,
            generation=self.generation,
            source_hz=SOURCE_HZ,
            video_source=VIDEO_SOURCE,
            inference_mode=INFERENCE_MODE,
            road_media_id=f"{media_prefix}:road",
            cabin_media_id=f"{media_prefix}:cabin",
        )

    def _media_frame(
        self,
        source: RecordedBundleFrame,
        *,
        stream_id: str,
        media_id: str,
        asset: BundleAsset,
        payload: bytes,
    ) -> TransportMediaFrame:
        rtp_timestamp = (
            source.source_media_timestamp_ms * RTP_CLOCK_RATE_HZ // 1000
        ) % RTP_MODULUS
        return TransportMediaFrame(
            stream_id=stream_id,
            media_id=media_id,
            source_sequence=source.source_sequence,
            rtp_timestamp=rtp_timestamp,
            rtp_clock_rate_hz=RTP_CLOCK_RATE_HZ,
            source_asset_id=asset.relative_id,
            content_type=asset.content_type,
            payload_sha256=asset.sha256,
            payload=payload,
        )

    def run(self, *, limit: int | None = None) -> SenderStats:
        if limit is None:
            count = len(self.bundle.frames)
        else:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            count = min(limit, len(self.bundle.frames))

        opened = False
        emitted = 0
        first_capture: int | None = None
        last_capture: int | None = None
        late_frames = 0
        max_lateness_ns = 0
        try:
            self.adapter.open(self.session)
            opened = True
            start_ns = self.clock.monotonic_ns()
            for offset, source in enumerate(self.bundle.frames[:count]):
                # Verify/prefetch before the deadline; capture time is sampled
                # only when both original media payloads are ready to emit.
                road_payload = source.road.read_verified()
                cabin_payload = source.cabin.read_verified()
                deadline_ns = start_ns + offset * PERIOD_NS
                remaining_ns = deadline_ns - self.clock.monotonic_ns()
                if remaining_ns > 0:
                    self.clock.sleep(remaining_ns / 1_000_000_000)
                emission_ns = self.clock.monotonic_ns()
                lateness_ns = max(0, emission_ns - deadline_ns)
                if lateness_ns > 0:
                    late_frames += 1
                    max_lateness_ns = max(max_lateness_ns, lateness_ns)

                road = self._media_frame(
                    source,
                    stream_id="road",
                    media_id=self.session.road_media_id,
                    asset=source.road,
                    payload=road_payload,
                )
                cabin = self._media_frame(
                    source,
                    stream_id="cabin",
                    media_id=self.session.cabin_media_id,
                    asset=source.cabin,
                    payload=cabin_payload,
                )
                capture_timestamp_ms = self.clock.time_ns() // 1_000_000
                frame = TransportFrame(
                    schema=FRAME_SCHEMA,
                    session_id=self.session_id,
                    generation=self.generation,
                    source_sequence=source.source_sequence,
                    video_source=VIDEO_SOURCE,
                    inference_mode=INFERENCE_MODE,
                    capture_timestamp_ms=capture_timestamp_ms,
                    source_media_timestamp_ms=source.source_media_timestamp_ms,
                    ego=source.ego,
                    road=road,
                    cabin=cabin,
                )
                self.adapter.emit(frame)
                emitted += 1
                if first_capture is None:
                    first_capture = capture_timestamp_ms
                last_capture = capture_timestamp_ms
        finally:
            if opened:
                self.adapter.close()
        return SenderStats(
            frames_emitted=emitted,
            first_capture_timestamp_ms=first_capture,
            last_capture_timestamp_ms=last_capture,
            late_frames=late_frames,
            max_lateness_ms=max_lateness_ns / 1_000_000.0,
        )


class JsonlInspectionAdapter:
    """Local no-network adapter; binary payloads are received but not printed."""

    def __init__(self, output: TextIO) -> None:
        self.output = output
        self._opened = False

    def _write(self, value: Mapping[str, object]) -> None:
        print(
            json.dumps(value, separators=(",", ":"), allow_nan=False),
            file=self.output,
            flush=True,
        )

    def open(self, session: TransportSession) -> None:
        if self._opened:
            raise RuntimeError("adapter is already open")
        self._opened = True
        self._write({"event": "session_open", **session.metadata()})

    def emit(self, frame: TransportFrame) -> None:
        if not self._opened:
            raise RuntimeError("adapter is not open")
        self._write({"event": "frame", **frame.metadata()})

    def close(self) -> None:
        if self._opened:
            self._write({"event": "session_close"})
        self._opened = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a truth-free CarSky demo bundle at 20 Hz as "
            "RECORDED_STREAM + LIVE_MODEL input"
        )
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate the complete bundle without emitting frames",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        bundle = TruthFreeRecordedBundle.load(args.bundle)
        if args.verify_only:
            print(
                json.dumps(
                    {
                        "event": "bundle_verified",
                        "schema": BUNDLE_SCHEMA,
                        "trip_id": bundle.trip_id,
                        "frames": len(bundle.frames),
                        "source_hz": SOURCE_HZ,
                        "truth_free": True,
                        "manifest": str(bundle.manifest_path),
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        if args.session_id is None:
            parser.error("--session-id is required unless --verify-only is used")
        adapter = JsonlInspectionAdapter(sys.stdout)
        sender = RecordedStreamSender(
            bundle,
            adapter,
            session_id=args.session_id,
            generation=args.generation,
        )
        stats = sender.run(limit=args.limit)
        print(
            json.dumps(
                {"event": "send_complete", **stats.metadata()},
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except (BundleValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA",
    "BundleValidationError",
    "FRAME_SCHEMA",
    "INFERENCE_MODE",
    "JsonlInspectionAdapter",
    "RecordedStreamSender",
    "SenderStats",
    "SOURCE_HZ",
    "TransportAdapter",
    "TransportFrame",
    "TransportMediaFrame",
    "TransportSession",
    "PersistentMediaTransform",
    "TransformingTransportAdapter",
    "TruthFreeRecordedBundle",
    "VIDEO_SOURCE",
]
