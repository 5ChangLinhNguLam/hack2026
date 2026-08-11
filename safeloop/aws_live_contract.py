"""Strict input and prepared-demo contracts for the AWS fast bridge.

The module is intentionally independent from FastAPI.  It accepts exactly one
road JPEG, one cabin JPEG and one ego sample per source tick, owns the decoded
pixels, and has no dataset/replay fallback.  The recorded path accepts only the
output contract of ``tools/prepare_carsky_demo.py`` and verifies every manifest
digest before a model can see a frame.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

import cv2
import numpy as np


INPUT_SCHEMA = "safeloop.input.v1"
BUNDLE_SCHEMA = "safeloop.carsky.demo-bundle.v1"
SOURCE_KINDS = frozenset({"LIVE_CAMERA", "RECORDED_STREAM"})
TELEMETRY_SOURCES = frozenset({"THIRD_PARTY", "RECORDED_DATA"})
MAX_INPUT_MESSAGE_BYTES = 2_500_000
MAX_JPEG_BYTES = 900_000
MAX_IMAGE_WIDTH = 1_920
MAX_IMAGE_HEIGHT = 1_080
MAX_IMAGE_PIXELS = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT
MAX_SESSION_ID_LENGTH = 64
TRIP_ID = re.compile(r"T(?:0[1-9]|10)-Sample", re.ASCII)
EGO_FIELDS = frozenset(
    {"speed_kmh", "longitudinal_accel", "lateral_accel"}
)
RUNTIME_METADATA_FIELDS = frozenset(
    {"trip_id", "duration_sec", "fps", "weather", "speed_limit_kmh"}
)
MANIFEST_KEYS = frozenset(
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
EXPECTED_MODALITIES = (
    "road_camera_left",
    "driver_camera",
    "ego_kinematics",
)
FORBIDDEN_PATH_WORDS = frozenset(
    {
        "calib",
        "depth",
        "event",
        "events",
        "gt",
        "image_3",
        "label",
        "label_2",
        "prediction",
        "predictions",
        "target",
        "targets",
    }
)


class InputContractError(ValueError):
    """An untrusted source message or prepared bundle is invalid."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InputContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise InputContractError(f"non-finite JSON constant: {value}")


def strict_json_loads(payload: str | bytes, *, max_bytes: int) -> Any:
    if isinstance(payload, str):
        raw = payload.encode("utf-8", errors="strict")
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise InputContractError("JSON payload must be text or bytes")
    if not raw or len(raw) > max_bytes:
        raise InputContractError(f"JSON payload must be 1..{max_bytes} bytes")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputContractError("payload is not strict UTF-8 JSON") from exc


def _exact_keys(value: object, *, field: str, expected: set[str] | frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputContractError(f"{field} must be an object")
    if set(value) != set(expected):
        raise InputContractError(
            f"{field} keys must be exactly {', '.join(sorted(expected))}"
        )
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InputContractError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise InputContractError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise InputContractError(f"{field} must be <= {maximum}")
    return number


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputContractError(f"{field} must be a non-negative integer")
    return value


def _session_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_SESSION_ID_LENGTH:
        raise InputContractError(
            f"session_id must be 1..{MAX_SESSION_ID_LENGTH} characters"
        )
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise InputContractError("session_id must use printable ASCII without spaces")
    return value


def _weather(value: object) -> Mapping[str, float]:
    if isinstance(value, str):
        aliases: dict[str, dict[str, float]] = {
            "clear": {},
            "cloudy": {"cloudiness": 80.0},
            "rain": {"precipitation": 80.0, "wetness": 70.0},
            "wet": {"wetness": 80.0},
            "fog": {"fog_density": 70.0},
        }
        normalized = value.strip().lower()
        if normalized not in aliases:
            raise InputContractError(
                "metadata.weather string must be clear/cloudy/rain/wet/fog"
            )
        return MappingProxyType(aliases[normalized])
    if not isinstance(value, Mapping):
        raise InputContractError("metadata.weather must be a string or object")
    allowed = {
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "fog_density",
        "fog_distance",
        "sun_altitude_angle",
        "sun_azimuth_angle",
        "wind_intensity",
        "wetness",
    }
    if not set(value).issubset(allowed):
        raise InputContractError("metadata.weather contains unsupported fields")
    result = {
        key: _finite_number(item, field=f"metadata.weather.{key}")
        for key, item in value.items()
    }
    return MappingProxyType(result)


def normalize_metadata(value: object, *, trip_id: str | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputContractError("metadata must be an object")
    allowed = {"speed_limit_kmh", "weather"}
    if trip_id is not None:
        allowed.update({"trip_id", "duration_sec", "fps"})
    if not set(value).issubset(allowed) or not {"speed_limit_kmh", "weather"}.issubset(value):
        raise InputContractError("metadata contains missing or unsupported fields")
    result: dict[str, Any] = {
        "speed_limit_kmh": _finite_number(
            value["speed_limit_kmh"],
            field="metadata.speed_limit_kmh",
            minimum=0.0,
            maximum=300.0,
        ),
        "weather": _weather(value["weather"]),
    }
    if trip_id is not None:
        result["trip_id"] = trip_id
        result["fps"] = _finite_number(
            value.get("fps", 20.0), field="metadata.fps", minimum=0.001, maximum=120.0
        )
        result["duration_sec"] = _finite_number(
            value.get("duration_sec", 0.0),
            field="metadata.duration_sec",
            minimum=0.0,
        )
    return MappingProxyType(result)


def normalize_ego(value: object) -> Mapping[str, float]:
    ego = _exact_keys(value, field="ego", expected=EGO_FIELDS)
    return MappingProxyType(
        {
            "speed_kmh": _finite_number(
                ego["speed_kmh"], field="ego.speed_kmh", minimum=-20.0, maximum=350.0
            ),
            "longitudinal_accel": _finite_number(
                ego["longitudinal_accel"],
                field="ego.longitudinal_accel",
                minimum=-50.0,
                maximum=50.0,
            ),
            "lateral_accel": _finite_number(
                ego["lateral_accel"],
                field="ego.lateral_accel",
                minimum=-50.0,
                maximum=50.0,
            ),
        }
    )


def decode_jpeg_bytes(payload: bytes, *, field: str) -> np.ndarray:
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_JPEG_BYTES:
        raise InputContractError(f"{field} JPEG must be 1..{MAX_JPEG_BYTES} bytes")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise InputContractError(f"{field} is not a valid three-channel JPEG")
    height, width = image.shape[:2]
    if (
        width <= 0
        or height <= 0
        or width > MAX_IMAGE_WIDTH
        or height > MAX_IMAGE_HEIGHT
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise InputContractError(f"{field} resolution exceeds the PoC limit")
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    image.setflags(write=False)
    return image


def _decode_b64(value: object, *, field: str) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise InputContractError(f"{field} must be non-empty base64")
    if len(value) > ((MAX_JPEG_BYTES + 2) // 3) * 4:
        raise InputContractError(f"{field} base64 exceeds the PoC limit")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InputContractError(f"{field} is malformed base64") from exc
    return decode_jpeg_bytes(payload, field=field)


@dataclass(frozen=True)
class InputTick:
    session_id: str
    source_sequence: int
    capture_timestamp_ms: int
    source_kind: str
    telemetry_source: str
    metadata: Mapping[str, Any]
    ego: Mapping[str, float]
    road_bgr: np.ndarray
    cabin_bgr: np.ndarray
    source_media_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise InputContractError("unsupported source_kind")
        if self.telemetry_source not in TELEMETRY_SOURCES:
            raise InputContractError("unsupported telemetry_source")
        expected = (
            "THIRD_PARTY" if self.source_kind == "LIVE_CAMERA" else "RECORDED_DATA"
        )
        if self.telemetry_source != expected:
            raise InputContractError("camera and telemetry provenance do not match")
        _session_id(self.session_id)
        _non_negative_int(self.source_sequence, field="source_sequence")
        _non_negative_int(self.capture_timestamp_ms, field="capture_timestamp_ms")
        for name, image in (("road_bgr", self.road_bgr), ("cabin_bgr", self.cabin_bgr)):
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or not image.flags.c_contiguous
                or image.flags.writeable
            ):
                raise InputContractError(f"{name} must be owned contiguous read-only BGR")


def parse_input_message(payload: str | bytes) -> InputTick:
    decoded = strict_json_loads(payload, max_bytes=MAX_INPUT_MESSAGE_BYTES)
    root = _exact_keys(
        decoded,
        field="input",
        expected={
            "schema",
            "session_id",
            "seq",
            "capture_ts_ms",
            "source_kind",
            "telemetry_source",
            "metadata",
            "ego",
            "road_jpeg_b64",
            "cabin_jpeg_b64",
        },
    )
    if root["schema"] != INPUT_SCHEMA:
        raise InputContractError("unsupported input schema")
    source_kind = root["source_kind"]
    telemetry_source = root["telemetry_source"]
    if source_kind != "LIVE_CAMERA" or telemetry_source != "THIRD_PARTY":
        raise InputContractError("WSS ingest accepts LIVE_CAMERA + THIRD_PARTY only")
    return InputTick(
        session_id=_session_id(root["session_id"]),
        source_sequence=_non_negative_int(root["seq"], field="seq"),
        capture_timestamp_ms=_non_negative_int(
            root["capture_ts_ms"], field="capture_ts_ms"
        ),
        source_kind=source_kind,
        telemetry_source=telemetry_source,
        metadata=normalize_metadata(root["metadata"]),
        ego=normalize_ego(root["ego"]),
        road_bgr=_decode_b64(root["road_jpeg_b64"], field="road_jpeg_b64"),
        cabin_bgr=_decode_b64(root["cabin_jpeg_b64"], field="cabin_jpeg_b64"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_contained_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise InputContractError("bundle manifest contains an unsafe path")
    if any(part.lower() in FORBIDDEN_PATH_WORDS for part in path.parts):
        raise InputContractError("bundle manifest contains a forbidden path")
    candidate = root.joinpath(*path.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise InputContractError("bundle files must be regular files without symlinks")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise InputContractError("bundle file escapes the demo root") from exc
    return candidate


@dataclass(frozen=True)
class DemoFrame:
    source_sequence: int
    source_media_timestamp_ms: int
    ego: Mapping[str, float]
    road_path: Path
    cabin_path: Path


@dataclass(frozen=True)
class PreparedDemoBundle:
    root: Path
    trip_id: str
    metadata: Mapping[str, Any]
    frames: tuple[DemoFrame, ...]
    manifest_sha256: str

    @classmethod
    def load(cls, bundle_root: str | Path) -> "PreparedDemoBundle":
        root = Path(bundle_root).resolve()
        if not root.is_dir() or root.is_symlink():
            raise InputContractError("demo bundle root must be a real directory")
        manifest_path = root / "BUNDLE_MANIFEST.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise InputContractError("BUNDLE_MANIFEST.json is required")
        manifest_bytes = manifest_path.read_bytes()
        manifest = strict_json_loads(manifest_bytes, max_bytes=2_000_000)
        _exact_keys(manifest, field="manifest", expected=MANIFEST_KEYS)
        trip_id = manifest["trip_id"]
        if not isinstance(trip_id, str) or TRIP_ID.fullmatch(trip_id) is None:
            raise InputContractError("demo trip_id is not allow-listed")
        if root.name != trip_id or manifest["schema"] != BUNDLE_SCHEMA:
            raise InputContractError("bundle schema/trip directory mismatch")
        if manifest["truth_free"] is not True:
            raise InputContractError("bundle must assert truth_free=true")
        if tuple(manifest["modalities"]) != EXPECTED_MODALITIES:
            raise InputContractError("bundle modalities do not match the runtime contract")
        frame_count = _non_negative_int(manifest["frames"], field="manifest.frames")
        if frame_count <= 0:
            raise InputContractError("bundle must contain frames")
        source_fps = _finite_number(
            manifest["source_fps"], field="manifest.source_fps", minimum=0.001
        )
        if not math.isclose(source_fps, 20.0, abs_tol=1e-6):
            raise InputContractError("recorded demo bundle must be 20 Hz")
        files = manifest["files"]
        if not isinstance(files, Mapping):
            raise InputContractError("manifest.files must be an object")
        expected_paths = {f"{trip_id}.json"}
        for index in range(frame_count):
            expected_paths.add(f"driver/frame_{index:06d}.jpg")
            expected_paths.add(f"kitti/image_2/{index:06d}.jpg")
        if set(files) != expected_paths:
            raise InputContractError("bundle inventory is incomplete or contains extras")
        resolved: dict[str, Path] = {}
        for relative, record in files.items():
            item = _exact_keys(
                record, field=f"manifest.files.{relative}", expected={"bytes", "sha256"}
            )
            path = _regular_contained_file(root, relative)
            payload = path.read_bytes()
            size = _non_negative_int(item["bytes"], field=f"manifest.files.{relative}.bytes")
            digest = item["sha256"]
            if len(payload) != size or not isinstance(digest, str) or _sha256_bytes(payload) != digest:
                raise InputContractError(f"bundle checksum mismatch: {relative}")
            resolved[relative] = path

        trip_payload = strict_json_loads(
            resolved[f"{trip_id}.json"].read_bytes(), max_bytes=2_000_000
        )
        trip = _exact_keys(
            trip_payload, field="trip", expected={"trip_id", "metadata", "frames"}
        )
        if trip["trip_id"] != trip_id or not isinstance(trip["frames"], list):
            raise InputContractError("trip JSON identity/frames are invalid")
        if len(trip["frames"]) != frame_count:
            raise InputContractError("trip frame count differs from manifest")
        raw_metadata = trip["metadata"]
        if not isinstance(raw_metadata, Mapping):
            raise InputContractError("trip metadata must be an object")
        projected = {
            key: raw_metadata[key]
            for key in RUNTIME_METADATA_FIELDS
            if key in raw_metadata
        }
        metadata = normalize_metadata(projected, trip_id=trip_id)
        frames: list[DemoFrame] = []
        last_timestamp_ms = -1
        for index, raw_frame in enumerate(trip["frames"]):
            frame = _exact_keys(
                raw_frame, field=f"frames[{index}]", expected={"frame_id", "timestamp", "ego"}
            )
            if frame["frame_id"] != index:
                raise InputContractError("recorded frame IDs must be contiguous")
            timestamp = _finite_number(
                frame["timestamp"], field=f"frames[{index}].timestamp", minimum=0.0
            )
            timestamp_ms = round(timestamp * 1_000.0)
            if timestamp_ms <= last_timestamp_ms:
                raise InputContractError("recorded timestamps must increase")
            last_timestamp_ms = timestamp_ms
            frames.append(
                DemoFrame(
                    source_sequence=index,
                    source_media_timestamp_ms=timestamp_ms,
                    ego=normalize_ego(frame["ego"]),
                    road_path=resolved[f"kitti/image_2/{index:06d}.jpg"],
                    cabin_path=resolved[f"driver/frame_{index:06d}.jpg"],
                )
            )
        return cls(
            root=root,
            trip_id=trip_id,
            metadata=metadata,
            frames=tuple(frames),
            manifest_sha256=_sha256_bytes(manifest_bytes),
        )

    def tick(
        self,
        index: int,
        *,
        session_id: str,
        capture_timestamp_ms: int,
    ) -> InputTick:
        frame = self.frames[index]
        return InputTick(
            session_id=_session_id(session_id),
            source_sequence=frame.source_sequence,
            capture_timestamp_ms=_non_negative_int(
                capture_timestamp_ms, field="capture_timestamp_ms"
            ),
            source_kind="RECORDED_STREAM",
            telemetry_source="RECORDED_DATA",
            metadata=self.metadata,
            ego=frame.ego,
            road_bgr=decode_jpeg_bytes(frame.road_path.read_bytes(), field="road_jpeg"),
            cabin_bgr=decode_jpeg_bytes(frame.cabin_path.read_bytes(), field="cabin_jpeg"),
            source_media_timestamp_ms=frame.source_media_timestamp_ms,
        )

    def iter_ticks(
        self,
        *,
        session_id: str,
        clock_ms: Callable[[], int],
    ) -> Iterator[InputTick]:
        for index in range(len(self.frames)):
            yield self.tick(
                index, session_id=session_id, capture_timestamp_ms=clock_ms()
            )


__all__ = [
    "BUNDLE_SCHEMA",
    "DemoFrame",
    "InputContractError",
    "InputTick",
    "MAX_IMAGE_HEIGHT",
    "MAX_IMAGE_WIDTH",
    "MAX_INPUT_MESSAGE_BYTES",
    "MAX_JPEG_BYTES",
    "PreparedDemoBundle",
    "decode_jpeg_bytes",
    "normalize_ego",
    "normalize_metadata",
    "parse_input_message",
    "strict_json_loads",
]
