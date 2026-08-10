"""Strict one-camera input boundary for C1 inference.

Runtime code must never open the source trip JSON because that document can
contain targets, events and ground-truth TTC.  The boundary is deliberately
split in two:

* ``safeloop.c1.prepare_manifest`` is a separate offline operation.  It reads
  the source JSON once and writes a sanitized manifest outside the dataset/Git
  artifacts.
* :class:`C1LeanTripLoader` consumes only that sanitized manifest and
  ``kitti/image_2``.  It has no source-JSON fallback.

The manifest carries fixed schema/camera/count metadata plus ``trip_id`` and,
per frame, ``frame_id``, ``timestamp`` and three explicitly allowed
ego-kinematics fields. Missing images are reported as integrity errors;
nothing is synthesized.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

_IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
_EGO_KINEMATICS_FIELDS = (
    "speed_kmh",
    "longitudinal_accel",
    "lateral_accel",
)
C1_MANIFEST_SCHEMA_VERSION = 1
C1_MANIFEST_SOURCE_CAMERA = "image_2"
_MANIFEST_ROOT_FIELDS = frozenset(
    ("schema_version", "trip_id", "source_camera", "expected_frames", "frames")
)
_MANIFEST_FRAME_FIELDS = frozenset(("frame_id", "timestamp", "ego"))


@dataclass(frozen=True, slots=True)
class C1Image2IntegrityReport:
    """Image/telemetry consistency for one trip, without repairing it."""

    trip_id: str
    telemetry_frame_count: int
    present_image_count: int
    missing_frame_ids: tuple[int, ...]
    duplicate_frame_ids: tuple[int, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_frame_ids and not self.duplicate_frame_ids

    @property
    def errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.missing_frame_ids:
            missing = ", ".join(str(frame_id) for frame_id in self.missing_frame_ids)
            errors.append(
                f"trip {self.trip_id}: missing image_2 for frame_id(s): {missing}"
            )
        if self.duplicate_frame_ids:
            duplicate = ", ".join(str(frame_id) for frame_id in self.duplicate_frame_ids)
            errors.append(
                f"trip {self.trip_id}: duplicate telemetry frame_id(s): {duplicate}"
            )
        return tuple(errors)

    def require_ok(self) -> None:
        """Raise an integrity error; deliberately performs no repair."""

        if not self.ok:
            raise C1Image2IntegrityError("; ".join(self.errors))


class C1Image2IntegrityError(FileNotFoundError):
    """Raised when sanitized telemetry cannot map safely to ``image_2``."""


@dataclass(frozen=True, slots=True)
class _C1FrameRecord:
    frame_id: int
    timestamp: float
    ego: Mapping[str, float]


class C1InputFrame:
    """One C1 inference frame with no accessor for a forbidden modality."""

    __slots__ = ("_frame_id", "_timestamp", "_ego", "_image_path", "_image")

    def __init__(
        self,
        *,
        frame_id: int,
        timestamp: float,
        ego: Mapping[str, float],
        image_path: Path,
    ) -> None:
        self._frame_id = frame_id
        self._timestamp = timestamp
        self._ego = ego
        self._image_path = image_path
        self._image: np.ndarray | None = None

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def ego(self) -> Mapping[str, float]:
        return self._ego

    def left(self) -> np.ndarray:
        """Decode the single permitted road-camera image (``image_2``)."""

        if self._image is None:
            image = cv2.imread(str(self._image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise C1Image2IntegrityError(
                    f"trip image_2 is unreadable for frame_id {self.frame_id}: "
                    f"{self._image_path}"
                )
            self._image = image
        return self._image


class C1LeanTripLoader:
    """Consume a sanitized manifest plus ``image_2`` only.

    ``manifest_path`` is mandatory.  There is intentionally no code path that
    discovers or opens ``<trip>.json``/``<trip>.json.gz`` at inference time.
    """

    def __init__(self, trip_dir: str | Path, manifest_path: str | Path) -> None:
        self._trip_dir = Path(trip_dir)
        if not self._trip_dir.is_dir():
            raise FileNotFoundError(f"Trip directory does not exist: {self._trip_dir}")

        self._manifest_path = Path(manifest_path)
        if not self._manifest_path.is_file():
            raise FileNotFoundError(
                f"Sanitized C1 manifest does not exist: {self._manifest_path}"
            )
        if self._manifest_path.resolve().is_relative_to(self._trip_dir.resolve()):
            raise ValueError(
                f"Sanitized C1 manifest must be outside the trip: {self._manifest_path}"
            )

        manifest = _load_manifest_document(self._manifest_path)
        self._trip_id, self._records = _validate_and_project_manifest(manifest)
        del manifest
        if self._trip_id != self._trip_dir.name:
            raise ValueError(
                f"Manifest trip_id {self._trip_id!r} does not match trip directory "
                f"{self._trip_dir.name!r}"
            )

        self._image_2_dir = self._trip_dir / "kitti" / "image_2"
        self._image_extension: str | None = None

    @property
    def trip_id(self) -> str:
        return self._trip_id

    @property
    def n_frames(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return self.n_frames

    def frame(self, index: int) -> C1InputFrame:
        """Return one narrow input frame by sanitized-manifest position."""

        self._check_index(index)
        record = self._records[index]
        image_path = self._find_image_2(record.frame_id)
        if image_path is None:
            raise C1Image2IntegrityError(
                f"trip {self.trip_id}: missing image_2 for frame_id "
                f"{record.frame_id}; no image was synthesized or substituted"
            )
        return C1InputFrame(
            frame_id=record.frame_id,
            timestamp=record.timestamp,
            ego=record.ego,
            image_path=image_path,
        )

    def iter_frames(self, start: int = 0, end: int | None = None) -> Iterator[C1InputFrame]:
        """Iterate a half-open sanitized telemetry range in recorded order."""

        stop = self.n_frames if end is None else end
        if not 0 <= start <= stop <= self.n_frames:
            raise ValueError(
                f"Invalid C1 frame range: start={start}, end={stop}, "
                f"n_frames={self.n_frames}"
            )
        for index in range(start, stop):
            yield self.frame(index)

    def check_image_2_integrity(self) -> C1Image2IntegrityReport:
        """Report missing images/duplicate IDs without changing the dataset."""

        frame_ids = tuple(record.frame_id for record in self._records)
        counts = Counter(frame_ids)
        duplicate = tuple(
            sorted(frame_id for frame_id, count in counts.items() if count > 1)
        )
        image_presence = {
            frame_id: self._find_image_2(frame_id) is not None
            for frame_id in set(frame_ids)
        }
        missing = tuple(
            sorted(
                frame_id
                for frame_id, present in image_presence.items()
                if not present
            )
        )
        return C1Image2IntegrityReport(
            trip_id=self.trip_id,
            telemetry_frame_count=self.n_frames,
            present_image_count=sum(image_presence[frame_id] for frame_id in frame_ids),
            missing_frame_ids=missing,
            duplicate_frame_ids=duplicate,
        )

    def _find_image_2(self, frame_id: int) -> Path | None:
        stem = f"{frame_id:06d}"
        if self._image_extension is not None:
            cached = self._image_2_dir / f"{stem}{self._image_extension}"
            if cached.is_file():
                return cached
        for extension in _IMAGE_EXTENSIONS:
            candidate = self._image_2_dir / f"{stem}{extension}"
            if candidate.is_file():
                self._image_extension = extension
                return candidate
        return None

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"C1 frame index {index} is outside [0, {self.n_frames}) "
                f"for trip {self.trip_id}"
            )


def _load_manifest_document(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_and_project_manifest(
    manifest: Any,
) -> tuple[str, tuple[_C1FrameRecord, ...]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("Sanitized C1 manifest root must be an object")
    root_fields = frozenset(manifest)
    if root_fields != _MANIFEST_ROOT_FIELDS:
        raise ValueError(
            "Sanitized C1 manifest root fields must be exactly "
            f"{sorted(_MANIFEST_ROOT_FIELDS)}, got {sorted(map(str, root_fields))}"
        )
    schema_version = manifest["schema_version"]
    trip_id = manifest["trip_id"]
    source_camera = manifest["source_camera"]
    expected_frames = manifest["expected_frames"]
    frames = manifest["frames"]
    if type(schema_version) is not int or schema_version != C1_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Sanitized C1 manifest schema_version must be "
            f"{C1_MANIFEST_SCHEMA_VERSION}"
        )
    if not isinstance(trip_id, str) or not trip_id:
        raise ValueError("Sanitized C1 manifest trip_id must be a non-empty string")
    if source_camera != C1_MANIFEST_SOURCE_CAMERA:
        raise ValueError(
            f"Sanitized C1 manifest source_camera must be "
            f"{C1_MANIFEST_SOURCE_CAMERA!r}"
        )
    if type(expected_frames) is not int or expected_frames < 0:
        raise ValueError(
            "Sanitized C1 manifest expected_frames must be a non-negative int"
        )
    if not isinstance(frames, list):
        raise ValueError("Sanitized C1 manifest frames must be a list")
    if len(frames) != expected_frames:
        raise ValueError(
            f"Sanitized C1 manifest expected_frames={expected_frames}, "
            f"but contains {len(frames)} frames"
        )

    records: list[_C1FrameRecord] = []
    previous_timestamp: float | None = None
    for position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"manifest frames[{position}] must be an object")
        frame_fields = frozenset(frame)
        if frame_fields != _MANIFEST_FRAME_FIELDS:
            raise ValueError(
                f"manifest frames[{position}] fields must be exactly "
                f"{sorted(_MANIFEST_FRAME_FIELDS)}"
            )
        ego = frame["ego"]
        if not isinstance(ego, Mapping):
            raise ValueError(f"manifest frames[{position}].ego must be an object")
        unexpected_ego = frozenset(ego).difference(_EGO_KINEMATICS_FIELDS)
        if unexpected_ego:
            raise ValueError(
                f"manifest frames[{position}].ego has forbidden fields: "
                f"{sorted(map(str, unexpected_ego))}"
            )
        frame_id = frame["frame_id"]
        if type(frame_id) is not int:
            raise ValueError(f"manifest frames[{position}].frame_id must be a true int")
        if frame_id != position:
            raise ValueError(
                "Sanitized C1 manifest frame_id values must be unique and "
                f"contiguous 0..{expected_frames - 1}; position {position} has {frame_id}"
            )
        timestamp = _finite_number(
            frame["timestamp"], f"manifest frames[{position}].timestamp"
        )
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                "Sanitized C1 manifest timestamps must be strictly increasing; "
                f"frames[{position}]={timestamp} follows {previous_timestamp}"
            )
        previous_timestamp = timestamp
        projected_ego = {
            key: _finite_number(value, f"manifest frames[{position}].ego.{key}")
            for key, value in ego.items()
        }
        records.append(
            _C1FrameRecord(
                frame_id=frame_id,
                timestamp=timestamp,
                ego=MappingProxyType(projected_ego),
            )
        )
    return trip_id, tuple(records)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric and not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


__all__ = [
    "C1Image2IntegrityError",
    "C1Image2IntegrityReport",
    "C1InputFrame",
    "C1LeanTripLoader",
    "C1_MANIFEST_SCHEMA_VERSION",
    "C1_MANIFEST_SOURCE_CAMERA",
]
