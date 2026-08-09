"""Dense face, eye, and mouth regions for the single-backbone visual model."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import numpy as np


REGION_NAMES = ("face", "left_eye", "right_eye", "mouth")
REGION_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvidenceRegionRecord:
    """Normalized regions and pose metadata for one source frame."""

    frame_id: int
    boxes: np.ndarray
    visibility: np.ndarray
    head_pose: np.ndarray

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float32)
        visibility = np.asarray(self.visibility, dtype=np.float32)
        head_pose = np.asarray(self.head_pose, dtype=np.float32)
        if boxes.shape != (len(REGION_NAMES), 4):
            raise ValueError("region boxes must have shape (4, 4)")
        if visibility.shape != (len(REGION_NAMES),):
            raise ValueError("region visibility must have shape (4,)")
        if head_pose.shape != (2,):
            raise ValueError("head pose must have shape (2,)")
        if not np.isfinite(boxes).all() or not np.isfinite(head_pose).all():
            raise ValueError("region boxes and head pose must be finite")
        if np.any((boxes < 0.0) | (boxes > 1.0)):
            raise ValueError("normalized region boxes must be in [0, 1]")
        if np.any(~np.isfinite(visibility)) or np.any(
            (visibility < 0.0) | (visibility > 1.0)
        ):
            raise ValueError("region visibility must be finite and in [0, 1]")
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        visible = visibility > 0.0
        if np.any(visible & ((widths <= 0.0) | (heights <= 0.0))):
            raise ValueError("visible region boxes must have positive area")

        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "boxes", boxes.copy())
        object.__setattr__(self, "visibility", visibility.copy())
        object.__setattr__(self, "head_pose", head_pose.copy())

    @classmethod
    def missing(cls, frame_id: int) -> "EvidenceRegionRecord":
        return cls(
            frame_id=int(frame_id),
            boxes=np.zeros((len(REGION_NAMES), 4), dtype=np.float32),
            visibility=np.zeros(len(REGION_NAMES), dtype=np.float32),
            head_pose=np.zeros(2, dtype=np.float32),
        )


def project_normalized_boxes_to_letterbox(
    boxes: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Project source-normalized boxes into absolute letterboxed coordinates."""

    values = np.asarray(boxes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape [regions, 4]")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("normalized boxes must be finite and in [0, 1]")
    source_width, source_height = (int(value) for value in source_size)
    target_width, target_height = (int(value) for value in target_size)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )
    resized_width = source_width * scale
    resized_height = source_height * scale
    offset_x = (target_width - resized_width) / 2.0
    offset_y = (target_height - resized_height) / 2.0
    projected = values.copy()
    projected[:, (0, 2)] = projected[:, (0, 2)] * resized_width + offset_x
    projected[:, (1, 3)] = projected[:, (1, 3)] * resized_height + offset_y
    return projected.astype(np.float32, copy=False)


def _cache_path(cache_dir: Path | str, session_name: str) -> Path:
    if not session_name:
        raise ValueError("session name is required")
    return Path(cache_dir) / f"{session_name}.regions.npz"


def save_evidence_region_cache(
    cache_dir: Path | str,
    session_name: str,
    records: Sequence[EvidenceRegionRecord],
) -> None:
    """Atomically save one dense region row for every supplied frame."""

    records = tuple(records)
    frame_ids = np.asarray([record.frame_id for record in records], dtype=np.int32)
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("region-cache frame IDs must be unique")
    if len(frame_ids) > 1 and np.any(np.diff(frame_ids) <= 0):
        raise ValueError("region-cache frame IDs must be increasing")
    boxes = np.stack([record.boxes for record in records]).astype(
        np.float32,
        copy=False,
    ) if records else np.empty((0, len(REGION_NAMES), 4), dtype=np.float32)
    visibility = np.stack([record.visibility for record in records]).astype(
        np.float32,
        copy=False,
    ) if records else np.empty((0, len(REGION_NAMES)), dtype=np.float32)
    head_pose = np.stack([record.head_pose for record in records]).astype(
        np.float32,
        copy=False,
    ) if records else np.empty((0, 2), dtype=np.float32)

    path = _cache_path(cache_dir, session_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            schema_version=np.asarray(
                [REGION_CACHE_SCHEMA_VERSION],
                dtype=np.int32,
            ),
            complete=np.asarray([True], dtype=np.bool_),
            frame_ids=frame_ids,
            boxes=boxes,
            visibility=visibility,
            head_pose=head_pose,
        )
    os.replace(temporary, path)


class EvidenceRegionStore:
    """Read a dense region cache with explicit missing-frame fallback."""

    def __init__(self, cache_dir: Path | str, session_name: str) -> None:
        path = _cache_path(cache_dir, session_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"evidence-region cache is missing for {session_name}"
            )
        with np.load(path, allow_pickle=False) as cache:
            schema = int(cache["schema_version"][0])
            complete = bool(cache["complete"][0])
            if schema != REGION_CACHE_SCHEMA_VERSION or not complete:
                raise ValueError(
                    f"incompatible evidence-region cache for {session_name}"
                )
            self.frame_ids = cache["frame_ids"].astype(np.int64, copy=True)
            self.boxes = cache["boxes"].astype(np.float32, copy=True)
            self.visibility = cache["visibility"].astype(np.float32, copy=True)
            self.head_pose = cache["head_pose"].astype(np.float32, copy=True)
        count = len(self.frame_ids)
        if (
            self.boxes.shape != (count, len(REGION_NAMES), 4)
            or self.visibility.shape != (count, len(REGION_NAMES))
            or self.head_pose.shape != (count, 2)
        ):
            raise ValueError(
                f"inconsistent evidence-region cache for {session_name}"
            )
        if len(np.unique(self.frame_ids)) != count:
            raise ValueError(
                f"duplicate frame IDs in evidence-region cache for {session_name}"
            )
        self._rows = {
            int(frame_id): row
            for row, frame_id in enumerate(self.frame_ids)
        }

    def get(self, frame_id: int) -> EvidenceRegionRecord:
        row = self._rows.get(int(frame_id))
        if row is None:
            return EvidenceRegionRecord.missing(frame_id)
        return EvidenceRegionRecord(
            frame_id=int(frame_id),
            boxes=self.boxes[row],
            visibility=self.visibility[row],
            head_pose=self.head_pose[row],
        )


__all__ = [
    "EvidenceRegionRecord",
    "EvidenceRegionStore",
    "REGION_CACHE_SCHEMA_VERSION",
    "REGION_NAMES",
    "project_normalized_boxes_to_letterbox",
    "save_evidence_region_cache",
]
