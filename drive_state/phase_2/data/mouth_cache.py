"""Mmap-friendly dense mouth-crop cache without a PyTorch dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def cache_paths(cache_dir: Path, session_name: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{session_name}.mouths.npy",
        cache_dir / f"{session_name}.meta.npz",
    )


def save_mouth_crop_cache(
    cache_dir: Path | str,
    session_name: str,
    *,
    mouths: np.ndarray,
    frame_ids: np.ndarray,
    visibility: np.ndarray,
) -> None:
    """Save one fixed-size mouth crop and visibility value per source frame."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mouths = np.asarray(mouths)
    frame_ids = np.asarray(frame_ids)
    visibility = np.asarray(visibility)
    if mouths.ndim != 4 or mouths.shape[1:] != (64, 96, 3) or mouths.dtype != np.uint8:
        raise ValueError("mouths must be uint8 [frame, 64, 96, 3]")
    if frame_ids.shape != (len(mouths),):
        raise ValueError("frame_ids must contain one ID per mouth")
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")
    if visibility.shape != (len(mouths),):
        raise ValueError("visibility must contain one value per mouth")
    if np.any((visibility < 0.0) | (visibility > 1.0)):
        raise ValueError("visibility values must be between 0 and 1")

    mouths_path, metadata_path = cache_paths(cache_dir, session_name)
    np.save(mouths_path, mouths, allow_pickle=False)
    np.savez(
        metadata_path,
        frame_ids=frame_ids.astype(np.int32, copy=False),
        visibility=visibility.astype(np.float32, copy=False),
        complete=np.array([True], dtype=np.bool_),
    )


class MouthCropStore:
    """Lazy, memory-mapped access to one complete session's mouth crops."""

    def __init__(self, cache_dir: Path | str, session_name: str) -> None:
        mouths_path, metadata_path = cache_paths(Path(cache_dir), session_name)
        if not mouths_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"mouth crop cache is missing for {session_name}")
        self.mouths = np.load(mouths_path, mmap_mode="r", allow_pickle=False)
        with np.load(metadata_path, allow_pickle=False) as metadata:
            self.frame_ids = metadata["frame_ids"].astype(np.int64, copy=True)
            self.visibility = metadata["visibility"].astype(np.float32, copy=True)
            complete = bool(metadata.get("complete", np.array([False]))[0])
        if not complete:
            raise ValueError(f"mouth crop cache is not marked complete: {metadata_path}")
        count = len(self.frame_ids)
        if self.mouths.shape != (count, 64, 96, 3) or self.mouths.dtype != np.uint8:
            raise ValueError(f"inconsistent mouth crop array for {session_name}")
        if self.visibility.shape != (count,):
            raise ValueError(f"inconsistent mouth visibility for {session_name}")
        if np.any((self.visibility < 0.0) | (self.visibility > 1.0)):
            raise ValueError(f"invalid mouth visibility for {session_name}")
        if len(np.unique(self.frame_ids)) != count:
            raise ValueError(f"duplicate frame ids in mouth crop cache for {session_name}")
        self._rows = {int(frame_id): row for row, frame_id in enumerate(self.frame_ids)}

    def rows_for(self, frame_ids: Sequence[int]) -> np.ndarray:
        missing = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in self._rows]
        if missing:
            raise ValueError(f"mouth crop cache is incomplete; missing frame ids {missing[:5]}")
        return np.array([self._rows[int(frame_id)] for frame_id in frame_ids], dtype=np.int64)
