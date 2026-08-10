"""Mmap-friendly dense eye-crop cache without a PyTorch dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def cache_paths(cache_dir: Path, session_name: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{session_name}.eyes.npy",
        cache_dir / f"{session_name}.meta.npz",
    )


def save_eye_crop_cache(
    cache_dir: Path | str,
    session_name: str,
    *,
    eyes: np.ndarray,
    frame_ids: np.ndarray,
    visibility: np.ndarray,
) -> None:
    """Save dense crops in mmap-friendly NPY form and small metadata in NPZ."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    eyes = np.asarray(eyes)
    frame_ids = np.asarray(frame_ids)
    visibility = np.asarray(visibility)
    if eyes.ndim != 5 or eyes.shape[1] != 2 or eyes.shape[-1] != 3:
        raise ValueError("eyes must have shape [frame, 2, height, width, 3]")
    if eyes.dtype != np.uint8:
        raise ValueError("eyes must be uint8")
    if frame_ids.shape != (eyes.shape[0],):
        raise ValueError("frame_ids must have one value per crop pair")
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")
    if visibility.shape != (eyes.shape[0], 2):
        raise ValueError("visibility must have shape [frame, 2]")
    if np.any(visibility < 0.0) or np.any(visibility > 1.0):
        raise ValueError("visibility values must be between 0 and 1")

    eyes_path, metadata_path = cache_paths(cache_dir, session_name)
    np.save(eyes_path, eyes, allow_pickle=False)
    np.savez(
        metadata_path,
        frame_ids=frame_ids.astype(np.int32, copy=False),
        visibility=visibility.astype(np.float32, copy=False),
        complete=np.array([True], dtype=np.bool_),
    )


class EyeCropStore:
    """Lazy, memory-mapped access to one complete session's eye crops."""

    def __init__(self, cache_dir: Path | str, session_name: str) -> None:
        eyes_path, metadata_path = cache_paths(Path(cache_dir), session_name)
        if not eyes_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"eye crop cache is missing for {session_name}")
        self.eyes = np.load(eyes_path, mmap_mode="r", allow_pickle=False)
        with np.load(metadata_path, allow_pickle=False) as metadata:
            self.frame_ids = metadata["frame_ids"].astype(np.int64, copy=True)
            self.visibility = metadata["visibility"].astype(np.float32, copy=True)
            complete = bool(metadata.get("complete", np.array([False]))[0])
        if not complete:
            raise ValueError(f"eye crop cache is not marked complete: {metadata_path}")
        if self.eyes.shape[0] != len(self.frame_ids) or self.visibility.shape != (
            len(self.frame_ids),
            2,
        ):
            raise ValueError(f"inconsistent eye crop cache arrays for {session_name}")
        if len(np.unique(self.frame_ids)) != len(self.frame_ids):
            raise ValueError(f"duplicate frame ids in eye crop cache for {session_name}")
        self._rows = {int(frame_id): row for row, frame_id in enumerate(self.frame_ids)}

    def rows_for(self, frame_ids: Sequence[int]) -> np.ndarray:
        missing = [frame_id for frame_id in frame_ids if frame_id not in self._rows]
        if missing:
            raise ValueError(f"eye crop cache is incomplete; missing frame ids {missing[:5]}")
        return np.array([self._rows[frame_id] for frame_id in frame_ids], dtype=np.int64)
