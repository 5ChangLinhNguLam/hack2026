"""Immutable, mmap-friendly fold-specific visual embedding sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .manifest import protocol_from_session, subject_from_session
from ..training.eye_trainer import SubjectSplitMetadata


SCHEMA_VERSION = 2


def cache_paths(cache_dir: Path, session_name: str) -> dict[str, Path]:
    return {
        "cabin": cache_dir / f"{session_name}.cabin.npy",
        "face": cache_dir / f"{session_name}.face.npy",
        "mouth": cache_dir / f"{session_name}.mouth.npy",
        "eye": cache_dir / f"{session_name}.eye.npy",
        "eye_evidence": cache_dir / f"{session_name}.eye_evidence.npy",
        "targets": cache_dir / f"{session_name}.targets.npy",
        "metadata": cache_dir / f"{session_name}.meta.npz",
    }


def _require_dtype(name: str, value: np.ndarray, dtype: np.dtype) -> None:
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _validate_monotonic(frame_ids: np.ndarray, timestamps: np.ndarray) -> None:
    if len(frame_ids) > 1 and np.any(np.diff(frame_ids.astype(np.int64)) <= 0):
        raise ValueError("frame_ids must be strictly increasing")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if len(timestamps) > 1 and np.any(np.diff(timestamps.astype(np.float64)) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")


def save_embedding_session(
    cache_dir: Path | str,
    *,
    session_name: str,
    subject_id: str,
    protocol: str,
    fps: float,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    cabin: np.ndarray,
    face: np.ndarray,
    mouth: np.ndarray,
    eye: np.ndarray,
    face_visibility: np.ndarray,
    mouth_visibility: np.ndarray,
    eye_visibility: np.ndarray,
    eye_evidence: np.ndarray,
    head_pose: np.ndarray,
    targets: np.ndarray,
    task_names: Sequence[str],
    spatial_fingerprint: str,
    mouth_fingerprint: str,
    eye_fingerprint: str,
    split: SubjectSplitMetadata,
    overwrite: bool = False,
) -> None:
    """Write a complete session whose producer checkpoints and split are explicit."""
    cache_dir = Path(cache_dir)
    paths = cache_paths(cache_dir, session_name)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"embedding cache already exists for {session_name}")

    arrays = {
        "frame_ids": np.asarray(frame_ids),
        "timestamps": np.asarray(timestamps),
        "cabin": np.asarray(cabin),
        "face": np.asarray(face),
        "mouth": np.asarray(mouth),
        "eye": np.asarray(eye),
        "face_visibility": np.asarray(face_visibility),
        "mouth_visibility": np.asarray(mouth_visibility),
        "eye_visibility": np.asarray(eye_visibility),
        "eye_evidence": np.asarray(eye_evidence),
        "head_pose": np.asarray(head_pose),
        "targets": np.asarray(targets),
    }
    frame_ids = arrays["frame_ids"]
    timestamps = arrays["timestamps"]
    cabin = arrays["cabin"]
    face = arrays["face"]
    mouth = arrays["mouth"]
    eye = arrays["eye"]
    face_visibility = arrays["face_visibility"]
    mouth_visibility = arrays["mouth_visibility"]
    eye_visibility = arrays["eye_visibility"]
    eye_evidence = arrays["eye_evidence"]
    head_pose = arrays["head_pose"]
    targets = arrays["targets"]
    count = len(frame_ids)
    task_names = tuple(str(name) for name in task_names)

    _require_dtype("frame_ids", frame_ids, np.dtype(np.int32))
    _require_dtype("timestamps", timestamps, np.dtype(np.float32))
    _require_dtype("cabin", cabin, np.dtype(np.float16))
    _require_dtype("face", face, np.dtype(np.float16))
    _require_dtype("mouth", mouth, np.dtype(np.float16))
    _require_dtype("eye", eye, np.dtype(np.float16))
    _require_dtype("face_visibility", face_visibility, np.dtype(np.float32))
    _require_dtype("mouth_visibility", mouth_visibility, np.dtype(np.float32))
    _require_dtype("eye_visibility", eye_visibility, np.dtype(np.float32))
    _require_dtype("eye_evidence", eye_evidence, np.dtype(np.float32))
    _require_dtype("head_pose", head_pose, np.dtype(np.float32))
    _require_dtype("targets", targets, np.dtype(np.int32))
    if frame_ids.shape != (count,) or timestamps.shape != (count,):
        raise ValueError("frame_ids and timestamps must be one-dimensional")
    if cabin.shape != (count, 256) or face.shape != (count, 256):
        raise ValueError("cabin and face embeddings must have shape [frame, 256]")
    if mouth.shape != (count, 64):
        raise ValueError("mouth embeddings must have shape [frame, 64]")
    if eye.shape != (count, 128):
        raise ValueError("eye embeddings must have shape [frame, 128]")
    if face_visibility.shape != (count,) or mouth_visibility.shape != (count,):
        raise ValueError("visibility arrays must have shape [frame]")
    if eye_visibility.shape != (count, 2):
        raise ValueError("eye_visibility must have shape [frame, 2]")
    if eye_evidence.shape != (count, 8):
        raise ValueError("eye_evidence must have shape [frame, 8]")
    if head_pose.shape != (count, 2):
        raise ValueError("head_pose must have shape [frame, 2]")
    if targets.shape != (count, len(task_names)):
        raise ValueError("targets must have one ordered column per task name")
    if len(set(task_names)) != len(task_names):
        raise ValueError("task_names must be unique")
    if not task_names:
        raise ValueError("task_names cannot be empty")
    if np.any((face_visibility < 0.0) | (face_visibility > 1.0)) or np.any(
        (mouth_visibility < 0.0) | (mouth_visibility > 1.0)
    ) or np.any(
        (eye_visibility < 0.0) | (eye_visibility > 1.0)
    ):
        raise ValueError("visibility values must be between 0 and 1")
    if not np.isfinite(head_pose).all() or not np.isfinite(eye_evidence).all():
        raise ValueError("head_pose and eye_evidence must be finite")
    bounded_evidence = eye_evidence[:, [0, 2, 3, 4, 5, 6, 7]]
    if np.any((bounded_evidence < 0.0) | (bounded_evidence > 1.0)):
        raise ValueError("eye probability and reliability evidence must be in [0, 1]")
    if np.any(eye_evidence[:, 1] < 0.0):
        raise ValueError("eye closure duration cannot be negative")
    _validate_monotonic(frame_ids, timestamps)
    if subject_from_session(session_name) != subject_id:
        raise ValueError("session subject is inconsistent with subject_id")
    if protocol_from_session(session_name) != protocol:
        raise ValueError("session protocol is inconsistent with protocol metadata")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not spatial_fingerprint or not mouth_fingerprint or not eye_fingerprint:
        raise ValueError("checkpoint fingerprints cannot be empty")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths["cabin"], cabin, allow_pickle=False)
    np.save(paths["face"], face, allow_pickle=False)
    np.save(paths["mouth"], mouth, allow_pickle=False)
    np.save(paths["eye"], eye, allow_pickle=False)
    np.save(paths["eye_evidence"], eye_evidence, allow_pickle=False)
    np.save(paths["targets"], targets, allow_pickle=False)
    np.savez(
        paths["metadata"],
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int32),
        complete=np.array([True], dtype=np.bool_),
        session_name=np.array([session_name]),
        subject_id=np.array([subject_id]),
        protocol=np.array([protocol]),
        fps=np.array([fps], dtype=np.float32),
        frame_ids=frame_ids,
        timestamps=timestamps,
        face_visibility=face_visibility,
        mouth_visibility=mouth_visibility,
        eye_visibility=eye_visibility,
        head_pose=head_pose,
        task_names=np.asarray(task_names),
        spatial_fingerprint=np.array([spatial_fingerprint]),
        mouth_fingerprint=np.array([mouth_fingerprint]),
        eye_fingerprint=np.array([eye_fingerprint]),
        split_train_subjects=np.asarray(split.train_subjects),
        split_validation_subject=np.array([split.validation_subject]),
        split_test_subject=np.array([split.test_subject]),
    )


class EmbeddingSessionStore:
    """Validated lazy access to one immutable, fold-bound embedding session."""

    def __init__(
        self,
        cache_dir: Path | str,
        session_name: str,
        *,
        expected_spatial_fingerprint: str | None = None,
        expected_mouth_fingerprint: str | None = None,
        expected_eye_fingerprint: str | None = None,
        expected_split: SubjectSplitMetadata | None = None,
    ) -> None:
        self.session_name = session_name
        paths = cache_paths(Path(cache_dir), session_name)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"embedding cache is incomplete for {session_name}: {missing}"
            )
        self.cabin = np.load(paths["cabin"], mmap_mode="r", allow_pickle=False)
        self.face = np.load(paths["face"], mmap_mode="r", allow_pickle=False)
        self.mouth = np.load(paths["mouth"], mmap_mode="r", allow_pickle=False)
        self.eye = np.load(paths["eye"], mmap_mode="r", allow_pickle=False)
        self.eye_evidence = np.load(
            paths["eye_evidence"], mmap_mode="r", allow_pickle=False
        )
        self.targets = np.load(paths["targets"], mmap_mode="r", allow_pickle=False)
        with np.load(paths["metadata"], allow_pickle=False) as metadata:
            schema_version = int(metadata["schema_version"][0])
            complete = bool(metadata.get("complete", np.array([False]))[0])
            stored_session = str(metadata["session_name"][0])
            self.subject_id = str(metadata["subject_id"][0])
            self.protocol = str(metadata["protocol"][0])
            self.fps = float(metadata["fps"][0])
            self.frame_ids = metadata["frame_ids"].astype(np.int32, copy=True)
            self.timestamps = metadata["timestamps"].astype(np.float32, copy=True)
            self.face_visibility = metadata["face_visibility"].astype(np.float32, copy=True)
            self.mouth_visibility = metadata["mouth_visibility"].astype(np.float32, copy=True)
            self.eye_visibility = metadata["eye_visibility"].astype(np.float32, copy=True)
            self.head_pose = metadata["head_pose"].astype(np.float32, copy=True)
            self.task_names = tuple(str(name) for name in metadata["task_names"])
            self.spatial_fingerprint = str(metadata["spatial_fingerprint"][0])
            self.mouth_fingerprint = str(metadata["mouth_fingerprint"][0])
            self.eye_fingerprint = str(metadata["eye_fingerprint"][0])
            self.split = SubjectSplitMetadata(
                tuple(str(value) for value in metadata["split_train_subjects"]),
                str(metadata["split_validation_subject"][0]),
                str(metadata["split_test_subject"][0]),
            )
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported embedding cache schema: {schema_version}")
        if not complete:
            raise ValueError(f"embedding cache is not marked complete: {session_name}")
        if stored_session != session_name:
            raise ValueError("embedding cache session metadata is inconsistent")
        if subject_from_session(session_name) != self.subject_id:
            raise ValueError("embedding cache subject metadata is inconsistent")
        if protocol_from_session(session_name) != self.protocol:
            raise ValueError("embedding cache protocol metadata is inconsistent")
        count = len(self.frame_ids)
        if self.cabin.shape != (count, 256) or self.cabin.dtype != np.float16:
            raise ValueError("invalid cabin embedding array")
        if self.face.shape != (count, 256) or self.face.dtype != np.float16:
            raise ValueError("invalid face embedding array")
        if self.mouth.shape != (count, 64) or self.mouth.dtype != np.float16:
            raise ValueError("invalid mouth embedding array")
        if self.eye.shape != (count, 128) or self.eye.dtype != np.float16:
            raise ValueError("invalid eye embedding array")
        if self.eye_evidence.shape != (count, 8) or self.eye_evidence.dtype != np.float32:
            raise ValueError("invalid eye evidence array")
        if self.targets.shape != (count, len(self.task_names)) or self.targets.dtype != np.int32:
            raise ValueError("invalid target embedding array")
        if self.timestamps.shape != (count,):
            raise ValueError("invalid timestamp metadata")
        if self.face_visibility.shape != (count,) or self.mouth_visibility.shape != (count,):
            raise ValueError("invalid visibility metadata")
        if self.eye_visibility.shape != (count, 2):
            raise ValueError("invalid eye visibility metadata")
        if self.head_pose.shape != (count, 2):
            raise ValueError("invalid head-pose metadata")
        _validate_monotonic(self.frame_ids, self.timestamps)
        if expected_spatial_fingerprint is not None and self.spatial_fingerprint != expected_spatial_fingerprint:
            raise ValueError("spatial checkpoint fingerprint mismatch")
        if expected_mouth_fingerprint is not None and self.mouth_fingerprint != expected_mouth_fingerprint:
            raise ValueError("mouth checkpoint fingerprint mismatch")
        if expected_eye_fingerprint is not None and self.eye_fingerprint != expected_eye_fingerprint:
            raise ValueError("eye checkpoint fingerprint mismatch")
        if expected_split is not None and self.split != expected_split:
            raise ValueError("embedding cache split metadata mismatch")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def task_column(self, name: str) -> int:
        try:
            return self.task_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown embedding target task: {name}") from error
