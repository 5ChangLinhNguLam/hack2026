"""Fold-specific visual embeddings with causal eye and PERCLOS evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..state.causal_evidence import (
    EVIDENCE_FEATURE_NAMES,
    CausalEvidenceAccumulator,
    PrimitiveFrameEvidence,
)
from ..state.perclos import OnlinePerclos
from .five_state_events import StateTargetRef
from .five_state_labels import IGNORE_INDEX
from .five_state_standardization import StandardizedSession

VISUAL_CACHE_SCHEMA_VERSION = 4
EYE_EVIDENCE_CONFIG: Mapping[str, float] = {
    "closure_threshold": 0.6,
    "closure_release_threshold": 0.3,
    "visibility_threshold": 0.2,
    "minimum_valid_fraction": 0.6,
    "slow_closure_seconds": 0.5,
    "microsleep_seconds": 2.0,
    "uncertain_gap_seconds": 0.1,
}


def build_causal_evidence(
    frames: Sequence[PrimitiveFrameEvidence],
    *,
    fps: float,
    windows_seconds: Sequence[float] = (10.0, 30.0, 60.0),
) -> np.ndarray:
    """Build dense evidence with the exact accumulator used by runtime."""

    accumulator = CausalEvidenceAccumulator(
        fps=fps,
        perclos_windows=windows_seconds,
        **EYE_EVIDENCE_CONFIG,
    )
    if not frames:
        return np.empty(
            (0, len(EVIDENCE_FEATURE_NAMES)),
            dtype=np.float32,
        )
    return np.stack(
        [accumulator.update(frame).vector for frame in frames]
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class CausalEyeEvidence:
    closure_duration: np.ndarray
    perclos: np.ndarray
    reliable: np.ndarray


def build_causal_eye_evidence(
    eye_probabilities: np.ndarray,
    face_visibility: np.ndarray,
    *,
    fps: float,
    windows_seconds: Sequence[float] = (10.0, 30.0, 60.0),
) -> CausalEyeEvidence:
    """Convert predicted eye phase probabilities into strictly past-only evidence."""

    probabilities = np.asarray(eye_probabilities, dtype=np.float32)
    visibility = np.asarray(face_visibility, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("eye probabilities must have shape [frames, 4]")
    count = probabilities.shape[0]
    if visibility.shape != (count,):
        raise ValueError("face visibility must align with eye probabilities")
    if fps <= 0.0:
        raise ValueError("FPS must be positive")
    windows = tuple(sorted({float(value) for value in windows_seconds}))
    if len(windows) != 3 or any(value <= 0.0 for value in windows):
        raise ValueError("exactly three positive PERCLOS windows are required")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError("eye probabilities must be finite normalized rows")
    if np.any(~np.isfinite(visibility)) or np.any(
        (visibility < 0.0) | (visibility > 1.0)
    ):
        raise ValueError("face visibility must be finite and between zero and one")

    tracker = OnlinePerclos(
        fps=fps,
        windows_seconds=windows,
        **EYE_EVIDENCE_CONFIG,
    )
    closure_duration = np.zeros(count, dtype=np.float32)
    perclos = np.zeros((count, len(windows)), dtype=np.float32)
    reliable = np.zeros((count, len(windows)), dtype=np.float32)
    for index in range(count):
        snapshot = tracker.update(
            closed_probability=float(probabilities[index, 2]),
            visibility=float(visibility[index]),
        )
        closure_duration[index] = snapshot.closure_duration_seconds
        for window_index, window in enumerate(windows):
            value = snapshot.values[window]
            perclos[index, window_index] = 0.0 if value is None else value
            reliable[index, window_index] = float(
                snapshot.reliable_by_window[window]
            )
    return CausalEyeEvidence(closure_duration, perclos, reliable)


@dataclass(frozen=True)
class VisualSessionCache:
    session: str
    subject: str
    protocol: str
    fps: float
    frame_ids: np.ndarray
    visual_embedding: np.ndarray
    evidence: np.ndarray
    region_visibility: np.ndarray
    raw_eye_probabilities: np.ndarray
    eye_phase_targets: np.ndarray
    head_pose: np.ndarray
    yawn_probability: np.ndarray
    distraction_probability: np.ndarray
    visual_checkpoint_fingerprint: str
    region_cache_fingerprint: str
    evidence_feature_names: tuple[str, ...]
    split: Mapping[str, object]

    def __post_init__(self) -> None:
        count = len(self.frame_ids)
        if (
            not self.session
            or not self.subject
            or not self.protocol
            or self.fps <= 0.0
        ):
            raise ValueError("visual cache identity, protocol, and FPS are required")
        if (
            not self.visual_checkpoint_fingerprint
            or not self.region_cache_fingerprint
        ):
            raise ValueError(
                "visual and region-cache fingerprints are required"
            )
        if self.frame_ids.shape != (count,):
            raise ValueError("visual cache frame IDs must be one-dimensional")
        if len(np.unique(self.frame_ids)) != count:
            raise ValueError("visual cache frame IDs must be unique")
        if count > 1 and np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("visual cache frame IDs must be increasing")
        if (
            self.visual_embedding.ndim != 2
            or self.visual_embedding.shape[0] != count
        ):
            raise ValueError("visual embeddings must have shape [frames, feature]")
        if self.evidence.shape != (
            count,
            len(self.evidence_feature_names),
        ):
            raise ValueError(
                "visual cache evidence must align with feature names"
            )
        if tuple(self.evidence_feature_names) != EVIDENCE_FEATURE_NAMES:
            raise ValueError("visual cache evidence feature contract mismatch")
        expected_shapes = {
            "region visibility": (self.region_visibility, (count, 4)),
            "raw eye probabilities": (self.raw_eye_probabilities, (count, 3)),
            "eye phase targets": (self.eye_phase_targets, (count,)),
            "head pose": (self.head_pose, (count, 2)),
            "yawn probability": (self.yawn_probability, (count,)),
            "distraction probability": (
                self.distraction_probability,
                (count,),
            ),
        }
        for name, (array, shape) in expected_shapes.items():
            if np.asarray(array).shape != shape:
                raise ValueError(f"visual cache {name} must have shape {shape}")
        finite_arrays = (
            self.visual_embedding,
            self.evidence,
            self.region_visibility,
            self.raw_eye_probabilities,
            self.head_pose,
            self.yawn_probability,
            self.distraction_probability,
        )
        if any(not np.isfinite(array).all() for array in finite_arrays):
            raise ValueError("visual cache arrays must be finite")
        probabilities = (
            self.region_visibility,
            self.raw_eye_probabilities,
            self.yawn_probability,
            self.distraction_probability,
        )
        if any(np.any((array < 0.0) | (array > 1.0)) for array in probabilities):
            raise ValueError("visual cache probabilities must be in [0, 1]")
        if count and not np.allclose(
            self.raw_eye_probabilities.sum(axis=1),
            1.0,
            atol=1e-4,
        ):
            raise ValueError("raw eye probability rows must be normalized")
        phases = np.asarray(self.eye_phase_targets)
        if np.any(~np.isin(phases, (IGNORE_INDEX, 0, 1, 2, 3))):
            raise ValueError("eye phase targets contain an invalid class")


class VisualWindowDataset(Dataset[dict[str, Tensor | str | int | float]]):
    """Return fixed causal evidence histories ending at referenced targets."""

    def __init__(
        self,
        caches: Sequence[VisualSessionCache],
        sessions: Sequence[StandardizedSession],
        references: Sequence[StateTargetRef],
        *,
        length: int,
    ) -> None:
        if length <= 0:
            raise ValueError("visual window length must be positive")
        if len(caches) != len(sessions):
            raise ValueError("visual caches and standardized sessions must align")
        self.caches = tuple(caches)
        self.sessions = tuple(sessions)
        self.references = tuple(references)
        self.length = length
        for cache, session in zip(self.caches, self.sessions, strict=True):
            if cache.session != session.session or cache.subject != session.subject:
                raise ValueError("visual cache session identity mismatch")
            if not math.isclose(cache.fps, session.fps):
                raise ValueError("visual cache session FPS mismatch")
            if not np.array_equal(
                cache.frame_ids,
                np.asarray(session.frame_ids, dtype=np.int32),
            ):
                raise ValueError("visual cache frame IDs do not align with targets")
        for reference in self.references:
            if not 0 <= reference.session_index < len(self.sessions):
                raise IndexError("visual target has invalid session index")
            session = self.sessions[reference.session_index]
            if not 0 <= reference.target_index < len(session.frame_ids):
                raise IndexError("visual target has invalid frame index")
            if (
                session.targets.event_ids[reference.target_index]
                != reference.event_id
            ):
                raise ValueError("visual target event ID mismatch")

    def __len__(self) -> int:
        return len(self.references)

    @staticmethod
    def _history(
        values: np.ndarray,
        *,
        start: int,
        stop: int,
        pad: int,
    ) -> Tensor:
        tensor = torch.from_numpy(
            np.asarray(values[start:stop], dtype=np.float32).copy()
        )
        if pad:
            tensor = torch.cat(
                (
                    torch.zeros(
                        (pad, *tensor.shape[1:]),
                        dtype=torch.float32,
                    ),
                    tensor,
                ),
                dim=0,
            )
        return tensor

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int | float]:
        reference = self.references[index]
        cache = self.caches[reference.session_index]
        session = self.sessions[reference.session_index]
        target = reference.target_index
        start = max(0, target - self.length + 1)
        stop = target + 1
        pad = self.length - (stop - start)
        target_values = torch.tensor(
            session.targets.targets[start:stop],
            dtype=torch.long,
        )
        confidence_values = torch.tensor(
            session.targets.confidence[start:stop],
            dtype=torch.float32,
        )
        if pad:
            target_values = torch.cat(
                (
                    torch.full(
                        (pad,),
                        IGNORE_INDEX,
                        dtype=torch.long,
                    ),
                    target_values,
                )
            )
            confidence_values = torch.cat(
                (
                    torch.zeros(pad, dtype=torch.float32),
                    confidence_values,
                )
            )
        return {
            "visual_embedding": self._history(
                cache.visual_embedding,
                start=start,
                stop=stop,
                pad=pad,
            ),
            "evidence": self._history(
                cache.evidence,
                start=start,
                stop=stop,
                pad=pad,
            ),
            "context_valid": torch.cat(
                (
                    torch.zeros(pad, dtype=torch.float32),
                    torch.ones(stop - start, dtype=torch.float32),
                )
            ),
            "five_state_targets": target_values,
            "confidence_sequence": confidence_values,
            "five_state_target": torch.tensor(
                session.targets.targets[target],
                dtype=torch.long,
            ),
            "confidence": torch.tensor(
                session.targets.confidence[target],
                dtype=torch.float32,
            ),
            "session": session.session,
            "event_id": reference.event_id,
            "target_frame_id": int(session.frame_ids[target]),
            "target_timestamp": target / session.fps,
        }


def _paths(cache_dir: Path, session: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{session}.visual.npz",
        cache_dir / f"{session}.visual.json",
    )


def save_visual_session_cache(
    cache_dir: Path | str,
    cache: VisualSessionCache,
) -> None:
    """Atomically write one dense visual session cache."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path, metadata_path = _paths(cache_dir, cache.session)
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    with temporary_arrays.open("wb") as stream:
        np.savez(
            stream,
            frame_ids=np.asarray(cache.frame_ids, dtype=np.int32),
            visual_embedding=np.asarray(
                cache.visual_embedding,
                dtype=np.float16,
            ),
            evidence=np.asarray(cache.evidence, dtype=np.float32),
            region_visibility=np.asarray(
                cache.region_visibility,
                dtype=np.float32,
            ),
            raw_eye_probabilities=np.asarray(
                cache.raw_eye_probabilities,
                dtype=np.float32,
            ),
            eye_phase_targets=np.asarray(
                cache.eye_phase_targets,
                dtype=np.int16,
            ),
            head_pose=np.asarray(cache.head_pose, dtype=np.float32),
            yawn_probability=np.asarray(
                cache.yawn_probability,
                dtype=np.float32,
            ),
            distraction_probability=np.asarray(
                cache.distraction_probability,
                dtype=np.float32,
            ),
        )
    os.replace(temporary_arrays, arrays_path)
    metadata = {
        "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "complete": True,
        "session": cache.session,
        "subject": cache.subject,
        "protocol": cache.protocol,
        "fps": cache.fps,
        "visual_checkpoint_fingerprint": cache.visual_checkpoint_fingerprint,
        "region_cache_fingerprint": cache.region_cache_fingerprint,
        "evidence_feature_names": list(cache.evidence_feature_names),
        "evidence_config": dict(EYE_EVIDENCE_CONFIG),
        "split": dict(cache.split),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)


def load_visual_session_cache(
    cache_dir: Path | str,
    session: str,
    *,
    expected_visual_fingerprint: str,
    expected_region_fingerprint: str,
    expected_fps: float,
    expected_split: Mapping[str, object],
) -> VisualSessionCache:
    """Load one cache and reject any incompatible producer metadata."""

    arrays_path, metadata_path = _paths(Path(cache_dir), session)
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"visual cache is missing for {session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    compatibility = {
        "schema_version": (
            metadata.get("schema_version"),
            VISUAL_CACHE_SCHEMA_VERSION,
        ),
        "complete": (metadata.get("complete"), True),
        "session": (metadata.get("session"), session),
        "visual checkpoint fingerprint": (
            metadata.get("visual_checkpoint_fingerprint"),
            expected_visual_fingerprint,
        ),
        "region cache fingerprint": (
            metadata.get("region_cache_fingerprint"),
            expected_region_fingerprint,
        ),
        "evidence feature names": (
            metadata.get("evidence_feature_names"),
            list(EVIDENCE_FEATURE_NAMES),
        ),
        "evidence config": (
            metadata.get("evidence_config"),
            dict(EYE_EVIDENCE_CONFIG),
        ),
        "split": (metadata.get("split"), dict(expected_split)),
    }
    mismatched = [
        name for name, (actual, expected) in compatibility.items() if actual != expected
    ]
    if not math.isclose(float(metadata.get("fps", 0.0)), expected_fps):
        mismatched.append("FPS")
    if mismatched:
        raise ValueError(f"visual cache mismatch: {', '.join(mismatched)}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        return VisualSessionCache(
            session=session,
            subject=str(metadata["subject"]),
            protocol=str(metadata["protocol"]),
            fps=float(metadata["fps"]),
            frame_ids=arrays["frame_ids"].copy(),
            visual_embedding=arrays["visual_embedding"].copy(),
            evidence=arrays["evidence"].copy(),
            region_visibility=arrays["region_visibility"].copy(),
            raw_eye_probabilities=arrays["raw_eye_probabilities"].copy(),
            eye_phase_targets=arrays["eye_phase_targets"].astype(
                np.int64,
                copy=True,
            ),
            head_pose=arrays["head_pose"].copy(),
            yawn_probability=arrays["yawn_probability"].copy(),
            distraction_probability=arrays[
                "distraction_probability"
            ].copy(),
            visual_checkpoint_fingerprint=str(
                metadata["visual_checkpoint_fingerprint"]
            ),
            region_cache_fingerprint=str(
                metadata["region_cache_fingerprint"]
            ),
            evidence_feature_names=tuple(
                str(value)
                for value in metadata["evidence_feature_names"]
            ),
            split=dict(metadata["split"]),
        )


__all__ = [
    "CausalEyeEvidence",
    "EYE_EVIDENCE_CONFIG",
    "VISUAL_CACHE_SCHEMA_VERSION",
    "VisualSessionCache",
    "VisualWindowDataset",
    "build_causal_eye_evidence",
    "build_causal_evidence",
    "load_visual_session_cache",
    "save_visual_session_cache",
]
