"""Immutable denoised ocular evidence produced by a fold-specific LSTM."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from ..models.ocular_lstm import (
    assemble_ocular_features,
    phase_calibrated_closed_probability,
)
from ..state.causal_evidence import (
    EVIDENCE_FEATURE_NAMES,
    CausalEvidenceAccumulator,
    PrimitiveFrameEvidence,
)
from .five_state_visual_cache import EYE_EVIDENCE_CONFIG, VisualSessionCache


OCULAR_CACHE_SCHEMA_VERSION = 2
LEGACY_OCULAR_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OcularSessionCache:
    session: str
    subject: str
    protocol: str
    fps: float
    frame_ids: np.ndarray
    phase_probabilities: np.ndarray
    closed_probability: np.ndarray
    progress: np.ndarray
    reliability: np.ndarray
    evidence: np.ndarray
    drowsy_active: np.ndarray
    microsleep_active: np.ndarray
    visual_checkpoint_fingerprint: str
    region_cache_fingerprint: str
    ocular_checkpoint_fingerprint: str
    evidence_feature_names: tuple[str, ...]
    split: Mapping[str, object]
    evidence_config: Mapping[str, float] = field(
        default_factory=lambda: dict(EYE_EVIDENCE_CONFIG)
    )

    def __post_init__(self) -> None:
        count = len(self.frame_ids)
        if (
            not self.session
            or not self.subject
            or not self.protocol
            or self.fps <= 0.0
        ):
            raise ValueError("ocular cache identity, protocol, and FPS are required")
        if not all(
            (
                self.visual_checkpoint_fingerprint,
                self.region_cache_fingerprint,
                self.ocular_checkpoint_fingerprint,
            )
        ):
            raise ValueError("ocular cache checkpoint fingerprints are required")
        shapes = {
            "frame IDs": (self.frame_ids, (count,)),
            "phase probabilities": (self.phase_probabilities, (count, 4)),
            "closed probability": (self.closed_probability, (count,)),
            "progress": (self.progress, (count,)),
            "reliability": (self.reliability, (count,)),
            "evidence": (
                self.evidence,
                (count, len(self.evidence_feature_names)),
            ),
            "drowsy active": (self.drowsy_active, (count,)),
            "microsleep active": (self.microsleep_active, (count,)),
        }
        for name, (array, shape) in shapes.items():
            if np.asarray(array).shape != shape:
                raise ValueError(f"ocular cache {name} must have shape {shape}")
        if count > 1 and np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("ocular cache frame IDs must be strictly increasing")
        if len(np.unique(self.frame_ids)) != count:
            raise ValueError("ocular cache frame IDs must be unique")
        finite = (
            self.phase_probabilities,
            self.closed_probability,
            self.progress,
            self.reliability,
            self.evidence,
        )
        if any(not np.isfinite(array).all() for array in finite):
            raise ValueError("ocular cache arrays must be finite")
        probabilities = (
            self.phase_probabilities,
            self.closed_probability,
            self.progress,
            self.reliability,
        )
        if any(np.any((array < 0.0) | (array > 1.0)) for array in probabilities):
            raise ValueError("ocular cache probabilities must be in [0, 1]")
        if count and not np.allclose(
            self.phase_probabilities.sum(axis=1),
            1.0,
            atol=1e-4,
        ):
            raise ValueError("ocular phase probability rows must be normalized")
        if tuple(self.evidence_feature_names) != EVIDENCE_FEATURE_NAMES:
            raise ValueError("ocular cache evidence feature contract mismatch")
        config = {
            str(name): float(value)
            for name, value in self.evidence_config.items()
        }
        if set(config) != set(EYE_EVIDENCE_CONFIG):
            raise ValueError("ocular cache evidence config keys mismatch")
        CausalEvidenceAccumulator(fps=self.fps, **config)
        object.__setattr__(self, "evidence_config", config)


@torch.inference_mode()
def build_ocular_session_cache(
    visual: VisualSessionCache,
    model: nn.Module,
    *,
    device: torch.device,
    embedding_dim: int,
    region_dim: int,
    ocular_checkpoint_fingerprint: str,
    split: Mapping[str, object],
    evidence_config: Mapping[str, float] = EYE_EVIDENCE_CONFIG,
    amp: bool = True,
) -> OcularSessionCache:
    """Run one complete session causally and replay the runtime accumulator."""

    if not ocular_checkpoint_fingerprint:
        raise ValueError("ocular checkpoint fingerprint is required")
    if dict(split) != dict(visual.split):
        raise ValueError("ocular cache split must match the visual cache")
    expected_dim = embedding_dim + 4 * region_dim
    if visual.visual_embedding.shape[1] != expected_dim:
        raise ValueError("visual embedding dimension does not match ocular config")
    visual_embedding = torch.from_numpy(
        np.asarray(visual.visual_embedding, dtype=np.float32)
    ).to(device)
    regions = visual_embedding[:, embedding_dim:].reshape(-1, 4, region_dim)
    features = assemble_ocular_features(
        region_embeddings=regions,
        raw_eye_probabilities=torch.from_numpy(
            np.asarray(visual.raw_eye_probabilities, dtype=np.float32)
        ).to(device),
        eye_visibility=torch.from_numpy(
            np.asarray(visual.region_visibility[:, 1:3], dtype=np.float32)
        ).to(device),
    )
    model = model.to(device).eval()
    with torch.autocast(
        device_type=device.type,
        enabled=amp and device.type == "cuda",
    ):
        output = model(features.unsqueeze(0))
    phase_tensor = output.phase_logits[0].float().softmax(dim=-1)
    learned_closed_tensor = output.closed_logits[0].float().sigmoid()
    closed_tensor = phase_calibrated_closed_probability(
        phase_tensor,
        learned_closed_tensor,
    )
    phase_probabilities = phase_tensor.cpu().numpy()
    closed_probability = closed_tensor.cpu().numpy()
    progress = output.progress[0].float().cpu().numpy()
    learned_reliability = (
        output.reliability_logits[0].float().sigmoid().cpu().numpy()
    )
    observed_reliability = np.asarray(
        visual.region_visibility[:, 1:3].mean(axis=1),
        dtype=np.float32,
    )
    reliability = np.clip(
        learned_reliability * observed_reliability,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    selected_evidence_config = {
        str(name): float(value)
        for name, value in evidence_config.items()
    }
    accumulator = CausalEvidenceAccumulator(
        fps=visual.fps,
        behavioral_gate_suppression=True,
        **selected_evidence_config,
    )
    snapshots = []
    for index in range(len(visual.frame_ids)):
        snapshots.append(
            accumulator.update(
                PrimitiveFrameEvidence(
                    closed_probability=float(closed_probability[index]),
                    eye_visibility=float(reliability[index]),
                    head_pose=(
                        float(visual.head_pose[index, 0]),
                        float(visual.head_pose[index, 1]),
                    ),
                    yawn_probability=float(visual.yawn_probability[index]),
                    mouth_visibility=float(visual.region_visibility[index, 3]),
                    distraction_probability=float(
                        visual.distraction_probability[index]
                    ),
                    face_visibility=float(visual.region_visibility[index, 0]),
                    timestamp=index / visual.fps,
                )
            )
        )
    evidence = (
        np.stack([snapshot.vector for snapshot in snapshots]).astype(
            np.float32,
            copy=False,
        )
        if snapshots
        else np.empty((0, len(EVIDENCE_FEATURE_NAMES)), dtype=np.float32)
    )
    microsleep_active = np.asarray(
        [snapshot.microsleep_active for snapshot in snapshots],
        dtype=np.bool_,
    )
    drowsy_active = np.asarray(
        [snapshot.drowsy_active for snapshot in snapshots],
        dtype=np.bool_,
    )
    return OcularSessionCache(
        session=visual.session,
        subject=visual.subject,
        protocol=visual.protocol,
        fps=visual.fps,
        frame_ids=visual.frame_ids.copy(),
        phase_probabilities=phase_probabilities.astype(np.float32, copy=False),
        closed_probability=closed_probability.astype(np.float32, copy=False),
        progress=progress.astype(np.float32, copy=False),
        reliability=reliability,
        evidence=evidence,
        drowsy_active=drowsy_active,
        microsleep_active=microsleep_active,
        visual_checkpoint_fingerprint=visual.visual_checkpoint_fingerprint,
        region_cache_fingerprint=visual.region_cache_fingerprint,
        ocular_checkpoint_fingerprint=ocular_checkpoint_fingerprint,
        evidence_feature_names=EVIDENCE_FEATURE_NAMES,
        split=dict(split),
        evidence_config=selected_evidence_config,
    )


def _paths(cache_dir: Path, session: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{session}.ocular.npz",
        cache_dir / f"{session}.ocular.json",
    )


def save_ocular_session_cache(
    cache_dir: Path | str,
    cache: OcularSessionCache,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path, metadata_path = _paths(cache_dir, cache.session)
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    with temporary_arrays.open("wb") as stream:
        np.savez(
            stream,
            frame_ids=np.asarray(cache.frame_ids, dtype=np.int32),
            phase_probabilities=np.asarray(
                cache.phase_probabilities,
                dtype=np.float32,
            ),
            closed_probability=np.asarray(
                cache.closed_probability,
                dtype=np.float32,
            ),
            progress=np.asarray(cache.progress, dtype=np.float32),
            reliability=np.asarray(cache.reliability, dtype=np.float32),
            evidence=np.asarray(cache.evidence, dtype=np.float32),
            drowsy_active=np.asarray(
                cache.drowsy_active,
                dtype=np.bool_,
            ),
            microsleep_active=np.asarray(
                cache.microsleep_active,
                dtype=np.bool_,
            ),
        )
    os.replace(temporary_arrays, arrays_path)
    metadata = {
        "schema_version": OCULAR_CACHE_SCHEMA_VERSION,
        "complete": True,
        "session": cache.session,
        "subject": cache.subject,
        "protocol": cache.protocol,
        "fps": cache.fps,
        "visual_checkpoint_fingerprint": cache.visual_checkpoint_fingerprint,
        "region_cache_fingerprint": cache.region_cache_fingerprint,
        "ocular_checkpoint_fingerprint": cache.ocular_checkpoint_fingerprint,
        "evidence_feature_names": list(cache.evidence_feature_names),
        "evidence_config": dict(cache.evidence_config),
        "split": dict(cache.split),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)


def load_ocular_session_cache(
    cache_dir: Path | str,
    session: str,
    *,
    expected_visual_fingerprint: str,
    expected_region_fingerprint: str,
    expected_ocular_fingerprint: str,
    expected_fps: float,
    expected_split: Mapping[str, object],
    expected_evidence_config: Mapping[str, float] = EYE_EVIDENCE_CONFIG,
    expected_schema_version: int = OCULAR_CACHE_SCHEMA_VERSION,
) -> OcularSessionCache:
    arrays_path, metadata_path = _paths(Path(cache_dir), session)
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"ocular cache is missing for {session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema version": int(expected_schema_version),
        "complete": True,
        "session": session,
        "visual checkpoint fingerprint": expected_visual_fingerprint,
        "region cache fingerprint": expected_region_fingerprint,
        "ocular checkpoint fingerprint": expected_ocular_fingerprint,
        "evidence feature names": list(EVIDENCE_FEATURE_NAMES),
        "evidence config": dict(expected_evidence_config),
        "split": dict(expected_split),
    }
    actual = {
        "schema version": metadata.get("schema_version"),
        "complete": metadata.get("complete"),
        "session": metadata.get("session"),
        "visual checkpoint fingerprint": metadata.get(
            "visual_checkpoint_fingerprint"
        ),
        "region cache fingerprint": metadata.get("region_cache_fingerprint"),
        "ocular checkpoint fingerprint": metadata.get(
            "ocular_checkpoint_fingerprint"
        ),
        "evidence feature names": metadata.get("evidence_feature_names"),
        "evidence config": metadata.get("evidence_config"),
        "split": metadata.get("split"),
    }
    mismatched = [
        name for name, value in expected.items() if actual.get(name) != value
    ]
    if not math.isclose(float(metadata.get("fps", 0.0)), expected_fps):
        mismatched.append("FPS")
    if mismatched:
        raise ValueError(f"ocular cache mismatch: {', '.join(mismatched)}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        drowsy_active = (
            arrays["drowsy_active"].astype(np.bool_, copy=True)
            if int(expected_schema_version) >= OCULAR_CACHE_SCHEMA_VERSION
            else np.zeros(len(arrays["frame_ids"]), dtype=np.bool_)
        )
        return OcularSessionCache(
            session=session,
            subject=str(metadata["subject"]),
            protocol=str(metadata["protocol"]),
            fps=float(metadata["fps"]),
            frame_ids=arrays["frame_ids"].copy(),
            phase_probabilities=arrays["phase_probabilities"].copy(),
            closed_probability=arrays["closed_probability"].copy(),
            progress=arrays["progress"].copy(),
            reliability=arrays["reliability"].copy(),
            evidence=arrays["evidence"].copy(),
            drowsy_active=drowsy_active,
            microsleep_active=arrays["microsleep_active"].astype(
                np.bool_,
                copy=True,
            ),
            visual_checkpoint_fingerprint=str(
                metadata["visual_checkpoint_fingerprint"]
            ),
            region_cache_fingerprint=str(
                metadata["region_cache_fingerprint"]
            ),
            ocular_checkpoint_fingerprint=str(
                metadata["ocular_checkpoint_fingerprint"]
            ),
            evidence_feature_names=tuple(
                str(value) for value in metadata["evidence_feature_names"]
            ),
            split=dict(metadata["split"]),
            evidence_config={
                str(name): float(value)
                for name, value in metadata["evidence_config"].items()
            },
        )


__all__ = [
    "LEGACY_OCULAR_CACHE_SCHEMA_VERSION",
    "OCULAR_CACHE_SCHEMA_VERSION",
    "OcularSessionCache",
    "build_ocular_session_cache",
    "load_ocular_session_cache",
    "save_ocular_session_cache",
]
