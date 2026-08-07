"""Generate frozen 10 FPS visual embeddings bound to one nested-LOSO fold."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config import EyeTemporalConfig
from ..data.eye_crop_cache import EyeCropStore
from ..data.manifest import load_sessions, make_nested_loso_folds
from ..data.mouth_cache import MouthCropStore
from ..data.primitive_dataset import PrimitiveFrameDataset
from ..data.primitive_labels import TASK_CLASS_COUNTS
from ..data.primitive_records import PrimitiveFrameRecord, load_primitive_records
from ..data.temporal_embeddings import (
    SCHEMA_VERSION,
    EmbeddingSessionStore,
    save_embedding_session,
)
from ..models.eye_temporal import EyeTemporalNet
from ..models.mouth_visual import MouthVisualNet
from ..models.spatial_multitask import SpatialPrimitiveNet
from ..state.perclos import OnlinePerclos
from ..training.eye_trainer import SubjectSplitMetadata


def checkpoint_fingerprint(path: Path | str) -> str:
    """SHA-256 over every byte in a checkpoint file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_sessions(fold) -> dict[str, tuple[object, ...]]:
    return {
        "train": tuple(fold.train),
        "validation": tuple(fold.validation),
        "test": tuple(fold.test),
    }


def _split_from_payload(payload: Mapping[str, object]) -> SubjectSplitMetadata:
    raw = payload.get("split")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no split metadata")
    try:
        return SubjectSplitMetadata(
            tuple(str(value) for value in raw["train_subjects"]),
            str(raw["validation_subject"]),
            str(raw["test_subject"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint has invalid split metadata") from error


def validate_checkpoint_split(
    payload: Mapping[str, object],
    expected: SubjectSplitMetadata,
    *,
    checkpoint_name: str,
) -> None:
    actual = _split_from_payload(payload)
    if actual != expected:
        raise ValueError(
            f"{checkpoint_name} checkpoint split mismatch: expected {expected}, got {actual}"
        )


class _SessionEmbeddingDataset(Dataset[dict[str, Tensor | str | int]]):
    def __init__(
        self,
        records: Sequence[PrimitiveFrameRecord],
        *,
        face_cache: Path,
        mouth_cache: Path,
    ) -> None:
        self.records = tuple(records)
        if not self.records:
            raise ValueError("cannot embed an empty session")
        sessions = {record.session_name for record in self.records}
        if len(sessions) != 1:
            raise ValueError("embedding dataset must contain exactly one session")
        self.primitive = PrimitiveFrameDataset(
            self.records, face_cache_dir=face_cache, strict_face_cache=True
        )
        self.mouth_store = MouthCropStore(mouth_cache, self.records[0].session_name)
        self.mouth_rows = self.mouth_store.rows_for(
            tuple(record.frame_id for record in self.records)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        sample = self.primitive[index]
        row = int(self.mouth_rows[index])
        array = np.asarray(self.mouth_store.mouths[row], dtype=np.float32).copy()
        mouth = torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)
        visibility = float(self.mouth_store.visibility[row])
        sample["mouth"] = mouth
        sample["mouth_visibility"] = torch.tensor(visibility, dtype=torch.float32)
        return sample


def _expected_split(fold) -> SubjectSplitMetadata:
    return SubjectSplitMetadata(
        tuple(sorted({session.subject_id for session in fold.train})),
        fold.validation_subject,
        fold.held_out_subject,
    )


def expected_mouth_split(fold) -> SubjectSplitMetadata:
    """Return the same fold restricted to subjects that actually have s5."""
    return SubjectSplitMetadata(
        tuple(
            sorted(
                {
                    session.subject_id
                    for session in fold.train
                    if session.protocol == "s5"
                }
            )
        ),
        fold.validation_subject,
        fold.held_out_subject,
    )


def _model_config(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    config = payload.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{name} checkpoint has no model_config")
    return config


@torch.inference_mode()
def _generate_eye_session_features(
    *,
    cache_dir: Path,
    session_name: str,
    target_frame_ids: np.ndarray,
    eye_model: EyeTemporalNet,
    device: torch.device,
    fps: float,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Generate causal eye features and online PERCLOS for one full session."""

    if chunk_size < 1:
        raise ValueError("eye feature chunk size must be positive")
    store = EyeCropStore(cache_dir, session_name)
    if len(store.frame_ids) > 1 and np.any(np.diff(store.frame_ids) <= 0):
        raise ValueError("eye cache frame IDs must be strictly increasing")
    target_rows = store.rows_for(tuple(int(value) for value in target_frame_ids))
    context = sum(int(block.left_padding) for block in eye_model.temporal)
    use_amp = device.type == "cuda"
    temporal_values: list[np.ndarray] = []
    closed_values: list[np.ndarray] = []
    count = len(store.frame_ids)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        context_start = max(0, start - context)
        crops = np.asarray(store.eyes[context_start:stop]).copy()
        eyes = (
            torch.from_numpy(crops)
            .permute(0, 1, 4, 2, 3)
            .float()
            .div_(127.5)
            .sub_(1.0)
            .unsqueeze(0)
            .to(device, non_blocking=True)
        )
        visibility = (
            torch.from_numpy(store.visibility[context_start:stop].copy())
            .unsqueeze(0)
            .to(device, non_blocking=True)
        )
        with torch.autocast(device_type=device.type, enabled=use_amp):
            visual = eye_model.encode_visual(eyes, visibility)
            temporal = eye_model.encode_temporal(visual)
            closed = eye_model.closedness_head(temporal).squeeze(-1).sigmoid()
        discard = start - context_start
        temporal_values.append(
            temporal[0, discard:].float().cpu().numpy().astype(np.float32)
        )
        closed_values.append(
            closed[0, discard:].float().cpu().numpy().astype(np.float32)
        )
    temporal_full = np.concatenate(temporal_values, axis=0)
    closed_full = np.concatenate(closed_values, axis=0)
    if temporal_full.shape != (count, 128) or closed_full.shape != (count,):
        raise ValueError("eye checkpoint dimensions do not match schema v2")

    windows = (10.0, 30.0, 60.0)
    perclos = OnlinePerclos(fps=fps, windows_seconds=windows)
    evidence = np.zeros((count, 8), dtype=np.float32)
    for index, closed_probability in enumerate(closed_full):
        snapshot = perclos.update(
            closed_probability=float(closed_probability),
            visibility=float(store.visibility[index].max()),
        )
        evidence[index, 0] = float(closed_probability)
        evidence[index, 1] = snapshot.closure_duration_seconds
        for offset, window in enumerate(windows):
            value = snapshot.slow_values[window]
            evidence[index, 2 + offset] = 0.0 if value is None else float(value)
            evidence[index, 5 + offset] = float(snapshot.reliable_by_window[window])
    return {
        "eye": temporal_full[target_rows].astype(np.float16),
        "eye_visibility": store.visibility[target_rows].astype(np.float32, copy=True),
        "eye_evidence": evidence[target_rows].astype(np.float32, copy=True),
    }


@torch.inference_mode()
def _generate_session(
    records: Sequence[PrimitiveFrameRecord],
    *,
    face_cache: Path,
    mouth_cache: Path,
    eye_cache: Path,
    spatial_model: SpatialPrimitiveNet,
    mouth_model: MouthVisualNet,
    eye_model: EyeTemporalNet,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    dataset = _SessionEmbeddingDataset(
        records, face_cache=face_cache, mouth_cache=mouth_cache
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    eye_values = _generate_eye_session_features(
        cache_dir=eye_cache,
        session_name=records[0].session_name,
        target_frame_ids=np.asarray([record.frame_id for record in records], dtype=np.int32),
        eye_model=eye_model,
        device=device,
        fps=20.0,
        chunk_size=batch_size,
    )
    use_amp = device.type == "cuda"
    cabin_values: list[np.ndarray] = []
    face_values: list[np.ndarray] = []
    mouth_values: list[np.ndarray] = []
    face_visibility_values: list[np.ndarray] = []
    mouth_visibility_values: list[np.ndarray] = []
    head_pose_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    for raw_batch in loader:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in raw_batch.items()
            if isinstance(value, Tensor)
        }
        with torch.autocast(device_type=device.type, enabled=use_amp):
            cabin_embedding = spatial_model.encode_cabin(batch["cabin"])
            face_embedding = spatial_model.encode_face(
                batch["face"], batch["face_visibility"], batch["head_pose"]
            )
            mouth_embedding = mouth_model.encode(batch["mouth"])
            mouth_embedding = mouth_embedding * batch["mouth_visibility"].reshape(-1, 1)
        cabin_values.append(cabin_embedding.float().cpu().numpy().astype(np.float16))
        face_values.append(face_embedding.float().cpu().numpy().astype(np.float16))
        mouth_values.append(mouth_embedding.float().cpu().numpy().astype(np.float16))
        face_visibility_values.append(batch["face_visibility"].float().cpu().numpy())
        mouth_visibility_values.append(batch["mouth_visibility"].float().cpu().numpy())
        head_pose_values.append(batch["head_pose"].float().cpu().numpy())
        target_values.append(
            torch.stack([batch[task] for task in TASK_CLASS_COUNTS], dim=1)
            .int()
            .cpu()
            .numpy()
        )
    return {
        "cabin": np.concatenate(cabin_values).astype(np.float16, copy=False),
        "face": np.concatenate(face_values).astype(np.float16, copy=False),
        "mouth": np.concatenate(mouth_values).astype(np.float16, copy=False),
        "face_visibility": np.concatenate(face_visibility_values).astype(np.float32, copy=False),
        "mouth_visibility": np.concatenate(mouth_visibility_values).astype(np.float32, copy=False),
        "head_pose": np.concatenate(head_pose_values).astype(np.float32, copy=False),
        "targets": np.concatenate(target_values).astype(np.int32, copy=False),
        **eye_values,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate immutable fold-specific DMD visual embeddings"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--face-cache", type=Path, required=True)
    parser.add_argument("--mouth-cache", type=Path, required=True)
    parser.add_argument("--eye-cache", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--mouth-checkpoint", type=Path, required=True)
    parser.add_argument("--eye-checkpoint", type=Path, required=True)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_stride < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("sample stride and batch size must be positive; workers cannot be negative")
    sessions = load_sessions(args.root / "labels_20fps/manifest_20fps.json")
    fold = next(
        (
            candidate
            for candidate in make_nested_loso_folds(sessions)
            if candidate.held_out_subject == args.held_out_subject
        ),
        None,
    )
    if fold is None:
        raise SystemExit(f"unknown held-out subject: {args.held_out_subject}")
    split = _expected_split(fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_payload = torch.load(args.spatial_checkpoint, map_location="cpu", weights_only=False)
    mouth_payload = torch.load(args.mouth_checkpoint, map_location="cpu", weights_only=False)
    eye_payload = torch.load(args.eye_checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_split(spatial_payload, split, checkpoint_name="spatial")
    validate_checkpoint_split(
        mouth_payload,
        expected_mouth_split(fold),
        checkpoint_name="mouth",
    )
    validate_checkpoint_split(
        eye_payload,
        expected_mouth_split(fold),
        checkpoint_name="eye",
    )
    spatial_config = _model_config(spatial_payload, "spatial")
    mouth_config = _model_config(mouth_payload, "mouth")
    eye_config = EyeTemporalConfig(**eye_payload["config"])
    spatial_dim = int(spatial_config.get("embedding_dim", 0))
    mouth_dim = int(mouth_config.get("embedding_dim", 0))
    if (
        spatial_dim != 256
        or mouth_dim != 64
        or eye_config.embedding_dim != 128
        or eye_config.tcn_channels != 128
    ):
        raise ValueError(
            "embedding schema requires spatial=256, mouth=64, and eye=128"
        )
    spatial_model = SpatialPrimitiveNet(pretrained=False, embedding_dim=spatial_dim).to(device)
    spatial_model.load_state_dict(spatial_payload["model_state"])
    spatial_model.eval()
    mouth_model = MouthVisualNet(embedding_dim=mouth_dim).to(device)
    mouth_model.load_state_dict(mouth_payload["model_state"])
    mouth_model.eval()
    eye_model = EyeTemporalNet(
        embedding_dim=eye_config.embedding_dim,
        temporal_channels=eye_config.tcn_channels,
        dilations=eye_config.tcn_dilations,
        visibility_floor=eye_config.visibility_floor,
    ).to(device)
    eye_model.load_state_dict(eye_payload["model_state"])
    eye_model.eval()
    spatial_fingerprint = checkpoint_fingerprint(args.spatial_checkpoint)
    mouth_fingerprint = checkpoint_fingerprint(args.mouth_checkpoint)
    eye_fingerprint = checkpoint_fingerprint(args.eye_checkpoint)

    output = args.output / fold.held_out_subject
    output.mkdir(parents=True, exist_ok=True)
    index_entries: list[dict[str, object]] = []
    for partition, partition_records in partition_sessions(fold).items():
        records = load_primitive_records(partition_records)
        by_session: dict[str, list[PrimitiveFrameRecord]] = {}
        for record in records:
            by_session.setdefault(record.session_name, []).append(record)
        for session_name in sorted(by_session):
            selected = tuple(by_session[session_name][:: args.sample_stride])
            frame_ids = np.asarray([record.frame_id for record in selected], dtype=np.int32)
            timestamps = np.asarray([record.timestamp for record in selected], dtype=np.float32)
            if not args.overwrite:
                try:
                    existing = EmbeddingSessionStore(
                        output,
                        session_name,
                        expected_spatial_fingerprint=spatial_fingerprint,
                        expected_mouth_fingerprint=mouth_fingerprint,
                        expected_eye_fingerprint=eye_fingerprint,
                        expected_split=split,
                    )
                except FileNotFoundError:
                    existing = None
                if existing is not None:
                    if not np.array_equal(existing.frame_ids, frame_ids):
                        raise ValueError(
                            f"existing embedding rows differ for {session_name}; use --overwrite"
                        )
                    index_entries.append(
                        {
                            "partition": partition,
                            "subject": existing.subject_id,
                            "session": session_name,
                            "frames": len(existing),
                        }
                    )
                    continue
            generated = _generate_session(
                selected,
                face_cache=args.face_cache,
                mouth_cache=args.mouth_cache,
                eye_cache=args.eye_cache,
                spatial_model=spatial_model,
                mouth_model=mouth_model,
                eye_model=eye_model,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            save_embedding_session(
                output,
                session_name=session_name,
                subject_id=selected[0].subject_id,
                protocol=selected[0].protocol,
                fps=20.0 / args.sample_stride,
                frame_ids=frame_ids,
                timestamps=timestamps,
                spatial_fingerprint=spatial_fingerprint,
                mouth_fingerprint=mouth_fingerprint,
                eye_fingerprint=eye_fingerprint,
                split=split,
                task_names=tuple(TASK_CLASS_COUNTS),
                overwrite=args.overwrite,
                **generated,
            )
            entry = {
                "partition": partition,
                "subject": selected[0].subject_id,
                "session": session_name,
                "frames": len(selected),
            }
            index_entries.append(entry)
            print(json.dumps(entry), flush=True)

    index = {
        "schema_version": SCHEMA_VERSION,
        "held_out_subject": fold.held_out_subject,
        "spatial_fingerprint": spatial_fingerprint,
        "mouth_fingerprint": mouth_fingerprint,
        "eye_fingerprint": eye_fingerprint,
        "split": asdict(split),
        "sample_stride": args.sample_stride,
        "fps": 20.0 / args.sample_stride,
        "sessions": index_entries,
    }
    temporary = output / "index.json.tmp"
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
