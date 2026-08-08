"""Staged LOSO training for MobileNetV3-Large and a causal LSTM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from ..data.evidence_regions import EvidenceRegionStore
from ..data.eye_sequences import PHASE_NAMES
from ..data.five_state_standardization import load_standardized_index
from ..data.five_state_standardization import load_standardized_session
from ..data.five_state_labels import IGNORE_INDEX
from ..data.five_state_events import (
    build_chronological_references,
    build_subject_event_epoch_references,
    extract_state_events,
)
from ..data.five_state_visual import (
    FiveStateFrameDataset,
    build_visual_training_indices,
    load_visual_records,
)
from ..data.five_state_visual_cache import (
    EYE_EVIDENCE_CONFIG,
    VISUAL_CACHE_SCHEMA_VERSION,
    VisualSessionCache,
    VisualWindowDataset,
    build_causal_evidence,
    load_visual_session_cache,
    save_visual_session_cache,
)
from ..data.four_state_labels import five_to_four_target
from ..data.four_state_windows import FourStateWindowDataset
from ..data.ocular_cache import (
    LEGACY_OCULAR_CACHE_SCHEMA_VERSION,
    OCULAR_CACHE_SCHEMA_VERSION,
    build_ocular_session_cache,
    load_ocular_session_cache,
    save_ocular_session_cache,
)
from ..data.ocular_windows import (
    OcularWindowDataset,
    build_chronological_ocular_references,
    build_ocular_epoch_references,
)
from ..data.method_a_mixed import (
    CoverageFirstBatchSampler,
    build_dmd_method_a_pools,
    build_nitymed_method_a_pools,
    select_method_a_adaptation_pools,
)
from ..data.nitymed import NitymedFrameDataset
from ..state.causal_evidence import (
    EVIDENCE_FEATURE_NAMES,
    PrimitiveFrameEvidence,
)
from ..models.mobilenet_lstm import (
    CausalFourStateLSTM,
    CausalFiveStateLSTM,
    MobileNetV3LargeVisualEncoder,
    initialize_phase_aware_temporal,
)
from ..models.ocular_lstm import CausalOcularLSTM
from ..metrics_events import (
    aggregate_loso_metrics,
    five_state_event_metrics,
)
from .precompute_embeddings import checkpoint_fingerprint
from ..training.eye_trainer import SubjectSplitMetadata
from ..training.mobilenet_lstm_trainer import (
    EvidenceLossWeights,
    effective_four_state_weights,
    evaluate_fused_temporal,
    evaluate_visual_frames,
    effective_number_weights,
    fused_temporal_selection_score,
    restore_visual_checkpoint,
    save_visual_checkpoint,
    evaluate_temporal,
    temporal_selection_score,
    train_four_state_epoch,
    train_temporal_epoch,
    train_visual_epoch,
)
from ..training.ocular_trainer import (
    OCULAR_CHECKPOINT_SCHEMA_VERSION,
    OcularInputAugmentation,
    OcularLossWeights,
    evaluate_ocular,
    restore_ocular_checkpoint,
    save_ocular_checkpoint,
    train_ocular_epoch,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one MobileNetV3-Large plus causal five-state LSTM"
    )
    parser.add_argument(
        "--stage",
        choices=(
            "visual",
            "visual-mixed",
            "cache",
            "ocular",
            "ocular-cache",
            "lstm",
            "evaluate",
            "loso",
        ),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--standardized-index", type=Path, required=True)
    parser.add_argument("--face-cache", type=Path)
    parser.add_argument("--region-cache", type=Path, required=True)
    parser.add_argument("--initial-visual-checkpoint", type=Path)
    parser.add_argument("--initial-temporal-checkpoint", type=Path)
    parser.add_argument("--nitymed-frame-manifest", type=Path)
    parser.add_argument("--nitymed-region-cache", type=Path)
    parser.add_argument("--nitymed-teacher-cache", type=Path)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--validation-subject")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--region-dim", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--ocular-sequence-length", type=int, default=60)
    parser.add_argument("--visual-epochs", type=int, default=12)
    parser.add_argument("--ocular-epochs", type=int, default=20)
    parser.add_argument(
        "--lstm-epochs",
        "--temporal-epochs",
        dest="temporal_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--visual-batch-size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lstm-batch-size",
        "--temporal-batch-size",
        dest="temporal_batch_size",
        type=int,
        default=128,
    )
    parser.add_argument("--ocular-batch-size", type=int, default=256)
    parser.add_argument("--ocular-channels", type=int, default=64)
    parser.add_argument("--ocular-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ocular-samples-per-epoch", type=int, default=20_000)
    parser.add_argument("--ocular-max-windows-per-event", type=int, default=8)
    parser.add_argument(
        "--ocular-input-corruption-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ocular-input-corruption-min-run-length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ocular-input-corruption-anchor-frames",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--ocular-input-corruption-max-block-length",
        type=int,
        default=15,
    )
    parser.add_argument("--uncertain-eye-gap-seconds", type=float, default=0.10)
    parser.add_argument("--microsleep-seconds", type=float, default=2.0)
    parser.add_argument("--microsleep-gate-ocular-cache", type=Path)
    parser.add_argument("--microsleep-gate-ocular-checkpoint", type=Path)
    parser.add_argument("--temporal-channels", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=40_000)
    parser.add_argument("--max-windows-per-event", type=int, default=8)
    parser.add_argument("--eye-sample-fraction", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mixed-consistency-weight", type=float, default=0.5)
    parser.add_argument(
        "--mixed-adaptation",
        choices=("all", "eye-yawn", "yawn"),
        default="all",
    )
    parser.add_argument(
        "--freeze-mixed-batch-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temporal-learning-rate", type=float, default=3e-4)
    parser.add_argument("--temporal-final-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--allow-confirmation-margin-checkpoint-reuse",
        action="store_true",
        help=(
            "allow a schema-1 primitive visual checkpoint only when schema 2 "
            "adds the microsleep confirmation field and changes nothing else"
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ocular_evidence_config(args: argparse.Namespace) -> dict[str, float]:
    config = dict(EYE_EVIDENCE_CONFIG)
    config["uncertain_gap_seconds"] = float(
        args.uncertain_eye_gap_seconds
    )
    config["microsleep_seconds"] = float(args.microsleep_seconds)
    return config


def _mixed_visual_collate(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor]:
    """Collate the common tensor contract and leave source metadata for audits."""

    if not samples:
        raise ValueError("cannot collate an empty mixed visual batch")
    keys = set(samples[0])
    for sample in samples[1:]:
        keys.intersection_update(sample)
    return {
        key: default_collate([sample[key] for sample in samples])
        for key in sorted(keys)
        if isinstance(samples[0][key], torch.Tensor)
    }


def _configure_mixed_trainable_parameters(
    model: MobileNetV3LargeVisualEncoder,
    *,
    adaptation: str,
) -> None:
    if adaptation == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    if adaptation not in {"eye-yawn", "yawn"}:
        raise ValueError(f"unknown mixed adaptation: {adaptation}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = (
        (model.eye_head, model.yawn_head)
        if adaptation == "eye-yawn"
        else (model.yawn_head,)
    )
    for head in heads:
        for parameter in head.parameters():
            parameter.requires_grad_(True)


def _mixed_loss_weights(args: argparse.Namespace) -> EvidenceLossWeights:
    if args.mixed_adaptation == "yawn":
        return EvidenceLossWeights(
            eye=0.0,
            visibility=0.0,
            yawn=1.0,
            distraction=0.0,
            pose=0.0,
            consistency=0.0,
        )
    if args.mixed_adaptation == "eye-yawn":
        return EvidenceLossWeights(
            eye=1.0,
            visibility=0.0,
            yawn=1.0,
            distraction=0.0,
            pose=0.0,
            consistency=0.0,
        )
    return EvidenceLossWeights(consistency=args.mixed_consistency_weight)


def _fold_path(index_path: Path, held_out_subject: str) -> Path:
    path = index_path.resolve().parent / "folds" / f"{held_out_subject}.json"
    if not path.is_file():
        raise ValueError(f"LOSO fold is missing for {held_out_subject}: {path}")
    return path


def _split(fold: Mapping[str, object]) -> SubjectSplitMetadata:
    return SubjectSplitMetadata(
        tuple(str(value) for value in fold["train_subjects"]),
        str(fold["validation_subject"]),
        str(fold["test_subject"]),
    )


def _validate_split(
    split: SubjectSplitMetadata,
    args: argparse.Namespace,
) -> None:
    train = set(split.train_subjects)
    if (
        split.test_subject != args.held_out_subject
        or split.test_subject in train
        or split.validation_subject in train
        or split.test_subject == split.validation_subject
    ):
        raise ValueError("LOSO fold contains subject leakage or mismatch")
    if (
        args.validation_subject is not None
        and split.validation_subject != args.validation_subject
    ):
        raise ValueError("LOSO fold validation subject mismatch")


def _region_stores(
    records: Sequence[object],
    cache_dir: Path,
) -> dict[str, EvidenceRegionStore]:
    sessions = sorted({str(getattr(record, "session")) for record in records})
    return {
        session: EvidenceRegionStore(cache_dir, session)
        for session in sessions
    }


def _region_cache_fingerprint(
    cache_dir: Path,
    session_names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for session in sorted(set(session_names)):
        path = cache_dir / f"{session}.regions.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"evidence-region cache is missing for {session}"
            )
        digest.update(session.encode("utf-8"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _visual_resume_best_score(
    checkpoint_payload: Mapping[str, object],
) -> float:
    metrics = checkpoint_payload.get("metrics")
    if not isinstance(metrics, Mapping) or "selection_score" not in metrics:
        raise ValueError(
            "visual resume checkpoint is missing its selection score"
        )
    return float(metrics["selection_score"])


def _visual_label_contract_mode(
    checkpoint: Mapping[str, object],
    standardized_index: Mapping[str, object],
    *,
    allow_confirmation_upgrade: bool,
) -> str | None:
    """Return the audited compatibility mode for primitive visual weights."""

    source_config = checkpoint.get("label_config")
    target_config = standardized_index.get("config")
    if not isinstance(source_config, Mapping) or not isinstance(
        target_config, Mapping
    ):
        return None
    source = dict(source_config)
    target = dict(target_config)
    source_schema = int(checkpoint.get("standardized_schema_version", 0))
    target_schema = int(standardized_index.get("schema_version", 0))
    if source_schema == target_schema and source == target:
        return "exact"
    if not allow_confirmation_upgrade:
        return None
    confirmation_field = "microsleep_confirmation_seconds"
    if source_schema != 1 or target_schema != 2:
        return None
    if confirmation_field in source or confirmation_field not in target:
        return None
    target_without_confirmation = dict(target)
    target_without_confirmation.pop(confirmation_field)
    if source != target_without_confirmation:
        return None
    return "schema1_to_schema2_confirmation_only"


def _prepare_visual(args: argparse.Namespace) -> int:
    index_path = args.standardized_index.resolve()
    index = load_standardized_index(index_path)
    fold_path = _fold_path(index_path, args.held_out_subject)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    split = _split(fold)
    _validate_split(split, args)
    train_records = load_visual_records(
        root=args.root,
        standardized_index=index_path,
        fold_manifest=fold_path,
        partition="train",
    )
    validation_records = load_visual_records(
        root=args.root,
        standardized_index=index_path,
        fold_manifest=fold_path,
        partition="validation",
    )
    image_size = (args.image_width, args.image_height)
    train_dataset = FiveStateFrameDataset(
        train_records,
        image_size=image_size,
        training=True,
        region_stores=_region_stores(train_records, args.region_cache),
    )
    validation_dataset = FiveStateFrameDataset(
        validation_records,
        image_size=image_size,
        training=False,
        region_stores=_region_stores(
            validation_records,
            args.region_cache,
        ),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3LargeVisualEncoder(
        embedding_dim=args.embedding_dim,
        region_dim=args.region_dim,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    model_config = {
        "embedding_dim": args.embedding_dim,
        "region_dim": args.region_dim,
        "pretrained": args.pretrained,
        "image_size": image_size,
        "eye_aperture_classes": 3,
        "region_order": ["face", "left_eye", "right_eye", "mouth"],
    }
    label_config = dict(index["config"])
    checkpoint_dir = (
        args.output.resolve() / args.held_out_subject / "visual"
    )
    checkpoint = checkpoint_dir / "best.pt"
    start_epoch = 1
    best_score = -1.0
    if args.resume:
        if not checkpoint.is_file():
            raise ValueError(f"visual resume checkpoint is missing: {checkpoint}")
        start_epoch, restored = restore_visual_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_split=split,
            expected_standardized_schema_version=int(index["schema_version"]),
            expected_target_fps=float(index["target_fps"]),
            expected_label_config=label_config,
            map_location=device,
        )
        best_score = _visual_resume_best_score(restored)
    stale_epochs = 0
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    for epoch in range(start_epoch, args.visual_epochs + 1):
        epoch_seed = args.seed + epoch - 1
        selected, audit = build_visual_training_indices(
            train_records,
            samples=args.samples_per_epoch,
            eye_fraction=args.eye_sample_fraction,
            seed=epoch_seed,
        )
        train_loader = DataLoader(
            Subset(train_dataset, selected),
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        train_metrics = train_visual_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            amp=args.amp,
        )
        validation_metrics = evaluate_visual_frames(
            model,
            validation_loader,
            device=device,
            amp=args.amp,
        )
        score = (
            float(validation_metrics["eye_aperture"]["macro_f1"])
            + float(validation_metrics["yawn"]["macro_f1"])
            + float(validation_metrics["distraction"]["macro_f1"])
        ) / 3.0 - float(validation_metrics["pose_mae"]) / 90.0
        validation_metrics["selection_score"] = score
        report = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(report), flush=True)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_visual_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                split=split,
                standardized_schema_version=int(index["schema_version"]),
                target_fps=float(index["target_fps"]),
                label_config=label_config,
                epoch=epoch,
                metrics=validation_metrics,
            )
            _atomic_json(
                checkpoint_dir / "metrics.json",
                {
                    **report,
                    "sampling_audit": {
                        "requested": audit.final_state.requested,
                        "final_state_counts": {
                            state.name.lower(): count
                            for state, count in audit.final_state.counts.items()
                        },
                        "missing_final_states": [
                            state.name.lower()
                            for state in audit.final_state.missing_states
                        ],
                        "eye_samples": audit.eye_samples,
                        "eye_phase_counts": dict(audit.eye_phase_counts),
                        "epoch_seed": epoch_seed,
                    },
                    "split": asdict(split),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("visual training did not produce a checkpoint")
    return 0


def _prepare_visual_mixed(args: argparse.Namespace) -> int:
    """Continue A0 on all eligible DMD and NITYMED primitive evidence."""

    required = {
        "initial visual checkpoint": args.initial_visual_checkpoint,
        "NITYMED frame manifest": args.nitymed_frame_manifest,
        "NITYMED region cache": args.nitymed_region_cache,
        "NITYMED teacher cache": args.nitymed_teacher_cache,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise ValueError(
            "visual-mixed requires " + ", ".join(missing)
        )
    assert args.initial_visual_checkpoint is not None
    assert args.nitymed_frame_manifest is not None
    assert args.nitymed_region_cache is not None
    assert args.nitymed_teacher_cache is not None
    for name, path in required.items():
        assert path is not None
        if name.endswith("checkpoint") or name.endswith("manifest"):
            if not path.is_file():
                raise ValueError(f"{name} is missing: {path}")
        elif not path.is_dir():
            raise ValueError(f"{name} is missing: {path}")

    index_path = args.standardized_index.resolve()
    index = load_standardized_index(index_path)
    fold_path = _fold_path(index_path, args.held_out_subject)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    split = _split(fold)
    _validate_split(split, args)
    train_records = load_visual_records(
        root=args.root,
        standardized_index=index_path,
        fold_manifest=fold_path,
        partition="train",
    )
    validation_records = load_visual_records(
        root=args.root,
        standardized_index=index_path,
        fold_manifest=fold_path,
        partition="validation",
    )
    train_regions = _region_stores(train_records, args.region_cache)
    image_size = (args.image_width, args.image_height)
    nitymed_dataset = NitymedFrameDataset(
        root=args.nitymed_frame_manifest.resolve().parent,
        frame_manifest=args.nitymed_frame_manifest,
        region_cache=args.nitymed_region_cache,
        teacher_cache=args.nitymed_teacher_cache,
        image_size=image_size,
        training=True,
        include_eye_targets=args.mixed_adaptation == "all",
    )
    expected_teacher_dim = args.embedding_dim + 4 * args.region_dim
    if nitymed_dataset.teacher_embedding_dim != expected_teacher_dim:
        raise ValueError(
            "NITYMED teacher embedding dimension does not match the visual model"
        )
    dmd_dataset = FiveStateFrameDataset(
        train_records,
        image_size=image_size,
        training=True,
        region_stores=train_regions,
        teacher_embedding_dim=nitymed_dataset.teacher_embedding_dim,
    )
    mixed_dataset = ConcatDataset((dmd_dataset, nitymed_dataset))
    validation_dataset = FiveStateFrameDataset(
        validation_records,
        image_size=image_size,
        training=False,
        region_stores=_region_stores(validation_records, args.region_cache),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    pools = {
        **build_dmd_method_a_pools(
            train_records,
            region_stores=train_regions,
            target_fps=float(index["target_fps"]),
        ),
        **build_nitymed_method_a_pools(
            nitymed_dataset.records,
            teacher_stores=nitymed_dataset.teacher_stores,
            dmd_offset=len(dmd_dataset),
        ),
    }
    pools = select_method_a_adaptation_pools(
        pools,
        adaptation=args.mixed_adaptation,
    )
    if not pools:
        raise ValueError("visual-mixed found no eligible primitive samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3LargeVisualEncoder(
        embedding_dim=args.embedding_dim,
        region_dim=args.region_dim,
        pretrained=args.pretrained,
    ).to(device)
    model_config = {
        "embedding_dim": args.embedding_dim,
        "region_dim": args.region_dim,
        "pretrained": args.pretrained,
        "image_size": image_size,
        "eye_aperture_classes": 3,
        "region_order": ["face", "left_eye", "right_eye", "mouth"],
    }
    label_config = dict(index["config"])
    if args.resume:
        _configure_mixed_trainable_parameters(
            model, adaptation=args.mixed_adaptation
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    checkpoint_dir = args.output.resolve() / args.held_out_subject / "visual"
    checkpoint = checkpoint_dir / "best.pt"
    start_epoch = 1
    best_score = -1.0
    pool_sizes = {name: len(indices) for name, indices in sorted(pools.items())}
    sampling_audit = {
        "strategy": "coverage_first",
        "pool_sizes": pool_sizes,
        "eligible_occurrences": sum(pool_sizes.values()),
        "adaptation": args.mixed_adaptation,
        "initial_visual_checkpoint": str(
            args.initial_visual_checkpoint.resolve()
        ),
        "initial_visual_fingerprint": checkpoint_fingerprint(
            args.initial_visual_checkpoint.resolve()
        ),
    }
    print(
        json.dumps(
            {
                "stage": "visual-mixed",
                **sampling_audit,
                "batches_per_epoch": math.ceil(
                    sampling_audit["eligible_occurrences"] / args.batch_size
                ),
            }
        ),
        flush=True,
    )
    if args.resume:
        if not checkpoint.is_file():
            raise ValueError(f"visual-mixed resume checkpoint is missing: {checkpoint}")
        start_epoch, restored = restore_visual_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_split=split,
            expected_standardized_schema_version=int(index["schema_version"]),
            expected_target_fps=float(index["target_fps"]),
            expected_label_config=label_config,
            map_location=device,
        )
        best_score = _visual_resume_best_score(restored)
    else:
        _, initial_payload = restore_visual_checkpoint(
            args.initial_visual_checkpoint.resolve(),
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_split=split,
            expected_standardized_schema_version=int(index["schema_version"]),
            expected_target_fps=float(index["target_fps"]),
            expected_label_config=label_config,
            map_location=device,
        )
        _configure_mixed_trainable_parameters(
            model, adaptation=args.mixed_adaptation
        )
        optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        best_score = _visual_resume_best_score(initial_payload)
        initial_metrics = dict(initial_payload["metrics"])
        save_visual_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            split=split,
            standardized_schema_version=int(index["schema_version"]),
            target_fps=float(index["target_fps"]),
            label_config=label_config,
            epoch=0,
            metrics=initial_metrics,
        )
        _atomic_json(
            checkpoint_dir / "metrics.json",
            {
                "epoch": 0,
                "train": None,
                "validation": initial_metrics,
                "sampling_audit": {
                    **sampling_audit,
                    "epoch_seed": None,
                },
                "split": asdict(split),
            },
        )

    stale_epochs = 0
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    for epoch in range(start_epoch, args.visual_epochs + 1):
        epoch_seed = args.seed + epoch - 1
        sampler = CoverageFirstBatchSampler(
            pools=pools,
            batch_size=args.batch_size,
            seed=epoch_seed,
        )
        train_loader = DataLoader(
            mixed_dataset,
            batch_sampler=sampler,
            collate_fn=_mixed_visual_collate,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
        train_metrics = train_visual_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            amp=args.amp,
            loss_weights=_mixed_loss_weights(args),
            freeze_batch_norm=args.freeze_mixed_batch_norm,
        )
        validation_metrics = evaluate_visual_frames(
            model, validation_loader, device=device, amp=args.amp
        )
        score = (
            float(validation_metrics["eye_aperture"]["macro_f1"])
            + float(validation_metrics["yawn"]["macro_f1"])
            + float(validation_metrics["distraction"]["macro_f1"])
        ) / 3.0 - float(validation_metrics["pose_mae"]) / 90.0
        validation_metrics["selection_score"] = score
        report = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(report), flush=True)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_visual_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                split=split,
                standardized_schema_version=int(index["schema_version"]),
                target_fps=float(index["target_fps"]),
                label_config=label_config,
                epoch=epoch,
                metrics=validation_metrics,
            )
            _atomic_json(
                checkpoint_dir / "metrics.json",
                {
                    **report,
                    "sampling_audit": {
                        **sampling_audit,
                        "epoch_seed": epoch_seed,
                    },
                    "split": asdict(split),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("visual-mixed training did not produce a checkpoint")
    return 0


@torch.inference_mode()
def _cache_visual(args: argparse.Namespace) -> int:
    index_path = args.standardized_index.resolve()
    index = load_standardized_index(index_path)
    fold_path = _fold_path(index_path, args.held_out_subject)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    split = _split(fold)
    _validate_split(split, args)
    checkpoint = (
        args.output.resolve()
        / args.held_out_subject
        / "visual"
        / "best.pt"
    )
    if not checkpoint.is_file():
        raise ValueError(f"visual checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_contract_mode = _visual_label_contract_mode(
        payload,
        index,
        allow_confirmation_upgrade=(
            args.allow_confirmation_margin_checkpoint_reuse
        ),
    )
    expected_model_config = {
        "embedding_dim": args.embedding_dim,
        "region_dim": args.region_dim,
        "pretrained": args.pretrained,
        "image_size": (args.image_width, args.image_height),
        "eye_aperture_classes": 3,
        "region_order": ["face", "left_eye", "right_eye", "mouth"],
    }
    compatibility = {
        "model_config": (payload.get("model_config"), expected_model_config),
        "split": (payload.get("split"), asdict(split)),
        "target_fps": (
            float(payload.get("target_fps", 0.0)),
            float(index["target_fps"]),
        ),
    }
    mismatched = [
        name for name, (actual, expected) in compatibility.items() if actual != expected
    ]
    if label_contract_mode is None:
        if payload.get("standardized_schema_version") != int(
            index["schema_version"]
        ):
            mismatched.append("standardized_schema_version")
        if payload.get("label_config") != dict(index["config"]):
            mismatched.append("label_config")
    if mismatched:
        raise ValueError(f"visual cache producer mismatch: {', '.join(mismatched)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3LargeVisualEncoder(
        embedding_dim=args.embedding_dim,
        region_dim=args.region_dim,
        pretrained=args.pretrained,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    visual_fingerprint = checkpoint_fingerprint(checkpoint)
    cache_dir = (
        args.output.resolve() / args.held_out_subject / "visual_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_mapping = asdict(split)
    all_session_names = tuple(
        str(name)
        for partition in ("train", "validation", "test")
        for name in fold[f"{partition}_sessions"]
    )
    region_fingerprint = _region_cache_fingerprint(
        args.region_cache,
        all_session_names,
    )
    session_entries: list[dict[str, object]] = []
    for partition in ("train", "validation", "test"):
        records = load_visual_records(
            root=args.root,
            standardized_index=index_path,
            fold_manifest=fold_path,
            partition=partition,
        )
        records_by_session: dict[str, list[object]] = {}
        for record in records:
            records_by_session.setdefault(record.session, []).append(record)
        for session_name, session_records in records_by_session.items():
            dataset = FiveStateFrameDataset(
                session_records,
                image_size=(args.image_width, args.image_height),
                training=False,
                region_stores={
                    session_name: EvidenceRegionStore(
                        args.region_cache,
                        session_name,
                    )
                },
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
            embedding_batches: list[np.ndarray] = []
            probability_batches: list[np.ndarray] = []
            visibility_batches: list[np.ndarray] = []
            pose_batches: list[np.ndarray] = []
            yawn_batches: list[np.ndarray] = []
            distraction_batches: list[np.ndarray] = []
            eye_phase_batches: list[np.ndarray] = []
            for raw_batch in loader:
                images = raw_batch["image"].to(device, non_blocking=True)
                region_boxes = raw_batch["region_boxes"].to(
                    device,
                    non_blocking=True,
                )
                region_visibility = raw_batch["region_visibility"].to(
                    device,
                    non_blocking=True,
                )
                with torch.autocast(
                    device_type=device.type,
                    enabled=args.amp and device.type == "cuda",
                ):
                    output = model(
                        images,
                        region_boxes,
                        region_visibility,
                    )
                embedding_batches.append(
                    output.embedding.float().cpu().numpy()
                )
                probability_batches.append(
                    output.eye_logits.float().softmax(dim=-1).cpu().numpy()
                )
                visibility_batches.append(
                    region_visibility.float().cpu().numpy()
                )
                pose_batches.append(
                    output.pose.float().mul(90.0).cpu().numpy()
                )
                yawn_batches.append(
                    output.yawn_logits.float()
                    .softmax(dim=-1)[:, 1]
                    .cpu()
                    .numpy()
                )
                distraction_batches.append(
                    output.distraction_logits.float()
                    .softmax(dim=-1)[:, 1]
                    .cpu()
                    .numpy()
                )
                eye_phase_batches.append(
                    raw_batch["eye_phase_target"].numpy()
                )
            embeddings = np.concatenate(embedding_batches, axis=0)
            eye_probabilities = np.concatenate(probability_batches, axis=0)
            region_visibility = np.concatenate(
                visibility_batches,
                axis=0,
            )
            head_pose = np.concatenate(pose_batches, axis=0)
            yawn_probabilities = np.concatenate(yawn_batches, axis=0)
            distraction_probabilities = np.concatenate(
                distraction_batches,
                axis=0,
            )
            eye_phase_targets = np.concatenate(
                eye_phase_batches,
                axis=0,
            )
            primitive = tuple(
                PrimitiveFrameEvidence(
                    closed_probability=float(eye_probabilities[row, 2]),
                    eye_visibility=float(
                        min(
                            region_visibility[row, 1],
                            region_visibility[row, 2],
                        )
                    ),
                    head_pose=(
                        float(head_pose[row, 0]),
                        float(head_pose[row, 1]),
                    ),
                    yawn_probability=float(yawn_probabilities[row]),
                    mouth_visibility=float(region_visibility[row, 3]),
                    distraction_probability=float(
                        distraction_probabilities[row]
                    ),
                    face_visibility=float(region_visibility[row, 0]),
                )
                for row in range(len(session_records))
            )
            evidence = build_causal_evidence(
                primitive,
                fps=float(index["target_fps"]),
            )
            cache = VisualSessionCache(
                session=session_name,
                subject=session_records[0].subject,
                protocol=session_records[0].protocol,
                fps=float(index["target_fps"]),
                frame_ids=np.asarray(
                    [record.frame_id for record in session_records],
                    dtype=np.int32,
                ),
                visual_embedding=embeddings,
                evidence=evidence,
                region_visibility=region_visibility,
                raw_eye_probabilities=eye_probabilities,
                eye_phase_targets=eye_phase_targets,
                head_pose=head_pose,
                yawn_probability=yawn_probabilities,
                distraction_probability=distraction_probabilities,
                visual_checkpoint_fingerprint=visual_fingerprint,
                region_cache_fingerprint=region_fingerprint,
                evidence_feature_names=EVIDENCE_FEATURE_NAMES,
                split=split_mapping,
            )
            save_visual_session_cache(cache_dir, cache)
            session_entries.append(
                {
                    "session": session_name,
                    "subject": cache.subject,
                    "partition": partition,
                    "frames": len(cache.frame_ids),
                }
            )
    _atomic_json(
        cache_dir / "index.json",
        {
            "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "target_fps": float(index["target_fps"]),
            "embedding_dim": args.embedding_dim,
            "region_dim": args.region_dim,
            "visual_embedding_dim": (
                args.embedding_dim + 4 * args.region_dim
            ),
            "evidence_config": dict(EYE_EVIDENCE_CONFIG),
            "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
            "visual_checkpoint": str(checkpoint),
            "visual_checkpoint_fingerprint": visual_fingerprint,
            "visual_checkpoint_label_contract": {
                "mode": label_contract_mode,
                "source_schema_version": int(
                    payload["standardized_schema_version"]
                ),
                "source_label_config": dict(payload["label_config"]),
                "target_schema_version": int(index["schema_version"]),
                "target_label_config": dict(index["config"]),
            },
            "region_cache_fingerprint": region_fingerprint,
            "split": split_mapping,
            "sessions": session_entries,
        },
    )
    return 0


def _standardized_sessions(
    index_path: Path,
    index: Mapping[str, object],
    session_names: Sequence[str],
):
    entries = index["sessions"]
    assert isinstance(entries, list)
    by_name = {str(entry["session"]): entry for entry in entries}
    sessions = []
    for name in session_names:
        entry = by_name.get(str(name))
        if entry is None:
            raise ValueError(f"standardized session is missing: {name}")
        sessions.append(
            load_standardized_session(
                index_path.parent / str(entry["csv"]),
                fps=float(index["target_fps"]),
                expected_session=str(name),
            )
        )
    return tuple(sessions)


def _visual_cache_context(args: argparse.Namespace):
    """Load one nested-LOSO fold and its immutable visual cache partition."""

    index_path = args.standardized_index.resolve()
    index = load_standardized_index(index_path)
    fold_path = _fold_path(index_path, args.held_out_subject)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    split = _split(fold)
    _validate_split(split, args)
    cache_dir = args.output.resolve() / args.held_out_subject / "visual_cache"
    cache_index_path = cache_dir / "index.json"
    if not cache_index_path.is_file():
        raise ValueError(f"visual cache index is missing: {cache_index_path}")
    cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
    json_split = {
        "train_subjects": list(split.train_subjects),
        "validation_subject": split.validation_subject,
        "test_subject": split.test_subject,
    }
    compatibility = {
        "schema_version": (
            int(cache_index.get("schema_version", 0)),
            VISUAL_CACHE_SCHEMA_VERSION,
        ),
        "target_fps": (
            float(cache_index.get("target_fps", 0.0)),
            float(index["target_fps"]),
        ),
        "embedding_dim": (
            int(cache_index.get("embedding_dim", 0)),
            args.embedding_dim,
        ),
        "region_dim": (
            int(cache_index.get("region_dim", 0)),
            args.region_dim,
        ),
        "visual_embedding_dim": (
            int(cache_index.get("visual_embedding_dim", 0)),
            args.embedding_dim + 4 * args.region_dim,
        ),
        "split": (cache_index.get("split"), json_split),
        "evidence_config": (
            cache_index.get("evidence_config"),
            dict(EYE_EVIDENCE_CONFIG),
        ),
        "evidence_feature_names": (
            cache_index.get("evidence_feature_names"),
            list(EVIDENCE_FEATURE_NAMES),
        ),
    }
    mismatched = [
        name
        for name, (actual, expected) in compatibility.items()
        if actual != expected
    ]
    if mismatched:
        raise ValueError(f"visual cache mismatch: {', '.join(mismatched)}")
    visual_fingerprint = str(cache_index["visual_checkpoint_fingerprint"])
    region_fingerprint = str(cache_index["region_cache_fingerprint"])
    partitions = {}
    for partition in ("train", "validation", "test"):
        names = tuple(str(value) for value in fold[f"{partition}_sessions"])
        sessions = _standardized_sessions(index_path, index, names)
        caches = tuple(
            load_visual_session_cache(
                cache_dir,
                name,
                expected_visual_fingerprint=visual_fingerprint,
                expected_region_fingerprint=region_fingerprint,
                expected_fps=float(index["target_fps"]),
                expected_split=json_split,
            )
            for name in names
        )
        partitions[partition] = (sessions, caches)
    return (
        index_path,
        index,
        fold,
        split,
        cache_index,
        partitions,
    )


def _ocular_loader(
    caches,
    references,
    *,
    length: int,
    embedding_dim: int,
    region_dim: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        OcularWindowDataset(
            caches,
            references,
            length=length,
            embedding_dim=embedding_dim,
            region_dim=region_dim,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _ocular_selection_score(metrics: Mapping[str, object]) -> float:
    closed = metrics.get("closed_binary")
    if not isinstance(closed, Mapping):
        raise ValueError("ocular metrics are missing closed-binary results")
    return (
        float(metrics["macro_f1"])
        + float(closed["macro_f1"])
        - float(metrics["progress_mae"])
    )


def _train_ocular(args: argparse.Namespace) -> int:
    _, index, _, split, cache_index, partitions = _visual_cache_context(args)
    visual_fingerprint = str(cache_index["visual_checkpoint_fingerprint"])
    train_caches = tuple(
        cache for cache in partitions["train"][1] if cache.protocol == "s5"
    )
    validation_caches = tuple(
        cache
        for cache in partitions["validation"][1]
        if cache.protocol == "s5"
    )
    if not train_caches or not validation_caches:
        raise ValueError("ocular training requires DMD s5 train and validation caches")
    validation_references = build_chronological_ocular_references(
        validation_caches
    )
    validation_loader = _ocular_loader(
        validation_caches,
        validation_references,
        length=args.ocular_sequence_length,
        embedding_dim=args.embedding_dim,
        region_dim=args.region_dim,
        batch_size=args.ocular_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    model_config = {
        "input_dim": 2 * args.region_dim + 5,
        "projection_dim": args.ocular_channels,
        "hidden_size": args.ocular_channels,
        "layers": 1,
        "dropout": 0.1,
    }
    loss_weights = OcularLossWeights()
    input_augmentation = OcularInputAugmentation(
        probability=args.ocular_input_corruption_probability,
        minimum_closed_run_frames=(
            args.ocular_input_corruption_min_run_length
        ),
        anchor_closed_frames=args.ocular_input_corruption_anchor_frames,
        maximum_block_frames=(
            args.ocular_input_corruption_max_block_length
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalOcularLSTM(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.ocular_learning_rate,
        weight_decay=1e-4,
    )
    ocular_dir = args.output.resolve() / args.held_out_subject / "ocular"
    checkpoint = ocular_dir / "best.pt"
    start_epoch = 1
    best_score = -math.inf
    if args.resume:
        if not checkpoint.is_file():
            raise ValueError(f"ocular resume checkpoint is missing: {checkpoint}")
        start_epoch, payload = restore_ocular_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_visual_checkpoint_fingerprint=visual_fingerprint,
            expected_split=split,
            expected_target_fps=float(index["target_fps"]),
            expected_microsleep_seconds=args.microsleep_seconds,
            expected_sequence_length=args.ocular_sequence_length,
            expected_loss_weights=loss_weights,
            expected_input_augmentation=input_augmentation,
            map_location=device,
        )
        best_score = float(
            payload["metrics"].get(
                "selection_score",
                _ocular_selection_score(payload["metrics"]),
            )
        )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    stale_epochs = 0
    for epoch in range(start_epoch, args.ocular_epochs + 1):
        epoch_seed = args.seed + epoch - 1
        references, audit = build_ocular_epoch_references(
            train_caches,
            samples=args.ocular_samples_per_epoch,
            max_windows_per_event=args.ocular_max_windows_per_event,
            seed=epoch_seed,
        )
        if not references:
            raise ValueError("ocular sampler produced no training windows")
        train_loader = _ocular_loader(
            train_caches,
            references,
            length=args.ocular_sequence_length,
            embedding_dim=args.embedding_dim,
            region_dim=args.region_dim,
            batch_size=args.ocular_batch_size,
            workers=args.workers,
            shuffle=True,
            seed=epoch_seed,
        )
        train_metrics = train_ocular_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_weights=loss_weights,
            input_augmentation=input_augmentation,
            input_region_dim=args.region_dim,
            augmentation_seed=epoch_seed,
            amp=args.amp,
            scaler=scaler,
        )
        validation_metrics = evaluate_ocular(
            model,
            validation_loader,
            device=device,
            loss_weights=loss_weights,
            amp=args.amp,
        )
        score = _ocular_selection_score(validation_metrics)
        validation_metrics["selection_score"] = score
        sampling_audit = {
            "requested": audit.requested,
            "actual": audit.actual,
            "bucket_counts": dict(audit.bucket_counts),
            "subject_counts": dict(audit.subject_counts),
            "windows_per_event": dict(audit.windows_per_event),
            "epoch_seed": epoch_seed,
        }
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": {
                        key: value
                        for key, value in validation_metrics.items()
                        if key != "rows"
                    },
                }
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            checkpoint_metrics = {
                key: value
                for key, value in validation_metrics.items()
                if key != "rows"
            }
            checkpoint_metrics["sampling_audit"] = sampling_audit
            save_ocular_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                visual_checkpoint_fingerprint=visual_fingerprint,
                split=split,
                target_fps=float(index["target_fps"]),
                microsleep_seconds=args.microsleep_seconds,
                sequence_length=args.ocular_sequence_length,
                loss_weights=loss_weights,
                epoch=epoch,
                metrics=checkpoint_metrics,
                input_augmentation=input_augmentation,
            )
            _atomic_json(
                ocular_dir / "metrics.json",
                {
                    **checkpoint_metrics,
                    "split": asdict(split),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("ocular training did not produce a checkpoint")
    return 0


@torch.inference_mode()
def _cache_ocular(args: argparse.Namespace) -> int:
    _, index, fold, split, cache_index, partitions = _visual_cache_context(args)
    checkpoint = (
        args.output.resolve() / args.held_out_subject / "ocular" / "best.pt"
    )
    if not checkpoint.is_file():
        raise ValueError(f"ocular checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", 0)) != OCULAR_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("ocular checkpoint schema mismatch")
    model_config = dict(payload["model_config"])
    expected = {
        "visual checkpoint fingerprint": (
            payload.get("visual_checkpoint_fingerprint"),
            cache_index["visual_checkpoint_fingerprint"],
        ),
        "split": (payload.get("split"), asdict(split)),
        "target FPS": (
            float(payload.get("target_fps", 0.0)),
            float(index["target_fps"]),
        ),
        "microsleep seconds": (
            float(payload.get("microsleep_seconds", 0.0)),
            float(args.microsleep_seconds),
        ),
        "sequence length": (
            int(payload.get("sequence_length", 0)),
            int(args.ocular_sequence_length),
        ),
    }
    mismatched = [
        name for name, values in expected.items() if values[0] != values[1]
    ]
    if mismatched:
        raise ValueError(f"ocular cache producer mismatch: {', '.join(mismatched)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalOcularLSTM(**model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    ocular_fingerprint = checkpoint_fingerprint(checkpoint)
    visual_fingerprint = str(cache_index["visual_checkpoint_fingerprint"])
    region_fingerprint = str(cache_index["region_cache_fingerprint"])
    split_mapping = dict(cache_index["split"])
    evidence_config = _ocular_evidence_config(args)
    ocular_dir = args.output.resolve() / args.held_out_subject / "ocular_cache"
    entries: list[dict[str, object]] = []
    expected_sessions = []
    for partition in ("train", "validation", "test"):
        sessions, visual_caches = partitions[partition]
        for session, visual in zip(sessions, visual_caches, strict=True):
            expected_sessions.append(session.session)
            cache = build_ocular_session_cache(
                visual,
                model,
                device=device,
                embedding_dim=args.embedding_dim,
                region_dim=args.region_dim,
                ocular_checkpoint_fingerprint=ocular_fingerprint,
                split=split_mapping,
                evidence_config=evidence_config,
                amp=args.amp,
            )
            save_ocular_session_cache(ocular_dir, cache)
            entries.append(
                {
                    "session": cache.session,
                    "subject": cache.subject,
                    "partition": partition,
                    "frames": len(cache.frame_ids),
                }
            )
    saved_sessions = {path.name.removesuffix(".ocular.npz") for path in ocular_dir.glob("*.ocular.npz")}
    if saved_sessions != set(expected_sessions):
        raise RuntimeError("ocular cache session set is partial or contains stale files")
    _atomic_json(
        ocular_dir / "index.json",
        {
            "schema_version": OCULAR_CACHE_SCHEMA_VERSION,
            "target_fps": float(index["target_fps"]),
            "embedding_dim": args.embedding_dim,
            "region_dim": args.region_dim,
            "visual_checkpoint_fingerprint": visual_fingerprint,
            "region_cache_fingerprint": region_fingerprint,
            "ocular_checkpoint": str(checkpoint),
            "ocular_checkpoint_fingerprint": ocular_fingerprint,
            "ocular_checkpoint_schema_version": OCULAR_CHECKPOINT_SCHEMA_VERSION,
            "evidence_config": evidence_config,
            "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
            "split": split_mapping,
            "sessions": entries,
        },
    )
    return 0


def _temporal_loader(
    caches,
    sessions,
    references,
    *,
    length: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        VisualWindowDataset(
            caches,
            sessions,
            references,
            length=length,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _four_state_loader(
    visual_caches,
    ocular_caches,
    sessions,
    references,
    *,
    gate_ocular_caches=None,
    length: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        FourStateWindowDataset(
            visual_caches,
            ocular_caches,
            sessions,
            references,
            length=length,
            gate_ocular_caches=gate_ocular_caches,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _save_temporal_checkpoint(
    path: Path,
    *,
    model: CausalFiveStateLSTM,
    optimizer: torch.optim.Optimizer,
    model_config: Mapping[str, object],
    split: SubjectSplitMetadata,
    visual_fingerprint: str,
    region_fingerprint: str,
    standardized_index: Mapping[str, object],
    sequence_length: int,
    temporal_loss_config: Mapping[str, float],
    epoch: int,
    metrics: Mapping[str, object],
    sampling_audit: Mapping[str, object],
) -> None:
    payload = {
        "schema_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": dict(model_config),
        "split": asdict(split),
        "visual_checkpoint_fingerprint": visual_fingerprint,
        "region_cache_fingerprint": region_fingerprint,
        "visual_cache_schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "evidence_config": dict(EYE_EVIDENCE_CONFIG),
        "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
        "standardized_schema_version": int(standardized_index["schema_version"]),
        "target_fps": float(standardized_index["target_fps"]),
        "label_config": dict(standardized_index["config"]),
        "sequence_length": sequence_length,
        "temporal_loss_config": dict(temporal_loss_config),
        "epoch": epoch,
        "metrics": dict(metrics),
        "sampling_audit": dict(sampling_audit),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_four_state_checkpoint(
    path: Path,
    *,
    model: CausalFourStateLSTM,
    optimizer: torch.optim.Optimizer,
    model_config: Mapping[str, object],
    split: SubjectSplitMetadata,
    visual_fingerprint: str,
    region_fingerprint: str,
    ocular_fingerprint: str,
    gate_ocular_fingerprint: str | None,
    gate_ocular_cache_schema_version: int | None,
    standardized_index: Mapping[str, object],
    sequence_length: int,
    temporal_loss_config: Mapping[str, float],
    evidence_config: Mapping[str, float],
    epoch: int,
    metrics: Mapping[str, object],
    sampling_audit: Mapping[str, object],
) -> None:
    dual_ocular = gate_ocular_fingerprint is not None
    payload = {
        "schema_version": 6,
        "model_type": "causal_four_state_lstm",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": dict(model_config),
        "split": asdict(split),
        "visual_checkpoint_fingerprint": visual_fingerprint,
        "region_cache_fingerprint": region_fingerprint,
        "visual_cache_schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "ocular_checkpoint_fingerprint": ocular_fingerprint,
        "microsleep_gate_ocular_checkpoint_fingerprint": (
            gate_ocular_fingerprint
        ),
        "ocular_cache_schema_version": OCULAR_CACHE_SCHEMA_VERSION,
        "microsleep_gate_ocular_cache_schema_version": (
            gate_ocular_cache_schema_version
        ),
        "evidence_config": dict(evidence_config),
        "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
        "ocular_phase_feature_names": list(PHASE_NAMES),
        "standardized_schema_version": int(standardized_index["schema_version"]),
        "target_fps": float(standardized_index["target_fps"]),
        "label_config": dict(standardized_index["config"]),
        "sequence_length": int(sequence_length),
        "temporal_loss_config": dict(temporal_loss_config),
        "microsleep_fusion": (
            "dual_ocular_drowsy_and_microsleep_gates"
            if dual_ocular
            else "drowsy_and_microsleep_gates"
        ),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "sampling_audit": dict(sampling_audit),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_legacy_temporal(args: argparse.Namespace) -> int:
    if args.temporal_final_loss_weight < 0.0:
        raise ValueError("temporal final loss weight must be non-negative")
    temporal_loss_config = {
        "history_weight": 1.0,
        "final_weight": float(args.temporal_final_loss_weight),
    }
    index_path = args.standardized_index.resolve()
    index = load_standardized_index(index_path)
    fold_path = _fold_path(index_path, args.held_out_subject)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    split = _split(fold)
    _validate_split(split, args)
    cache_dir = (
        args.output.resolve() / args.held_out_subject / "visual_cache"
    )
    cache_index_path = cache_dir / "index.json"
    if not cache_index_path.is_file():
        raise ValueError(f"visual cache index is missing: {cache_index_path}")
    cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
    json_split = {
        "train_subjects": list(split.train_subjects),
        "validation_subject": split.validation_subject,
        "test_subject": split.test_subject,
    }
    compatibility = {
        "schema_version": (
            int(cache_index.get("schema_version", 0)),
            VISUAL_CACHE_SCHEMA_VERSION,
        ),
        "target_fps": (
            float(cache_index.get("target_fps", 0.0)),
            float(index["target_fps"]),
        ),
        "embedding_dim": (
            int(cache_index.get("embedding_dim", 0)),
            args.embedding_dim,
        ),
        "region_dim": (
            int(cache_index.get("region_dim", 0)),
            args.region_dim,
        ),
        "visual_embedding_dim": (
            int(cache_index.get("visual_embedding_dim", 0)),
            args.embedding_dim + 4 * args.region_dim,
        ),
        "split": (cache_index.get("split"), json_split),
        "evidence_config": (
            cache_index.get("evidence_config"),
            dict(EYE_EVIDENCE_CONFIG),
        ),
        "evidence_feature_names": (
            cache_index.get("evidence_feature_names"),
            list(EVIDENCE_FEATURE_NAMES),
        ),
    }
    mismatched = [
        name for name, (actual, expected) in compatibility.items() if actual != expected
    ]
    if mismatched:
        raise ValueError(f"temporal cache mismatch: {', '.join(mismatched)}")
    visual_fingerprint = str(cache_index["visual_checkpoint_fingerprint"])
    region_fingerprint = str(cache_index["region_cache_fingerprint"])
    cache_split = cache_index["split"]

    partitions = {}
    for partition in ("train", "validation", "test"):
        names = tuple(str(value) for value in fold[f"{partition}_sessions"])
        sessions = _standardized_sessions(index_path, index, names)
        caches = tuple(
            load_visual_session_cache(
                cache_dir,
                name,
                expected_visual_fingerprint=visual_fingerprint,
                expected_region_fingerprint=region_fingerprint,
                expected_fps=float(index["target_fps"]),
                expected_split=cache_split,
            )
            for name in names
        )
        partitions[partition] = (sessions, caches)

    train_sessions, train_caches = partitions["train"]
    validation_sessions, validation_caches = partitions["validation"]
    test_sessions, test_caches = partitions["test"]
    validation_references = build_chronological_references(validation_sessions)
    test_references = build_chronological_references(test_sessions)
    validation_loader = _temporal_loader(
        validation_caches,
        validation_sessions,
        validation_references,
        length=args.sequence_length,
        batch_size=args.temporal_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = _temporal_loader(
        test_caches,
        test_sessions,
        test_references,
        length=args.sequence_length,
        batch_size=args.temporal_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    input_dim = (
        args.embedding_dim
        + 4 * args.region_dim
        + len(EVIDENCE_FEATURE_NAMES)
        + 1
    )
    model_config = {
        "input_dim": input_dim,
        "projection_dim": args.temporal_channels,
        "hidden_size": args.temporal_channels,
        "layers": 2,
        "dropout": 0.2,
        "input_dropout": 0.25,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalFiveStateLSTM(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.temporal_learning_rate,
        weight_decay=1e-4,
    )
    temporal_dir = (
        args.output.resolve() / args.held_out_subject / "temporal"
    )
    checkpoint = temporal_dir / "best.pt"
    start_epoch = 1
    best_score = -1.0
    if args.resume:
        if not checkpoint.is_file():
            raise ValueError(f"temporal resume checkpoint is missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        resume_expected = {
            "model_config": model_config,
            "split": asdict(split),
            "visual_checkpoint_fingerprint": visual_fingerprint,
            "region_cache_fingerprint": region_fingerprint,
            "visual_cache_schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "evidence_config": dict(EYE_EVIDENCE_CONFIG),
            "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
            "standardized_schema_version": int(index["schema_version"]),
            "target_fps": float(index["target_fps"]),
            "label_config": dict(index["config"]),
            "sequence_length": args.sequence_length,
            "temporal_loss_config": temporal_loss_config,
        }
        resume_mismatch = [
            name
            for name, expected in resume_expected.items()
            if payload.get(name) != expected
        ]
        if resume_mismatch:
            raise ValueError(
                f"temporal resume mismatch: {', '.join(resume_mismatch)}"
            )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(
            payload["metrics"].get(
                "selection_score",
                temporal_selection_score(payload["metrics"]),
            )
        )

    stale_epochs = 0
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    event_counts = torch.ones(5, dtype=torch.float32)
    extracted = extract_state_events(train_sessions)
    for state in range(5):
        count = sum(event.target == state for event in extracted)
        event_counts[state] = max(count, 1)
    class_weights = effective_number_weights(event_counts)
    for epoch in range(start_epoch, args.temporal_epochs + 1):
        epoch_seed = args.seed + epoch - 1
        train_references, audit = build_subject_event_epoch_references(
            train_sessions,
            samples=args.samples_per_epoch,
            max_windows_per_event=args.max_windows_per_event,
            seed=epoch_seed,
        )
        train_loader = _temporal_loader(
            train_caches,
            train_sessions,
            train_references,
            length=args.sequence_length,
            batch_size=args.temporal_batch_size,
            workers=args.workers,
            shuffle=True,
            seed=epoch_seed,
        )
        sampling_audit = {
            "requested": audit.requested,
            "actual": audit.actual,
            "counts": {
                state.name.lower(): count
                for state, count in audit.counts.items()
            },
            "missing_states": [
                state.name.lower() for state in audit.missing_states
            ],
            "windows_per_event": dict(audit.windows_per_event),
            "subject_counts": dict(audit.subject_counts),
            "epoch_seed": epoch_seed,
        }
        train_metrics = train_temporal_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            amp=args.amp,
            class_weights=class_weights,
            final_loss_weight=args.temporal_final_loss_weight,
        )
        validation_metrics = evaluate_temporal(
            model,
            validation_loader,
            device=device,
            amp=args.amp,
        )
        score = temporal_selection_score(validation_metrics)
        validation_metrics["selection_score"] = score
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": {
                        key: value
                        for key, value in validation_metrics.items()
                        if key != "rows"
                    },
                }
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            _save_temporal_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                split=split,
                visual_fingerprint=visual_fingerprint,
                region_fingerprint=region_fingerprint,
                standardized_index=index,
                sequence_length=args.sequence_length,
                temporal_loss_config=temporal_loss_config,
                epoch=epoch,
                metrics={
                    key: value
                    for key, value in validation_metrics.items()
                    if key != "rows"
                },
                sampling_audit=sampling_audit,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("temporal training did not produce a checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_metrics = evaluate_temporal(
        model,
        test_loader,
        device=device,
        amp=args.amp,
    )
    rows = test_metrics.pop("rows")
    test_metrics["events"] = five_state_event_metrics(
        targets=[int(row["target"]) for row in rows],
        predictions=[int(row["prediction"]) for row in rows],
        event_ids=[str(row["event_id"]) for row in rows],
        session_ids=[str(row["session"]) for row in rows],
        frame_ids=[int(row["frame_id"]) for row in rows],
        fps=float(index["target_fps"]),
    )
    _atomic_json(temporal_dir / "outer_metrics.json", test_metrics)
    predictions = temporal_dir / "outer_predictions.jsonl"
    temporary_predictions = predictions.with_suffix(".jsonl.tmp")
    with temporary_predictions.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary_predictions.replace(predictions)
    return 0


def _train_temporal(args: argparse.Namespace) -> int:
    """Train the learned non-event head and fuse the cached microsleep gate."""

    if args.temporal_final_loss_weight < 0.0:
        raise ValueError("temporal final loss weight must be non-negative")
    (
        _,
        index,
        fold,
        split,
        visual_index,
        visual_partitions,
    ) = _visual_cache_context(args)
    ocular_dir = args.output.resolve() / args.held_out_subject / "ocular_cache"
    ocular_index_path = ocular_dir / "index.json"
    if not ocular_index_path.is_file():
        raise ValueError(f"ocular cache index is missing: {ocular_index_path}")
    ocular_index = json.loads(ocular_index_path.read_text(encoding="utf-8"))
    visual_fingerprint = str(visual_index["visual_checkpoint_fingerprint"])
    region_fingerprint = str(visual_index["region_cache_fingerprint"])
    ocular_fingerprint = str(
        ocular_index.get("ocular_checkpoint_fingerprint", "")
    )
    evidence_config = _ocular_evidence_config(args)
    expected_sessions = {
        str(value)
        for partition in ("train", "validation", "test")
        for value in fold[f"{partition}_sessions"]
    }
    indexed_sessions = {
        str(entry["session"])
        for entry in ocular_index.get("sessions", [])
        if isinstance(entry, Mapping) and "session" in entry
    }
    compatibility = {
        "schema_version": (
            int(ocular_index.get("schema_version", 0)),
            OCULAR_CACHE_SCHEMA_VERSION,
        ),
        "target_fps": (
            float(ocular_index.get("target_fps", 0.0)),
            float(index["target_fps"]),
        ),
        "embedding_dim": (
            int(ocular_index.get("embedding_dim", 0)),
            args.embedding_dim,
        ),
        "region_dim": (
            int(ocular_index.get("region_dim", 0)),
            args.region_dim,
        ),
        "visual_checkpoint_fingerprint": (
            ocular_index.get("visual_checkpoint_fingerprint"),
            visual_fingerprint,
        ),
        "region_cache_fingerprint": (
            ocular_index.get("region_cache_fingerprint"),
            region_fingerprint,
        ),
        "split": (
            ocular_index.get("split"),
            visual_index.get("split"),
        ),
        "evidence_config": (
            ocular_index.get("evidence_config"),
            evidence_config,
        ),
        "evidence_feature_names": (
            ocular_index.get("evidence_feature_names"),
            list(EVIDENCE_FEATURE_NAMES),
        ),
        "sessions": (indexed_sessions, expected_sessions),
    }
    mismatched = [
        name
        for name, (actual, expected) in compatibility.items()
        if actual != expected
    ]
    if not ocular_fingerprint:
        mismatched.append("ocular_checkpoint_fingerprint")
    if mismatched:
        raise ValueError(f"temporal ocular cache mismatch: {', '.join(mismatched)}")

    gate_ocular_dir = args.microsleep_gate_ocular_cache
    gate_checkpoint = args.microsleep_gate_ocular_checkpoint
    gate_ocular_index: Mapping[str, object] | None = None
    gate_ocular_fingerprint: str | None = None
    gate_ocular_cache_schema_version: int | None = None
    if gate_ocular_dir is not None and gate_checkpoint is not None:
        gate_ocular_dir = gate_ocular_dir.resolve()
        gate_checkpoint = gate_checkpoint.resolve()
        gate_index_path = gate_ocular_dir / "index.json"
        if not gate_index_path.is_file():
            raise ValueError(
                f"microsleep-gate ocular cache index is missing: {gate_index_path}"
            )
        if not gate_checkpoint.is_file():
            raise ValueError(
                "microsleep-gate ocular checkpoint is missing: "
                f"{gate_checkpoint}"
            )
        gate_ocular_index = json.loads(
            gate_index_path.read_text(encoding="utf-8")
        )
        gate_ocular_cache_schema_version = int(
            gate_ocular_index.get("schema_version", 0)
        )
        gate_payload = torch.load(
            gate_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        gate_ocular_fingerprint = checkpoint_fingerprint(gate_checkpoint)
        gate_indexed_sessions = {
            str(entry["session"])
            for entry in gate_ocular_index.get("sessions", [])
            if isinstance(entry, Mapping) and "session" in entry
        }
        gate_compatibility = {
            "checkpoint_schema_version": (
                int(gate_payload.get("schema_version", 0)),
                OCULAR_CHECKPOINT_SCHEMA_VERSION,
            ),
            "cache_schema_version": (
                gate_ocular_cache_schema_version,
                (
                    gate_ocular_cache_schema_version
                    if gate_ocular_cache_schema_version
                    in (
                        LEGACY_OCULAR_CACHE_SCHEMA_VERSION,
                        OCULAR_CACHE_SCHEMA_VERSION,
                    )
                    else -1
                ),
            ),
            "target_fps": (
                float(gate_ocular_index.get("target_fps", 0.0)),
                float(index["target_fps"]),
            ),
            "embedding_dim": (
                int(gate_ocular_index.get("embedding_dim", 0)),
                args.embedding_dim,
            ),
            "region_dim": (
                int(gate_ocular_index.get("region_dim", 0)),
                args.region_dim,
            ),
            "visual_checkpoint_fingerprint": (
                gate_ocular_index.get("visual_checkpoint_fingerprint"),
                visual_fingerprint,
            ),
            "region_cache_fingerprint": (
                gate_ocular_index.get("region_cache_fingerprint"),
                region_fingerprint,
            ),
            "split": (
                gate_ocular_index.get("split"),
                visual_index.get("split"),
            ),
            "evidence_config": (
                gate_ocular_index.get("evidence_config"),
                evidence_config,
            ),
            "evidence_feature_names": (
                gate_ocular_index.get("evidence_feature_names"),
                list(EVIDENCE_FEATURE_NAMES),
            ),
            "sessions": (gate_indexed_sessions, expected_sessions),
            "ocular_checkpoint_fingerprint": (
                gate_ocular_index.get("ocular_checkpoint_fingerprint"),
                gate_ocular_fingerprint,
            ),
            "checkpoint_visual_fingerprint": (
                gate_payload.get("visual_checkpoint_fingerprint"),
                visual_fingerprint,
            ),
            "checkpoint_split": (
                gate_payload.get("split"),
                asdict(split),
            ),
            "checkpoint_target_fps": (
                float(gate_payload.get("target_fps", 0.0)),
                float(index["target_fps"]),
            ),
            "checkpoint_microsleep_seconds": (
                float(gate_payload.get("microsleep_seconds", 0.0)),
                float(args.microsleep_seconds),
            ),
        }
        gate_mismatched = [
            name
            for name, (actual, expected) in gate_compatibility.items()
            if actual != expected
        ]
        if gate_mismatched:
            raise ValueError(
                "microsleep-gate ocular artifacts mismatch: "
                + ", ".join(gate_mismatched)
            )

    partitions = {}
    for partition in ("train", "validation", "test"):
        sessions, visual_caches = visual_partitions[partition]
        ocular_caches = tuple(
            load_ocular_session_cache(
                ocular_dir,
                session.session,
                expected_visual_fingerprint=visual_fingerprint,
                expected_region_fingerprint=region_fingerprint,
                expected_ocular_fingerprint=ocular_fingerprint,
                expected_fps=float(index["target_fps"]),
                expected_split=visual_index["split"],
                expected_evidence_config=evidence_config,
            )
            for session in sessions
        )
        gate_ocular_caches = None
        if gate_ocular_index is not None:
            assert gate_ocular_dir is not None
            assert gate_ocular_fingerprint is not None
            gate_ocular_caches = tuple(
                load_ocular_session_cache(
                    gate_ocular_dir,
                    session.session,
                    expected_visual_fingerprint=visual_fingerprint,
                    expected_region_fingerprint=region_fingerprint,
                    expected_ocular_fingerprint=gate_ocular_fingerprint,
                    expected_fps=float(index["target_fps"]),
                    expected_split=visual_index["split"],
                    expected_evidence_config=evidence_config,
                    expected_schema_version=(
                        gate_ocular_cache_schema_version
                    ),
                )
                for session in sessions
            )
        partitions[partition] = (
            sessions,
            visual_caches,
            ocular_caches,
            gate_ocular_caches,
        )

    train_sessions, train_visual, train_ocular, train_gate_ocular = partitions[
        "train"
    ]
    (
        validation_sessions,
        validation_visual,
        validation_ocular,
        validation_gate_ocular,
    ) = partitions["validation"]
    test_sessions, test_visual, test_ocular, test_gate_ocular = partitions[
        "test"
    ]
    validation_references = build_chronological_references(validation_sessions)
    test_references = build_chronological_references(test_sessions)
    validation_loader = _four_state_loader(
        validation_visual,
        validation_ocular,
        validation_sessions,
        validation_references,
        gate_ocular_caches=validation_gate_ocular,
        length=args.sequence_length,
        batch_size=args.temporal_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = _four_state_loader(
        test_visual,
        test_ocular,
        test_sessions,
        test_references,
        gate_ocular_caches=test_gate_ocular,
        length=args.sequence_length,
        batch_size=args.temporal_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    input_dim = (
        args.embedding_dim
        + 4 * args.region_dim
        + len(PHASE_NAMES)
        + len(EVIDENCE_FEATURE_NAMES)
        + 1
    )
    model_config = {
        "input_dim": input_dim,
        "projection_dim": args.temporal_channels,
        "hidden_size": args.temporal_channels,
        "layers": 2,
        "dropout": 0.2,
        "input_dropout": 0.25,
    }
    temporal_loss_config = {
        "history_weight": 1.0,
        "final_weight": float(args.temporal_final_loss_weight),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalFourStateLSTM(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.temporal_learning_rate,
        weight_decay=1e-4,
    )
    temporal_dir = args.output.resolve() / args.held_out_subject / "temporal"
    checkpoint = temporal_dir / "best.pt"
    start_epoch = 1
    best_score = -math.inf
    if args.resume and args.initial_temporal_checkpoint is not None:
        raise ValueError(
            "temporal resume and initial checkpoint are mutually exclusive"
        )
    if args.resume:
        if not checkpoint.is_file():
            raise ValueError(f"temporal resume checkpoint is missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        dual_ocular = gate_ocular_fingerprint is not None
        resume_expected = {
            "schema_version": 6,
            "model_type": "causal_four_state_lstm",
            "model_config": model_config,
            "split": asdict(split),
            "visual_checkpoint_fingerprint": visual_fingerprint,
            "region_cache_fingerprint": region_fingerprint,
            "visual_cache_schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "ocular_checkpoint_fingerprint": ocular_fingerprint,
            "microsleep_gate_ocular_checkpoint_fingerprint": (
                gate_ocular_fingerprint
            ),
            "ocular_cache_schema_version": OCULAR_CACHE_SCHEMA_VERSION,
            "microsleep_gate_ocular_cache_schema_version": (
                gate_ocular_cache_schema_version
            ),
            "evidence_config": evidence_config,
            "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
            "ocular_phase_feature_names": list(PHASE_NAMES),
            "standardized_schema_version": int(index["schema_version"]),
            "target_fps": float(index["target_fps"]),
            "label_config": dict(index["config"]),
            "sequence_length": args.sequence_length,
            "temporal_loss_config": temporal_loss_config,
            "microsleep_fusion": (
                "dual_ocular_drowsy_and_microsleep_gates"
                if dual_ocular
                else "drowsy_and_microsleep_gates"
            ),
        }
        resume_mismatch = [
            name
            for name, expected in resume_expected.items()
            if payload.get(name) != expected
        ]
        if resume_mismatch:
            raise ValueError(
                f"temporal resume mismatch: {', '.join(resume_mismatch)}"
            )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(
            payload["metrics"].get(
                "selection_score",
                fused_temporal_selection_score(payload["metrics"]),
            )
        )
    elif args.initial_temporal_checkpoint is not None:
        initial_path = args.initial_temporal_checkpoint.resolve()
        if not initial_path.is_file():
            raise ValueError(
                f"initial temporal checkpoint is missing: {initial_path}"
            )
        payload = torch.load(
            initial_path,
            map_location=device,
            weights_only=False,
        )
        dual_ocular = gate_ocular_fingerprint is not None
        source_model_config = dict(model_config)
        source_model_config["input_dim"] = int(model_config["input_dim"]) - len(
            PHASE_NAMES
        )
        initial_expected = {
            "schema_version": 4 if dual_ocular else 3,
            "model_type": "causal_four_state_lstm",
            "model_config": source_model_config,
            "split": asdict(split),
            "visual_checkpoint_fingerprint": visual_fingerprint,
            "region_cache_fingerprint": region_fingerprint,
            "visual_cache_schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "ocular_checkpoint_fingerprint": ocular_fingerprint,
            "microsleep_gate_ocular_checkpoint_fingerprint": (
                gate_ocular_fingerprint
            ),
            "ocular_cache_schema_version": 1,
            "evidence_config": evidence_config,
            "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
            "standardized_schema_version": int(index["schema_version"]),
            "target_fps": float(index["target_fps"]),
            "label_config": dict(index["config"]),
            "sequence_length": args.sequence_length,
            "temporal_loss_config": temporal_loss_config,
            "microsleep_fusion": (
                "dual_ocular_deterministic_two_second_gate"
                if dual_ocular
                else "deterministic_two_second_gate"
            ),
        }
        initial_mismatch = [
            name
            for name, expected in initial_expected.items()
            if payload.get(name) != expected
        ]
        if initial_mismatch:
            raise ValueError(
                "initial temporal mismatch: " + ", ".join(initial_mismatch)
            )
        initialize_phase_aware_temporal(
            model,
            payload["model_state"],
            visual_embedding_dim=(
                args.embedding_dim + 4 * args.region_dim
            ),
            evidence_dim=len(EVIDENCE_FEATURE_NAMES),
        )

    event_counts = torch.ones(4, dtype=torch.float32)
    for event in extract_state_events(train_sessions):
        target = five_to_four_target(int(event.target))
        if target != IGNORE_INDEX:
            event_counts[target] += 1.0
    class_weights = effective_four_state_weights(event_counts)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if args.initial_temporal_checkpoint is not None:
        initial_metrics = evaluate_fused_temporal(
            model,
            validation_loader,
            device=device,
            amp=args.amp,
            fps=float(index["target_fps"]),
        )
        best_score = fused_temporal_selection_score(initial_metrics)
        initial_metrics["selection_score"] = best_score
        print(
            json.dumps(
                {
                    "epoch": 0,
                    "initial_validation": {
                        key: value
                        for key, value in initial_metrics.items()
                        if key != "rows"
                    },
                }
            ),
            flush=True,
        )
        _save_four_state_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            split=split,
            visual_fingerprint=visual_fingerprint,
            region_fingerprint=region_fingerprint,
            ocular_fingerprint=ocular_fingerprint,
            gate_ocular_fingerprint=gate_ocular_fingerprint,
            gate_ocular_cache_schema_version=(
                gate_ocular_cache_schema_version
            ),
            standardized_index=index,
            sequence_length=args.sequence_length,
            temporal_loss_config=temporal_loss_config,
            evidence_config=evidence_config,
            epoch=0,
            metrics={
                key: value
                for key, value in initial_metrics.items()
                if key != "rows"
            },
            sampling_audit={
                "initial_temporal_checkpoint": str(
                    args.initial_temporal_checkpoint.resolve()
                )
            },
        )
    stale_epochs = 0
    for epoch in range(start_epoch, args.temporal_epochs + 1):
        epoch_seed = args.seed + epoch - 1
        candidate_references, audit = build_subject_event_epoch_references(
            train_sessions,
            samples=args.samples_per_epoch,
            max_windows_per_event=args.max_windows_per_event,
            seed=epoch_seed,
        )
        train_references = tuple(
            reference
            for reference in candidate_references
            if five_to_four_target(
                int(
                    train_sessions[reference.session_index].targets.targets[
                        reference.target_index
                    ]
                )
            )
            != IGNORE_INDEX
        )
        if not train_references:
            raise ValueError("four-state sampler produced no non-microsleep windows")
        train_loader = _four_state_loader(
            train_visual,
            train_ocular,
            train_sessions,
            train_references,
            gate_ocular_caches=train_gate_ocular,
            length=args.sequence_length,
            batch_size=args.temporal_batch_size,
            workers=args.workers,
            shuffle=True,
            seed=epoch_seed,
        )
        sampling_audit = {
            "requested": audit.requested,
            "sampled_before_microsleep_mask": audit.actual,
            "actual": len(train_references),
            "counts": {
                state.name.lower(): count
                for state, count in audit.counts.items()
            },
            "missing_states": [
                state.name.lower() for state in audit.missing_states
            ],
            "windows_per_event": dict(audit.windows_per_event),
            "subject_counts": dict(audit.subject_counts),
            "epoch_seed": epoch_seed,
        }
        train_metrics = train_four_state_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            amp=args.amp,
            class_weights=class_weights,
            final_loss_weight=args.temporal_final_loss_weight,
        )
        validation_metrics = evaluate_fused_temporal(
            model,
            validation_loader,
            device=device,
            amp=args.amp,
            fps=float(index["target_fps"]),
        )
        score = fused_temporal_selection_score(validation_metrics)
        validation_metrics["selection_score"] = score
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": {
                        key: value
                        for key, value in validation_metrics.items()
                        if key != "rows"
                    },
                }
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            _save_four_state_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                split=split,
                visual_fingerprint=visual_fingerprint,
                region_fingerprint=region_fingerprint,
                ocular_fingerprint=ocular_fingerprint,
                gate_ocular_fingerprint=gate_ocular_fingerprint,
                gate_ocular_cache_schema_version=(
                    gate_ocular_cache_schema_version
                ),
                standardized_index=index,
                sequence_length=args.sequence_length,
                temporal_loss_config=temporal_loss_config,
                evidence_config=evidence_config,
                epoch=epoch,
                metrics={
                    key: value
                    for key, value in validation_metrics.items()
                    if key != "rows"
                },
                sampling_audit=sampling_audit,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("four-state temporal training did not produce a checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("model_type") != "causal_four_state_lstm":
        raise ValueError("temporal checkpoint model-type mismatch")
    model.load_state_dict(payload["model_state"])
    test_metrics = evaluate_fused_temporal(
        model,
        test_loader,
        device=device,
        amp=args.amp,
        fps=float(index["target_fps"]),
    )
    rows = test_metrics.pop("rows")
    _atomic_json(temporal_dir / "outer_metrics.json", test_metrics)
    predictions = temporal_dir / "outer_predictions.jsonl"
    temporary_predictions = predictions.with_suffix(".jsonl.tmp")
    with temporary_predictions.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary_predictions.replace(predictions)
    return 0


def _evaluate_saved_temporal(args: argparse.Namespace) -> int:
    metrics_path = (
        args.output.resolve()
        / args.held_out_subject
        / "temporal"
        / "outer_metrics.json"
    )
    predictions_path = metrics_path.with_name("outer_predictions.jsonl")
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise ValueError(
            "outer evaluation is missing; run the LSTM stage first"
        )
    print(metrics_path.read_text(encoding="utf-8").strip(), flush=True)
    return 0


def _run_loso(args: argparse.Namespace) -> int:
    index = load_standardized_index(args.standardized_index.resolve())
    entries = index["sessions"]
    assert isinstance(entries, list)
    subjects = sorted({str(entry["subject"]) for entry in entries})
    if not subjects:
        raise ValueError("LOSO requires standardized subjects")
    for subject in subjects:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.held_out_subject = subject
        fold_args.validation_subject = None
        fold_dir = args.output.resolve() / subject
        if not (fold_dir / "visual" / "best.pt").is_file():
            _prepare_visual(fold_args)
        if not (fold_dir / "visual_cache" / "index.json").is_file():
            _cache_visual(fold_args)
        if not (fold_dir / "ocular" / "best.pt").is_file():
            _train_ocular(fold_args)
        if not (fold_dir / "ocular_cache" / "index.json").is_file():
            _cache_ocular(fold_args)
        temporal_dir = fold_dir / "temporal"
        outer_complete = (
            (temporal_dir / "outer_metrics.json").is_file()
            and (temporal_dir / "outer_predictions.jsonl").is_file()
        )
        if not outer_complete:
            _train_temporal(fold_args)
            _evaluate_saved_temporal(fold_args)
    fold_metrics = {
        subject: json.loads(
            (
                args.output.resolve()
                / subject
                / "temporal"
                / "outer_metrics.json"
            ).read_text(encoding="utf-8")
        )
        for subject in subjects
    }
    summary = aggregate_loso_metrics(fold_metrics)
    _atomic_json(args.output.resolve() / "loso_summary.json", summary)
    csv_path = args.output.resolve() / "loso_summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "subject",
                "accuracy",
                "macro_f1",
                "supported_macro_f1",
                "support",
            ),
        )
        writer.writeheader()
        for subject in subjects:
            metrics = fold_metrics[subject]
            writer.writerow(
                {
                    "subject": subject,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "supported_macro_f1": metrics[
                        "supported_macro_f1"
                    ],
                    "support": metrics["support"],
                }
            )
    temporary_csv.replace(csv_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate_paths = (
        args.microsleep_gate_ocular_cache,
        args.microsleep_gate_ocular_checkpoint,
    )
    if any(path is not None for path in gate_paths) and not all(
        path is not None for path in gate_paths
    ):
        raise ValueError(
            "microsleep-gate ocular cache and checkpoint must be supplied together"
        )
    if all(path is not None for path in gate_paths) and args.stage not in (
        "lstm",
        "evaluate",
    ):
        raise ValueError(
            "separate microsleep-gate artifacts are only valid for the LSTM "
            "or evaluation stage"
        )
    positive = (
        args.embedding_dim,
        args.region_dim,
        args.image_width,
        args.image_height,
        args.sequence_length,
        args.ocular_sequence_length,
        args.visual_epochs,
        args.ocular_epochs,
        args.temporal_epochs,
        args.batch_size,
        args.ocular_batch_size,
        args.ocular_channels,
        args.ocular_samples_per_epoch,
        args.ocular_max_windows_per_event,
        args.temporal_batch_size,
        args.temporal_channels,
        args.samples_per_epoch,
        args.max_windows_per_event,
        args.patience,
    )
    if any(value <= 0 for value in positive) or args.workers < 0:
        raise ValueError("training sizes must be positive and workers non-negative")
    if (
        not math.isfinite(args.ocular_learning_rate)
        or args.ocular_learning_rate <= 0.0
        or not math.isfinite(args.microsleep_seconds)
        or args.microsleep_seconds <= 0.0
        or not math.isfinite(args.uncertain_eye_gap_seconds)
        or args.uncertain_eye_gap_seconds < 0.0
    ):
        raise ValueError("ocular timing and learning-rate options are invalid")
    if not math.isclose(
        args.microsleep_seconds,
        float(EYE_EVIDENCE_CONFIG["microsleep_seconds"]),
    ):
        raise ValueError(
            "the public causal microsleep contract is fixed at 2.0 seconds"
        )
    if not 0.0 <= args.eye_sample_fraction < 1.0:
        raise ValueError("eye sample fraction must be in [0, 1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == "visual":
        return _prepare_visual(args)
    if args.stage == "visual-mixed":
        return _prepare_visual_mixed(args)
    if args.stage == "cache":
        return _cache_visual(args)
    if args.stage == "ocular":
        return _train_ocular(args)
    if args.stage == "ocular-cache":
        return _cache_ocular(args)
    if args.stage == "lstm":
        return _train_temporal(args)
    if args.stage == "evaluate":
        return _evaluate_saved_temporal(args)
    return _run_loso(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
