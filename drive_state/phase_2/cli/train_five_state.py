"""Train one exclusive DMD-only five-state head with nested LOSO isolation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.embedding_windows import (
    EmbeddingWindowDataset,
    build_five_state_target_index,
    chronological_target_index,
)
from ..data.five_state_labels import (
    WeakStateTargets,
    build_weak_state_targets,
    weak_state_support,
)
from ..data.manifest import load_sessions
from ..data.temporal_embeddings import SCHEMA_VERSION, EmbeddingSessionStore
from ..losses.five_state_loss import (
    FiveStateLoss,
    compute_five_state_class_weights,
)
from ..models.five_state_temporal import FiveStateFusionTCN, FiveStateLastFrameMLP
from ..training.eye_trainer import SubjectSplitMetadata
from ..training.five_state_trainer import (
    evaluate_five_state,
    load_five_state_checkpoint,
    restore_five_state_training_checkpoint,
    save_five_state_checkpoint,
    train_five_state_epoch,
)


def load_aligned_weak_targets(
    labels_csv: Path | str,
    *,
    protocol: str,
    fps: float,
    frame_ids: np.ndarray,
) -> WeakStateTargets:
    """Build at native rate, then select exact embedding frame IDs."""

    with Path(labels_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    row_by_frame: dict[int, int] = {}
    for index, row in enumerate(rows):
        frame_id = int(row["frame_id"])
        if frame_id in row_by_frame:
            raise ValueError(f"duplicate label frame ID: {frame_id}")
        row_by_frame[frame_id] = index
    missing = [int(value) for value in frame_ids if int(value) not in row_by_frame]
    if missing:
        raise ValueError(f"embedding cache references missing frame IDs: {missing[:5]}")
    native = build_weak_state_targets(protocol, rows, fps=fps)
    indices = [row_by_frame[int(value)] for value in frame_ids]
    return WeakStateTargets(
        targets=tuple(native.targets[index] for index in indices),
        confidence=tuple(native.confidence[index] for index in indices),
        reasons=tuple(native.reasons[index] for index in indices),
    )


def _split(raw: Mapping[str, object]) -> SubjectSplitMetadata:
    return SubjectSplitMetadata(
        tuple(str(value) for value in raw["train_subjects"]),
        str(raw["validation_subject"]),
        str(raw["test_subject"]),
    )


def _load_stores(
    cache_dir: Path, held_out_subject: str
) -> tuple[
    dict[str, tuple[EmbeddingSessionStore, ...]],
    SubjectSplitMetadata,
    dict[str, str],
]:
    with (cache_dir / "index.json").open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"five-state training requires embedding schema {SCHEMA_VERSION}"
        )
    if index.get("held_out_subject") != held_out_subject:
        raise ValueError("embedding index held-out subject mismatch")
    split = _split(index["split"])
    if split.test_subject != held_out_subject:
        raise ValueError("embedding split test subject mismatch")
    fingerprints = {
        name: str(index[f"{name}_fingerprint"])
        for name in ("spatial", "mouth", "eye")
    }
    partitions: dict[str, list[EmbeddingSessionStore]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[str] = set()
    for entry in index["sessions"]:
        partition = str(entry["partition"])
        session_name = str(entry["session"])
        if partition not in partitions:
            raise ValueError(f"unknown embedding partition: {partition}")
        if session_name in seen:
            raise ValueError(f"duplicate embedding session: {session_name}")
        seen.add(session_name)
        store = EmbeddingSessionStore(
            cache_dir,
            session_name,
            expected_spatial_fingerprint=fingerprints["spatial"],
            expected_mouth_fingerprint=fingerprints["mouth"],
            expected_eye_fingerprint=fingerprints["eye"],
            expected_split=split,
        )
        partitions[partition].append(store)
    if any(not values for values in partitions.values()):
        raise ValueError("embedding index must contain train, validation, and test")
    if any(store.subject_id not in set(split.train_subjects) for store in partitions["train"]):
        raise ValueError("non-training subject routed into train partition")
    if any(store.subject_id != split.validation_subject for store in partitions["validation"]):
        raise ValueError("wrong subject routed into validation partition")
    if any(store.subject_id != split.test_subject for store in partitions["test"]):
        raise ValueError("wrong subject routed into test partition")
    return {name: tuple(values) for name, values in partitions.items()}, split, fingerprints


def _targets_for(
    stores: Sequence[EmbeddingSessionStore], sessions_by_name: Mapping[str, object]
) -> dict[str, WeakStateTargets]:
    targets: dict[str, WeakStateTargets] = {}
    for store in stores:
        session = sessions_by_name.get(store.session_name)
        if session is None:
            raise ValueError(f"embedding session missing from manifest: {store.session_name}")
        targets[store.session_name] = load_aligned_weak_targets(
            session.labels_csv,
            protocol=session.protocol,
            fps=session.fps,
            frame_ids=store.frame_ids,
        )
    return targets


def _loader(
    stores: Sequence[EmbeddingSessionStore],
    references,
    targets: Mapping[str, WeakStateTargets],
    *,
    sequence_length: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        EmbeddingWindowDataset(
            stores,
            references,
            length=sequence_length,
            five_state_targets=targets,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _model(kind: str, *, input_dim: int, channels: int):
    if kind == "mlp":
        return FiveStateLastFrameMLP(input_dim=input_dim, channels=channels)
    if kind == "tcn":
        return FiveStateFusionTCN(input_dim=input_dim, channels=channels)
    raise ValueError("model kind must be mlp or tcn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one causal DMD five-state classifier"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("mlp", "tcn"), default="tcn")
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt, or best.pt when no last checkpoint exists",
    )
    return parser


def _without_rows(metrics: Mapping[str, object]) -> dict[str, object]:
    return {name: value for name, value in metrics.items() if name != "rows"}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = (
        args.sequence_length,
        args.channels,
        args.batch_size,
        args.samples_per_epoch,
        args.epochs,
        args.patience,
    )
    if any(value < 1 for value in positive) or args.workers < 0:
        raise ValueError("training sizes must be positive and workers non-negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sessions = load_sessions(args.root / "labels_20fps/manifest_20fps.json")
    sessions_by_name = {session.name: session for session in sessions}
    stores, split, fingerprints = _load_stores(
        args.embedding_cache, args.held_out_subject
    )
    all_stores = tuple(
        store for partition in ("train", "validation", "test") for store in stores[partition]
    )
    all_targets = _targets_for(all_stores, sessions_by_name)
    train_targets = {
        store.session_name: all_targets[store.session_name] for store in stores["train"]
    }
    validation_targets = {
        store.session_name: all_targets[store.session_name]
        for store in stores["validation"]
    }
    test_targets = {
        store.session_name: all_targets[store.session_name] for store in stores["test"]
    }
    support = weak_state_support(train_targets.values())
    class_weights = compute_five_state_class_weights(
        [target for result in train_targets.values() for target in result.targets]
    )
    train_references = build_five_state_target_index(
        stores["train"],
        train_targets,
        samples=args.samples_per_epoch,
        seed=args.seed,
    )
    validation_references = chronological_target_index(stores["validation"])
    test_references = chronological_target_index(stores["test"])
    train_loader = _loader(
        stores["train"],
        train_references,
        train_targets,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = _loader(
        stores["validation"],
        validation_references,
        validation_targets,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = _loader(
        stores["test"],
        test_references,
        test_targets,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    input_dim = 718
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(args.model_kind, input_dim=input_dim, channels=args.channels).to(device)
    criterion = FiveStateLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    run_dir = args.output / args.held_out_subject / args.model_kind
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    best_score = -1.0
    stale = 0
    dilations = () if args.model_kind == "mlp" else (1, 2, 4, 8, 16, 32)
    start_epoch = 1
    if args.resume:
        resume_path = last_path if last_path.exists() else best_path
        if not resume_path.exists():
            raise FileNotFoundError(f"no five-state checkpoint to resume in {run_dir}")
        start_epoch, best_score, stale = restore_five_state_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_model_kind=args.model_kind,
            expected_input_dim=input_dim,
            expected_channels=args.channels,
            expected_dilations=dilations,
            expected_sequence_length=args.sequence_length,
            expected_split=split,
            expected_source_fingerprints=fingerprints,
            expected_class_weights=class_weights,
            map_location=device,
        )
        print(
            json.dumps(
                {
                    "resumed_from": str(resume_path),
                    "next_epoch": start_epoch,
                    "best_score": best_score,
                    "stale_epochs": stale,
                }
            ),
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_five_state_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
        )
        validation_metrics = evaluate_five_state(
            model, criterion, validation_loader, device=device
        )
        score = float(validation_metrics["macro_f1"])
        report = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": _without_rows(validation_metrics),
        }
        print(json.dumps(report), flush=True)
        if score > best_score:
            best_score = score
            stale = 0
            save_five_state_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                model_kind=args.model_kind,
                input_dim=input_dim,
                channels=args.channels,
                dilations=dilations,
                sequence_length=args.sequence_length,
                split=split,
                source_fingerprints=fingerprints,
                epoch=epoch,
                metrics=_without_rows(validation_metrics),
                class_weights=class_weights,
                best_score=best_score,
                stale_epochs=stale,
            )
        else:
            stale += 1
        save_five_state_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            model_kind=args.model_kind,
            input_dim=input_dim,
            channels=args.channels,
            dilations=dilations,
            sequence_length=args.sequence_length,
            split=split,
            source_fingerprints=fingerprints,
            epoch=epoch,
            metrics=_without_rows(validation_metrics),
            class_weights=class_weights,
            best_score=best_score,
            stale_epochs=stale,
        )
        if stale >= args.patience:
            break
    load_five_state_checkpoint(best_path, model=model, map_location=device)
    test_metrics = evaluate_five_state(model, criterion, test_loader, device=device)
    rows = test_metrics.pop("rows")
    summary = {
        "split": asdict(split),
        "model_kind": args.model_kind,
        "train_support": {
            "frames": {state.name.lower(): support.frames[state] for state in support.frames},
            "events": {state.name.lower(): support.events[state] for state in support.events},
            "ignored": support.ignored,
        },
        "class_weights": class_weights,
        "outer_test": test_metrics,
    }
    (run_dir / "outer_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (run_dir / "outer_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
