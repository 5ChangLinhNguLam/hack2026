from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.manifest import load_sessions, make_nested_loso_folds
from ..data.mouth_dataset import MouthFrameDataset, build_yawn_event_sample_weights
from ..data.primitive_records import load_primitive_records
from ..losses.yawn_loss import YawnHierarchicalLoss
from ..models.mouth_visual import MouthVisualNet
from ..training.eye_trainer import SubjectSplitMetadata
from ..training.mouth_trainer import (
    evaluate_mouth,
    save_mouth_checkpoint,
    train_mouth_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fold-specific DMD s5 visual mouth specialist"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mouth-cache", type=Path, required=True)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    return parser


def _s5(records):
    return tuple(record for record in records if record.protocol == "s5")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.patience < 1 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("epochs, batch size, and patience must be positive; workers cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    train_records = _s5(load_primitive_records(fold.train))
    validation_records = _s5(load_primitive_records(fold.validation))
    test_records = _s5(load_primitive_records(fold.test))
    if not train_records or not validation_records or not test_records:
        raise SystemExit("nested LOSO mouth training requires s5 data in every split")

    datasets = {
        "train": MouthFrameDataset(train_records, cache_dir=args.mouth_cache),
        "validation": MouthFrameDataset(validation_records, cache_dir=args.mouth_cache),
        "test": MouthFrameDataset(test_records, cache_dir=args.mouth_cache),
    }
    weights = build_yawn_event_sample_weights(train_records)
    samples_per_epoch = args.samples_per_epoch or len(train_records)
    if samples_per_epoch < args.batch_size:
        raise SystemExit("samples per epoch must be at least one batch")
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        datasets["train"], sampler=sampler, drop_last=True, **loader_options
    )
    validation_loader = DataLoader(
        datasets["validation"], shuffle=False, **loader_options
    )
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MouthVisualNet(embedding_dim=args.embedding_dim).to(device)
    criterion = YawnHierarchicalLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    split = SubjectSplitMetadata(
        train_subjects=tuple(sorted({record.subject_id for record in train_records})),
        validation_subject=fold.validation_subject,
        test_subject=fold.held_out_subject,
    )
    checkpoint = args.output / fold.held_out_subject / "best_mouth.pt"
    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_mouth_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
        )
        validation_metrics = evaluate_mouth(
            model, criterion, validation_loader, device=device
        )
        score = float(validation_metrics["binary"]["macro_f1"])
        print(
            json.dumps(
                {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_mouth_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                split=split,
                epoch=epoch,
                metrics=validation_metrics,
                model_config={
                    "embedding_dim": args.embedding_dim,
                    "input_size": (96, 64),
                    "sampling": "50pct_unique_yawn_events_50pct_no_yawn",
                    "visibility_threshold": criterion.visibility_threshold,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    json.dumps(
                        {"early_stopping": True, "epoch": epoch, "best_binary_macro_f1": best_score}
                    ),
                    flush=True,
                )
                break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_metrics = evaluate_mouth(model, criterion, test_loader, device=device)
    print(
        json.dumps({"outer_test_subject": fold.held_out_subject, "test": test_metrics}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
