from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_eye_config
from ..data.eye_dataset import EyeClipDataset
from ..data.eye_sequences import EyeClipRef, build_eye_clip_index, load_eye_sequences
from ..data.manifest import load_sessions, make_nested_loso_folds
from ..losses.eye_loss import EyeMultiTaskLoss
from ..models.eye_temporal import EyeTemporalNet
from ..training.eye_trainer import (
    SubjectSplitMetadata,
    compute_phase_class_weights,
    evaluate_eye,
    load_eye_checkpoint,
    save_eye_checkpoint,
    train_eye_epoch,
)


def _evaluation_index(sequences, sequence_length: int) -> tuple[EyeClipRef, ...]:
    references: list[EyeClipRef] = []
    for sequence_index, sequence in enumerate(sequences):
        maximum = len(sequence) - sequence_length
        if maximum < 0:
            continue
        starts = list(range(0, maximum + 1, sequence_length))
        if starts[-1] != maximum:
            starts.append(maximum)
        references.extend(EyeClipRef(sequence_index, start) for start in starts)
    return tuple(references)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the causal four-phase DMD eye model")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eye-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eye_temporal.yaml"))
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_eye_config(args.config)
    s5_sessions = tuple(
        session
        for session in load_sessions(args.root / "labels_20fps/manifest_20fps.json")
        if session.protocol == "s5"
    )
    folds = make_nested_loso_folds(s5_sessions)
    fold = next(
        (fold for fold in folds if fold.held_out_subject == args.held_out_subject), None
    )
    if fold is None:
        available = [fold.held_out_subject for fold in folds]
        raise SystemExit(f"held-out subject has no s5 session; choose one of {available}")

    train_sequences = load_eye_sequences(fold.train)
    validation_sequences = load_eye_sequences(fold.validation)
    test_sequences = load_eye_sequences(fold.test)
    train_index = build_eye_clip_index(
        train_sequences,
        sequence_length=config.sequence_length,
        transition_fraction=config.transition_fraction,
        seed=args.seed,
    )
    validation_index = _evaluation_index(validation_sequences, config.sequence_length)
    test_index = _evaluation_index(test_sequences, config.sequence_length)
    datasets = {
        "train": EyeClipDataset(
            train_sequences,
            train_index,
            cache_dir=args.eye_cache,
            sequence_length=config.sequence_length,
        ),
        "validation": EyeClipDataset(
            validation_sequences,
            validation_index,
            cache_dir=args.eye_cache,
            sequence_length=config.sequence_length,
        ),
        "test": EyeClipDataset(
            test_sequences,
            test_index,
            cache_dir=args.eye_cache,
            sequence_length=config.sequence_length,
        ),
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(datasets["validation"], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeTemporalNet(
        embedding_dim=config.embedding_dim,
        temporal_channels=config.tcn_channels,
        dilations=config.tcn_dilations,
        visibility_floor=config.visibility_floor,
    ).to(device)
    class_weights = compute_phase_class_weights(
        [target for sequence in train_sequences for target in sequence.phase_targets if target >= 0]
    )
    criterion = EyeMultiTaskLoss(
        phase_weight=config.phase_weight,
        closedness_weight=config.closed_weight,
        moving_weight=config.moving_weight,
        phase_class_weights=class_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    split = SubjectSplitMetadata(
        train_subjects=tuple(sorted({sequence.subject_id for sequence in train_sequences})),
        validation_subject=fold.validation_subject,
        test_subject=fold.held_out_subject,
    )
    checkpoint = args.output / fold.held_out_subject / "best_eye.pt"
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_eye_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
        )
        validation_metrics = evaluate_eye(
            model, criterion, validation_loader, device=device
        )
        score = float(validation_metrics["transition_macro_f1"])
        print(
            json.dumps(
                {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            save_eye_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                split=split,
                epoch=epoch,
                metrics=validation_metrics,
            )
    load_eye_checkpoint(checkpoint, model=model, optimizer=None, map_location=device)
    test_metrics = evaluate_eye(model, criterion, test_loader, device=device)
    print(json.dumps({"outer_test_subject": fold.held_out_subject, "test": test_metrics}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
