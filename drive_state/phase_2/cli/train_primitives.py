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
from ..data.primitive_dataset import PrimitiveFrameDataset
from ..data.primitive_records import (
    build_balanced_sample_weights,
    load_primitive_records,
)
from ..losses.primitive_loss import MaskedPrimitiveLoss
from ..models.spatial_multitask import SpatialPrimitiveNet
from ..training.eye_trainer import SubjectSplitMetadata
from ..training.primitive_trainer import (
    FINAL_STATE_SELECTION_WEIGHTS,
    evaluate_primitives,
    save_primitive_checkpoint,
    train_primitive_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DMD cabin/face primitive heads")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--face-cache", type=Path, required=True)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=120_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--max-class-boost", type=float, default=5.0)
    parser.add_argument("--event-power", type=float, default=0.5)
    parser.add_argument("--action-distraction-weight", type=float, default=1.0)
    parser.add_argument(
        "--distraction-consistency-weight", type=float, default=0.25
    )
    parser.add_argument("--patience", type=int, default=3)
    pretraining = parser.add_mutually_exclusive_group()
    pretraining.add_argument(
        "--pretrained",
        action="store_true",
        help="Opt in to external ImageNet weights (not DMD-only compliant)",
    )
    pretraining.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(pretrained=False)
    parser.add_argument(
        "--allow-missing-face-cache",
        action="store_true",
        help="Debug only: substitute invisible black faces for missing sessions",
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.patience < 1:
        raise SystemExit("--patience must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sessions = load_sessions(args.root / "labels_20fps/manifest_20fps.json")
    folds = make_nested_loso_folds(sessions)
    fold = next(
        (fold for fold in folds if fold.held_out_subject == args.held_out_subject), None
    )
    if fold is None:
        raise SystemExit(
            f"unknown subject; choose one of {[fold.held_out_subject for fold in folds]}"
        )
    train_records = load_primitive_records(fold.train)
    validation_records = load_primitive_records(fold.validation)
    test_records = load_primitive_records(fold.test)
    datasets = {
        "train": PrimitiveFrameDataset(
            train_records,
            face_cache_dir=args.face_cache,
            strict_face_cache=not args.allow_missing_face_cache,
        ),
        "validation": PrimitiveFrameDataset(
            validation_records,
            face_cache_dir=args.face_cache,
            strict_face_cache=not args.allow_missing_face_cache,
        ),
        "test": PrimitiveFrameDataset(
            test_records,
            face_cache_dir=args.face_cache,
            strict_face_cache=not args.allow_missing_face_cache,
        ),
    }
    weights = build_balanced_sample_weights(
        train_records,
        max_class_boost=args.max_class_boost,
        event_power=args.event_power,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=generator,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(datasets["train"], sampler=sampler, drop_last=True, **loader_options)
    validation_loader = DataLoader(datasets["validation"], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialPrimitiveNet(
        pretrained=args.pretrained, embedding_dim=args.embedding_dim
    ).to(device)
    criterion = MaskedPrimitiveLoss(
        task_weights={
            "driver_action": 0.75,
            "distraction": 2.0,
            "road_gaze": 1.5,
            "hands_using_wheel": 0.25,
            "talking": 0.25,
            "yawn": 2.0,
            "blink": 0.25,
            "gaze_zone": 0.5,
            "hands_on_wheel": 0.25,
            "moving_hands": 0.25,
        },
        action_distraction_weight=args.action_distraction_weight,
        distraction_consistency_weight=args.distraction_consistency_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    split = SubjectSplitMetadata(
        train_subjects=tuple(sorted({record.subject_id for record in train_records})),
        validation_subject=fold.validation_subject,
        test_subject=fold.held_out_subject,
    )
    checkpoint = args.output / fold.held_out_subject / "best_primitives.pt"
    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_primitive_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
        )
        validation_metrics = evaluate_primitives(
            model, criterion, validation_loader, device=device
        )
        score = float(validation_metrics["final_state_priority_score"])
        print(
            json.dumps(
                {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_primitive_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                split=split,
                epoch=epoch,
                metrics=validation_metrics,
                model_config={
                    "embedding_dim": args.embedding_dim,
                    "pretrained": args.pretrained,
                    "sampling": {
                        "max_class_boost": args.max_class_boost,
                        "event_power": args.event_power,
                    },
                    "hierarchical_distraction": {
                        "action_distraction_weight": args.action_distraction_weight,
                        "consistency_weight": args.distraction_consistency_weight,
                    },
                    "selection_weights": FINAL_STATE_SELECTION_WEIGHTS,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "epoch": epoch,
                            "best_final_state_priority_score": best_score,
                        }
                    ),
                    flush=True,
                )
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_metrics = evaluate_primitives(model, criterion, test_loader, device=device)
    print(json.dumps({"outer_test_subject": fold.held_out_subject, "test": test_metrics}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
