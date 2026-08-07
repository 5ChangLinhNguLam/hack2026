"""Run the approved frozen-embedding MLP-versus-causal-TCN experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.embedding_windows import (
    EmbeddingWindowDataset,
    build_cabin_target_index,
    build_face_mouth_target_index,
    chronological_target_index,
)
from ..data.temporal_embeddings import EmbeddingSessionStore
from ..distraction import action_distraction_probability, pool_distraction_probabilities
from ..losses.embedding_temporal_loss import CabinTemporalLoss, FaceMouthTemporalLoss
from ..models.embedding_temporal import (
    CabinEmbeddingTCN,
    CabinLastFrameMLP,
    FaceMouthEmbeddingTCN,
    FaceMouthLastFrameMLP,
)
from ..training.embedding_trainer import (
    calibrate_binary_logits,
    embedding_metrics,
    load_embedding_checkpoint,
    predict_embedding_model,
    save_embedding_checkpoint,
    train_embedding_epoch,
)
from ..training.eye_trainer import SubjectSplitMetadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare last-frame MLPs with causal TCNs on frozen DMD embeddings"
    )
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    return parser


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
    index_path = cache_dir / "index.json"
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != 1:
        raise ValueError(f"unsupported embedding index schema: {index.get('schema_version')}")
    if index.get("held_out_subject") != held_out_subject:
        raise ValueError("embedding index held-out subject mismatch")
    split = _split(index["split"])
    if split.test_subject != held_out_subject:
        raise ValueError("embedding split test subject mismatch")
    fingerprints = {
        "spatial": str(index["spatial_fingerprint"]),
        "mouth": str(index["mouth_fingerprint"]),
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
            raise ValueError(f"duplicate embedding index session: {session_name}")
        seen.add(session_name)
        store = EmbeddingSessionStore(
            cache_dir,
            session_name,
            expected_spatial_fingerprint=fingerprints["spatial"],
            expected_mouth_fingerprint=fingerprints["mouth"],
            expected_split=split,
        )
        if partition == "train" and store.subject_id not in set(split.train_subjects):
            raise ValueError("non-training subject routed into embedding train partition")
        if partition == "validation" and store.subject_id != split.validation_subject:
            raise ValueError("wrong subject routed into embedding validation partition")
        if partition == "test" and store.subject_id != split.test_subject:
            raise ValueError("wrong subject routed into embedding test partition")
        partitions[partition].append(store)
    if any(not values for values in partitions.values()):
        raise ValueError("embedding index must contain train, validation, and test sessions")
    return (
        {name: tuple(values) for name, values in partitions.items()},
        split,
        fingerprints,
    )


def _loader(
    stores: Sequence[EmbeddingSessionStore],
    references,
    *,
    length: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        EmbeddingWindowDataset(stores, references, length=length),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _sigmoid(logit: float, temperature: float = 1.0) -> float:
    value = logit / temperature
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _logit(probability: float) -> float:
    value = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _fit_calibration(
    predictions: Mapping[str, object], specialist: str
) -> dict[str, dict[str, float]]:
    rows = [dict(row) for row in predictions["rows"]]
    if specialist == "cabin":
        distraction = [row for row in rows if int(row["distraction"]) != -100]
        road = [row for row in rows if int(row["road_gaze"]) != -100]
        direct = calibrate_binary_logits(
            logits=[float(row["direct_distraction_logit"]) for row in distraction],
            targets=[int(row["distraction"]) for row in distraction],
        )
        action = calibrate_binary_logits(
            logits=[float(row["action_distraction_logit"]) for row in distraction],
            targets=[int(row["distraction"]) for row in distraction],
        )
        road_config = calibrate_binary_logits(
            logits=[float(row["road_gaze_logit"]) for row in road],
            targets=[int(row["road_gaze"]) for row in road],
        )
        fused_logits = []
        for row in distraction:
            probability = pool_distraction_probabilities(
                _sigmoid(float(row["direct_distraction_logit"]), direct["temperature"]),
                _sigmoid(float(row["action_distraction_logit"]), action["temperature"]),
                _sigmoid(float(row["road_gaze_logit"]), road_config["temperature"]),
            )
            fused_logits.append(_logit(float(probability)))
        fused = calibrate_binary_logits(
            logits=fused_logits,
            targets=[int(row["distraction"]) for row in distraction],
        )
        return {
            "direct_distraction": direct,
            "action_distraction": action,
            "road_gaze": road_config,
            "fused_distraction": fused,
        }
    if specialist == "face_mouth":
        valid = [
            row
            for row in rows
            if int(row["yawn"]) != -100 and float(row["mouth_visibility"]) >= 0.2
        ]
        targets = [int(int(row["yawn"]) > 0) for row in valid]
        return {
            "binary_yawn": calibrate_binary_logits(
                logits=[float(row["binary_yawn_logit"]) for row in valid],
                targets=targets,
            ),
            "type_derived_yawn": calibrate_binary_logits(
                logits=[float(row["type_derived_yawn_logit"]) for row in valid],
                targets=targets,
            ),
        }
    raise ValueError("specialist must be cabin or face_mouth")


def _make_model(kind: str, specialist: str, channels: int) -> torch.nn.Module:
    if specialist == "cabin":
        return (
            CabinLastFrameMLP(input_dim=256, channels=channels)
            if kind == "mlp"
            else CabinEmbeddingTCN(input_dim=256, channels=channels)
        )
    return (
        FaceMouthLastFrameMLP(input_dim=324, channels=channels)
        if kind == "mlp"
        else FaceMouthEmbeddingTCN(input_dim=324, channels=channels)
    )


def _run_model(
    *,
    kind: str,
    specialist: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    output: Path,
    split: SubjectSplitMetadata,
    fingerprints: Mapping[str, str],
    channels: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    model = _make_model(kind, specialist, channels).to(device)
    criterion = CabinTemporalLoss().to(device) if specialist == "cabin" else FaceMouthTemporalLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    checkpoint = output / f"best_{specialist}_{kind}.pt"
    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_embedding_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            specialist=specialist,
            device=device,
            scaler=scaler,
        )
        validation_predictions = predict_embedding_model(
            model,
            criterion,
            validation_loader,
            specialist=specialist,
            device=device,
        )
        calibration = _fit_calibration(validation_predictions, specialist)
        validation_metrics = embedding_metrics(
            validation_predictions,
            specialist=specialist,
            calibration=calibration,
        )
        score = float(validation_metrics["score"])
        print(
            json.dumps(
                {
                    "model_kind": kind,
                    "specialist": specialist,
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "calibration": calibration,
                }
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_embedding_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_kind=kind,
                specialist=specialist,
                input_dim=256 if specialist == "cabin" else 324,
                channels=channels,
                split=split,
                source_fingerprints=fingerprints,
                epoch=epoch,
                metrics=validation_metrics,
                calibration=calibration,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    json.dumps(
                        {
                            "model_kind": kind,
                            "specialist": specialist,
                            "early_stopping": True,
                            "epoch": epoch,
                            "best_score": best_score,
                        }
                    ),
                    flush=True,
                )
                break
    payload = load_embedding_checkpoint(checkpoint, model=model, map_location=device)
    outer_predictions = predict_embedding_model(
        model,
        criterion,
        test_loader,
        specialist=specialist,
        device=device,
    )
    outer_metrics = embedding_metrics(
        outer_predictions,
        specialist=specialist,
        calibration=payload["calibration"],
    )
    result = {
        "checkpoint": str(checkpoint),
        "best_epoch": int(payload["epoch"]),
        "validation": payload["validation_metrics"],
        "calibration": payload["calibration"],
        "outer_test": outer_metrics,
    }
    print(
        json.dumps(
            {
                "model_kind": kind,
                "specialist": specialist,
                "outer_test_subject": split.test_subject,
                "test": outer_metrics,
            }
        ),
        flush=True,
    )
    return model, result


def _hard_negative_scores(
    model: torch.nn.Module,
    stores: Sequence[EmbeddingSessionStore],
    *,
    sequence_length: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = _loader(
        stores,
        chronological_target_index(stores),
        length=sequence_length,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=0,
    )
    predictions = predict_embedding_model(
        model,
        FaceMouthTemporalLoss(),
        loader,
        specialist="face_mouth",
        device=device,
    )
    result = {
        store.session_name: np.full(len(store), -np.inf, dtype=np.float32)
        for store in stores
    }
    for row in predictions["rows"]:
        result[str(row["session"])][int(row["target_index"])] = _sigmoid(
            float(row["binary_yawn_logit"])
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(
        value < 1
        for value in (
            args.sequence_length,
            args.batch_size,
            args.samples_per_epoch,
            args.epochs,
            args.patience,
            args.channels,
        )
    ) or args.workers < 0:
        raise SystemExit("temporal dimensions/counts must be positive; workers cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    partitions, split, fingerprints = _load_stores(
        args.embedding_cache, args.held_out_subject
    )
    output = args.output / args.held_out_subject
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cabin_train_refs = build_cabin_target_index(
        partitions["train"], samples=args.samples_per_epoch, seed=args.seed
    )
    cabin_train_loader = _loader(
        partitions["train"],
        cabin_train_refs,
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    cabin_validation_loader = _loader(
        partitions["validation"],
        chronological_target_index(partitions["validation"]),
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    cabin_test_loader = _loader(
        partitions["test"],
        chronological_target_index(partitions["test"]),
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    cabin_mlp_model, cabin_mlp = _run_model(
        kind="mlp",
        specialist="cabin",
        train_loader=cabin_train_loader,
        validation_loader=cabin_validation_loader,
        test_loader=cabin_test_loader,
        output=output,
        split=split,
        fingerprints=fingerprints,
        channels=args.channels,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )
    del cabin_mlp_model
    cabin_tcn_model, cabin_tcn = _run_model(
        kind="tcn",
        specialist="cabin",
        train_loader=cabin_train_loader,
        validation_loader=cabin_validation_loader,
        test_loader=cabin_test_loader,
        output=output,
        split=split,
        fingerprints=fingerprints,
        channels=args.channels,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed + 1,
    )
    del cabin_tcn_model

    train_face_stores = tuple(store for store in partitions["train"] if store.protocol == "s5")
    validation_face_stores = tuple(store for store in partitions["validation"] if store.protocol == "s5")
    test_face_stores = tuple(store for store in partitions["test"] if store.protocol == "s5")
    if not train_face_stores or not validation_face_stores or not test_face_stores:
        raise ValueError("face-mouth temporal experiment requires s5 in every partition")
    face_mlp_refs = build_face_mouth_target_index(
        train_face_stores, samples=args.samples_per_epoch, seed=args.seed
    )
    face_mlp_loader = _loader(
        train_face_stores,
        face_mlp_refs,
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    face_validation_loader = _loader(
        validation_face_stores,
        chronological_target_index(validation_face_stores),
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    face_test_loader = _loader(
        test_face_stores,
        chronological_target_index(test_face_stores),
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    face_mlp_model, face_mlp = _run_model(
        kind="mlp",
        specialist="face_mouth",
        train_loader=face_mlp_loader,
        validation_loader=face_validation_loader,
        test_loader=face_test_loader,
        output=output,
        split=split,
        fingerprints=fingerprints,
        channels=args.channels,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed + 2,
    )
    hard_scores = _hard_negative_scores(
        face_mlp_model,
        train_face_stores,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    del face_mlp_model
    face_tcn_refs = build_face_mouth_target_index(
        train_face_stores,
        samples=args.samples_per_epoch,
        seed=args.seed,
        hard_negative_scores=hard_scores,
    )
    face_tcn_loader = _loader(
        train_face_stores,
        face_tcn_refs,
        length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    face_tcn_model, face_tcn = _run_model(
        kind="tcn",
        specialist="face_mouth",
        train_loader=face_tcn_loader,
        validation_loader=face_validation_loader,
        test_loader=face_test_loader,
        output=output,
        split=split,
        fingerprints=fingerprints,
        channels=args.channels,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed + 3,
    )
    del face_tcn_model

    cabin_mlp_test = cabin_mlp["outer_test"]
    cabin_tcn_test = cabin_tcn["outer_test"]
    face_mlp_test = face_mlp["outer_test"]
    face_tcn_test = face_tcn["outer_test"]
    gates = {
        "fused_distraction_at_least_0_65": float(cabin_tcn_test["fused_distraction"]["macro_f1"]) >= 0.65,
        "fused_distraction_gain_at_least_0_03": float(cabin_tcn_test["fused_distraction"]["macro_f1"]) - float(cabin_mlp_test["fused_distraction"]["macro_f1"]) >= 0.03,
        "road_gaze_drop_at_most_0_02": float(cabin_tcn_test["road_gaze"]["macro_f1"]) >= float(cabin_mlp_test["road_gaze"]["macro_f1"]) - 0.02,
        "binary_yawn_at_least_0_55": float(face_tcn_test["binary_yawn"]["macro_f1"]) >= 0.55,
        "binary_yawn_gain_at_least_0_05": float(face_tcn_test["binary_yawn"]["macro_f1"]) - float(face_mlp_test["binary_yawn"]["macro_f1"]) >= 0.05,
        "distraction_event_f1_improves": float(cabin_tcn_test["events"]["event_f1"]) > float(cabin_mlp_test["events"]["event_f1"]),
        "yawn_event_f1_improves": float(face_tcn_test["events"]["event_f1"]) > float(face_mlp_test["events"]["event_f1"]),
    }
    comparison = {
        "held_out_subject": split.test_subject,
        "split": asdict(split),
        "source_fingerprints": fingerprints,
        "cabin": {"mlp": cabin_mlp, "tcn": cabin_tcn},
        "face_mouth": {"mlp": face_mlp, "tcn": face_tcn},
        "acceptance_gates": gates,
        "model_metric_gates_pass": all(gates.values()),
        "runtime_gate_pending": True,
    }
    temporary = output / "comparison.json.tmp"
    temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output / "comparison.json")
    print(json.dumps({"comparison": comparison}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
