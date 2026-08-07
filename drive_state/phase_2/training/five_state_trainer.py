"""Training, natural evaluation, metrics, and checkpoints for five states."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..data.five_state_labels import FiveState, IGNORE_INDEX
from ..losses.five_state_loss import FiveStateLoss
from .eye_trainer import SubjectSplitMetadata


def _tensor_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def train_five_state_epoch(
    model: nn.Module,
    criterion: FiveStateLoss,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    use_amp = amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    total = 0.0
    batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["fused"])
            loss = criterion(output, batch)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        total += float(loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("five-state training loader produced no batches")
    return {"loss": total / batches}


def five_state_metrics(
    *, targets: Sequence[int], predictions: Sequence[int]
) -> dict[str, object]:
    if len(targets) != len(predictions):
        raise ValueError("five-state targets and predictions must align")
    classes = len(FiveState)
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        if target == IGNORE_INDEX:
            continue
        confusion[int(FiveState(int(target)))][int(FiveState(int(prediction)))] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for state in FiveState:
        index = int(state)
        true_positive = confusion[index][index]
        predicted = sum(row[index] for row in confusion)
        support = sum(confusion[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[state.name.lower()] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(classes))
    return {
        "support": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / classes,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@torch.inference_mode()
def evaluate_five_state(
    model: nn.Module,
    criterion: FiveStateLoss,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    amp: bool = True,
) -> dict[str, object]:
    model.eval()
    use_amp = amp and device.type == "cuda"
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["fused"])
            loss = criterion(output, batch)
        losses.append(float(loss))
        probabilities = output.logits[:, -1].float().softmax(dim=-1).cpu()
        batch_targets = batch["five_state_target"].cpu()
        sessions = list(raw_batch["session"])
        frame_ids = raw_batch["target_frame_id"].tolist()
        timestamps = raw_batch["target_timestamp"].tolist()
        for index, session in enumerate(sessions):
            identity = (str(session), int(frame_ids[index]))
            if identity in seen:
                raise ValueError(f"natural evaluation reference repeated: {identity}")
            seen.add(identity)
            target = int(batch_targets[index])
            prediction = int(probabilities[index].argmax())
            if target != IGNORE_INDEX:
                targets.append(target)
                predictions.append(prediction)
            rows.append(
                {
                    "session": str(session),
                    "frame_id": int(frame_ids[index]),
                    "timestamp": float(timestamps[index]),
                    "target": target,
                    "prediction": prediction,
                    "probabilities": probabilities[index].tolist(),
                }
            )
    metrics = five_state_metrics(targets=targets, predictions=predictions)
    metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
    metrics["rows"] = rows
    return metrics


def save_five_state_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_kind: str,
    input_dim: int,
    channels: int,
    dilations: Sequence[int],
    sequence_length: int,
    split: SubjectSplitMetadata,
    source_fingerprints: Mapping[str, str],
    epoch: int,
    metrics: Mapping[str, object],
    class_weights: Sequence[float],
    best_score: float | None = None,
    stale_epochs: int = 0,
) -> None:
    if model_kind not in {"mlp", "tcn"}:
        raise ValueError("model_kind must be mlp or tcn")
    required = {"spatial", "mouth", "eye"}
    if set(source_fingerprints) != required or any(
        not source_fingerprints[name] for name in required
    ):
        raise ValueError("all three source checkpoint fingerprints are required")
    if stale_epochs < 0:
        raise ValueError("stale_epochs must be non-negative")
    if best_score is None:
        best_score = float(metrics["macro_f1"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_kind": model_kind,
        "model_config": {
            "input_dim": int(input_dim),
            "channels": int(channels),
            "dilations": tuple(int(value) for value in dilations),
            "sequence_length": int(sequence_length),
        },
        "split": asdict(split),
        "source_fingerprints": dict(source_fingerprints),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "class_weights": tuple(float(value) for value in class_weights),
        "training_state": {
            "best_score": float(best_score),
            "stale_epochs": int(stale_epochs),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_five_state_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported five-state checkpoint schema")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def restore_five_state_training_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_model_kind: str,
    expected_input_dim: int,
    expected_channels: int,
    expected_dilations: Sequence[int],
    expected_sequence_length: int,
    expected_split: SubjectSplitMetadata,
    expected_source_fingerprints: Mapping[str, str],
    expected_class_weights: Sequence[float],
    map_location: str | torch.device = "cpu",
) -> tuple[int, float, int]:
    """Restore one compatible run and return next epoch plus early-stop state."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported five-state checkpoint schema")
    expected_config = {
        "input_dim": int(expected_input_dim),
        "channels": int(expected_channels),
        "dilations": tuple(int(value) for value in expected_dilations),
        "sequence_length": int(expected_sequence_length),
    }
    compatibility = {
        "model_kind": (payload.get("model_kind"), expected_model_kind),
        "model_config": (payload.get("model_config"), expected_config),
        "split": (payload.get("split"), asdict(expected_split)),
        "source_fingerprints": (
            payload.get("source_fingerprints"),
            dict(expected_source_fingerprints),
        ),
        "class_weights": (
            tuple(payload.get("class_weights", ())),
            tuple(float(value) for value in expected_class_weights),
        ),
    }
    mismatched = [
        name for name, (actual, expected) in compatibility.items() if actual != expected
    ]
    if mismatched:
        raise ValueError(f"resume checkpoint mismatch: {', '.join(mismatched)}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    training_state = payload.get("training_state", {})
    best_score = float(
        training_state.get("best_score", payload["metrics"]["macro_f1"])
    )
    stale_epochs = int(training_state.get("stale_epochs", 0))
    if stale_epochs < 0:
        raise ValueError("resume checkpoint has negative stale epoch count")
    return int(payload["epoch"]) + 1, best_score, stale_epochs


__all__ = [
    "evaluate_five_state",
    "five_state_metrics",
    "load_five_state_checkpoint",
    "restore_five_state_training_checkpoint",
    "save_five_state_checkpoint",
    "train_five_state_epoch",
]
