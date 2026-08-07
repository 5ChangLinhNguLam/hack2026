"""T4-friendly eye training loop with subject-safe checkpoint metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..config import EyeTemporalConfig
from ..losses.eye_loss import EyeMultiTaskLoss
from ..metrics import phase_classification_metrics


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SubjectSplitMetadata:
    train_subjects: tuple[str, ...]
    validation_subject: str
    test_subject: str

    def __post_init__(self) -> None:
        train = set(self.train_subjects)
        if len(train) != len(self.train_subjects):
            raise ValueError("train_subjects contains duplicates")
        if (
            self.validation_subject in train
            or self.test_subject in train
            or self.validation_subject == self.test_subject
        ):
            raise ValueError("subject split overlaps across train/validation/test")


def compute_phase_class_weights(
    phase_targets: Sequence[int], *, beta: float = 0.9999
) -> tuple[float, float, float, float]:
    """Effective-number weights normalized to mean one."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    counts = [sum(int(target == phase) for target in phase_targets) for phase in range(4)]
    if any(count == 0 for count in counts):
        raise ValueError(f"all four eye phases must occur in training data; counts={counts}")
    raw = [(1.0 - beta) / (1.0 - beta**count) for count in counts]
    scale = 4.0 / sum(raw)
    return tuple(weight * scale for weight in raw)  # type: ignore[return-value]


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def train_eye_epoch(
    model: nn.Module,
    criterion: EyeMultiTaskLoss,
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
    totals = {"loss": 0.0, "phase_loss": 0.0, "closedness_loss": 0.0, "moving_loss": 0.0}
    batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["eyes"], batch["visibility"])
            losses = criterion(output, batch)
        if use_amp:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        totals["loss"] += float(losses.total.detach())
        totals["phase_loss"] += float(losses.phase.detach())
        totals["closedness_loss"] += float(losses.closedness.detach())
        totals["moving_loss"] += float(losses.moving.detach())
        batches += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def evaluate_eye(
    model: nn.Module,
    criterion: EyeMultiTaskLoss,
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
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["eyes"], batch["visibility"])
            breakdown = criterion(output, batch)
        losses.append(float(breakdown.total))
        valid = batch["valid_mask"].bool()
        targets.extend(batch["phase"][valid].cpu().tolist())
        predictions.extend(output.phase_logits.argmax(dim=-1)[valid].cpu().tolist())
    if not losses:
        raise ValueError("evaluation loader produced no batches")
    metrics = phase_classification_metrics(targets, predictions)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


def save_eye_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: EyeTemporalConfig,
    split: SubjectSplitMetadata,
    epoch: int,
    metrics: Mapping[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": type(model).__name__,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "split": asdict(split),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_eye_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {
        key: payload[key]
        for key in ("model_type", "config", "split", "epoch", "metrics")
    }
