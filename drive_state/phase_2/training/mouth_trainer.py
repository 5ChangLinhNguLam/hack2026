"""Training, evaluation, and checkpointing for the visual mouth specialist."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from ..data.primitive_labels import IGNORE_INDEX
from ..losses.yawn_loss import YawnHierarchicalLoss
from ..models.mouth_visual import MouthVisualOutput
from .eye_trainer import SubjectSplitMetadata


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def _classification_summary(
    targets: list[int], predictions: list[int], classes: int
) -> dict[str, float | int]:
    if not targets:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0}
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target, prediction] += 1
    f1: list[float] = []
    for index in range(classes):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "support": len(targets),
        "accuracy": float(confusion.diagonal().sum()) / len(targets),
        "macro_f1": sum(f1) / len(f1),
    }


def train_mouth_epoch(
    model: nn.Module,
    criterion: YawnHierarchicalLoss,
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
    totals = {
        "loss": 0.0,
        "direct_loss": 0.0,
        "derived_loss": 0.0,
        "consistency_loss": 0.0,
        "type_loss": 0.0,
    }
    batches = 0
    support = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["mouth"])
            breakdown = criterion(output, batch)
        if use_amp:
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        totals["loss"] += float(breakdown.total.detach())
        totals["direct_loss"] += float(breakdown.direct.detach())
        totals["derived_loss"] += float(breakdown.derived.detach())
        totals["consistency_loss"] += float(breakdown.consistency.detach())
        totals["type_loss"] += float(breakdown.type.detach())
        support += breakdown.support
        batches += 1
    if batches == 0:
        raise ValueError("mouth training loader produced no batches")
    result = {name: value / batches for name, value in totals.items()}
    result["support"] = float(support)
    return result


@torch.inference_mode()
def evaluate_mouth(
    model: nn.Module,
    criterion: YawnHierarchicalLoss,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    amp: bool = True,
) -> dict[str, object]:
    model.eval()
    use_amp = amp and device.type == "cuda"
    losses: list[float] = []
    binary_targets: list[int] = []
    binary_predictions: list[int] = []
    derived_predictions: list[int] = []
    type_targets: list[int] = []
    type_predictions: list[int] = []
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output: MouthVisualOutput = model(batch["mouth"])
            breakdown = criterion(output, batch)
        losses.append(float(breakdown.total))
        mask = batch["yawn"].ne(IGNORE_INDEX) & batch["mouth_visibility"].ge(
            criterion.visibility_threshold
        )
        target = batch["yawn"][mask]
        binary_targets.extend(target.gt(0).long().cpu().tolist())
        binary_predictions.extend(
            output.binary_logits[mask].ge(0.0).long().cpu().tolist()
        )
        derived_probability = output.type_logits.float().softmax(dim=-1)[:, 1:].sum(-1)
        derived_predictions.extend(
            derived_probability[mask].ge(0.5).long().cpu().tolist()
        )
        type_targets.extend(target.cpu().tolist())
        type_predictions.extend(output.type_logits[mask].argmax(dim=-1).cpu().tolist())
    if not losses:
        raise ValueError("mouth evaluation loader produced no batches")
    return {
        "loss": sum(losses) / len(losses),
        "binary": _classification_summary(binary_targets, binary_predictions, 2),
        "derived_binary": _classification_summary(
            binary_targets, derived_predictions, 2
        ),
        "type": _classification_summary(type_targets, type_predictions, 3),
    }


def save_mouth_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    split: SubjectSplitMetadata,
    epoch: int,
    metrics: Mapping[str, object],
    model_config: Mapping[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model_type": type(model).__name__,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "split": asdict(split),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "model_config": dict(model_config),
        },
        temporary,
    )
    os.replace(temporary, path)
