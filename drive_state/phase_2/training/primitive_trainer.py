"""Training/evaluation helpers for cabin and face primitive heads."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from ..data.primitive_labels import IGNORE_INDEX, TASK_CLASS_COUNTS
from ..distraction import (
    action_distraction_probability,
    pool_distraction_probabilities,
)
from ..losses.primitive_loss import MaskedPrimitiveLoss
from ..models.spatial_multitask import FACE_TASKS
from .eye_trainer import SubjectSplitMetadata


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def train_primitive_epoch(
    model: nn.Module,
    criterion: MaskedPrimitiveLoss,
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
    totals = {"loss": 0.0, **{f"{task}_loss": 0.0 for task in TASK_CLASS_COUNTS}}
    batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                batch["cabin"],
                batch["face"],
                batch["face_visibility"],
                batch["head_pose"],
            )
            losses = criterion(outputs, batch)
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
        for task, loss in losses.per_task.items():
            key = f"{task}_loss"
            totals.setdefault(key, 0.0)
            totals[key] += float(loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("primitive training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def _classification_summary(
    targets: list[int], predictions: list[int], classes: int
) -> dict[str, float | int]:
    if not targets:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0}
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target][prediction] += 1
    f1: list[float] = []
    for index in range(classes):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "support": len(targets),
        "accuracy": sum(confusion[index][index] for index in range(classes)) / len(targets),
        "macro_f1": sum(f1) / len(f1),
    }


FINAL_STATE_SELECTION_WEIGHTS = {
    "fused_distraction": 0.50,
    "road_gaze": 0.20,
    "yawn": 0.30,
}


def final_state_priority_score(metrics: Mapping[str, object]) -> float:
    derived = metrics["derived"]
    tasks = metrics["tasks"]
    if not isinstance(derived, Mapping) or not isinstance(tasks, Mapping):
        raise ValueError("primitive metrics must contain task and derived mappings")
    components = {
        "fused_distraction": derived["fused_distraction"],
        "road_gaze": tasks["road_gaze"],
        "yawn": tasks["yawn"],
    }
    weighted_score = 0.0
    supported_weight = 0.0
    for name, weight in FINAL_STATE_SELECTION_WEIGHTS.items():
        metric = components[name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"metric {name} must be a mapping")
        if int(metric["support"]) > 0:
            weighted_score += weight * float(metric["macro_f1"])
            supported_weight += weight
    return weighted_score / supported_weight if supported_weight else 0.0


@torch.inference_mode()
def evaluate_primitives(
    model: nn.Module,
    criterion: MaskedPrimitiveLoss,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    amp: bool = True,
) -> dict[str, object]:
    model.eval()
    use_amp = amp and device.type == "cuda"
    losses: list[float] = []
    targets = {task: [] for task in TASK_CLASS_COUNTS}
    predictions = {task: [] for task in TASK_CLASS_COUNTS}
    derived_targets = {
        "action_distraction": [],
        "fused_distraction": [],
    }
    derived_predictions = {
        "action_distraction": [],
        "fused_distraction": [],
    }
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                batch["cabin"],
                batch["face"],
                batch["face_visibility"],
                batch["head_pose"],
            )
            losses.append(float(criterion(outputs, batch).total))
        for task in TASK_CLASS_COUNTS:
            mask = batch[task].ne(IGNORE_INDEX)
            if task in FACE_TASKS:
                mask = mask & batch["face_visibility"].ge(
                    criterion.face_visibility_threshold
                )
            targets[task].extend(batch[task][mask].cpu().tolist())
            predictions[task].extend(outputs[task].argmax(dim=1)[mask].cpu().tolist())
        hierarchical_mask = batch["distraction"].ne(IGNORE_INDEX) & batch[
            "driver_action"
        ].ne(IGNORE_INDEX)
        action_probability = action_distraction_probability(
            outputs["driver_action"].float()
        )
        direct_probability = outputs["distraction"].float().softmax(dim=-1)[:, 1]
        off_road_probability = outputs["road_gaze"].float().softmax(dim=-1)[:, 1]
        fused_probability = pool_distraction_probabilities(
            direct_probability, action_probability, off_road_probability
        )
        assert isinstance(fused_probability, Tensor)
        binary_targets = batch["distraction"][hierarchical_mask].cpu().tolist()
        derived_targets["action_distraction"].extend(binary_targets)
        derived_targets["fused_distraction"].extend(binary_targets)
        derived_predictions["action_distraction"].extend(
            action_probability[hierarchical_mask].ge(0.5).long().cpu().tolist()
        )
        derived_predictions["fused_distraction"].extend(
            fused_probability[hierarchical_mask].ge(0.5).long().cpu().tolist()
        )
    if not losses:
        raise ValueError("primitive evaluation loader produced no batches")
    task_metrics = {
        task: _classification_summary(
            targets[task], predictions[task], TASK_CLASS_COUNTS[task]
        )
        for task in TASK_CLASS_COUNTS
    }
    derived_metrics = {
        name: _classification_summary(
            derived_targets[name], derived_predictions[name], 2
        )
        for name in derived_targets
    }
    supported = [
        float(metric["macro_f1"])
        for metric in task_metrics.values()
        if int(metric["support"]) > 0
    ]
    metrics: dict[str, object] = {
        "loss": sum(losses) / len(losses),
        "mean_task_macro_f1": sum(supported) / len(supported) if supported else 0.0,
        "tasks": task_metrics,
        "derived": derived_metrics,
    }
    metrics["final_state_priority_score"] = final_state_priority_score(metrics)
    return metrics


def save_primitive_checkpoint(
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
