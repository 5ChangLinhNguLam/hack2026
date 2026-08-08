"""Training and checkpoint helpers for MobileNetV3-Large plus causal LSTM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.five_state_labels import IGNORE_INDEX
from ..data.four_state_labels import FourState, compose_five_state_probabilities
from ..metrics_events import five_state_event_metrics
from ..models.mobilenet_lstm import (
    CausalFourStateLSTM,
    CausalFiveStateLSTM,
    VisualFrameOutput,
    assemble_temporal_features,
)
from .eye_trainer import SubjectSplitMetadata
from .five_state_trainer import five_state_metrics

VISUAL_CHECKPOINT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class EvidenceLossWeights:
    eye: float = 1.0
    visibility: float = 0.25
    yawn: float = 1.0
    distraction: float = 1.0
    pose: float = 0.25
    consistency: float = 0.10

    def __post_init__(self) -> None:
        if any(
            value < 0.0
            for value in (
                self.eye,
                self.visibility,
                self.yawn,
                self.distraction,
                self.pose,
                self.consistency,
            )
        ):
            raise ValueError("evidence loss weights must be non-negative")


DEFAULT_EVIDENCE_WEIGHTS = EvidenceLossWeights()


class VisualEvidenceLoss(NamedTuple):
    total: Tensor
    eye_aperture: Tensor
    eye_visibility: Tensor
    yawn: Tensor
    distraction: Tensor
    pose: Tensor
    consistency: Tensor


def _weighted_mean(
    values: Tensor,
    mask: Tensor,
    sample_weight: Tensor,
) -> Tensor:
    valid = mask.to(dtype=torch.bool)
    if not valid.any():
        return values.sum() * 0.0
    weights = sample_weight[valid].to(dtype=values.dtype)
    denominator = valid.sum().to(dtype=values.dtype)
    return (values[valid] * weights).sum() / denominator


def _masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    sample_weight: Tensor,
) -> Tensor:
    safe_targets = targets.masked_fill(~mask.to(dtype=torch.bool), 0)
    losses = F.cross_entropy(logits, safe_targets, reduction="none")
    return _weighted_mean(losses, mask, sample_weight)


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sample_weight: Tensor,
) -> Tensor:
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    losses = losses.flatten(1).mean(dim=1)
    return _weighted_mean(losses, mask, sample_weight)


def visual_evidence_loss(
    output: VisualFrameOutput,
    batch: Mapping[str, Tensor],
    *,
    weights: EvidenceLossWeights = DEFAULT_EVIDENCE_WEIGHTS,
) -> VisualEvidenceLoss:
    """Train observable evidence only; final five states remain temporal."""

    sample_weight = batch.get(
        "supervision_weight",
        output.embedding.new_ones(output.embedding.shape[0]),
    )
    eye_targets = batch["eye_aperture_target"]
    eye_aperture = _masked_cross_entropy(
        output.eye_logits,
        eye_targets,
        eye_targets != IGNORE_INDEX,
        sample_weight,
    )
    visibility_losses = F.binary_cross_entropy_with_logits(
        output.eye_visibility_logits,
        batch["region_visibility"][:, 1:3].to(
            dtype=output.eye_visibility_logits.dtype
        ),
        reduction="none",
    )
    eye_visibility = _weighted_mean(
        visibility_losses.mean(dim=1),
        torch.ones_like(sample_weight, dtype=torch.bool),
        sample_weight,
    )
    yawn = _masked_cross_entropy(
        output.yawn_logits,
        batch["yawn_target"],
        batch["yawn_mask"],
        sample_weight,
    )
    distraction = _masked_cross_entropy(
        output.distraction_logits,
        batch["distraction_target"],
        batch["distraction_mask"],
        sample_weight,
    )
    face_valid = batch.get(
        "pose_mask", batch["region_visibility"][:, 0] > 0.0
    )
    pose = _masked_smooth_l1(
        output.pose,
        batch["head_pose_target"].to(dtype=output.pose.dtype).div(90.0),
        face_valid,
        sample_weight,
    )
    teacher_embedding = batch.get("teacher_embedding")
    consistency_mask = batch.get("consistency_mask")
    if (
        teacher_embedding is None
        or consistency_mask is None
        or not consistency_mask.to(dtype=torch.bool).any()
    ):
        consistency = output.embedding.sum() * 0.0
    else:
        if teacher_embedding.shape != output.embedding.shape:
            raise ValueError(
                "teacher embedding shape must match the visual embedding shape"
            )
        consistency_losses = 1.0 - F.cosine_similarity(
            output.embedding,
            teacher_embedding.detach().to(dtype=output.embedding.dtype),
            dim=1,
        )
        consistency = _weighted_mean(
            consistency_losses,
            consistency_mask,
            sample_weight,
        )
    total = (
        weights.eye * eye_aperture
        + weights.visibility * eye_visibility
        + weights.yawn * yawn
        + weights.distraction * distraction
        + weights.pose * pose
        + weights.consistency * consistency
    )
    return VisualEvidenceLoss(
        total=total,
        eye_aperture=eye_aperture,
        eye_visibility=eye_visibility,
        yawn=yawn,
        distraction=distraction,
        pose=pose,
        consistency=consistency,
    )


def _tensor_batch(
    batch: Mapping[str, object],
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def train_visual_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_weights: EvidenceLossWeights = DEFAULT_EVIDENCE_WEIGHTS,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
    freeze_batch_norm: bool = False,
) -> dict[str, float]:
    """Train all visual encoder parameters for one frame-pretraining epoch."""

    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    use_amp = amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    totals = {
        "loss": 0.0,
        "eye_aperture_loss": 0.0,
        "eye_visibility_loss": 0.0,
        "yawn_loss": 0.0,
        "distraction_loss": 0.0,
        "pose_loss": 0.0,
        "consistency_loss": 0.0,
    }
    batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(
                batch["image"],
                batch["region_boxes"],
                batch["region_visibility"],
            )
            losses = visual_evidence_loss(
                output,
                batch,
                weights=loss_weights,
            )
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
        totals["eye_aperture_loss"] += float(
            losses.eye_aperture.detach()
        )
        totals["eye_visibility_loss"] += float(
            losses.eye_visibility.detach()
        )
        totals["yawn_loss"] += float(losses.yawn.detach())
        totals["distraction_loss"] += float(losses.distraction.detach())
        totals["pose_loss"] += float(losses.pose.detach())
        totals["consistency_loss"] += float(losses.consistency.detach())
        batches += 1
    if batches == 0:
        raise ValueError("visual training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def temporal_five_state_loss(
    model: CausalFiveStateLSTM,
    batch: Mapping[str, Tensor],
    *,
    class_weights: Tensor | None = None,
    final_loss_weight: float = 1.0,
) -> Tensor:
    """Train histories while explicitly protecting each sampled final state."""

    if final_loss_weight < 0.0:
        raise ValueError("final loss weight must be non-negative")

    features = assemble_temporal_features(
        visual_embedding=batch["visual_embedding"],
        evidence=batch["evidence"],
        context_valid=batch["context_valid"],
    )
    logits = model(features).logits
    history_loss = _temporal_loss_from_logits(
        logits,
        batch,
        class_weights=class_weights,
    )
    if final_loss_weight == 0.0:
        return history_loss
    final_loss = _temporal_loss_from_logits(
        logits[:, -1],
        batch,
        class_weights=class_weights,
    )
    return history_loss + final_loss_weight * final_loss


def effective_number_weights(
    event_counts: Tensor,
    *,
    beta: float = 0.999,
) -> Tensor:
    """Return mean-one class weights from independent event counts."""

    if event_counts.ndim != 1 or event_counts.numel() != 5:
        raise ValueError("event counts must contain exactly five classes")
    if not 0.0 <= beta < 1.0:
        raise ValueError("effective-number beta must be in [0, 1)")
    counts = event_counts.to(dtype=torch.float64)
    if torch.any(counts <= 0.0):
        raise ValueError("event counts must be positive")
    if beta == 0.0:
        return torch.ones_like(counts, dtype=torch.float32)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return (weights / weights.mean()).to(dtype=torch.float32)


def effective_four_state_weights(
    event_counts: Tensor,
    *,
    beta: float = 0.999,
) -> Tensor:
    """Return mean-one weights for the four learned non-event states."""

    if event_counts.ndim != 1 or event_counts.numel() != 4:
        raise ValueError("event counts must contain exactly four classes")
    if not 0.0 <= beta < 1.0:
        raise ValueError("effective-number beta must be in [0, 1)")
    counts = event_counts.to(dtype=torch.float64)
    if torch.any(counts <= 0.0):
        raise ValueError("event counts must be positive")
    if beta == 0.0:
        return torch.ones_like(counts, dtype=torch.float32)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return (weights / weights.mean()).to(dtype=torch.float32)


def _four_state_loss_from_logits(
    logits: Tensor,
    batch: Mapping[str, Tensor],
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    if logits.ndim not in (2, 3) or logits.shape[-1] != 4:
        raise ValueError(
            "four-state logits must have shape [batch, 4] "
            "or [batch, time, 4]"
        )
    if logits.ndim == 3:
        targets = batch["four_state_targets"]
        confidence = batch["confidence_sequence"]
        if targets.shape != logits.shape[:2] or confidence.shape != targets.shape:
            raise ValueError("four-state target histories must align with logits")
    else:
        targets = batch["four_state_target"]
        confidence = batch["confidence"]
    valid = targets.ne(IGNORE_INDEX)
    if not valid.any():
        return logits.sum() * 0.0
    weights = confidence[valid].to(dtype=logits.dtype).clamp_min(0.0)
    class_weight = (
        None
        if class_weights is None
        else class_weights.to(device=logits.device, dtype=logits.dtype)
    )
    per_item = F.cross_entropy(
        logits[valid],
        targets[valid],
        reduction="none",
        weight=class_weight,
    )
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-8)


def temporal_four_state_loss(
    model: CausalFourStateLSTM,
    batch: Mapping[str, Tensor],
    *,
    class_weights: Tensor | None = None,
    final_loss_weight: float = 1.0,
) -> Tensor:
    """Train non-microsleep histories while masking event-active frames."""

    _, loss = _temporal_four_state_logits_and_loss(
        model,
        batch,
        class_weights=class_weights,
        final_loss_weight=final_loss_weight,
    )
    return loss


def _temporal_four_state_logits_and_loss(
    model: CausalFourStateLSTM,
    batch: Mapping[str, Tensor],
    *,
    class_weights: Tensor | None = None,
    final_loss_weight: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return causal sequence logits with their masked training loss."""

    if final_loss_weight < 0.0:
        raise ValueError("final loss weight must be non-negative")
    features = assemble_temporal_features(
        visual_embedding=batch["visual_embedding"],
        evidence=batch["evidence"],
        context_valid=batch["context_valid"],
        ocular_phase_probabilities=batch["ocular_phase_probabilities"],
    )
    logits = model(features).logits
    history = _four_state_loss_from_logits(
        logits,
        batch,
        class_weights=class_weights,
    )
    if final_loss_weight == 0.0:
        return logits, history
    final = _four_state_loss_from_logits(
        logits[:, -1],
        batch,
        class_weights=class_weights,
    )
    return logits, history + final_loss_weight * final


def train_four_state_epoch(
    model: CausalFourStateLSTM,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
    class_weights: Tensor | None = None,
    final_loss_weight: float = 1.0,
) -> dict[str, float]:
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")
    model.train()
    use_amp = amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    total = 0.0
    batches = 0
    correct = 0
    support = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, loss = _temporal_four_state_logits_and_loss(
                model,
                batch,
                class_weights=class_weights,
                final_loss_weight=final_loss_weight,
            )
        final_targets = batch["four_state_target"]
        valid_final = final_targets != IGNORE_INDEX
        if valid_final.any():
            final_predictions = logits[:, -1].detach().argmax(dim=-1)
            correct += int(
                (final_predictions[valid_final] == final_targets[valid_final])
                .sum()
                .item()
            )
            support += int(valid_final.sum().item())
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
        raise ValueError("four-state training loader produced no batches")
    return {
        "loss": total / batches,
        "accuracy": correct / support if support else 0.0,
        "support": support,
    }


def _temporal_loss_from_logits(
    logits: Tensor,
    batch: Mapping[str, Tensor],
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Compute the final-timestamp loss from an already evaluated sequence."""

    if logits.ndim not in (2, 3) or logits.shape[-1] != 5:
        raise ValueError(
            "temporal logits must have shape [batch, 5] "
            "or [batch, time, 5]"
        )
    if logits.ndim == 3:
        targets = batch["five_state_targets"]
        confidence = batch["confidence_sequence"]
        if targets.shape != logits.shape[:2] or confidence.shape != targets.shape:
            raise ValueError("temporal target histories must align with logits")
    else:
        targets = batch["five_state_target"]
        confidence = batch["confidence"]
    valid = targets != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    weights = confidence[valid].to(dtype=logits.dtype).clamp_min(0.0)
    weight = (
        None
        if class_weights is None
        else class_weights.to(device=logits.device, dtype=logits.dtype)
    )
    per_item = F.cross_entropy(
        logits[valid],
        targets[valid],
        reduction="none",
        weight=weight,
    )
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-8)


def train_temporal_epoch(
    model: CausalFiveStateLSTM,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
    class_weights: Tensor | None = None,
    final_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Train the causal LSTM on final targets from cached evidence windows."""

    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")
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
            loss = temporal_five_state_loss(
                model,
                batch,
                class_weights=class_weights,
                final_loss_weight=final_loss_weight,
            )
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
        raise ValueError("temporal training loader produced no batches")
    return {"loss": total / batches}


def _classification_metrics(
    targets: list[int],
    predictions: list[int],
    *,
    classes: int,
) -> dict[str, object]:
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target][prediction] += 1
    f1: list[float] = []
    for index in range(classes):
        true_positive = confusion[index][index]
        predicted = sum(row[index] for row in confusion)
        support = sum(confusion[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    support = len(targets)
    return {
        "support": support,
        "accuracy": (
            sum(confusion[index][index] for index in range(classes)) / support
            if support
            else 0.0
        ),
        "macro_f1": sum(f1) / classes,
        "confusion_matrix": confusion,
    }


def temporal_selection_score(
    metrics: Mapping[str, object],
    *,
    microsleep_weight: float = 0.5,
) -> float:
    """Prefer five-state checkpoints that do not discard microsleep."""

    if microsleep_weight < 0.0:
        raise ValueError("microsleep weight must be non-negative")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        raise ValueError("temporal metrics are missing per-class results")
    microsleep = per_class.get("microsleep")
    if not isinstance(microsleep, Mapping):
        raise ValueError("temporal metrics are missing microsleep results")
    return float(metrics["macro_f1"]) + microsleep_weight * float(
        microsleep["f1"]
    )


def fused_temporal_selection_score(
    metrics: Mapping[str, object],
    *,
    microsleep_event_weight: float = 0.5,
    microsleep_false_rate_weight: float = 0.1,
) -> float:
    """Score a fused checkpoint without treating frame support as event support."""

    if microsleep_event_weight < 0.0 or microsleep_false_rate_weight < 0.0:
        raise ValueError("fused selection weights must be non-negative")
    score = float(metrics["macro_f1"])
    events = metrics.get("events")
    if not isinstance(events, Mapping):
        return score
    per_class = events.get("per_class")
    if not isinstance(per_class, Mapping):
        return score
    microsleep = per_class.get("microsleep")
    if not isinstance(microsleep, Mapping) or int(microsleep.get("events", 0)) == 0:
        return score
    raw_precision = microsleep.get("event_precision")
    raw_recall = microsleep.get("event_recall")
    precision = 0.0 if raw_precision is None else float(raw_precision)
    recall = 0.0 if raw_recall is None else float(raw_recall)
    event_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    false_rate = float(microsleep.get("false_events_per_minute", 0.0))
    return (
        score
        + microsleep_event_weight * event_f1
        - microsleep_false_rate_weight * false_rate
    )


@torch.inference_mode()
def evaluate_visual_frames(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    loss_weights: EvidenceLossWeights = DEFAULT_EVIDENCE_WEIGHTS,
    amp: bool = True,
) -> dict[str, object]:
    """Evaluate the representation-pretraining heads at natural frequency."""

    model.eval()
    use_amp = amp and device.type == "cuda"
    losses: list[float] = []
    eye_targets: list[int] = []
    eye_predictions: list[int] = []
    yawn_targets: list[int] = []
    yawn_predictions: list[int] = []
    distraction_targets: list[int] = []
    distraction_predictions: list[int] = []
    pose_errors: list[float] = []
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(
                batch["image"],
                batch["region_boxes"],
                batch["region_visibility"],
            )
            loss = visual_evidence_loss(
                output,
                batch,
                weights=loss_weights,
            )
        losses.append(float(loss.total))
        eye_batch_predictions = output.eye_logits.argmax(dim=-1)
        for target, prediction in zip(
            batch["eye_aperture_target"].tolist(),
            eye_batch_predictions.tolist(),
            strict=True,
        ):
            if target != IGNORE_INDEX:
                eye_targets.append(int(target))
                eye_predictions.append(int(prediction))
        yawn_mask = batch["yawn_mask"].to(dtype=torch.bool)
        yawn_targets.extend(
            int(value) for value in batch["yawn_target"][yawn_mask].tolist()
        )
        yawn_predictions.extend(
            int(value)
            for value in output.yawn_logits.argmax(dim=-1)[yawn_mask].tolist()
        )
        distraction_mask = batch["distraction_mask"].to(dtype=torch.bool)
        distraction_targets.extend(
            int(value)
            for value in batch["distraction_target"][
                distraction_mask
            ].tolist()
        )
        distraction_predictions.extend(
            int(value)
            for value in output.distraction_logits.argmax(dim=-1)[
                distraction_mask
            ].tolist()
        )
        face_valid = batch["region_visibility"][:, 0] > 0.0
        if face_valid.any():
            errors = (
                output.pose[face_valid] * 90.0
                - batch["head_pose_target"][face_valid]
            ).abs()
            pose_errors.extend(float(value) for value in errors.flatten())
    eye_metrics = _classification_metrics(
        eye_targets,
        eye_predictions,
        classes=3,
    )
    return {
        "support": eye_metrics["support"],
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "eye_aperture": eye_metrics,
        "yawn": _classification_metrics(
            yawn_targets,
            yawn_predictions,
            classes=2,
        ),
        "distraction": _classification_metrics(
            distraction_targets,
            distraction_predictions,
            classes=2,
        ),
        "pose_mae": (
            sum(pose_errors) / len(pose_errors)
            if pose_errors
            else 0.0
        ),
    }


@torch.inference_mode()
def evaluate_temporal(
    model: CausalFiveStateLSTM,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    amp: bool = True,
) -> dict[str, object]:
    """Evaluate every natural target once and distinguish absent classes."""

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
            features = assemble_temporal_features(
                visual_embedding=batch["visual_embedding"],
                evidence=batch["evidence"],
                context_valid=batch["context_valid"],
            )
            output = model(features)
            loss = _temporal_loss_from_logits(output.logits, batch)
        losses.append(float(loss))
        probabilities = output.logits[:, -1].float().softmax(dim=-1).cpu()
        batch_targets = batch["five_state_target"].cpu()
        sessions = [str(value) for value in raw_batch["session"]]
        event_ids = [
            str(value)
            for value in raw_batch.get(
                "event_id",
                [""] * len(sessions),
            )
        ]
        frame_ids = raw_batch["target_frame_id"].tolist()
        timestamps = raw_batch["target_timestamp"].tolist()
        for index, session in enumerate(sessions):
            identity = (session, int(frame_ids[index]))
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
                    "session": session,
                    "event_id": event_ids[index],
                    "frame_id": int(frame_ids[index]),
                    "timestamp": float(timestamps[index]),
                    "target": target,
                    "prediction": prediction,
                    "probabilities": probabilities[index].tolist(),
                }
            )
    metrics = five_state_metrics(targets=targets, predictions=predictions)
    per_class = metrics["per_class"]
    assert isinstance(per_class, Mapping)
    supported = [
        float(value["f1"])
        for value in per_class.values()
        if isinstance(value, Mapping) and int(value["support"]) > 0
    ]
    metrics["supported_classes"] = len(supported)
    metrics["supported_macro_f1"] = (
        sum(supported) / len(supported) if supported else 0.0
    )
    metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
    metrics["rows"] = rows
    return metrics


@torch.inference_mode()
def evaluate_fused_temporal(
    model: CausalFourStateLSTM,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    amp: bool = True,
    fps: float | None = None,
) -> dict[str, object]:
    """Evaluate the four-state head after the causal microsleep override."""

    if fps is not None and (not math.isfinite(fps) or fps <= 0.0):
        raise ValueError("fused evaluation FPS must be positive and finite")
    model.eval()
    use_amp = amp and device.type == "cuda"
    losses: list[float] = []
    five_targets: list[int] = []
    five_predictions: list[int] = []
    four_targets: list[int] = []
    four_predictions: list[int] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            features = assemble_temporal_features(
                visual_embedding=batch["visual_embedding"],
                evidence=batch["evidence"],
                context_valid=batch["context_valid"],
                ocular_phase_probabilities=batch[
                    "ocular_phase_probabilities"
                ],
            )
            output = model(features)
            loss = _four_state_loss_from_logits(output.logits, batch)
        losses.append(float(loss))
        four_probabilities = output.logits[:, -1].float().softmax(dim=-1)
        fused_probabilities = compose_five_state_probabilities(
            four_probabilities,
            batch["microsleep_active"].to(dtype=torch.bool),
            drowsy_active=batch["drowsy_active"].to(dtype=torch.bool),
        )
        four_probabilities = four_probabilities.cpu()
        fused_probabilities = fused_probabilities.cpu()
        batch_five_targets = batch["five_state_target"].cpu()
        batch_four_targets = batch["four_state_target"].cpu()
        microsleep_active = batch["microsleep_active"].to(
            dtype=torch.bool
        ).cpu()
        drowsy_active = batch["drowsy_active"].to(dtype=torch.bool).cpu()
        sessions = [str(value) for value in raw_batch["session"]]
        event_ids = [
            str(value)
            for value in raw_batch.get("event_id", [""] * len(sessions))
        ]
        frame_ids = raw_batch["target_frame_id"].tolist()
        timestamps = raw_batch["target_timestamp"].tolist()
        for index, session in enumerate(sessions):
            frame_id = int(frame_ids[index])
            identity = (session, frame_id)
            if identity in seen:
                raise ValueError(
                    f"natural evaluation reference repeated: {identity}"
                )
            seen.add(identity)
            five_target = int(batch_five_targets[index])
            four_target = int(batch_four_targets[index])
            five_prediction = int(fused_probabilities[index].argmax())
            four_prediction = int(four_probabilities[index].argmax())
            if five_target != IGNORE_INDEX:
                five_targets.append(five_target)
                five_predictions.append(five_prediction)
            if four_target != IGNORE_INDEX:
                four_targets.append(four_target)
                four_predictions.append(four_prediction)
            rows.append(
                {
                    "session": session,
                    "event_id": event_ids[index],
                    "frame_id": frame_id,
                    "timestamp": float(timestamps[index]),
                    "target": five_target,
                    "four_state_target": four_target,
                    "prediction": five_prediction,
                    "four_state_prediction": four_prediction,
                    "microsleep_active": bool(microsleep_active[index]),
                    "drowsy_active": bool(drowsy_active[index]),
                    "four_probabilities": four_probabilities[index].tolist(),
                    "probabilities": fused_probabilities[index].tolist(),
                }
            )

    metrics = five_state_metrics(
        targets=five_targets,
        predictions=five_predictions,
    )
    per_class = metrics["per_class"]
    assert isinstance(per_class, Mapping)
    supported = [
        float(value["f1"])
        for value in per_class.values()
        if isinstance(value, Mapping) and int(value["support"]) > 0
    ]
    metrics["supported_classes"] = len(supported)
    metrics["supported_macro_f1"] = (
        sum(supported) / len(supported) if supported else 0.0
    )
    four_metrics = _classification_metrics(
        four_targets,
        four_predictions,
        classes=len(FourState),
    )
    four_metrics["class_names"] = [state.name.lower() for state in FourState]
    metrics["four_state"] = four_metrics
    metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
    metrics["rows"] = rows
    if fps is not None:
        metrics["events"] = five_state_event_metrics(
            targets=[int(row["target"]) for row in rows],
            predictions=[int(row["prediction"]) for row in rows],
            event_ids=[str(row["event_id"]) for row in rows],
            session_ids=[str(row["session"]) for row in rows],
            frame_ids=[int(row["frame_id"]) for row in rows],
            fps=fps,
        )
        event_block = metrics["events"]
        assert isinstance(event_block, Mapping)
        event_per_class = event_block["per_class"]
        assert isinstance(event_per_class, Mapping)
        microsleep = event_per_class["microsleep"]
        assert isinstance(microsleep, Mapping)
        metrics["microsleep_selection_supported"] = (
            int(microsleep["events"]) > 0
        )
    else:
        metrics["microsleep_selection_supported"] = False
    return metrics


def save_visual_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: Mapping[str, object],
    split: SubjectSplitMetadata,
    standardized_schema_version: int,
    target_fps: float,
    label_config: Mapping[str, object],
    epoch: int,
    metrics: Mapping[str, object],
) -> None:
    if standardized_schema_version <= 0 or target_fps <= 0.0 or epoch < 0:
        raise ValueError("visual checkpoint metadata is invalid")
    payload = {
        "schema_version": VISUAL_CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": dict(model_config),
        "split": asdict(split),
        "standardized_schema_version": int(standardized_schema_version),
        "target_fps": float(target_fps),
        "label_config": dict(label_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_visual_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_model_config: Mapping[str, object],
    expected_split: SubjectSplitMetadata,
    expected_standardized_schema_version: int,
    expected_target_fps: float,
    expected_label_config: Mapping[str, object],
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[str, object]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    expected = {
        "schema_version": VISUAL_CHECKPOINT_SCHEMA_VERSION,
        "model_config": dict(expected_model_config),
        "split": asdict(expected_split),
        "standardized_schema_version": int(
            expected_standardized_schema_version
        ),
        "label_config": dict(expected_label_config),
    }
    mismatched = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if not math.isclose(
        float(payload.get("target_fps", 0.0)),
        float(expected_target_fps),
    ):
        mismatched.append("target_fps")
    if mismatched:
        raise ValueError(
            f"visual resume checkpoint mismatch: {', '.join(mismatched)}"
        )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"]) + 1, payload


__all__ = [
    "DEFAULT_EVIDENCE_WEIGHTS",
    "EvidenceLossWeights",
    "VISUAL_CHECKPOINT_SCHEMA_VERSION",
    "VisualEvidenceLoss",
    "effective_four_state_weights",
    "effective_number_weights",
    "evaluate_fused_temporal",
    "evaluate_temporal",
    "evaluate_visual_frames",
    "restore_visual_checkpoint",
    "save_visual_checkpoint",
    "fused_temporal_selection_score",
    "temporal_four_state_loss",
    "temporal_five_state_loss",
    "temporal_selection_score",
    "train_temporal_epoch",
    "train_four_state_epoch",
    "train_visual_epoch",
    "visual_evidence_loss",
]
