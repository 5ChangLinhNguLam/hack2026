"""Training and checkpoint utilities for the causal ocular LSTM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.five_state_labels import IGNORE_INDEX
from ..models.ocular_lstm import CausalOcularLSTM, OcularLSTMOutput
from .eye_trainer import SubjectSplitMetadata


OCULAR_CHECKPOINT_SCHEMA_VERSION = 2
PHASE_NAMES = ("open", "closing", "close", "opening")


@dataclass(frozen=True)
class OcularLossWeights:
    phase: float = 1.0
    closed: float = 0.5
    progress: float = 0.5
    reliability: float = 0.25
    transition: float = 0.1

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in asdict(self).values()
        ):
            raise ValueError("ocular loss weights must be finite and non-negative")


@dataclass(frozen=True)
class OcularInputAugmentation:
    """Bounded false-open corruption inside supervised long closures."""

    probability: float = 0.5
    minimum_closed_run_frames: int = 10
    anchor_closed_frames: int = 3
    maximum_block_frames: int = 15

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("ocular augmentation probability must be in [0, 1]")
        if min(
            self.minimum_closed_run_frames,
            self.anchor_closed_frames,
            self.maximum_block_frames,
        ) <= 0:
            raise ValueError("ocular augmentation frame counts must be positive")
        if self.anchor_closed_frames >= self.minimum_closed_run_frames:
            raise ValueError("ocular augmentation anchor must be shorter than a closed run")


def augment_ocular_features(
    features: Tensor,
    phase_targets: Tensor,
    context_valid: Tensor,
    *,
    region_dim: int,
    config: OcularInputAugmentation = OcularInputAugmentation(),
    generator: torch.Generator | None = None,
) -> Tensor:
    """Inject causal false-open head evidence without changing image embeddings."""

    if features.ndim != 3 or region_dim <= 0:
        raise ValueError("ocular features must have shape [batch, time, features]")
    if features.shape[-1] != 2 * region_dim + 5:
        raise ValueError("ocular feature dimension does not match the region dimension")
    if phase_targets.shape != features.shape[:2] or context_valid.shape != features.shape[:2]:
        raise ValueError("ocular augmentation targets must align with features")
    augmented = features.clone()
    raw_start = 2 * region_dim
    false_open = features.new_tensor((1.0, 0.0, 0.0))
    for batch_index in range(features.shape[0]):
        if float(torch.rand((), generator=generator)) >= config.probability:
            continue
        closed = (
            phase_targets[batch_index].eq(2)
            & context_valid[batch_index].gt(0.0)
        ).detach().cpu().tolist()
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate((*closed, False)):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if index - start >= config.minimum_closed_run_frames:
                    runs.append((start, index))
                start = None
        if not runs:
            continue
        run_index = int(torch.randint(len(runs), (), generator=generator))
        run_start, run_stop = runs[run_index]
        first_corruptible = run_start + config.anchor_closed_frames
        available = run_stop - first_corruptible
        maximum = min(config.maximum_block_frames, available)
        block_length = 1 + int(torch.randint(maximum, (), generator=generator))
        start_choices = available - block_length + 1
        block_start = first_corruptible + int(
            torch.randint(start_choices, (), generator=generator)
        )
        augmented[
            batch_index,
            block_start : block_start + block_length,
            raw_start : raw_start + 3,
        ] = false_open
    return augmented


class OcularLoss(NamedTuple):
    total: Tensor
    phase: Tensor
    closed: Tensor
    progress: Tensor
    reliability: Tensor
    transition: Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = mask.to(dtype=torch.bool)
    if not selected.any():
        return values.sum() * 0.0
    return values[selected].mean()


def ocular_loss(
    output: OcularLSTMOutput,
    batch: Mapping[str, Tensor],
    *,
    weights: OcularLossWeights = OcularLossWeights(),
    phase_weights: Tensor | None = None,
) -> OcularLoss:
    """Compute masked phase, closure, progress, reliability, and order losses."""

    logits = output.phase_logits.float()
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("ocular phase logits must have shape [batch, time, 4]")
    shape = logits.shape[:2]
    phase_targets = batch["phase_targets"].long()
    closed_targets = batch["closed_targets"].float()
    progress_targets = batch["progress_targets"].float()
    reliability_targets = batch["reliability_targets"].float()
    context_valid = batch["context_valid"].gt(0.0)
    arrays = (
        phase_targets,
        closed_targets,
        progress_targets,
        reliability_targets,
        context_valid,
        output.closed_logits,
        output.progress,
        output.reliability_logits,
    )
    if any(value.shape != shape for value in arrays):
        raise ValueError("ocular targets and outputs must align with phase logits")
    valid_phase = context_valid & phase_targets.ne(IGNORE_INDEX)
    safe_phase = phase_targets.masked_fill(~valid_phase, 0)
    phase_per_item = F.cross_entropy(
        logits.transpose(1, 2),
        safe_phase,
        reduction="none",
        weight=(
            None
            if phase_weights is None
            else phase_weights.to(device=logits.device, dtype=logits.dtype)
        ),
    )
    phase = _masked_mean(phase_per_item, valid_phase)
    closed_per_item = F.binary_cross_entropy_with_logits(
        output.closed_logits.float(),
        closed_targets,
        reduction="none",
    )
    closed = _masked_mean(closed_per_item, valid_phase)
    progress_per_item = F.smooth_l1_loss(
        output.progress.float(),
        progress_targets,
        reduction="none",
    )
    progress_mask = (
        valid_phase
        & phase_targets.eq(2)
        & reliability_targets.gt(0.0)
    )
    progress = _masked_mean(progress_per_item, progress_mask)
    reliability_per_item = F.binary_cross_entropy_with_logits(
        output.reliability_logits.float(),
        reliability_targets.clamp(0.0, 1.0),
        reduction="none",
    )
    reliability = _masked_mean(reliability_per_item, context_valid)
    probabilities = logits.softmax(dim=-1)
    forbidden = (
        probabilities[:, :-1, 0] * probabilities[:, 1:, 2]
        + probabilities[:, :-1, 2] * probabilities[:, 1:, 0]
    )
    pair_mask = valid_phase[:, :-1] & valid_phase[:, 1:]
    transition = _masked_mean(forbidden, pair_mask)
    total = (
        weights.phase * phase
        + weights.closed * closed
        + weights.progress * progress
        + weights.reliability * reliability
        + weights.transition * transition
    )
    return OcularLoss(
        total,
        phase,
        closed,
        progress,
        reliability,
        transition,
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


def train_ocular_epoch(
    model: CausalOcularLSTM,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_weights: OcularLossWeights = OcularLossWeights(),
    phase_weights: Tensor | None = None,
    input_augmentation: OcularInputAugmentation | None = None,
    input_region_dim: int | None = None,
    augmentation_seed: int = 0,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")
    if input_augmentation is not None and (
        input_region_dim is None or input_region_dim <= 0
    ):
        raise ValueError("ocular augmentation requires a positive region dimension")
    model.train()
    use_amp = amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    augmentation_generator = torch.Generator().manual_seed(augmentation_seed)
    totals = {name: 0.0 for name in OcularLoss._fields}
    batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        ocular_features = batch["ocular_features"]
        if input_augmentation is not None:
            ocular_features = augment_ocular_features(
                ocular_features,
                batch["phase_targets"],
                batch["context_valid"],
                region_dim=int(input_region_dim),
                config=input_augmentation,
                generator=augmentation_generator,
            )
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(ocular_features)
            losses = ocular_loss(
                output,
                batch,
                weights=loss_weights,
                phase_weights=phase_weights,
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
        for name, value in zip(OcularLoss._fields, losses, strict=True):
            totals[name] += float(value.detach())
        batches += 1
    if batches == 0:
        raise ValueError("ocular training loader produced no batches")
    return {
        ("loss" if name == "total" else f"{name}_loss"): value / batches
        for name, value in totals.items()
    }


def _classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    classes: int,
) -> dict[str, object]:
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target)][int(prediction)] += 1
    per_class: dict[int, dict[str, float | int]] = {}
    f1_values = []
    for index in range(classes):
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
        per_class[index] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return {
        "support": len(targets),
        "accuracy": (
            sum(confusion[index][index] for index in range(classes))
            / len(targets)
            if targets
            else 0.0
        ),
        "macro_f1": sum(f1_values) / classes,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


@torch.inference_mode()
def evaluate_ocular(
    model: CausalOcularLSTM,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    loss_weights: OcularLossWeights = OcularLossWeights(),
    phase_weights: Tensor | None = None,
    amp: bool = True,
) -> dict[str, object]:
    model.eval()
    use_amp = amp and device.type == "cuda"
    losses = []
    phase_targets: list[int] = []
    phase_predictions: list[int] = []
    closed_targets: list[int] = []
    closed_predictions: list[int] = []
    progress_errors: list[float] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["ocular_features"])
            result = ocular_loss(
                output,
                batch,
                weights=loss_weights,
                phase_weights=phase_weights,
            )
        losses.append(float(result.total))
        final_phase = batch["phase_targets"][:, -1].cpu()
        final_reliability = batch["reliability_targets"][:, -1].cpu()
        phase_probability = output.phase_logits[:, -1].float().softmax(-1).cpu()
        closed_probability = output.closed_logits[:, -1].float().sigmoid().cpu()
        progress = output.progress[:, -1].float().cpu()
        progress_target = batch["progress_targets"][:, -1].cpu()
        sessions = [str(value) for value in raw_batch["session"]]
        event_ids = [str(value) for value in raw_batch["event_id"]]
        frame_ids = raw_batch["target_frame_id"].tolist()
        timestamps = raw_batch["target_timestamp"].tolist()
        for index, session in enumerate(sessions):
            identity = (session, int(frame_ids[index]))
            if identity in seen:
                raise ValueError(f"ocular evaluation reference repeated: {identity}")
            seen.add(identity)
            target = int(final_phase[index])
            prediction = int(phase_probability[index].argmax())
            if target != IGNORE_INDEX:
                phase_targets.append(target)
                phase_predictions.append(prediction)
                closed_target = int(target == 2)
                closed_targets.append(closed_target)
                closed_predictions.append(int(closed_probability[index] >= 0.5))
                if closed_target and float(final_reliability[index]) > 0.0:
                    progress_errors.append(
                        abs(float(progress[index] - progress_target[index]))
                    )
            rows.append(
                {
                    "session": session,
                    "event_id": event_ids[index],
                    "frame_id": int(frame_ids[index]),
                    "timestamp": float(timestamps[index]),
                    "target": target,
                    "prediction": prediction,
                    "phase_probabilities": phase_probability[index].tolist(),
                    "closed_probability": float(closed_probability[index]),
                    "progress": float(progress[index]),
                    "reliability": float(
                        output.reliability_logits[index, -1].float().sigmoid().cpu()
                    ),
                }
            )
    phase_metrics = _classification_metrics(
        phase_targets,
        phase_predictions,
        classes=4,
    )
    per_class = phase_metrics.pop("per_class")
    return {
        **phase_metrics,
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "per_phase": {
            name: per_class[index] for index, name in enumerate(PHASE_NAMES)
        },
        "closed_binary": _classification_metrics(
            closed_targets,
            closed_predictions,
            classes=2,
        ),
        "progress_mae": (
            sum(progress_errors) / len(progress_errors)
            if progress_errors
            else 0.0
        ),
        "rows": rows,
    }


def save_ocular_checkpoint(
    path: Path | str,
    *,
    model: CausalOcularLSTM,
    optimizer: torch.optim.Optimizer,
    model_config: Mapping[str, object],
    visual_checkpoint_fingerprint: str,
    split: SubjectSplitMetadata,
    target_fps: float,
    microsleep_seconds: float,
    sequence_length: int,
    loss_weights: OcularLossWeights,
    epoch: int,
    metrics: Mapping[str, object],
    input_augmentation: OcularInputAugmentation = OcularInputAugmentation(),
) -> None:
    if (
        not visual_checkpoint_fingerprint
        or target_fps <= 0.0
        or microsleep_seconds <= 0.0
        or sequence_length <= 0
        or epoch < 0
    ):
        raise ValueError("ocular checkpoint metadata is invalid")
    payload = {
        "schema_version": OCULAR_CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": dict(model_config),
        "visual_checkpoint_fingerprint": visual_checkpoint_fingerprint,
        "split": asdict(split),
        "target_fps": float(target_fps),
        "microsleep_seconds": float(microsleep_seconds),
        "sequence_length": int(sequence_length),
        "loss_weights": asdict(loss_weights),
        "input_augmentation": asdict(input_augmentation),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_ocular_checkpoint(
    path: Path | str,
    *,
    model: CausalOcularLSTM,
    optimizer: torch.optim.Optimizer,
    expected_model_config: Mapping[str, object],
    expected_visual_checkpoint_fingerprint: str,
    expected_split: SubjectSplitMetadata,
    expected_target_fps: float,
    expected_microsleep_seconds: float,
    expected_sequence_length: int,
    expected_loss_weights: OcularLossWeights,
    expected_input_augmentation: OcularInputAugmentation = OcularInputAugmentation(),
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[str, object]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    expected = {
        "schema_version": OCULAR_CHECKPOINT_SCHEMA_VERSION,
        "model_config": dict(expected_model_config),
        "visual checkpoint fingerprint": (
            expected_visual_checkpoint_fingerprint
        ),
        "split": asdict(expected_split),
        "sequence_length": int(expected_sequence_length),
        "loss_weights": asdict(expected_loss_weights),
        "input augmentation": asdict(expected_input_augmentation),
    }
    actual = {
        "schema_version": payload.get("schema_version"),
        "model_config": payload.get("model_config"),
        "visual checkpoint fingerprint": payload.get(
            "visual_checkpoint_fingerprint"
        ),
        "split": payload.get("split"),
        "sequence_length": payload.get("sequence_length"),
        "loss_weights": payload.get("loss_weights"),
        "input augmentation": payload.get("input_augmentation"),
    }
    mismatched = [
        name for name, value in expected.items() if actual.get(name) != value
    ]
    if not math.isclose(
        float(payload.get("target_fps", 0.0)),
        float(expected_target_fps),
    ):
        mismatched.append("target FPS")
    if not math.isclose(
        float(payload.get("microsleep_seconds", 0.0)),
        float(expected_microsleep_seconds),
    ):
        mismatched.append("microsleep seconds")
    if mismatched:
        raise ValueError(f"ocular checkpoint mismatch: {', '.join(mismatched)}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"]) + 1, payload


__all__ = [
    "OCULAR_CHECKPOINT_SCHEMA_VERSION",
    "OcularInputAugmentation",
    "OcularLoss",
    "OcularLossWeights",
    "augment_ocular_features",
    "evaluate_ocular",
    "ocular_loss",
    "restore_ocular_checkpoint",
    "save_ocular_checkpoint",
    "train_ocular_epoch",
]
