"""Masked, class-balanced multi-task objective for temporal eye phases."""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..models.eye_temporal import EyeTemporalOutput


class EyeLossBreakdown(NamedTuple):
    total: Tensor
    phase: Tensor
    closedness: Tensor
    moving: Tensor


def _zero_with_gradient(tensor: Tensor) -> Tensor:
    return tensor.sum() * 0.0


class EyeMultiTaskLoss(nn.Module):
    """Focal four-phase loss plus unambiguous closedness and motion tasks."""

    def __init__(
        self,
        *,
        phase_weight: float = 1.0,
        closedness_weight: float = 0.4,
        moving_weight: float = 0.2,
        phase_class_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if len(phase_class_weights) != 4 or any(weight <= 0 for weight in phase_class_weights):
            raise ValueError("phase_class_weights must contain four positive values")
        if min(phase_weight, closedness_weight, moving_weight, focal_gamma) < 0.0:
            raise ValueError("loss weights and focal_gamma cannot be negative")
        self.phase_weight = phase_weight
        self.closedness_weight = closedness_weight
        self.moving_weight = moving_weight
        self.focal_gamma = focal_gamma
        self.register_buffer(
            "phase_class_weights", torch.tensor(phase_class_weights, dtype=torch.float32)
        )

    def _phase_loss(self, logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool() & targets.ne(-100)
        if not mask.any():
            return _zero_with_gradient(logits)
        selected_logits = logits[mask]
        selected_targets = targets[mask]
        cross_entropy = F.cross_entropy(
            selected_logits,
            selected_targets,
            weight=self.phase_class_weights,
            reduction="none",
        )
        correct_probability = selected_logits.softmax(dim=-1).gather(
            1, selected_targets.unsqueeze(1)
        ).squeeze(1)
        return (((1.0 - correct_probability) ** self.focal_gamma) * cross_entropy).mean()

    @staticmethod
    def _binary_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        if not mask.any():
            return _zero_with_gradient(logits)
        return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])

    def forward(
        self,
        output: EyeTemporalOutput,
        batch: Mapping[str, Tensor],
    ) -> EyeLossBreakdown:
        valid = batch["valid_mask"].bool()
        phase = self._phase_loss(output.phase_logits, batch["phase"], valid)
        closedness = self._binary_loss(
            output.closedness_logits,
            batch["closedness"],
            valid & batch["closedness_mask"].bool(),
        )
        moving = self._binary_loss(output.moving_logits, batch["moving"], valid)
        total = (
            self.phase_weight * phase
            + self.closedness_weight * closedness
            + self.moving_weight * moving
        )
        return EyeLossBreakdown(total, phase, closedness, moving)
