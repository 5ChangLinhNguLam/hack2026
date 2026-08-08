"""Masked multi-task classification objective for protocol-separated labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.primitive_labels import IGNORE_INDEX, TASK_CLASS_COUNTS
from ..distraction import action_distraction_probability
from ..models.spatial_multitask import FACE_TASKS


@dataclass(frozen=True)
class PrimitiveLossBreakdown:
    total: Tensor
    per_task: Mapping[str, Tensor]


class MaskedPrimitiveLoss(nn.Module):
    def __init__(
        self,
        *,
        task_weights: Mapping[str, float] | None = None,
        face_visibility_threshold: float = 0.2,
        action_distraction_weight: float = 1.0,
        distraction_consistency_weight: float = 0.25,
    ) -> None:
        super().__init__()
        unknown = set(task_weights or {}).difference(TASK_CLASS_COUNTS)
        if unknown:
            raise ValueError(f"unknown task weights: {sorted(unknown)}")
        self.task_weights = {
            task: float((task_weights or {}).get(task, 1.0)) for task in TASK_CLASS_COUNTS
        }
        if any(weight < 0.0 for weight in self.task_weights.values()):
            raise ValueError("task weights cannot be negative")
        if not 0.0 <= face_visibility_threshold <= 1.0:
            raise ValueError("face_visibility_threshold must be between 0 and 1")
        self.face_visibility_threshold = face_visibility_threshold
        if action_distraction_weight < 0.0 or distraction_consistency_weight < 0.0:
            raise ValueError("hierarchical distraction weights cannot be negative")
        self.action_distraction_weight = float(action_distraction_weight)
        self.distraction_consistency_weight = float(
            distraction_consistency_weight
        )

    def forward(
        self, outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]
    ) -> PrimitiveLossBreakdown:
        if set(outputs) != set(TASK_CLASS_COUNTS):
            raise ValueError("primitive model outputs do not match configured tasks")
        per_task: dict[str, Tensor] = {}
        for task in TASK_CLASS_COUNTS:
            target = batch[task]
            mask = target.ne(IGNORE_INDEX)
            if task in FACE_TASKS:
                mask = mask & batch["face_visibility"].ge(
                    self.face_visibility_threshold
                )
            if mask.any():
                per_task[task] = F.cross_entropy(outputs[task][mask], target[mask])
            else:
                per_task[task] = outputs[task].sum() * 0.0
        hierarchical_mask = batch["distraction"].ne(IGNORE_INDEX) & batch[
            "driver_action"
        ].ne(IGNORE_INDEX)
        action_distraction = action_distraction_probability(
            outputs["driver_action"]
        )
        direct_distraction = outputs["distraction"].softmax(dim=-1)[:, 1]
        if hierarchical_mask.any():
            binary_target = batch["distraction"][hierarchical_mask].to(
                dtype=action_distraction.dtype
            )
            action_distraction_logits = torch.logit(
                action_distraction[hierarchical_mask].clamp(1e-6, 1.0 - 1e-6)
            )
            per_task["action_distraction"] = F.binary_cross_entropy_with_logits(
                action_distraction_logits,
                binary_target,
            )
            per_task["distraction_consistency"] = F.mse_loss(
                direct_distraction[hierarchical_mask],
                action_distraction[hierarchical_mask],
            )
        else:
            zero = (action_distraction.sum() + direct_distraction.sum()) * 0.0
            per_task["action_distraction"] = zero
            per_task["distraction_consistency"] = zero
        total = sum(
            self.task_weights[task] * per_task[task] for task in TASK_CLASS_COUNTS
        )
        total = (
            total
            + self.action_distraction_weight * per_task["action_distraction"]
            + self.distraction_consistency_weight
            * per_task["distraction_consistency"]
        )
        return PrimitiveLossBreakdown(total=total, per_task=per_task)
