"""Final-timestamp hierarchical objectives for cached-embedding specialists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.primitive_labels import IGNORE_INDEX, TASK_CLASS_COUNTS
from ..distraction import action_distraction_probability
from ..models.embedding_temporal import CabinTemporalOutput, FaceMouthTemporalOutput


@dataclass(frozen=True)
class TemporalLossBreakdown:
    total: Tensor
    components: Mapping[str, Tensor]
    support: Mapping[str, int]


def _zero_cabin(output: CabinTemporalOutput) -> Tensor:
    values = (
        output.distraction_logits.sum()
        + output.action_logits.sum()
        + output.road_gaze_logits.sum()
    )
    for tensor in output.auxiliary_logits.values():
        values = values + tensor.sum()
    return values.float() * 0.0


class CabinTemporalLoss(nn.Module):
    """Hierarchical distraction plus road/action/low-weight auxiliary losses."""

    auxiliary_tasks = (
        "hands_using_wheel",
        "talking",
        "hands_on_wheel",
        "moving_hands",
    )

    def forward(
        self,
        output: CabinTemporalOutput,
        batch: Mapping[str, Tensor],
    ) -> TemporalLossBreakdown:
        direct_logits = output.distraction_logits[:, -1].float()
        action_logits = output.action_logits[:, -1].float()
        road_logits = output.road_gaze_logits[:, -1].float()
        zero = _zero_cabin(output)
        components: dict[str, Tensor] = {}
        support: dict[str, int] = {}

        distraction_mask = batch["distraction"].ne(IGNORE_INDEX)
        support["distraction"] = int(distraction_mask.sum().item())
        if distraction_mask.any():
            binary_target = batch["distraction"][distraction_mask].float()
            components["direct_distraction"] = F.binary_cross_entropy_with_logits(
                direct_logits[distraction_mask], binary_target
            )
        else:
            components["direct_distraction"] = zero

        action_mask = batch["driver_action"].ne(IGNORE_INDEX)
        support["driver_action"] = int(action_mask.sum().item())
        components["action"] = (
            F.cross_entropy(action_logits[action_mask], batch["driver_action"][action_mask])
            if action_mask.any()
            else zero
        )
        hierarchical_mask = distraction_mask & action_mask
        derived_probability = action_distraction_probability(action_logits)
        if hierarchical_mask.any():
            binary_target = batch["distraction"][hierarchical_mask].float()
            derived_logits = torch.logit(
                derived_probability[hierarchical_mask].float().clamp(1e-6, 1.0 - 1e-6)
            )
            components["action_derived_distraction"] = F.binary_cross_entropy_with_logits(
                derived_logits, binary_target
            )
            components["distraction_consistency"] = F.mse_loss(
                direct_logits[hierarchical_mask].sigmoid(),
                derived_probability[hierarchical_mask].float(),
            )
        else:
            components["action_derived_distraction"] = zero
            components["distraction_consistency"] = zero

        road_mask = batch["road_gaze"].ne(IGNORE_INDEX)
        support["road_gaze"] = int(road_mask.sum().item())
        components["road_gaze"] = (
            F.binary_cross_entropy_with_logits(
                road_logits[road_mask], batch["road_gaze"][road_mask].float()
            )
            if road_mask.any()
            else zero
        )

        for task in self.auxiliary_tasks:
            logits = output.auxiliary_logits[task][:, -1].float()
            mask = batch[task].ne(IGNORE_INDEX)
            support[task] = int(mask.sum().item())
            if not mask.any():
                components[task] = zero
            elif TASK_CLASS_COUNTS[task] == 2:
                components[task] = F.binary_cross_entropy_with_logits(
                    logits[mask], batch[task][mask].float()
                )
            else:
                components[task] = F.cross_entropy(logits[mask], batch[task][mask])

        total = (
            2.0 * components["direct_distraction"]
            + components["action_derived_distraction"]
            + 0.25 * components["distraction_consistency"]
            + 0.5 * components["action"]
            + components["road_gaze"]
            + 0.1 * sum(components[task] for task in self.auxiliary_tasks)
        )
        return TemporalLossBreakdown(total=total, components=components, support=support)


class FaceMouthTemporalLoss(nn.Module):
    def __init__(self, *, visibility_threshold: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be between 0 and 1")
        self.visibility_threshold = visibility_threshold

    def forward(
        self,
        output: FaceMouthTemporalOutput,
        batch: Mapping[str, Tensor],
    ) -> TemporalLossBreakdown:
        direct_logits = output.binary_yawn_logits[:, -1].float()
        type_logits = output.yawn_type_logits[:, -1].float()
        visibility = batch["mouth_visibility"]
        if visibility.ndim == 2:
            visibility = visibility[:, -1]
        mask = batch["yawn"].ne(IGNORE_INDEX) & visibility.ge(
            self.visibility_threshold
        )
        support = {"yawn": int(mask.sum().item())}
        if mask.any():
            target = batch["yawn"][mask]
            binary_target = target.gt(0).float()
            direct = F.binary_cross_entropy_with_logits(
                direct_logits[mask], binary_target
            )
            type_probability = type_logits[mask].softmax(dim=-1)[:, 1:].sum(dim=-1)
            derived_logits = torch.logit(type_probability.clamp(1e-6, 1.0 - 1e-6))
            derived = F.binary_cross_entropy_with_logits(
                derived_logits, binary_target
            )
            consistency = F.mse_loss(
                direct_logits[mask].sigmoid(), type_probability
            )
            type_loss = F.cross_entropy(type_logits[mask], target)
        else:
            zero = (direct_logits.sum() + type_logits.sum()) * 0.0
            direct = derived = consistency = type_loss = zero
        components = {
            "direct_yawn": direct,
            "type_derived_yawn": derived,
            "yawn_consistency": consistency,
            "yawn_type": type_loss,
        }
        total = 2.0 * direct + derived + 0.25 * consistency + 0.5 * type_loss
        return TemporalLossBreakdown(total=total, components=components, support=support)
