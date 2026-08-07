"""Hierarchical direct/type-derived yawn objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.primitive_labels import IGNORE_INDEX
from ..models.mouth_visual import MouthVisualOutput


@dataclass(frozen=True)
class YawnLossBreakdown:
    total: Tensor
    direct: Tensor
    derived: Tensor
    consistency: Tensor
    type: Tensor
    support: int


class YawnHierarchicalLoss(nn.Module):
    def __init__(self, *, visibility_threshold: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be between 0 and 1")
        self.visibility_threshold = visibility_threshold

    def forward(
        self,
        output: MouthVisualOutput,
        batch: Mapping[str, Tensor],
    ) -> YawnLossBreakdown:
        target = batch["yawn"]
        mask = target.ne(IGNORE_INDEX) & batch["mouth_visibility"].ge(
            self.visibility_threshold
        )
        support = int(mask.sum().item())
        direct_logits = output.binary_logits.float()
        type_logits = output.type_logits.float()
        if support:
            binary_target = target[mask].gt(0).to(dtype=torch.float32)
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
            type_loss = F.cross_entropy(type_logits[mask], target[mask])
        else:
            zero = (direct_logits.sum() + type_logits.sum()) * 0.0
            direct = derived = consistency = type_loss = zero
        total = 2.0 * direct + derived + 0.25 * consistency + 0.5 * type_loss
        return YawnLossBreakdown(
            total=total,
            direct=direct,
            derived=derived,
            consistency=consistency,
            type=type_loss,
            support=support,
        )
