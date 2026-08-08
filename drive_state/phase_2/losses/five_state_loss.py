"""Confidence-weighted masked objective for one exclusive five-state head."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.five_state_labels import FiveState, IGNORE_INDEX
from ..models.five_state_temporal import FiveStateOutput


def compute_five_state_class_weights(
    targets: Sequence[int], *, max_boost: float = 5.0
) -> tuple[float, float, float, float, float]:
    if max_boost < 1.0:
        raise ValueError("max_boost must be at least one")
    counts = [0] * len(FiveState)
    for target in targets:
        if target != IGNORE_INDEX:
            counts[FiveState(int(target))] += 1
    largest = max(counts, default=0)
    if largest == 0:
        raise ValueError("cannot compute class weights without valid targets")
    return tuple(
        min(math.sqrt(largest / count), max_boost) if count else max_boost
        for count in counts
    )  # type: ignore[return-value]


class FiveStateLoss(nn.Module):
    def __init__(self, *, class_weights: Sequence[float]) -> None:
        super().__init__()
        if len(class_weights) != len(FiveState) or any(
            not math.isfinite(value) or value <= 0.0 for value in class_weights
        ):
            raise ValueError("class_weights must contain five finite positive values")
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(
        self, output: FiveStateOutput, batch: Mapping[str, Tensor]
    ) -> Tensor:
        logits = output.logits[:, -1].float()
        targets = batch["five_state_target"]
        confidence = batch["five_state_confidence"].float()
        if targets.ndim > 1:
            targets = targets[:, -1]
        if confidence.ndim > 1:
            confidence = confidence[:, -1]
        mask = targets.ne(IGNORE_INDEX) & confidence.gt(0.0)
        if not mask.any():
            return logits.sum() * 0.0
        selected_targets = targets[mask].long()
        per_example = F.cross_entropy(
            logits[mask], selected_targets, reduction="none"
        )
        effective = confidence[mask] * self.class_weights[selected_targets]
        return (per_example * effective).sum() / effective.sum().clamp_min(1e-6)


__all__ = ["FiveStateLoss", "compute_five_state_class_weights"]
