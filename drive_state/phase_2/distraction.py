"""Shared hierarchical-distraction probabilities for training and runtime."""

from __future__ import annotations

from torch import Tensor

from .data.primitive_labels import DISTRACTING_ACTIONS, DRIVER_ACTIONS


DISTRACTING_ACTION_INDICES = tuple(
    index for index, action in enumerate(DRIVER_ACTIONS) if action in DISTRACTING_ACTIONS
)
DISTRACTION_POOL_WEIGHTS = (0.50, 0.35, 0.15)


def action_distraction_probability(action_logits: Tensor) -> Tensor:
    if action_logits.shape[-1] != len(DRIVER_ACTIONS):
        raise ValueError(
            f"action logits must have {len(DRIVER_ACTIONS)} classes in the last dimension"
        )
    return action_logits.softmax(dim=-1)[..., DISTRACTING_ACTION_INDICES].sum(dim=-1)


def pool_distraction_probabilities(
    direct: Tensor | float,
    action: Tensor | float,
    off_road: Tensor | float,
) -> Tensor | float:
    """Pool calibrated evidence without max-pooling one noisy specialist."""
    direct_weight, action_weight, road_weight = DISTRACTION_POOL_WEIGHTS
    pooled = direct_weight * direct + action_weight * action + road_weight * off_road
    if isinstance(pooled, Tensor):
        return pooled.clamp(0.0, 1.0)
    return max(0.0, min(1.0, pooled))
