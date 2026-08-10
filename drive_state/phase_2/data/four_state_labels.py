"""Four-state learning targets with causal microsleep output fusion."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor

from .five_state_labels import FiveState, IGNORE_INDEX


class FourState(IntEnum):
    ALERT = 0
    DROWSY = 1
    YAWNING = 2
    DISTRACTION = 3


_FIVE_TO_FOUR = {
    int(FiveState.ALERT): int(FourState.ALERT),
    int(FiveState.DROWSY): int(FourState.DROWSY),
    int(FiveState.MICROSLEEP): IGNORE_INDEX,
    int(FiveState.YAWNING): int(FourState.YAWNING),
    int(FiveState.DISTRACTION): int(FourState.DISTRACTION),
    IGNORE_INDEX: IGNORE_INDEX,
}
_FOUR_TO_FIVE = (0, 1, 3, 4)


def five_to_four_target(target: int) -> int:
    """Mask microsleep and preserve the four non-event state identities."""

    try:
        return _FIVE_TO_FOUR[int(target)]
    except KeyError as error:
        raise ValueError(f"invalid five-state target: {target}") from error


def four_to_five_target(target: int) -> int:
    """Map a learned four-state class back into the public class space."""

    try:
        index = int(target)
        if index < 0:
            raise IndexError(index)
        return _FOUR_TO_FIVE[index]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid four-state target: {target}") from error


def compose_five_state_probabilities(
    four_probabilities: Tensor,
    microsleep_active: Tensor,
    *,
    drowsy_active: Tensor | None = None,
) -> Tensor:
    """Fuse causal physiology gates into one exclusive five-state output."""

    if four_probabilities.ndim != 2 or four_probabilities.shape[-1] != 4:
        raise ValueError("four-state probabilities must have shape [batch, 4]")
    if microsleep_active.shape != four_probabilities.shape[:1]:
        raise ValueError("microsleep gate must have shape [batch]")
    if (
        drowsy_active is not None
        and drowsy_active.shape != four_probabilities.shape[:1]
    ):
        raise ValueError("drowsy gate must have shape [batch]")
    if not torch.isfinite(four_probabilities).all():
        raise ValueError("four-state probabilities must be finite")
    if torch.any(four_probabilities < 0.0) or not torch.allclose(
        four_probabilities.sum(dim=-1),
        torch.ones(
            four_probabilities.shape[0],
            device=four_probabilities.device,
            dtype=four_probabilities.dtype,
        ),
        atol=1e-5,
    ):
        raise ValueError("four-state probabilities must be normalized")
    fused = four_probabilities.new_zeros((four_probabilities.shape[0], 5))
    fused[:, (0, 1, 3, 4)] = four_probabilities
    if drowsy_active is not None:
        drowsy_gate = drowsy_active.to(device=fused.device, dtype=torch.bool)
        fused[drowsy_gate] = 0.0
        fused[drowsy_gate, int(FiveState.DROWSY)] = 1.0
    gate = microsleep_active.to(device=fused.device, dtype=torch.bool)
    fused[gate] = 0.0
    fused[gate, int(FiveState.MICROSLEEP)] = 1.0
    return fused


__all__ = [
    "FourState",
    "compose_five_state_probabilities",
    "five_to_four_target",
    "four_to_five_target",
]
