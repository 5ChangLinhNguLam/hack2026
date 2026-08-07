"""One-output causal temporal models for exclusive five-state classification."""

from __future__ import annotations

from typing import NamedTuple, Sequence

from torch import Tensor, nn

from .eye_temporal import CausalTemporalBlock


class FiveStateOutput(NamedTuple):
    logits: Tensor


class FiveStateLastFrameMLP(nn.Module):
    def __init__(self, *, input_dim: int = 718, channels: int = 192) -> None:
        super().__init__()
        if input_dim <= 0 or channels <= 0:
            raise ValueError("input_dim and channels must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, 5),
        )

    def forward(self, sequence: Tensor) -> FiveStateOutput:
        if sequence.ndim != 3:
            raise ValueError("fused sequence must have shape [batch, time, feature]")
        return FiveStateOutput(self.network(sequence[:, -1:]))


class FiveStateFusionTCN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 718,
        channels: int = 192,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or channels <= 0:
            raise ValueError("input_dim and channels must be positive")
        if not dilations or any(value <= 0 for value in dilations):
            raise ValueError("dilations must contain positive values")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            *(CausalTemporalBlock(channels, value, dropout) for value in dilations)
        )
        self.head = nn.Linear(channels, 5)

    def forward(self, sequence: Tensor) -> FiveStateOutput:
        if sequence.ndim != 3:
            raise ValueError("fused sequence must have shape [batch, time, feature]")
        features = self.temporal(self.input(sequence))
        return FiveStateOutput(self.head(features))


__all__ = ["FiveStateFusionTCN", "FiveStateLastFrameMLP", "FiveStateOutput"]
