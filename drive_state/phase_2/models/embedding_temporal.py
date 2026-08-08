"""Matched last-frame MLP and causal TCN heads over frozen visual embeddings."""

from __future__ import annotations

from typing import Callable, Mapping, NamedTuple

from torch import Tensor, nn

from ..data.primitive_labels import TASK_CLASS_COUNTS
from .eye_temporal import CausalTemporalBlock


class CabinTemporalOutput(NamedTuple):
    distraction_logits: Tensor
    action_logits: Tensor
    road_gaze_logits: Tensor
    auxiliary_logits: Mapping[str, Tensor]


class FaceMouthTemporalOutput(NamedTuple):
    binary_yawn_logits: Tensor
    yawn_type_logits: Tensor


class CabinOutputHeads(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.distraction = nn.Linear(channels, 1)
        self.action = nn.Linear(channels, TASK_CLASS_COUNTS["driver_action"])
        self.road_gaze = nn.Linear(channels, 1)
        self.auxiliary = nn.ModuleDict(
            {
                "hands_using_wheel": nn.Linear(
                    channels, TASK_CLASS_COUNTS["hands_using_wheel"]
                ),
                "talking": nn.Linear(channels, 1),
                "hands_on_wheel": nn.Linear(
                    channels, TASK_CLASS_COUNTS["hands_on_wheel"]
                ),
                "moving_hands": nn.Linear(channels, 1),
            }
        )

    def forward(self, features: Tensor) -> CabinTemporalOutput:
        return CabinTemporalOutput(
            distraction_logits=self.distraction(features).squeeze(-1),
            action_logits=self.action(features),
            road_gaze_logits=self.road_gaze(features).squeeze(-1),
            auxiliary_logits={
                task: head(features).squeeze(-1)
                if TASK_CLASS_COUNTS[task] == 2
                else head(features)
                for task, head in self.auxiliary.items()
            },
        )


class FaceMouthOutputHeads(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.binary_yawn = nn.Linear(channels, 1)
        self.yawn_type = nn.Linear(channels, TASK_CLASS_COUNTS["yawn"])

    def forward(self, features: Tensor) -> FaceMouthTemporalOutput:
        return FaceMouthTemporalOutput(
            binary_yawn_logits=self.binary_yawn(features).squeeze(-1),
            yawn_type_logits=self.yawn_type(features),
        )


class LastFrameMLP(nn.Module):
    """Apply a nonlinear projection only to the final cached timestamp."""

    def __init__(
        self,
        input_dim: int,
        channels: int,
        output_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        if input_dim <= 0 or channels <= 0:
            raise ValueError("input_dim and channels must be positive")
        self.input = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )
        self.heads = output_factory(channels)

    def forward(self, sequence: Tensor):
        if sequence.ndim != 3:
            raise ValueError("embedding sequence must have shape [batch, time, feature]")
        features = self.input(sequence[:, -1:])
        return self.heads(features)


class CabinLastFrameMLP(LastFrameMLP):
    def __init__(self, *, input_dim: int = 256, channels: int = 128) -> None:
        super().__init__(input_dim, channels, CabinOutputHeads)


class FaceMouthLastFrameMLP(LastFrameMLP):
    def __init__(self, *, input_dim: int = 324, channels: int = 128) -> None:
        super().__init__(input_dim, channels, FaceMouthOutputHeads)


class _EmbeddingTCN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        channels: int,
        output_factory: Callable[[int], nn.Module],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or channels <= 0:
            raise ValueError("input_dim and channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            CausalTemporalBlock(channels, 1, dropout),
            CausalTemporalBlock(channels, 2, dropout),
            CausalTemporalBlock(channels, 4, dropout),
        )
        self.heads = output_factory(channels)

    def forward(self, sequence: Tensor):
        if sequence.ndim != 3:
            raise ValueError("embedding sequence must have shape [batch, time, feature]")
        return self.heads(self.temporal(self.input(sequence)))


class CabinEmbeddingTCN(_EmbeddingTCN):
    def __init__(
        self,
        *,
        input_dim: int = 256,
        channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            channels=channels,
            output_factory=CabinOutputHeads,
            dropout=dropout,
        )


class FaceMouthEmbeddingTCN(_EmbeddingTCN):
    def __init__(
        self,
        *,
        input_dim: int = 324,
        channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            channels=channels,
            output_factory=FaceMouthOutputHeads,
            dropout=dropout,
        )
