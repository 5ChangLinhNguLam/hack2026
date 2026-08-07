"""A small shared-eye CNN followed by a strictly causal temporal network."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EyeTemporalOutput(NamedTuple):
    phase_logits: Tensor
    closedness_logits: Tensor
    moving_logits: Tensor


class EyeEncoder(nn.Module):
    """One encoder shared by both canonicalized anatomical eyes."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, embedding_dim)

    def forward(self, eyes: Tensor) -> Tensor:
        return self.projection(self.features(eyes).flatten(1))


class CausalTemporalBlock(nn.Module):
    """Residual TCN block whose output at t can only see frames <= t."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=0,
        )
        self.left_padding = 2 * dilation
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        residual = sequence
        features = self.norm(sequence).transpose(1, 2)
        features = F.pad(features, (self.left_padding, 0))
        features = self.convolution(features).transpose(1, 2)
        return residual + self.dropout(self.activation(features))


class EyeTemporalNet(nn.Module):
    """Predict open/closing/close/opening from two-eye causal clips.

    Input shape is ``[batch, time, eye, channel, height, width]`` and eye must
    have size two.  The driver's right eye is flipped horizontally before the
    shared encoder, placing both eyes in a common anatomical orientation.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        temporal_channels: int = 128,
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
        visibility_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or temporal_channels <= 0:
            raise ValueError("channel dimensions must be positive")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        if not 0.0 <= visibility_floor <= 1.0:
            raise ValueError("visibility_floor must be between 0 and 1")

        self.eye_encoder = EyeEncoder(embedding_dim)
        self.visibility_floor = visibility_floor
        self.eye_fusion = nn.Sequential(
            nn.Linear(embedding_dim + 2, embedding_dim),
            nn.SiLU(),
        )
        self.temporal_input = nn.Linear(embedding_dim, temporal_channels)
        self.temporal = nn.Sequential(
            *(CausalTemporalBlock(temporal_channels, dilation, dropout) for dilation in dilations)
        )
        self.phase_head = nn.Linear(temporal_channels, 4)
        self.closedness_head = nn.Linear(temporal_channels, 1)
        self.moving_head = nn.Linear(temporal_channels, 1)

    def _encode_and_fuse(self, eyes: Tensor, visibility: Tensor) -> Tensor:
        batch, time, eye_count, channels, height, width = eyes.shape
        if eye_count != 2 or channels != 3:
            raise ValueError("eyes must have shape [batch, time, 2, 3, height, width]")
        if visibility.shape != (batch, time, 2):
            raise ValueError("visibility must have shape [batch, time, 2]")

        canonical = torch.stack(
            (eyes[:, :, 0], torch.flip(eyes[:, :, 1], dims=(-1,))), dim=2
        )
        encoded = self.eye_encoder(canonical.reshape(batch * time * 2, 3, height, width))
        encoded = encoded.reshape(batch, time, 2, -1)

        weights = visibility.to(dtype=encoded.dtype).clamp(0.0, 1.0)
        weights = torch.where(
            weights >= self.visibility_floor, weights, torch.zeros_like(weights)
        )
        weighted = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled = weighted / denominator
        return self.eye_fusion(torch.cat((pooled, weights), dim=-1))

    def encode_visual(self, eyes: Tensor, visibility: Tensor) -> Tensor:
        """Return one visibility-aware shared-eye embedding per timestamp."""

        if eyes.ndim != 6:
            raise ValueError("eyes must be a six-dimensional tensor")
        return self._encode_and_fuse(eyes, visibility)

    def encode_temporal(self, visual: Tensor) -> Tensor:
        """Apply the strictly causal temporal trunk to visual eye embeddings."""

        if visual.ndim != 3:
            raise ValueError("visual eye features must have shape [batch, time, feature]")
        return self.temporal(self.temporal_input(visual))

    def forward(self, eyes: Tensor, visibility: Tensor) -> EyeTemporalOutput:
        fused = self.encode_visual(eyes, visibility)
        temporal = self.encode_temporal(fused)
        return EyeTemporalOutput(
            phase_logits=self.phase_head(temporal),
            closedness_logits=self.closedness_head(temporal).squeeze(-1),
            moving_logits=self.moving_head(temporal).squeeze(-1),
        )
