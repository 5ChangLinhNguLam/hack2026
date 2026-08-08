"""Small visual mouth specialist with hierarchical yawn heads."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class MouthVisualOutput(NamedTuple):
    embedding: Tensor
    binary_logits: Tensor
    type_logits: Tensor


class MouthVisualNet(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, embedding_dim)
        self.binary_head = nn.Linear(embedding_dim, 1)
        self.type_head = nn.Linear(embedding_dim, 3)

    def encode(self, mouth: Tensor) -> Tensor:
        if mouth.ndim != 4 or mouth.shape[1] != 3:
            raise ValueError("mouth input must have shape [batch, 3, height, width]")
        return self.projection(self.features(mouth).flatten(1))

    def forward(self, mouth: Tensor) -> MouthVisualOutput:
        embedding = self.encode(mouth)
        return MouthVisualOutput(
            embedding=embedding,
            binary_logits=self.binary_head(embedding).squeeze(-1),
            type_logits=self.type_head(embedding),
        )
