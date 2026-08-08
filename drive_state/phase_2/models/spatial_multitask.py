"""Lightweight cabin/face encoders with protocol-native primitive heads."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from ..data.primitive_labels import TASK_CLASS_COUNTS


CABIN_TASKS = (
    "driver_action",
    "distraction",
    "road_gaze",
    "hands_using_wheel",
    "talking",
    "hands_on_wheel",
    "moving_hands",
)
FACE_TASKS = ("yawn", "blink", "gaze_zone")


class MobileNetV3SmallEncoder(nn.Module):
    def __init__(self, embedding_dim: int, *, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        feature_channels = backbone.classifier[0].in_features
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(feature_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.projection(self.pool(self.features(image)).flatten(1))


class SpatialPrimitiveNet(nn.Module):
    """Separate full-cabin and face backbones with multiple masked heads."""

    cabin_tasks = CABIN_TASKS
    face_tasks = FACE_TASKS

    def __init__(self, *, pretrained: bool = True, embedding_dim: int = 256) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.cabin_encoder = MobileNetV3SmallEncoder(embedding_dim, pretrained=pretrained)
        self.face_encoder = MobileNetV3SmallEncoder(embedding_dim, pretrained=pretrained)
        self.face_fusion = nn.Sequential(
            nn.Linear(embedding_dim + 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(embedding_dim, TASK_CLASS_COUNTS[task])
                for task in TASK_CLASS_COUNTS
            }
        )

    def encode_cabin(self, cabin: Tensor) -> Tensor:
        if cabin.ndim != 4:
            raise ValueError("cabin input must have shape [batch, 3, height, width]")
        return self.cabin_encoder(cabin)

    def encode_face(
        self,
        face: Tensor,
        face_visibility: Tensor,
        head_pose: Tensor,
    ) -> Tensor:
        if face.ndim != 4:
            raise ValueError("face input must have shape [batch, 3, height, width]")
        batch = face.shape[0]
        visibility = face_visibility.reshape(batch, 1).to(dtype=face.dtype).clamp(0.0, 1.0)
        if head_pose.shape != (batch, 2):
            raise ValueError("head_pose must have shape [batch, 2]")
        encoded = self.face_encoder(face) * visibility
        pose = head_pose.to(dtype=encoded.dtype).clamp(-90.0, 90.0) / 90.0
        return self.face_fusion(
            torch.cat((encoded, pose * visibility, visibility), dim=1)
        )

    def forward(
        self,
        cabin: Tensor,
        face: Tensor,
        face_visibility: Tensor,
        head_pose: Tensor,
    ) -> dict[str, Tensor]:
        if cabin.ndim != 4 or face.ndim != 4:
            raise ValueError("cabin and face inputs must have shape [batch, 3, height, width]")
        cabin_embedding = self.encode_cabin(cabin)
        face_embedding = self.encode_face(face, face_visibility, head_pose)
        outputs = {
            task: self.heads[task](cabin_embedding) for task in CABIN_TASKS
        }
        outputs.update({task: self.heads[task](face_embedding) for task in FACE_TASKS})
        return outputs
