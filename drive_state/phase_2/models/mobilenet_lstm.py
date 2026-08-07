"""Single-backbone visual encoder and causal temporal state model."""

from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
from torch import Tensor, nn


class VisualFrameOutput(NamedTuple):
    embedding: Tensor
    region_embeddings: Tensor
    eye_logits: Tensor
    eye_visibility_logits: Tensor
    yawn_logits: Tensor
    distraction_logits: Tensor
    pose: Tensor


class FiveStateLSTMOutput(NamedTuple):
    logits: Tensor
    hidden: tuple[Tensor, Tensor]


class FourStateLSTMOutput(NamedTuple):
    logits: Tensor
    hidden: tuple[Tensor, Tensor]


def causal_closure_features(closure_duration: Tensor) -> Tensor:
    """Expose smooth causal proximity to slow-closure and microsleep bounds."""

    duration = closure_duration.to(dtype=torch.float32).clamp_min(0.0)
    normalized = duration.clamp_max(2.0).div(2.0)
    slow_strength = duration.sub(0.5).div(1.5).clamp(0.0, 1.0)
    microsleep_strength = duration.sub(1.5).div(0.5).clamp(0.0, 1.0)
    return torch.stack(
        (normalized, slow_strength, microsleep_strength),
        dim=-1,
    )


def fixed_count_roi_align(
    feature_map: Tensor,
    boxes: Tensor,
    *,
    image_size: tuple[int, int],
    output_size: tuple[int, int],
) -> Tensor:
    """Pool a fixed number of absolute input-image boxes per batch item."""

    if feature_map.ndim != 4:
        raise ValueError("feature map must have shape [batch, channel, height, width]")
    if boxes.ndim != 3 or boxes.shape[0] != feature_map.shape[0] or boxes.shape[2] != 4:
        raise ValueError("ROI boxes must have shape [batch, regions, 4]")
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("ROI input image dimensions must be positive")
    safe = boxes.to(
        device=feature_map.device,
        dtype=feature_map.dtype,
    ).clone()
    if not torch.isfinite(safe).all():
        raise ValueError("ROI boxes must be finite")
    if torch.any(
        (safe[..., (0, 2)] < 0.0)
        | (safe[..., (0, 2)] > float(image_width))
    ) or torch.any(
        (safe[..., (1, 3)] < 0.0)
        | (safe[..., (1, 3)] > float(image_height))
    ):
        raise ValueError("ROI boxes must lie inside the input image")
    safe[..., 0] = safe[..., 0].clamp(0.0, float(image_width - 1))
    safe[..., 1] = safe[..., 1].clamp(0.0, float(image_height - 1))
    safe[..., 2] = safe[..., 2].clamp(1.0, float(image_width))
    safe[..., 3] = safe[..., 3].clamp(1.0, float(image_height))
    safe[..., 2] = torch.maximum(safe[..., 2], safe[..., 0] + 1.0)
    safe[..., 3] = torch.maximum(safe[..., 3], safe[..., 1] + 1.0)
    batch, regions, _ = safe.shape
    batch_ids = torch.arange(
        batch,
        device=safe.device,
        dtype=safe.dtype,
    ).repeat_interleave(regions)
    rois = torch.cat(
        (batch_ids.unsqueeze(1), safe.reshape(-1, 4)),
        dim=1,
    )
    scale_x = feature_map.shape[-1] / float(image_width)
    scale_y = feature_map.shape[-2] / float(image_height)
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError("ROI feature map must use one isotropic spatial scale")
    from torchvision.ops import roi_align

    return roi_align(
        feature_map,
        rois,
        output_size=output_size,
        spatial_scale=scale_x,
        aligned=True,
    )


class MobileNetV3LargeVisualEncoder(nn.Module):
    """Encode one full cabin frame with one MobileNetV3-Large backbone."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        region_dim: int = 64,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or region_dim <= 0:
            raise ValueError("embedding and region dimensions must be positive")
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            mobilenet_v3_large,
        )

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        feature_channels = backbone.classifier[0].in_features
        self.features = backbone.features
        self.roi_feature_index = 3
        roi_channels = self.features[self.roi_feature_index].out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.global_projection = nn.Sequential(
            nn.Linear(feature_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
        )
        self.region_pool_size = (3, 3)
        self.region_projection = nn.Sequential(
            nn.Linear(
                roi_channels
                * self.region_pool_size[0]
                * self.region_pool_size[1],
                region_dim,
            ),
            nn.LayerNorm(region_dim),
            nn.Hardswish(),
        )
        self.eye_head = nn.Linear(region_dim * 2, 3)
        self.eye_visibility_head = nn.Linear(region_dim * 2, 2)
        self.yawn_head = nn.Linear(region_dim, 2)
        self.distraction_head = nn.Linear(
            embedding_dim + region_dim,
            2,
        )
        self.pose_head = nn.Linear(region_dim, 2)
        self.visual_embedding_dim = embedding_dim + 4 * region_dim

    def forward(
        self,
        image: Tensor,
        region_boxes: Tensor,
        region_visibility: Tensor,
    ) -> VisualFrameOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image input must have shape [batch, 3, height, width]")
        batch = image.shape[0]
        if region_boxes.shape != (batch, 4, 4):
            raise ValueError("region boxes must have shape [batch, 4, 4]")
        if region_visibility.shape != (batch, 4):
            raise ValueError("region visibility must have shape [batch, 4]")
        if not torch.isfinite(region_boxes).all():
            raise ValueError("region boxes must be finite")
        if not torch.isfinite(region_visibility).all() or torch.any(
            (region_visibility < 0.0) | (region_visibility > 1.0)
        ):
            raise ValueError("region visibility must be finite and in [0, 1]")

        features = image
        roi_map: Tensor | None = None
        for index, block in enumerate(self.features):
            features = block(features)
            if index == self.roi_feature_index:
                roi_map = features
        assert roi_map is not None
        global_embedding = self.global_projection(
            self.pool(features).flatten(1)
        )
        pooled_regions = fixed_count_roi_align(
            roi_map,
            region_boxes,
            image_size=(image.shape[-1], image.shape[-2]),
            output_size=self.region_pool_size,
        )
        region_embeddings = self.region_projection(
            pooled_regions.flatten(1)
        ).reshape(batch, 4, -1)
        visibility = region_visibility.to(
            device=region_embeddings.device,
            dtype=region_embeddings.dtype,
        )
        region_embeddings = region_embeddings * visibility.unsqueeze(-1)
        embedding = torch.cat(
            (global_embedding, region_embeddings.flatten(1)),
            dim=-1,
        )
        eyes = torch.cat(
            (region_embeddings[:, 1], region_embeddings[:, 2]),
            dim=-1,
        )
        return VisualFrameOutput(
            embedding=embedding,
            region_embeddings=region_embeddings,
            eye_logits=self.eye_head(eyes),
            eye_visibility_logits=self.eye_visibility_head(eyes),
            yawn_logits=self.yawn_head(region_embeddings[:, 3]),
            distraction_logits=self.distraction_head(
                torch.cat(
                    (global_embedding, region_embeddings[:, 0]),
                    dim=-1,
                )
            ),
            pose=torch.tanh(self.pose_head(region_embeddings[:, 0])),
        )


def assemble_temporal_features(
    *,
    visual_embedding: Tensor,
    evidence: Tensor,
    context_valid: Tensor,
    ocular_phase_probabilities: Tensor | None = None,
) -> Tensor:
    """Combine cached visual and causal evidence with an explicit pad mask."""

    if visual_embedding.ndim != 3:
        raise ValueError(
            "visual embedding must have shape [batch, time, feature]"
        )
    batch, time, _ = visual_embedding.shape
    if evidence.ndim != 3 or evidence.shape[:2] != (batch, time):
        raise ValueError("evidence must have shape [batch, time, feature]")
    if context_valid.shape != (batch, time):
        raise ValueError("context validity must have shape [batch, time]")
    if ocular_phase_probabilities is not None:
        if ocular_phase_probabilities.shape != (batch, time, 4):
            raise ValueError(
                "ocular phase probabilities must have shape [batch, time, 4]"
            )
        if not torch.isfinite(ocular_phase_probabilities).all() or torch.any(
            ocular_phase_probabilities < 0.0
        ):
            raise ValueError(
                "ocular phase probabilities must be finite and non-negative"
            )
    if not torch.isfinite(visual_embedding).all() or not torch.isfinite(
        evidence
    ).all():
        raise ValueError("temporal inputs must be finite")
    if not torch.isfinite(context_valid).all() or torch.any(
        (context_valid < 0.0) | (context_valid > 1.0)
    ):
        raise ValueError("context validity must be finite and in [0, 1]")
    dtype = visual_embedding.dtype
    valid = context_valid.to(dtype=dtype).unsqueeze(-1)
    parts = [visual_embedding * valid]
    if ocular_phase_probabilities is not None:
        phase = ocular_phase_probabilities.to(dtype=dtype)
        valid_rows = context_valid.to(dtype=torch.bool)
        if valid_rows.any() and not torch.allclose(
            phase.sum(dim=-1)[valid_rows],
            torch.ones_like(phase.sum(dim=-1)[valid_rows]),
            atol=1e-4,
        ):
            raise ValueError(
                "valid ocular phase probabilities must be normalized"
            )
        parts.append(phase * valid)
    parts.extend((evidence.to(dtype=dtype) * valid, valid))
    return torch.cat(
        tuple(parts),
        dim=-1,
    )


class CausalFiveStateLSTM(nn.Module):
    """Unidirectional LSTM with one exclusive five-logit output head."""

    def __init__(
        self,
        *,
        input_dim: int = 540,
        projection_dim: int = 256,
        hidden_size: int = 256,
        layers: int = 2,
        dropout: float = 0.2,
        input_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if min(input_dim, projection_dim, hidden_size, layers) <= 0:
            raise ValueError("LSTM dimensions and layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= input_dropout < 1.0:
            raise ValueError("input_dropout must be in [0, 1)")
        self.input = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
            nn.Dropout(input_dropout),
        )
        self.temporal = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 5)

    def forward(
        self,
        sequence: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> FiveStateLSTMOutput:
        if sequence.ndim != 3:
            raise ValueError("temporal input must have shape [batch, time, feature]")
        features = self.input(sequence)
        temporal, next_hidden = self.temporal(features, hidden)
        return FiveStateLSTMOutput(self.head(temporal), next_hidden)

    def step(
        self,
        features: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if features.ndim != 2:
            raise ValueError("step input must have shape [batch, feature]")
        output = self(features.unsqueeze(1), hidden)
        return output.logits[:, 0], output.hidden


class CausalFourStateLSTM(nn.Module):
    """Unidirectional LSTM for non-microsleep frame-state learning."""

    def __init__(
        self,
        *,
        input_dim: int = 540,
        projection_dim: int = 256,
        hidden_size: int = 256,
        layers: int = 2,
        dropout: float = 0.2,
        input_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if min(input_dim, projection_dim, hidden_size, layers) <= 0:
            raise ValueError("LSTM dimensions and layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= input_dropout < 1.0:
            raise ValueError("input_dropout must be in [0, 1)")
        self.input = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
            nn.Dropout(input_dropout),
        )
        self.temporal = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 4)

    def forward(
        self,
        sequence: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> FourStateLSTMOutput:
        if sequence.ndim != 3:
            raise ValueError("temporal input must have shape [batch, time, feature]")
        features = self.input(sequence)
        temporal, next_hidden = self.temporal(features, hidden)
        return FourStateLSTMOutput(self.head(temporal), next_hidden)

    def step(
        self,
        features: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if features.ndim != 2:
            raise ValueError("step input must have shape [batch, feature]")
        output = self(features.unsqueeze(1), hidden)
        return output.logits[:, 0], output.hidden


def initialize_phase_aware_temporal(
    model: CausalFourStateLSTM,
    source_state: Mapping[str, Tensor],
    *,
    visual_embedding_dim: int,
    evidence_dim: int,
) -> None:
    """Widen a legacy temporal input with four zero-impact phase channels."""

    if visual_embedding_dim <= 0 or evidence_dim <= 0:
        raise ValueError("warm-start feature dimensions must be positive")
    weight_name = "input.0.weight"
    if weight_name not in source_state:
        raise ValueError("source temporal state lacks its input projection")
    target_state = model.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError("source and phase-aware temporal state keys differ")
    source_weight = source_state[weight_name]
    target_weight = target_state[weight_name]
    source_input_dim = visual_embedding_dim + evidence_dim + 1
    target_input_dim = source_input_dim + 4
    if source_weight.shape != (target_weight.shape[0], source_input_dim):
        raise ValueError("source temporal input projection has the wrong shape")
    if target_weight.shape != (source_weight.shape[0], target_input_dim):
        raise ValueError("phase-aware temporal input projection has the wrong shape")
    with torch.no_grad():
        for name, source in source_state.items():
            if name == weight_name:
                continue
            if target_state[name].shape != source.shape:
                raise ValueError(f"temporal warm-start shape mismatch: {name}")
            target_state[name].copy_(source)
        target_weight.zero_()
        target_weight[:, :visual_embedding_dim].copy_(
            source_weight[:, :visual_embedding_dim]
        )
        target_weight[:, visual_embedding_dim + 4 :].copy_(
            source_weight[:, visual_embedding_dim:]
        )
    model.load_state_dict(target_state)


__all__ = [
    "CausalFourStateLSTM",
    "CausalFiveStateLSTM",
    "FourStateLSTMOutput",
    "FiveStateLSTMOutput",
    "MobileNetV3LargeVisualEncoder",
    "VisualFrameOutput",
    "assemble_temporal_features",
    "causal_closure_features",
    "fixed_count_roi_align",
    "initialize_phase_aware_temporal",
]
