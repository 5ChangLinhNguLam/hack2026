"""Small causal temporal model for denoising binocular eye evidence."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class OcularLSTMOutput(NamedTuple):
    phase_logits: Tensor
    closed_logits: Tensor
    progress: Tensor
    reliability_logits: Tensor
    hidden: tuple[Tensor, Tensor]


class OcularStepOutput(NamedTuple):
    phase_logits: Tensor
    closed_logits: Tensor
    progress: Tensor
    reliability_logits: Tensor


def assemble_ocular_features(
    *,
    region_embeddings: Tensor,
    raw_eye_probabilities: Tensor,
    eye_visibility: Tensor,
) -> Tensor:
    """Join left/right ROI embeddings with raw eye and visibility evidence."""

    if region_embeddings.ndim not in (3, 4) or region_embeddings.shape[-2] != 4:
        raise ValueError(
            "region embeddings must have shape [..., 4, region_dim]"
        )
    prefix = region_embeddings.shape[:-2]
    if raw_eye_probabilities.shape != (*prefix, 3):
        raise ValueError("raw eye probabilities must have shape [..., 3]")
    if eye_visibility.shape != (*prefix, 2):
        raise ValueError("eye visibility must have shape [..., 2]")
    if not torch.isfinite(region_embeddings).all():
        raise ValueError("region embeddings must be finite")
    if (
        not torch.isfinite(raw_eye_probabilities).all()
        or torch.any(raw_eye_probabilities < 0.0)
        or not torch.allclose(
            raw_eye_probabilities.sum(dim=-1),
            torch.ones(
                prefix,
                device=raw_eye_probabilities.device,
                dtype=raw_eye_probabilities.dtype,
            ),
            atol=1e-4,
        )
    ):
        raise ValueError("raw eye probabilities must be finite normalized rows")
    if not torch.isfinite(eye_visibility).all() or torch.any(
        (eye_visibility < 0.0) | (eye_visibility > 1.0)
    ):
        raise ValueError("eye visibility must be finite and in [0, 1]")
    left_right = region_embeddings[..., 1:3, :].flatten(-2)
    return torch.cat(
        (left_right, raw_eye_probabilities, eye_visibility),
        dim=-1,
    )


def phase_calibrated_closed_probability(
    phase_probabilities: Tensor,
    learned_closed_probability: Tensor,
    *,
    transition_weight: float = 0.5,
) -> Tensor:
    """Recover closure evidence when the binary head is under-confident.

    Closing and opening are partially occluded eye states, so they contribute
    half as much as a fully closed phase.  The dedicated binary head is kept
    whenever it is more confident than this phase-derived projection.
    """

    if phase_probabilities.ndim < 1 or phase_probabilities.shape[-1] != 4:
        raise ValueError("ocular phase probabilities must end in four classes")
    if learned_closed_probability.shape != phase_probabilities.shape[:-1]:
        raise ValueError("learned closure probability must match ocular phases")
    if not 0.0 <= transition_weight <= 1.0:
        raise ValueError("ocular transition weight must be in [0, 1]")
    if (
        not torch.isfinite(phase_probabilities).all()
        or not torch.isfinite(learned_closed_probability).all()
        or torch.any(phase_probabilities < 0.0)
        or torch.any(
            (learned_closed_probability < 0.0)
            | (learned_closed_probability > 1.0)
        )
        or not torch.allclose(
            phase_probabilities.sum(dim=-1),
            torch.ones_like(learned_closed_probability),
            atol=1e-4,
        )
    ):
        raise ValueError("ocular probabilities must be finite and normalized")
    projected = phase_probabilities[..., 2] + transition_weight * (
        phase_probabilities[..., 1] + phase_probabilities[..., 3]
    )
    return torch.maximum(learned_closed_probability, projected).clamp(0.0, 1.0)


class CausalOcularLSTM(nn.Module):
    """Unidirectional eye-phase and closure-continuity estimator."""

    def __init__(
        self,
        *,
        input_dim: int,
        projection_dim: int = 64,
        hidden_size: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, projection_dim, hidden_size, layers) <= 0:
            raise ValueError("ocular LSTM dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("ocular dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.input = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
        )
        self.temporal = nn.LSTM(
            projection_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.phase_head = nn.Linear(hidden_size, 4)
        self.closed_head = nn.Linear(hidden_size, 1)
        self.progress_head = nn.Linear(hidden_size, 1)
        self.reliability_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        sequence: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> OcularLSTMOutput:
        if (
            sequence.ndim != 3
            or sequence.shape[-1] != self.input_dim
            or sequence.shape[1] <= 0
        ):
            raise ValueError(
                "ocular sequence must have shape [batch, time, input_dim]"
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("ocular sequence must be finite")
        temporal, next_hidden = self.temporal(self.input(sequence), hidden)
        return OcularLSTMOutput(
            self.phase_head(temporal),
            self.closed_head(temporal).squeeze(-1),
            self.progress_head(temporal).sigmoid().squeeze(-1),
            self.reliability_head(temporal).squeeze(-1),
            next_hidden,
        )

    def step(
        self,
        features: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[OcularStepOutput, tuple[Tensor, Tensor]]:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "ocular step input must have shape [batch, input_dim]"
            )
        output = self(features.unsqueeze(1), hidden)
        return (
            OcularStepOutput(
                output.phase_logits[:, 0],
                output.closed_logits[:, 0],
                output.progress[:, 0],
                output.reliability_logits[:, 0],
            ),
            output.hidden,
        )


__all__ = [
    "CausalOcularLSTM",
    "OcularLSTMOutput",
    "OcularStepOutput",
    "assemble_ocular_features",
    "phase_calibrated_closed_probability",
]
