"""Small causal GRU heads for monocular TTC estimation and physics fusion."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_features import FEATURE_DIM, inverse_ttc


PARAMETER_BUDGET = 500_000
ALLOWED_SEQUENCE_LENGTHS = (8, 16, 24)
ALLOWED_HIDDEN_SIZES = (32, 64)
ALLOWED_GRU_LAYERS = (1, 2)


@dataclass(frozen=True)
class TemporalModelConfig:
    sequence_length: int = 16
    feature_dim: int = FEATURE_DIM
    hidden_size: int = 64
    gru_layers: int = 1
    dropout: float = 0.10
    residual_limit: float = 2.0
    max_inverse_ttc: float = 10.0

    def __post_init__(self) -> None:
        if self.sequence_length not in ALLOWED_SEQUENCE_LENGTHS:
            raise ValueError(
                f"sequence_length must be one of {ALLOWED_SEQUENCE_LENGTHS}"
            )
        if self.feature_dim != FEATURE_DIM:
            raise ValueError(f"feature_dim must match the frozen schema ({FEATURE_DIM})")
        if self.hidden_size not in ALLOWED_HIDDEN_SIZES:
            raise ValueError(f"hidden_size must be one of {ALLOWED_HIDDEN_SIZES}")
        if self.gru_layers not in ALLOWED_GRU_LAYERS:
            raise ValueError(f"gru_layers must be one of {ALLOWED_GRU_LAYERS}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.residual_limit <= 0.0 or self.max_inverse_ttc <= 0.0:
            raise ValueError("residual_limit and max_inverse_ttc must be positive")

    @property
    def tag(self) -> str:
        return f"seq{self.sequence_length}-h{self.hidden_size}-l{self.gru_layers}"

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "TemporalModelConfig":
        return cls(**values)  # type: ignore[arg-type]


DEFAULT_CANDIDATES: tuple[TemporalModelConfig, ...] = (
    TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1),
    TemporalModelConfig(sequence_length=16, hidden_size=64, gru_layers=1),
    TemporalModelConfig(sequence_length=24, hidden_size=64, gru_layers=2),
)


@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-feature statistics fitted on training trips only."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != (FEATURE_DIM,) or self.scale.shape != (FEATURE_DIM,):
            raise ValueError(f"Normalizer arrays must have shape ({FEATURE_DIM},)")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Normalizer statistics must be finite")
        if np.any(self.scale <= 0.0):
            raise ValueError("Normalizer scale must be positive")

    @classmethod
    def fit(cls, frame_features: Iterable[np.ndarray]) -> "FeatureNormalizer":
        arrays = [np.asarray(values, dtype=np.float32) for values in frame_features]
        if not arrays:
            raise ValueError("Cannot fit a normalizer without training features")
        if any(values.ndim != 2 or values.shape[1] != FEATURE_DIM for values in arrays):
            raise ValueError("Every feature matrix must be [frames, FEATURE_DIM]")
        combined = np.concatenate(arrays, axis=0).astype(np.float64)
        mean = combined.mean(axis=0)
        scale = combined.std(axis=0)
        # Constant/binary-absent features remain unchanged rather than being
        # amplified by a tiny denominator.
        scale[scale < 1e-6] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != FEATURE_DIM:
            raise ValueError(f"Last feature dimension must be {FEATURE_DIM}")
        transformed = (values - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Normalized features contain non-finite values")
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, Sequence[float]]) -> "FeatureNormalizer":
        return cls(
            np.asarray(values["mean"], dtype=np.float32),
            np.asarray(values["scale"], dtype=np.float32),
        )


class CausalTemporalTTC(nn.Module):
    """A compact causal GRU with temporal and physics-residual heads.

    The GRU sees only label-free detector/tracker geometry and ego features.
    Physics inverse-TTC is *not* an input to this network; it is added to the
    learned residual outside ``forward`` for the fusion ablation.

    Output channels are, in order:

    1. temporal-only inverse TTC (non-negative),
    2. signed raw residual for physics fusion,
    3. temporal-only danger logit,
    4. fusion danger logit.
    """

    def __init__(self, config: TemporalModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_size),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.temporal_inverse_head = nn.Linear(config.hidden_size, 1)
        self.residual_head = nn.Linear(config.hidden_size, 1)
        self.temporal_danger_head = nn.Linear(config.hidden_size, 1)
        self.fusion_danger_head = nn.Linear(config.hidden_size, 1)
        self._initialize_heads()
        if self.parameter_count >= PARAMETER_BUDGET:
            raise ValueError(
                f"Temporal model has {self.parameter_count:,} parameters; "
                f"budget is < {PARAMETER_BUDGET:,}"
            )

    def _initialize_heads(self) -> None:
        # Begin conservatively near 'no hazard'.  Fusion begins exactly at the
        # physics baseline because its residual weights and bias are zero.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.constant_(self.temporal_inverse_head.bias, -3.0)
        nn.init.constant_(self.temporal_danger_head.bias, -2.0)
        nn.init.constant_(self.fusion_danger_head.bias, -2.0)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                "features must have shape [batch, time, "
                f"{self.config.feature_dim}]"
            )
        projected = self.input_projection(features)
        hidden, _ = self.gru(projected)
        hidden = self.output_dropout(hidden)
        temporal_inverse = F.softplus(self.temporal_inverse_head(hidden))
        temporal_inverse = temporal_inverse.clamp(max=self.config.max_inverse_ttc)
        return torch.cat(
            (
                temporal_inverse,
                self.residual_head(hidden),
                self.temporal_danger_head(hidden),
                self.fusion_danger_head(hidden),
            ),
            dim=-1,
        )


def gather_current(sequence_outputs: Tensor, lengths: Tensor) -> Tensor:
    """Gather the last valid causal output from left-aligned windows."""

    if sequence_outputs.ndim != 3:
        raise ValueError("sequence_outputs must be [batch, time, channels]")
    if lengths.ndim != 1 or len(lengths) != len(sequence_outputs):
        raise ValueError("lengths must contain one value per batch item")
    if torch.any(lengths < 1) or torch.any(lengths > sequence_outputs.shape[1]):
        raise ValueError("Every sequence length must be in [1, time]")
    indices = lengths.to(device=sequence_outputs.device, dtype=torch.long) - 1
    batch = torch.arange(len(sequence_outputs), device=sequence_outputs.device)
    return sequence_outputs[batch, indices]


def fused_inverse_ttc(
    physics_inverse: Tensor,
    raw_residual: Tensor,
    *,
    residual_limit: float = 2.0,
    max_inverse_ttc: float = 10.0,
) -> Tensor:
    """Add a bounded signed residual, retaining a physically meaningful base."""

    residual = residual_limit * torch.tanh(raw_residual)
    return torch.clamp(physics_inverse + residual, min=0.0, max=max_inverse_ttc)


def decode_current_outputs(
    current_outputs: Tensor,
    physics_inverse: Tensor,
    config: TemporalModelConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return temporal inverse, fused inverse, and their danger logits."""

    if current_outputs.shape[-1] != 4:
        raise ValueError("Expected four temporal output channels")
    temporal_inverse = current_outputs[..., 0]
    fused_inverse = fused_inverse_ttc(
        physics_inverse,
        current_outputs[..., 1],
        residual_limit=config.residual_limit,
        max_inverse_ttc=config.max_inverse_ttc,
    )
    return (
        temporal_inverse,
        fused_inverse,
        current_outputs[..., 2],
        current_outputs[..., 3],
    )


def calibrated_ttc(
    predicted_inverse: np.ndarray,
    danger_logit: np.ndarray,
    *,
    danger_threshold: float,
    max_finite_ttc_s: float = 10.0,
    danger_ttc_s: float = 1.999,
) -> np.ndarray:
    """Decode regression + train-fold-calibrated danger classification.

    The official evaluator defines danger as ``predicted_ttc < 2``.  This
    decoder makes that classification agree with the separately calibrated
    danger head while preserving the regressed TTC everywhere else.
    """

    if not 0.0 < danger_threshold < 1.0:
        raise ValueError("danger_threshold must be strictly between 0 and 1")
    if not 0.0 < danger_ttc_s < 2.0:
        raise ValueError("danger_ttc_s must be in (0, 2)")
    inverse = np.asarray(predicted_inverse, dtype=np.float64)
    logits = np.asarray(danger_logit, dtype=np.float64)
    if inverse.shape != logits.shape:
        raise ValueError("predicted_inverse and danger_logit shapes must match")
    probability = np.empty_like(logits)
    positive = logits >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    probability[~positive] = exp_values / (1.0 + exp_values)
    danger = probability >= danger_threshold

    output = np.full(inverse.shape, np.inf, dtype=np.float64)
    finite = np.isfinite(inverse) & (inverse >= 1.0 / max_finite_ttc_s)
    output[finite] = 1.0 / np.maximum(inverse[finite], 1e-6)
    # Keep official danger classification exactly aligned with the learned
    # probability gate.  2.0 itself is non-danger because evaluator uses '<'.
    output[danger] = np.minimum(output[danger], danger_ttc_s)
    output[~danger & (output < 2.0)] = 2.0
    return output


def build_causal_windows(
    normalized_frame_features: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create left-aligned, past-only windows for every frame.

    For frame ``i``, the window contains only ``max(0, i-L+1)..i``.  Padding
    is placed *after* the current frame; ``gather_current`` reads at
    ``length-1``, so padded future positions can never affect a prediction.
    """

    values = np.asarray(normalized_frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must be [frames, {FEATURE_DIM}]")
    if sequence_length not in ALLOWED_SEQUENCE_LENGTHS:
        raise ValueError(f"sequence_length must be one of {ALLOWED_SEQUENCE_LENGTHS}")
    n_frames = len(values)
    windows = np.zeros((n_frames, sequence_length, FEATURE_DIM), dtype=np.float32)
    lengths = np.empty(n_frames, dtype=np.int64)
    for frame_id in range(n_frames):
        start = max(0, frame_id - sequence_length + 1)
        history = values[start : frame_id + 1]
        lengths[frame_id] = len(history)
        windows[frame_id, : len(history)] = history
    return windows, lengths


@dataclass(frozen=True)
class TemporalPrediction:
    temporal_ttc_s: float
    fused_ttc_s: float
    temporal_danger_probability: float
    fusion_danger_probability: float


class CausalTemporalPredictor:
    """Stateful inference wrapper; it has no API through which to receive GT."""

    def __init__(
        self,
        model: CausalTemporalTTC,
        normalizer: FeatureNormalizer,
        *,
        temporal_threshold: float,
        fusion_threshold: float,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.temporal_threshold = temporal_threshold
        self.fusion_threshold = fusion_threshold
        self.device = torch.device(device)
        self._history: deque[np.ndarray] = deque(maxlen=model.config.sequence_length)

    def reset(self) -> None:
        self._history.clear()

    @torch.inference_mode()
    def step(self, feature_row: np.ndarray, physics_ttc_s: float) -> TemporalPrediction:
        row = np.asarray(feature_row, dtype=np.float32)
        if row.shape != (FEATURE_DIM,):
            raise ValueError(f"feature_row must have shape ({FEATURE_DIM},)")
        self._history.append(self.normalizer.transform(row))
        window = np.zeros(
            (1, self.model.config.sequence_length, FEATURE_DIM), dtype=np.float32
        )
        history = np.stack(tuple(self._history), axis=0)
        window[0, : len(history)] = history
        features = torch.from_numpy(window).to(self.device)
        lengths = torch.tensor([len(history)], device=self.device)
        current = gather_current(self.model(features), lengths)
        physics_inverse = torch.from_numpy(
            inverse_ttc(np.asarray([physics_ttc_s], dtype=np.float32))
        ).to(self.device)
        temporal_inv, fused_inv, temporal_logit, fusion_logit = decode_current_outputs(
            current, physics_inverse, self.model.config
        )
        temporal_np = temporal_inv.cpu().numpy()
        fused_np = fused_inv.cpu().numpy()
        temporal_logit_np = temporal_logit.cpu().numpy()
        fusion_logit_np = fusion_logit.cpu().numpy()
        temporal_ttc = calibrated_ttc(
            temporal_np,
            temporal_logit_np,
            danger_threshold=self.temporal_threshold,
        )[0]
        fusion_ttc = calibrated_ttc(
            fused_np,
            fusion_logit_np,
            danger_threshold=self.fusion_threshold,
        )[0]
        return TemporalPrediction(
            temporal_ttc_s=float(temporal_ttc),
            fused_ttc_s=float(fusion_ttc),
            temporal_danger_probability=float(torch.sigmoid(temporal_logit)[0].item()),
            fusion_danger_probability=float(torch.sigmoid(fusion_logit)[0].item()),
        )


def checkpoint_payload(
    model: CausalTemporalTTC,
    normalizer: FeatureNormalizer,
    *,
    temporal_threshold: float,
    fusion_threshold: float,
    train_trip_ids: Sequence[str],
    outer_test_trip_id: str,
    seed: int,
) -> dict[str, object]:
    """Serializable model state with explicit split provenance (no labels)."""

    if outer_test_trip_id in train_trip_ids:
        raise ValueError("Outer-test trip cannot appear in checkpoint train provenance")
    return {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "normalizer": normalizer.to_dict(),
        "temporal_threshold": float(temporal_threshold),
        "fusion_threshold": float(fusion_threshold),
        "train_trip_ids": list(train_trip_ids),
        "outer_test_trip_id": outer_test_trip_id,
        "seed": int(seed),
        "parameter_count": model.parameter_count,
    }


def export_temporal_onnx(
    model: CausalTemporalTTC,
    output_path: str | Path,
    *,
    opset_version: int = 20,
) -> Path:
    """Export the fixed-window causal head using PyTorch's ONNX exporter.

    Input is deliberately fixed to ``[1, sequence_length, 48]`` for the edge
    runtime.  The pinned onnxscript/onnx-ir/ONNX Runtime stack validates the
    dynamo exporter at opset 20.  Physics fusion and threshold calibration
    remain lightweight host-side operations around the four raw channels.
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_device = next(model.parameters()).device
    export_model = model.to("cpu").eval()
    dummy = torch.zeros(
        1,
        model.config.sequence_length,
        model.config.feature_dim,
        dtype=torch.float32,
    )
    try:
        torch.onnx.export(
            export_model,
            (dummy,),
            str(destination),
            input_names=["features"],
            output_names=["temporal_outputs"],
            opset_version=opset_version,
            dynamo=True,
            external_data=False,
        )
    finally:
        model.to(original_device)
    return destination
