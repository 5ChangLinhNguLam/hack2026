"""Importable StudentTTC architecture extracted from the C1 notebook.

The checkpoint was trained with a MobileOne image backbone, causal temporal
shift, scalar telemetry fusion and a causal TCN.  Keep this module aligned with
the architecture metadata embedded in ``C1/student_ttc.pth``.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import Version
import torch
from torch import nn
from torch.nn import functional as F

try:
    if Version(importlib.metadata.version("timm")) < Version("1.0.0"):
        raise RuntimeError("C1 runtime requires timm>=1.0.0")
    import timm
except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "C1 runtime requires timm>=1.0.0; install the c1-runtime extra"
    ) from exc

from .features import FEATURES_USED


SHIFT_FRACTION = 8
TCN_DIMENSION = 256
DEFAULT_TCN_DILATIONS = (1, 2, 4, 8)
TTC_CEILING_SECONDS = 10.0
SCALAR_DIMENSION = 64
DEPTH_SIZE = (28, 28)


def inverse_to_ttc(
    inverse_ttc: torch.Tensor,
    *,
    ceiling_seconds: float = TTC_CEILING_SECONDS,
) -> torch.Tensor:
    return torch.where(
        inverse_ttc > 1.0 / ceiling_seconds,
        1.0 / inverse_ttc.clamp(min=1e-6),
        torch.full_like(inverse_ttc, float("inf")),
    )


def causal_shift(
    values: torch.Tensor,
    frames: int,
    *,
    fraction: int = SHIFT_FRACTION,
) -> torch.Tensor:
    batch_frames, channels, height, width = values.shape
    shifted_channels = max(1, channels // fraction)
    values = values.view(batch_frames // frames, frames, channels, height, width)
    result = values.clone()
    result[:, 1:, :shifted_channels] = values[:, :-1, :shifted_channels]
    result[:, 0, :shifted_channels] = 0
    return result.view(batch_frames, channels, height, width)


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        dimension: int = TCN_DIMENSION,
        dilations: Sequence[int] = DEFAULT_TCN_DILATIONS,
    ) -> None:
        super().__init__()
        # Attribute names intentionally match the training notebook/checkpoint.
        self.dil = tuple(int(value) for value in dilations)
        self.proj = nn.Conv1d(input_dimension, dimension, 1)
        self.dw = nn.ModuleList(
            nn.Conv1d(
                dimension,
                dimension,
                3,
                dilation=dilation,
                groups=dimension,
            )
            for dilation in self.dil
        )
        self.pw = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(dimension, dimension, 1),
                nn.BatchNorm1d(dimension),
                nn.ReLU(),
            )
            for _ in self.dil
        )

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(self.dil)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(values)
        for dilation, depthwise, pointwise in zip(self.dil, self.dw, self.pw):
            hidden = hidden + pointwise(
                depthwise(F.pad(hidden, (2 * dilation, 0)))
            )
        return hidden


class DepthHead(nn.Module):
    """Training-only auxiliary head retained for architecture completeness."""

    def __init__(self, input_channels: int, size: tuple[int, int] = DEPTH_SIZE) -> None:
        super().__init__()
        self.size = size
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            feature_map, size=self.size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(self.net(resized)).squeeze(1)


class StudentTTC(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = "mobileone_s2",
        pretrained: bool = False,
        shift_at: Sequence[int] = (1, 2, 3),
        n_scalar: int = len(FEATURES_USED),
        auxiliary_depth: bool = False,
        tcn_dilations: Sequence[int] = DEFAULT_TCN_DILATIONS,
    ) -> None:
        super().__init__()
        self.n_scalar = int(n_scalar)
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        self._mode = "off"
        self._temporal_length = 1
        self.reset()

        with torch.no_grad():
            vector, feature_map = self._frames(
                torch.zeros(1, 3, 224, 224), want_map=True
            )
            visual_dimension = vector.shape[1]
            map_channels = feature_map.shape[1]

        module_names = [item["module"] for item in self.net.feature_info]
        modules = dict(self.net.named_modules())
        self.shift_pts = [
            module_names[index]
            for index in shift_at
            if 0 <= int(index) < len(module_names)
        ]
        for name in self.shift_pts:
            modules[name].register_forward_hook(self._make_hook(name))

        if self.n_scalar:
            self.scalar = nn.Sequential(
                nn.Linear(self.n_scalar, 64),
                nn.ReLU(),
                nn.Linear(64, SCALAR_DIMENSION),
            )
        self.tcn = CausalTCN(
            visual_dimension + (SCALAR_DIMENSION if self.n_scalar else 0),
            dilations=tcn_dilations,
        )
        self.head_cls = nn.Conv1d(TCN_DIMENSION, 1, 1)
        self.head_ttc = nn.Conv1d(TCN_DIMENSION, 1, 1)
        self.head_depth = DepthHead(map_channels) if auxiliary_depth else None

    @property
    def context_frames(self) -> int:
        return self.tcn.receptive_field

    def _make_hook(self, key: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            if self._mode == "batch":
                return causal_shift(output, self._temporal_length)
            if self._mode == "stream":
                return self._shift_stream(key, output)
            return None

        return hook

    def _frames(
        self, values: torch.Tensor, *, want_map: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.net.forward_features(values)
        vector = self.net.forward_head(feature_map, pre_logits=True)
        return (vector, feature_map) if want_map else vector

    def _fuse(
        self,
        visual: torch.Tensor,
        scalar: torch.Tensor | None,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        hidden = visual.view(batch, frames, -1)
        if self.n_scalar:
            if scalar is None:
                scalar = hidden.new_zeros(batch, frames, self.n_scalar)
            encoded = self.scalar(scalar.to(hidden.dtype).flatten(0, 1)).view(
                batch, frames, -1
            )
            hidden = torch.cat((hidden, encoded), dim=2)
        return hidden.transpose(1, 2)

    def forward(
        self, clip: torch.Tensor, scalar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames = clip.shape[:2]
        self._mode = "batch"
        self._temporal_length = frames
        try:
            visual = self._frames(clip.flatten(0, 1))
        finally:
            self._mode = "off"
        hidden = self.tcn(self._fuse(visual, scalar, batch, frames))
        return (
            self.head_cls(hidden).squeeze(1),
            F.softplus(self.head_ttc(hidden)).squeeze(1),
        )

    def reset(self) -> None:
        self._shift_cache: dict[str, torch.Tensor] = {}
        self._feature_buffer: torch.Tensor | None = None
        self._observations = 0

    def _shift_stream(self, key: str, values: torch.Tensor) -> torch.Tensor:
        count = max(1, values.shape[1] // SHIFT_FRACTION)
        previous = self._shift_cache.get(key)
        result = values.clone()
        self._shift_cache[key] = values[:, :count].detach().clone()
        result[:, :count] = (
            torch.zeros_like(values[:, :count]) if previous is None else previous
        )
        return result

    @torch.no_grad()
    def step(
        self, frame: torch.Tensor, scalar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._mode = "stream"
        try:
            visual = self._frames(frame)
        finally:
            self._mode = "off"
        fused = self._fuse(
            visual,
            None if scalar is None else scalar.view(1, 1, -1),
            1,
            1,
        )
        if self._feature_buffer is None:
            self._feature_buffer = torch.zeros(
                1,
                fused.shape[1],
                self.context_frames,
                device=fused.device,
                dtype=fused.dtype,
            )
        self._feature_buffer = torch.cat(
            (self._feature_buffer[:, :, 1:], fused), dim=2
        )
        self._observations = min(self._observations + 1, self.context_frames)
        hidden = self.tcn(
            self._feature_buffer[:, :, -self._observations :]
        )[:, :, -1:]
        return (
            self.head_cls(hidden).flatten(),
            F.softplus(self.head_ttc(hidden)).flatten(),
        )

    @torch.no_grad()
    def predict(
        self, frame: torch.Tensor, scalar: torch.Tensor | None = None
    ) -> tuple[float, float]:
        logit, inverse_ttc = self.step(frame, scalar)
        return (
            float(torch.sigmoid(logit)),
            float(inverse_to_ttc(inverse_ttc)),
        )


def load_student_ttc(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[StudentTTC, Mapping[str, Any]]:
    """Load the repository checkpoint and fail closed on contract drift."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("C1 checkpoint must contain a metadata mapping")
    required = {"state", "backbone", "shift_at", "n_scalar", "feat_use"}
    missing_metadata = sorted(required - set(checkpoint))
    if missing_metadata:
        raise ValueError(f"C1 checkpoint missing metadata: {missing_metadata}")
    feature_order = tuple(str(value) for value in checkpoint["feat_use"])
    if feature_order != FEATURES_USED:
        raise ValueError(
            "C1 checkpoint feature order differs from runtime: "
            f"checkpoint={feature_order}, runtime={FEATURES_USED}"
        )
    if int(checkpoint["n_scalar"]) != len(FEATURES_USED):
        raise ValueError(
            "C1 checkpoint scalar width differs from runtime feature contract"
        )

    model = StudentTTC(
        backbone=str(checkpoint["backbone"]),
        pretrained=False,
        shift_at=tuple(int(value) for value in checkpoint["shift_at"]),
        n_scalar=int(checkpoint["n_scalar"]),
        auxiliary_depth=False,
        tcn_dilations=tuple(
            int(value)
            for value in checkpoint.get("tcn_dil", DEFAULT_TCN_DILATIONS)
        ),
    )
    state = {
        str(key): value
        for key, value in checkpoint["state"].items()
        if not str(key).startswith("head_depth.")
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.reset()
    return model, checkpoint


__all__ = [
    "CausalTCN",
    "DepthHead",
    "StudentTTC",
    "TTC_CEILING_SECONDS",
    "causal_shift",
    "inverse_to_ttc",
    "load_student_ttc",
]
