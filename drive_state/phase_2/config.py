from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EyeTemporalConfig:
    fps: float = 20.0
    sequence_length: int = 15
    eye_height: int = 48
    eye_width: int = 96
    embedding_dim: int = 128
    tcn_channels: int = 128
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    phase_weight: float = 1.0
    closed_weight: float = 0.4
    moving_weight: float = 0.2
    undefined_index: int = -100
    transition_fraction: float = 0.5
    visibility_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.sequence_length < 3 or self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd and at least 3")
        if self.eye_height <= 0 or self.eye_width <= 0:
            raise ValueError("eye crop dimensions must be positive")
        if self.embedding_dim <= 0 or self.tcn_channels <= 0:
            raise ValueError("model dimensions must be positive")
        if not self.tcn_dilations or any(d <= 0 for d in self.tcn_dilations):
            raise ValueError("tcn_dilations must contain positive integers")
        if any(weight < 0 for weight in (self.phase_weight, self.closed_weight, self.moving_weight)):
            raise ValueError("loss weights must be non-negative")
        if not 0 <= self.transition_fraction <= 1:
            raise ValueError("transition_fraction must be in [0, 1]")
        if not 0 <= self.visibility_floor <= 1:
            raise ValueError("visibility_floor must be in [0, 1]")


def load_eye_config(path: Path | str) -> EyeTemporalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    allowed = {field.name for field in fields(EyeTemporalConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    if "tcn_dilations" in raw:
        raw["tcn_dilations"] = tuple(int(value) for value in raw["tcn_dilations"])
    return EyeTemporalConfig(**raw)
