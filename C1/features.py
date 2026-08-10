"""Causal scalar feature contract used by the C1 StudentTTC checkpoint.

This module is the importable source of truth for inference.  The normalization
matches the ``# @arch`` feature cell that produced ``student_ttc.pth``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "speed",
    "accel",
    "jerk",
    "lat_accel",
    "speed_ratio",
    "has_limit",
    "n_targets",
    "d_targets",
    "cloud",
    "rain",
    "wet",
    "fog",
    "sun",
)

# Order is part of the checkpoint contract.  Do not sort or otherwise alter it.
FEATURES_USED = (
    "speed",
    "accel",
    "jerk",
    "lat_accel",
    "speed_ratio",
    "has_limit",
    "cloud",
    "rain",
    "wet",
    "fog",
    "sun",
)


def _numbers(values: Sequence[object], default: float = 0.0) -> np.ndarray:
    source = np.asarray(values, dtype=object)
    result = np.full(len(source), float(default), dtype=np.float32)
    for index, value in enumerate(source):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result[index] = number
    return result


def build_scalars(
    speed_kmh: Sequence[object],
    accel: Sequence[object],
    jerk: Sequence[object],
    lat_accel: Sequence[object],
    n_targets: Sequence[object],
    *,
    speed_limit_kmh: object = None,
    weather: Mapping[str, object] | None = None,
    feature_order: Sequence[str] = FEATURES_USED,
) -> np.ndarray:
    """Return normalized scalar features with shape ``(T, N)``.

    All transforms are causal and use fixed constants; no statistics from the
    complete trip are consulted.
    """

    unknown = tuple(name for name in feature_order if name not in FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown C1 scalar feature(s): {unknown}")

    speed = _numbers(speed_kmh)
    acceleration = _numbers(accel)
    jerk_values = _numbers(jerk)
    lateral = _numbers(lat_accel)
    target_count = _numbers(n_targets)
    lengths = {
        len(speed),
        len(acceleration),
        len(jerk_values),
        len(lateral),
        len(target_count),
    }
    if len(lengths) != 1:
        raise ValueError("all C1 scalar input sequences must have equal length")

    try:
        speed_limit = float(speed_limit_kmh) if speed_limit_kmh not in (None, "", 0) else 0.0
    except (TypeError, ValueError):
        speed_limit = 0.0
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        speed_limit = 0.0

    conditions = weather or {}
    count = len(speed)
    ones = np.ones(count, dtype=np.float32)
    delta_targets = np.concatenate(([0.0], np.diff(target_count))).astype(np.float32)
    raw_speed_kmh = _numbers(speed_kmh)

    columns = {
        "speed": speed / 3.6 / 20.0,
        "accel": np.tanh(acceleration / 3.0),
        "jerk": np.tanh(jerk_values / 10.0),
        "lat_accel": np.tanh(np.abs(lateral) / 2.0),
        "speed_ratio": (
            np.clip(raw_speed_kmh / speed_limit, 0.0, 2.0) / 2.0
            if speed_limit > 0.0
            else np.zeros(count, dtype=np.float32)
        ),
        "has_limit": ones * (1.0 if speed_limit > 0.0 else 0.0),
        "n_targets": np.log1p(target_count) / 3.0,
        "d_targets": np.tanh(delta_targets),
        "cloud": ones * float(conditions.get("w_cloud", 0.0)) / 100.0,
        "rain": ones * float(conditions.get("w_rain", 0.0)) / 100.0,
        "wet": ones * float(conditions.get("w_wet", 0.0)) / 100.0,
        "fog": ones * float(conditions.get("w_fog", 0.0)) / 100.0,
        "sun": ones
        * float(
            np.clip(
                (float(conditions.get("w_sun_alt", 75.0)) + 90.0) / 165.0,
                0.0,
                1.0,
            )
        ),
    }
    if count == 0:
        return np.empty((0, len(feature_order)), dtype=np.float32)
    return np.stack(
        [np.asarray(columns[name], dtype=np.float32) for name in feature_order], axis=1
    )


def weather_features(metadata: Mapping[str, Any]) -> dict[str, float]:
    weather = metadata.get("weather") or {}
    return {
        "w_cloud": float(weather.get("cloudiness", 0.0)),
        "w_rain": float(weather.get("precipitation", 0.0)),
        "w_wet": max(
            float(weather.get("wetness", 0.0)),
            float(weather.get("precipitation_deposits", 0.0)),
        ),
        "w_fog": float(weather.get("fog_density", 0.0)),
        "w_sun_alt": float(weather.get("sun_altitude_angle", 75.0)),
    }


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


@dataclass
class ScalarFeatureStream:
    """Build one checkpoint-compatible feature vector per sampled C1 frame."""

    metadata: Mapping[str, Any]
    sample_hz: float = 10.0
    feature_order: tuple[str, ...] = FEATURES_USED

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_hz) or self.sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive and finite")
        self.reset()

    def reset(self) -> None:
        self._previous_speed_mps: float | None = None
        self._previous_acceleration = 0.0

    def step(
        self,
        ego: Mapping[str, Any] | None,
        *,
        target_count: int = 0,
    ) -> np.ndarray:
        ego_values = ego or {}
        speed_kmh = _finite(ego_values.get("speed_kmh"))
        speed_mps = speed_kmh / 3.6
        if self._previous_speed_mps is None:
            acceleration = 0.0
            jerk = 0.0
        else:
            acceleration = (
                speed_mps - self._previous_speed_mps
            ) * self.sample_hz
            jerk = (
                acceleration - self._previous_acceleration
            ) * self.sample_hz

        self._previous_speed_mps = speed_mps
        self._previous_acceleration = acceleration
        features = build_scalars(
            [speed_kmh],
            [acceleration],
            [jerk],
            [_finite(ego_values.get("lateral_accel"))],
            [target_count],
            speed_limit_kmh=self.metadata.get("speed_limit_kmh"),
            weather=weather_features(self.metadata),
            feature_order=self.feature_order,
        )
        return features[0]


__all__ = [
    "FEATURE_NAMES",
    "FEATURES_USED",
    "ScalarFeatureStream",
    "build_scalars",
    "weather_features",
]
