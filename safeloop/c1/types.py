"""Small, framework-independent data contracts for the C1 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in source-image pixels


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class TrackRisk:
    track_id: int
    label: str
    confidence: float
    bbox: BBox
    collision_relevant: bool
    predicted_ttc_s: float
    scale_ttc_s: float
    range_ttc_s: float
    range_closing_speed_mps: float
    range_trend_confidence: float
    lateral_ttc_s: float
    ego_fallback_ttc_s: float
    estimated_distance_m: float


@dataclass(frozen=True)
class C1FramePrediction:
    frame_id: int
    timestamp: float
    predicted_ttc_s: float
    is_warning: bool
    risks: Tuple[TrackRisk, ...]
    latency_ms: float

    def submission_row(self) -> dict[str, object]:
        ttc: object = "inf"
        if math.isfinite(self.predicted_ttc_s):
            ttc = round(self.predicted_ttc_s, 3)
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "predicted_ttc": ttc,
        }
