"""Confidence-aware pseudo labels for the ten redacted C1 trips.

The teacher consumes only detections from ``image_2`` and allowed ego
kinematics.  It deliberately has no API for ground truth, event schedules,
targets, stereo images or recorded depth.  Its CSV output is named
``pseudo_ttc`` so it cannot be mistaken for organizer ground truth.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tripkit import TripLoader

from .tracker import MonocularTTCTracker, TrackerConfig
from .types import Detection, TrackRisk

ROOT = Path(__file__).resolve().parents[2]
PSEUDO_SOURCE = "pseudo_rgb_physics_ensemble_v1"


@dataclass(frozen=True)
class PseudoLabel:
    frame_id: int
    timestamp: float
    pseudo_ttc_s: float
    confidence: float
    support_count: int
    danger_vote_count: int
    agreement: float
    source: str = PSEUDO_SOURCE

    def row(self) -> dict[str, object]:
        ttc: object = "inf"
        if math.isfinite(self.pseudo_ttc_s):
            ttc = round(self.pseudo_ttc_s, 3)
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "pseudo_ttc": ttc,
            "confidence": round(self.confidence, 4),
            "support_count": self.support_count,
            "danger_vote_count": self.danger_vote_count,
            "agreement": round(self.agreement, 4),
            "label_source": self.source,
        }


def build_teacher_trackers(
    *, focal_y_px: float, focal_x_px: float, principal_x_px: float
) -> list[MonocularTTCTracker]:
    strict_range = {
        "min_history": 2,
        "enable_range_ttc": True,
        "min_range_history": 3,
        "range_recent_observations": 6,
        "min_range_decreasing_fraction": 0.80,
        "max_range_slope_relative_mad": 0.50,
        "min_range_box_height_px": 20.0,
    }
    configs = [
        TrackerConfig(min_history=2),
        TrackerConfig(**strict_range),
        TrackerConfig(
            **strict_range,
            enable_ego_fallback=True,
            min_fallback_box_height_px=20.0,
        ),
    ]
    return [
        MonocularTTCTracker(
            config,
            focal_y_px=focal_y_px,
            focal_x_px=focal_x_px,
            principal_x_px=principal_x_px,
        )
        for config in configs
    ]


class PseudoLabelEnsemble:
    """Fuse three independent physical TTC configurations.

    A single finite vote remains low-confidence.  Agreement is measured in
    inverse TTC because urgency ``1/TTC`` is bounded around normal driving and
    is also the organizer's stable evaluation quantity.
    """

    def __init__(self, trackers: Sequence[MonocularTTCTracker]) -> None:
        if len(trackers) < 2:
            raise ValueError("Pseudo-label ensemble cần ít nhất 2 tracker")
        self.trackers = tuple(trackers)

    def reset(self) -> None:
        for tracker in self.trackers:
            tracker.reset()

    def update(
        self,
        detections: Sequence[Detection] | None,
        *,
        frame_id: int,
        timestamp: float,
        image_shape: tuple[int, int],
        ego_speed_kmh: float,
    ) -> PseudoLabel:
        risks_by_teacher: list[list[TrackRisk]] = []
        for tracker in self.trackers:
            kwargs = {
                "timestamp": timestamp,
                "image_shape": image_shape,
                "ego_speed_kmh": ego_speed_kmh,
            }
            risks = (
                tracker.update(detections, **kwargs)
                if detections is not None
                else tracker.predict(**kwargs)
            )
            risks_by_teacher.append(risks)
        return self._fuse(frame_id, timestamp, risks_by_teacher)

    def _fuse(
        self,
        frame_id: int,
        timestamp: float,
        risks_by_teacher: Sequence[Sequence[TrackRisk]],
    ) -> PseudoLabel:
        urgencies: list[float] = []
        confidences: list[float] = []
        danger_votes = 0
        relevant_without_ttc = 0
        for risks in risks_by_teacher:
            finite = [risk for risk in risks if math.isfinite(risk.predicted_ttc_s)]
            if not finite:
                relevant_without_ttc += int(any(risk.collision_relevant for risk in risks))
                continue
            best = min(finite, key=lambda risk: risk.predicted_ttc_s)
            urgency = 1.0 / max(0.1, best.predicted_ttc_s)
            urgencies.append(urgency)
            confidences.append(best.confidence)
            danger_votes += int(best.predicted_ttc_s < 2.0)

        support = len(urgencies)
        if support == 0:
            # Agreement on normality is useful but is less trustworthy when
            # road users are present and the geometry simply failed to fit.
            confidence = 0.80 if relevant_without_ttc == 0 else 0.55
            return PseudoLabel(
                frame_id,
                timestamp,
                float("inf"),
                confidence,
                0,
                0,
                1.0 if relevant_without_ttc == 0 else 0.5,
            )

        median_urgency = float(np.median(urgencies))
        relative_spread = float(
            np.median(np.abs(np.asarray(urgencies) - median_urgency))
            / max(median_urgency, 1e-3)
        )
        agreement = 1.0 / (1.0 + relative_spread)
        support_ratio = support / len(risks_by_teacher)
        detector_confidence = float(np.mean(confidences))
        confidence = support_ratio * agreement * detector_confidence
        if support == 1:
            confidence *= 0.35
        pseudo_ttc = 1.0 / max(median_urgency, 1e-3)
        return PseudoLabel(
            frame_id,
            timestamp,
            pseudo_ttc,
            min(1.0, confidence),
            support,
            danger_votes,
            agreement,
        )


def load_detection_cache(path: str | Path) -> tuple[int, Mapping[int, list[Detection]]]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    stride = int(payload.get("stride") or 1)
    rows: dict[int, list[Detection]] = {}
    for row in payload.get("rows") or []:
        rows[int(row["frame_id"])] = [
            Detection(
                int(item["class_id"]),
                str(item["label"]),
                float(item["confidence"]),
                tuple(float(value) for value in item["bbox"]),
            )
            for item in row.get("detections") or []
        ]
    return stride, rows


def generate_pseudo_labels(
    loader: TripLoader,
    detections_by_frame: Mapping[int, list[Detection]],
    ensemble: PseudoLabelEnsemble,
    *,
    output_csv: str | Path,
    detection_confidence: float = 0.20,
) -> dict[str, object]:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ensemble.reset()
    labels: list[PseudoLabel] = []
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "frame_id",
                "timestamp",
                "pseudo_ttc",
                "confidence",
                "support_count",
                "danger_vote_count",
                "agreement",
                "label_source",
            ],
        )
        writer.writeheader()
        for frame_id in range(loader.n_frames):
            raw = loader.raw_frame(frame_id)
            cached = detections_by_frame.get(frame_id)
            detections = (
                [item for item in cached if item.confidence >= detection_confidence]
                if cached is not None
                else None
            )
            ego = raw.get("ego") or {}
            label = ensemble.update(
                detections,
                frame_id=frame_id,
                timestamp=float(raw.get("timestamp", frame_id / loader.fps)),
                image_shape=(loader.calib.height, loader.calib.width),
                ego_speed_kmh=float(ego.get("speed_kmh") or 0.0),
            )
            writer.writerow(label.row())
            labels.append(label)

    high_confidence = sum(item.confidence >= 0.70 for item in labels)
    finite = sum(math.isfinite(item.pseudo_ttc_s) for item in labels)
    danger = sum(item.pseudo_ttc_s < 2.0 for item in labels)
    return {
        "trip_id": loader.trip_id,
        "frames": len(labels),
        "finite_pseudo_ttc": finite,
        "danger_pseudo_frames": danger,
        "high_confidence_frames": high_confidence,
        "high_confidence_fraction": round(high_confidence / max(1, len(labels)), 4),
        "output": str(output_csv),
        "label_source": PSEUDO_SOURCE,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sinh pseudo-label C1 từ image_2; không dùng ground truth/event/depth."
    )
    parser.add_argument("trip_dir")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loader = TripLoader(args.trip_dir)
        cache_path = args.cache or (
            ROOT / "predictions/c1_detection_cache"
            / f"{loader.trip_id}.stride3.conf020.json.gz"
        )
        _, detections = load_detection_cache(cache_path)
        ensemble = PseudoLabelEnsemble(build_teacher_trackers(
            focal_y_px=loader.calib.fy,
            focal_x_px=loader.calib.fx,
            principal_x_px=loader.calib.cx,
        ))
        output = args.output or (
            ROOT / "predictions/c1_pseudo_labels" / f"{loader.trip_id}.csv"
        )
        report = generate_pseudo_labels(
            loader,
            detections,
            ensemble,
            output_csv=output,
            detection_confidence=args.confidence,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Lỗi sinh pseudo-label: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
