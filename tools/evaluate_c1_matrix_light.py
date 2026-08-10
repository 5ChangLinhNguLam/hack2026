#!/usr/bin/env python3
"""Offline matrix-light evaluation on the six labelled development trips."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safeloop.c1.matrix_light import MatrixLightController, MatrixLightOfflineEvaluator
from safeloop.c1.lane import LaneDetector
from safeloop.c1.pseudo_label import load_detection_cache
from safeloop.c1.tracker import MonocularTTCTracker, TrackerConfig
from safeloop.c1.types import BBox, C1FramePrediction
from tripkit import TripLoader
from tripkit.types import KittiLabel


def _label_class(label: KittiLabel) -> str:
    return {"Pedestrian": "walker", "Cyclist": "bike", "Car": "vehicle"}.get(
        label.type, label.type.lower()
    )


def _project_bbox(label: KittiLabel, loader: TripLoader) -> BBox:
    z = max(label.z, 0.1)
    center_x = loader.calib.fx * label.x / z + loader.calib.cx
    bottom_y = loader.calib.fy * label.y / z + loader.calib.cy
    top_y = loader.calib.fy * (label.y - label.height) / z + loader.calib.cy
    half_width = loader.calib.fx * label.width / (2.0 * z)
    return (
        max(0.0, center_x - half_width),
        max(0.0, top_y),
        min(float(loader.calib.width), center_x + half_width),
        min(float(loader.calib.height), bottom_y),
    )


def _danger_bboxes(loader: TripLoader, frame_id: int) -> list[BBox]:
    raw = loader.raw_frame(frame_id)
    targets = [
        target for target in raw.get("targets") or []
        if target.get("in_collision_cone")
        and math.isfinite(float(target.get("ttc_2d", float("inf"))))
        and float(target["ttc_2d"]) <= 3.0
    ]
    labels = [label for label in loader.frame(frame_id, cache_images=False).labels if label.z > 0]
    boxes: list[BBox] = []
    used: set[int] = set()
    for target in targets:
        candidates = [
            (abs(label.z - float(target.get("longitudinal_distance") or label.z)), index)
            for index, label in enumerate(labels)
            if index not in used and _label_class(label) == target.get("target_class")
        ]
        if not candidates:
            continue
        _, index = min(candidates)
        used.add(index)
        boxes.append(_project_bbox(labels[index], loader))
    return boxes


def evaluate_trip(trip_dir: Path, cache_dir: Path) -> dict[str, object]:
    loader = TripLoader(trip_dir)
    _, cache = load_detection_cache(
        cache_dir / f"{loader.trip_id}.stride3.conf020.json.gz"
    )
    tracker = MonocularTTCTracker(
        TrackerConfig(
            min_history=2,
            enable_range_ttc=True,
            min_range_history=3,
            min_range_decreasing_fraction=0.80,
            max_range_slope_relative_mad=0.50,
            min_range_box_height_px=20.0,
        ),
        focal_y_px=loader.calib.fy,
        focal_x_px=loader.calib.fx,
        principal_x_px=loader.calib.cx,
    )
    controller = MatrixLightController(
        focal_x_px=loader.calib.fx,
        focal_y_px=loader.calib.fy,
        principal_x_px=loader.calib.cx,
        principal_y_px=loader.calib.cy,
    )
    evaluator = MatrixLightOfflineEvaluator(
        (loader.calib.height, loader.calib.width),
        (controller.config.rows, controller.config.columns),
    )
    lane_detector = LaneDetector()
    for frame_id in range(loader.n_frames):
        raw = loader.raw_frame(frame_id)
        kwargs = {
            "timestamp": float(raw.get("timestamp", frame_id / loader.fps)),
            "image_shape": (loader.calib.height, loader.calib.width),
            "ego_speed_kmh": float((raw.get("ego") or {}).get("speed_kmh") or 0.0),
        }
        risks = (
            tracker.update(
                [item for item in cache[frame_id] if item.confidence >= 0.25], **kwargs
            )
            if frame_id in cache else tracker.predict(**kwargs)
        )
        finite = [risk.predicted_ttc_s for risk in risks if math.isfinite(risk.predicted_ttc_s)]
        ttc = min(finite, default=float("inf"))
        prediction = C1FramePrediction(
            frame_id,
            kwargs["timestamp"],
            ttc,
            ttc < 2.0,
            tuple(risks),
            0.0,
        )
        image = cv2.imread(str(loader.left_path(frame_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(loader.left_path(frame_id))
        lane = lane_detector.detect(image)
        light_frame = controller.compute(image, prediction, lane)
        evaluator.add(light_frame, _danger_bboxes(loader, frame_id))
    return evaluator.report().__dict__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("predictions/c1_detection_cache")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("predictions/c1_matrix_light/evaluation.json")
    )
    args = parser.parse_args()
    per_trip = {
        trip_dir.name: evaluate_trip(trip_dir, args.cache_dir)
        for trip_dir in sorted(args.data_dir.glob("T*-Sample"))
    }
    totals = {
        "frame_precision_macro": round(float(np.mean([
            report["frame_precision"] for report in per_trip.values()
        ])), 4),
        "frame_recall_macro": round(float(np.mean([
            report["frame_recall"] for report in per_trip.values()
        ])), 4),
        "target_precision_macro": round(float(np.mean([
            report["target_precision"] for report in per_trip.values()
        ])), 4),
        "gt_coverage_95_recall_macro": round(float(np.mean([
            report["gt_coverage_95_recall"] for report in per_trip.values()
        ])), 4),
        "note": "6 development trips; projected KITTI boxes; not hidden-test accuracy.",
    }
    payload = {"per_trip": per_trip, "macro": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
