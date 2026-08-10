#!/usr/bin/env python3
"""Run the no-ground-truth lane consistency proxy on all available trips."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safeloop.c1.lane import LaneDetector, LaneSelfEvaluator  # noqa: E402
from tripkit import TripLoader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("predictions/c1_lane/evaluation.json")
    )
    args = parser.parse_args()
    per_trip: dict[str, object] = {}
    for trip_dir in sorted(args.data_dir.glob("T*")):
        if not trip_dir.is_dir() or trip_dir.name == "Demo":
            continue
        loader = TripLoader(trip_dir)
        detector = LaneDetector()
        evaluator = LaneSelfEvaluator(loader.calib.width)
        started = time.perf_counter()
        for frame_id in range(loader.n_frames):
            image = cv2.imread(str(loader.left_path(frame_id)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(loader.left_path(frame_id))
            evaluator.add(detector.detect(image))
        report = evaluator.report().__dict__
        report["throughput_fps"] = round(
            loader.n_frames / max(time.perf_counter() - started, 1e-9), 2
        )
        per_trip[loader.trip_id] = report
        print(loader.trip_id, report, flush=True)
    payload = {
        "per_trip": per_trip,
        "macro_proxy_score": round(float(np.mean([
            item["proxy_score"] for item in per_trip.values()
        ])), 1),
        "macro_valid_fraction": round(float(np.mean([
            item["valid_fraction"] for item in per_trip.values()
        ])), 4),
        "note": "Temporal/geometry proxy only; dataset has no independent lane ground truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
