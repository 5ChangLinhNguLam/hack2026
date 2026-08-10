"""Replay C1 TTC + virtual matrix light + lane geometry from one camera."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from tripkit import TripLoader, TripReplayer

from .detector import OpenCVDnnYoloDetector
from .lane import LaneDetector, LaneSelfEvaluator, render_lane
from .matrix_light import MatrixLightController, render_matrix_light
from .pipeline import MonocularC1Pipeline, render_hud
from .tracker import MonocularTTCTracker, TrackerConfig

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-camera replay: YOLO/TTC + lane + virtual matrix-light. "
            "Không dùng image_3/depth/driver/event/ground truth khi inference."
        )
    )
    parser.add_argument("trip_dir")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--detector-stride", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loader = TripLoader(args.trip_dir)
        end = loader.n_frames if args.end is None else args.end
        if args.limit is not None:
            end = min(end, args.start + args.limit)
        replayer = TripReplayer(loader, start=args.start, end=end)
        detector = OpenCVDnnYoloDetector(
            ROOT / "models/yolo11s.onnx",
            ROOT / "models/driver-objects.labels",
            confidence_threshold=args.confidence,
            device="auto",
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
        c1 = MonocularC1Pipeline(detector, tracker, detector_stride=args.detector_stride)
        lane = LaneDetector()
        lane_eval = LaneSelfEvaluator(loader.calib.width)
        light = MatrixLightController(
            focal_x_px=loader.calib.fx,
            focal_y_px=loader.calib.fy,
            principal_x_px=loader.calib.cx,
            principal_y_px=loader.calib.cy,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi khởi tạo perception replay: {exc}", file=sys.stderr)
        return 2

    output = args.output or ROOT / "predictions/c1_perception" / f"{loader.trip_id}.csv"
    report_path = args.report or output.with_suffix(".report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    latencies: list[float] = []
    coverages: list[float] = []
    target_frames = 0
    modes: Counter[str] = Counter()
    c1.reset()
    lane.reset()
    started = time.perf_counter()
    fields = [
        "frame_id", "timestamp", "predicted_ttc", "lane_confidence",
        "lane_center_offset", "lane_departure", "matrix_mode",
        "matrix_targets", "matrix_min_geometric_coverage",
    ]
    try:
        with output.open("w", encoding="utf-8", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            for count, bundle in enumerate(replayer, start=1):
                frame_started = time.perf_counter()
                prediction, image = c1.predict(bundle)
                lane_estimate = lane.detect(image)
                light_frame = light.compute(image, prediction, lane_estimate)
                lane_eval.add(lane_estimate)
                modes[light_frame.mode] += 1
                if light_frame.targets:
                    target_frames += 1
                    coverages.extend(target.geometric_coverage for target in light_frame.targets)
                row = prediction.submission_row()
                row.update({
                    "lane_confidence": round(lane_estimate.confidence, 4),
                    "lane_center_offset": round(lane_estimate.lane_center_offset, 4),
                    "lane_departure": int(lane_estimate.departure_warning),
                    "matrix_mode": light_frame.mode,
                    "matrix_targets": len(light_frame.targets),
                    "matrix_min_geometric_coverage": (
                        round(min(target.geometric_coverage for target in light_frame.targets), 4)
                        if light_frame.targets else ""
                    ),
                })
                csv_writer.writerow(row)
                latencies.append((time.perf_counter() - frame_started) * 1000.0)

                if args.video or args.show:
                    panel = render_hud(image, prediction)
                    panel = render_lane(panel, lane_estimate)
                    panel = render_matrix_light(panel, light_frame)
                    if args.video:
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(args.video),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                loader.fps,
                                (panel.shape[1], panel.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"Không mở được video: {args.video}")
                        writer.write(panel)
                    if args.show:
                        cv2.imshow("SafeLoop C1 perception", panel)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                if count % 200 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"Perception: {count} frame, {count/elapsed:.1f} fps", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Lỗi perception replay: {exc}", file=sys.stderr)
        return 3
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    lane_report = lane_eval.report()
    report = {
        "trip_id": loader.trip_id,
        "frames": len(latencies),
        "throughput_fps": round(len(latencies) / max(elapsed, 1e-9), 2),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "lane": lane_report.__dict__,
        "matrix_light": {
            "simulation_only": True,
            "target_frames": target_frames,
            "mode_frames": dict(modes),
            "mean_geometric_coverage": round(float(np.mean(coverages)), 4) if coverages else None,
            "min_geometric_coverage": round(float(np.min(coverages)), 4) if coverages else None,
            "coverage_at_least_95_fraction": (
                round(sum(value >= 0.95 for value in coverages) / len(coverages), 4)
                if coverages else None
            ),
            "selection_accuracy": None,
            "selection_accuracy_note": "Cần danger-object ground truth độc lập; không tự chấm bằng predicted box.",
        },
        "output": str(output),
        "video": str(args.video) if args.video else None,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.evaluate:
        if args.start != 0 or end != loader.n_frames or len(latencies) != loader.n_frames:
            print("Không chấm C1: --evaluate yêu cầu chạy đủ trip.", file=sys.stderr)
            return 4
        if not loader.has_gt():
            print("Không chấm C1: trip không có ground truth.", file=sys.stderr)
            return 4
        from team_kit.evaluation import evaluate, print_report
        print_report(evaluate(output, loader.trip_dir.parent, None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
