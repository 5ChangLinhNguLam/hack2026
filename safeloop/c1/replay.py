"""Run the untrained monocular C1 baseline on a TripReplayer stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tripkit import TripLoader, TripReplayer
from tripkit.replayer import MODES

from .detector import OpenCVDnnYoloDetector
from .pipeline import MonocularC1Pipeline, run_replay
from .tracker import MonocularTTCTracker, TrackerConfig

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m safeloop.c1.replay",
        description=(
            "Baseline C1 end-to-end: 1 camera RGB -> YOLO -> IoU tracker -> "
            "monocular TTC -> CSV/HUD. Không dùng image_3/depth/camera tài xế."
        ),
    )
    parser.add_argument("trip_dir", help="thư mục trip, ví dụ data/T01-Sample")
    parser.add_argument("--mode", choices=MODES, default="fast")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV; mặc định predictions/c1_monocular_base/<trip>.csv",
    )
    parser.add_argument("--video", type=Path, default=None, help="ghi HUD thành MP4")
    parser.add_argument("--show", action="store_true", help="mở HUD live; q/ESC để thoát")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11s.onnx")
    parser.add_argument("--labels", type=Path, default=ROOT / "models/driver-objects.labels")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--history-size", type=int, default=12)
    parser.add_argument(
        "--detector-stride",
        type=int,
        default=1,
        help="chạy detector mỗi N frame, tracker vẫn xuất TTC 20 Hz; mặc định 1",
    )
    parser.add_argument(
        "--ego-fallback",
        action="store_true",
        help="dùng speed + kích thước vật thể làm TTC dự phòng; mặc định tắt",
    )
    parser.add_argument(
        "--no-range-ttc",
        action="store_true",
        help="tắt range-TTC đã hiệu chỉnh; mặc định bật",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="chấm bằng GT sau khi chạy đủ toàn bộ trip Sample",
    )
    parser.add_argument("--quiet", action="store_true", help="không in progress mỗi 50 frame")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        print("Lỗi: --limit phải >= 0", file=sys.stderr)
        return 2

    try:
        loader = TripLoader(args.trip_dir)
        end = loader.n_frames if args.end is None else args.end
        if args.limit is not None:
            end = min(end, args.start + args.limit)
        replayer = TripReplayer(
            loader,
            mode=args.mode,
            speed=args.speed,
            start=args.start,
            end=end,
        )
        detector = OpenCVDnnYoloDetector(
            args.model,
            args.labels,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            device=args.device,
        )
        tracker = MonocularTTCTracker(
            TrackerConfig(
                history_size=args.history_size,
                min_history=min(2, args.history_size),
                enable_range_ttc=not args.no_range_ttc,
                min_range_history=min(3, args.history_size),
                range_recent_observations=max(3, min(6, args.history_size)),
                min_range_decreasing_fraction=0.80,
                max_range_slope_relative_mad=0.50,
                min_range_box_height_px=20.0,
                enable_ego_fallback=args.ego_fallback,
            ),
            focal_y_px=loader.calib.fy,
            focal_x_px=loader.calib.fx,
            principal_x_px=loader.calib.cx,
        )
        pipeline = MonocularC1Pipeline(
            detector,
            tracker,
            detector_stride=args.detector_stride,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi khởi tạo C1: {exc}", file=sys.stderr)
        return 2

    output = args.output or ROOT / "predictions/c1_monocular_base" / f"{loader.trip_id}.csv"
    try:
        stats = run_replay(
            replayer,
            pipeline,
            output_csv=output,
            output_video=args.video,
            fps=loader.fps,
            show=args.show,
            progress_every=0 if args.quiet else 50,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Lỗi chạy C1: {exc}", file=sys.stderr)
        return 3

    print(
        f"C1 MONOCULAR BASE: {stats.frame_count} frame, device={detector.device}, "
        f"detector_stride={args.detector_stride}, "
        f"{stats.elapsed_s:.2f}s, {stats.throughput_fps:.1f} fps, "
        f"latency mean/p95={stats.mean_latency_ms:.1f}/{stats.p95_latency_ms:.1f}ms"
    )
    print(
        f"  finite TTC={stats.finite_ttc_frames}, warning={stats.warning_frames}, "
        f"CSV={stats.output_csv}"
    )
    if stats.output_video is not None:
        print(f"  HUD video={stats.output_video}")

    if args.evaluate:
        if args.start != 0 or end != loader.n_frames or stats.frame_count != loader.n_frames:
            print(
                "Không chấm: --evaluate chỉ hợp lệ khi chạy đủ toàn bộ trip từ frame 0.",
                file=sys.stderr,
            )
            return 4
        if not loader.has_gt():
            print("Không chấm: trip này không có ground truth.", file=sys.stderr)
            return 4
        from team_kit.evaluation import evaluate, print_report

        report = evaluate(Path(output), loader.trip_dir.parent, None)
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
