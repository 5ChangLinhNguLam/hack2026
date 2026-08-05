"""Replay ego telemetry as versioned NDJSON or CarSky REST signals.

Local smoke test::

    python3 -m safeloop.replay_telemetry data/T01-Sample --limit 5

Publish to a deployed CarSky KUKSA signal node::

    python3 -m safeloop.replay_telemetry /data/T01d --mode realtime \
        --sink carsky-rest
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tripkit import TripLoader, TripReplayer
from tripkit.replayer import MODES

from .telemetry import (
    CarSkyRestClient,
    CarSkyRestSink,
    CarSkySignalPaths,
    JsonLinesSink,
    TelemetryContractError,
    publish_replay,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m safeloop.replay_telemetry",
        description="Phát ego telemetry theo SafeLoop contract v1.",
    )
    parser.add_argument("trip_dir", help="thư mục trip, vd data/T01-Sample")
    parser.add_argument("--mode", choices=MODES, default="fast")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sink", choices=("jsonl", "carsky-rest"), default="jsonl")
    parser.add_argument(
        "--output",
        default="-",
        help="NDJSON output path; '-' là stdout (chỉ dùng với --sink jsonl)",
    )
    carsky = parser.add_argument_group("CarSky REST")
    carsky.add_argument(
        "--carsky-url",
        default=os.getenv("A8_URL"),
        help="base URL của tenant; mặc định A8_URL (bắt buộc với carsky-rest)",
    )
    carsky.add_argument(
        "--carsky-room-id",
        default=os.getenv("A8_ROOM_ID"),
        help="Device ID/roomId; mặc định A8_ROOM_ID",
    )
    carsky.add_argument(
        "--carsky-node-key",
        default=os.getenv("A8_NODE_KEY"),
        help="signal node key từ GET /api/v1/signals/{roomId}",
    )
    carsky.add_argument(
        "--carsky-speed-path",
        default=os.getenv("A8_SPEED_PATH", "Vehicle.Speed"),
    )
    carsky.add_argument(
        "--carsky-longitudinal-accel-path",
        default=os.getenv(
            "A8_LONGITUDINAL_ACCEL_PATH", "Vehicle.Acceleration.Longitudinal"
        ),
    )
    carsky.add_argument(
        "--carsky-lateral-accel-path",
        default=os.getenv("A8_LATERAL_ACCEL_PATH", "Vehicle.Acceleration.Lateral"),
    )
    carsky.add_argument(
        "--carsky-trip-id-path",
        default=os.getenv("A8_TRIP_ID_PATH"),
        help="optional custom signal path cho trip_id",
    )
    carsky.add_argument(
        "--carsky-frame-id-path",
        default=os.getenv("A8_FRAME_ID_PATH"),
        help="optional custom signal path cho frame_id",
    )
    carsky.add_argument(
        "--carsky-timestamp-path",
        default=os.getenv("A8_TIMESTAMP_PATH"),
        help="optional custom signal path cho timestamp_ms",
    )
    carsky.add_argument("--carsky-timeout", type=float, default=10.0)
    carsky.add_argument(
        "--carsky-skip-signal-validation",
        action="store_true",
        help="bỏ kiểm tra path trước replay; chỉ dùng để debug",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.sink != "jsonl" and args.output != "-":
        print("Lỗi: --output chỉ dùng với --sink jsonl", file=sys.stderr)
        return 2
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
    except (FileNotFoundError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2

    output_stream = None
    try:
        if args.sink == "carsky-rest":
            sink = CarSkyRestSink(
                CarSkyRestClient(
                    args.carsky_url or "",
                    os.getenv("A8_API_KEY", ""),
                    args.carsky_room_id or "",
                    args.carsky_node_key or "",
                    timeout_s=args.carsky_timeout,
                ),
                CarSkySignalPaths(
                    speed_kmh=args.carsky_speed_path,
                    longitudinal_accel_mps2=args.carsky_longitudinal_accel_path,
                    lateral_accel_mps2=args.carsky_lateral_accel_path,
                    trip_id=args.carsky_trip_id_path,
                    frame_id=args.carsky_frame_id_path,
                    timestamp_ms=args.carsky_timestamp_path,
                ),
                validate_signals=not args.carsky_skip_signal_validation,
            )
        elif args.output == "-":
            sink = JsonLinesSink(sys.stdout)
        else:
            output_path = Path(args.output)
            output_stream = output_path.open("w", encoding="utf-8")
            sink = JsonLinesSink(output_stream)

        stats = publish_replay(replayer, sink)
        sink.close()
    except (OSError, RuntimeError, TelemetryContractError, ValueError) as exc:
        print(f"Lỗi phát telemetry: {exc}", file=sys.stderr)
        return 3
    finally:
        if output_stream is not None:
            output_stream.close()

    rate = stats.count / stats.elapsed_s if stats.elapsed_s > 0 else float("inf")
    print(
        f"Telemetry: {stats.count} message, frame "
        f"{stats.first_frame_id}..{stats.last_frame_id}, "
        f"{stats.elapsed_s:.3f}s, {rate:.1f} msg/s, sink={args.sink}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
