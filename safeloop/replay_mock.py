"""Replay SafeLoop's deterministic C1/C2/Risk mock locally or to CarSky."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tripkit import TripLoader, TripReplayer
from tripkit.replayer import MODES

from .mock_pipeline import MockCarSkyRestSink, MockJsonLinesSink, publish_mock_replay
from .telemetry import CarSkyRestClient, TelemetryContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m safeloop.replay_mock",
        description="Phát mock C1 + C2 + Risk Fusion; mọi output đều có mock=true.",
    )
    parser.add_argument("trip_dir", help="thư mục trip, ví dụ data/T01-Sample")
    parser.add_argument("--mode", choices=MODES, default="fast")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sink", choices=("jsonl", "carsky-rest"), default="jsonl")
    parser.add_argument("--output", default="-", help="NDJSON path hoặc '-' cho stdout")
    parser.add_argument("--carsky-url", default=os.getenv("A8_URL"))
    parser.add_argument("--carsky-room-id", default=os.getenv("A8_ROOM_ID"))
    parser.add_argument("--carsky-node-key", default=os.getenv("A8_NODE_KEY"))
    parser.add_argument("--carsky-timeout", type=float, default=10.0)
    parser.add_argument("--carsky-skip-signal-validation", action="store_true")
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
            loader, mode=args.mode, speed=args.speed, start=args.start, end=end
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2

    output_stream = None
    try:
        if args.sink == "carsky-rest":
            sink = MockCarSkyRestSink(
                CarSkyRestClient(
                    args.carsky_url or "",
                    os.getenv("A8_API_KEY", ""),
                    args.carsky_room_id or "",
                    args.carsky_node_key or "",
                    timeout_s=args.carsky_timeout,
                ),
                validate_signals=not args.carsky_skip_signal_validation,
            )
        elif args.output == "-":
            sink = MockJsonLinesSink(sys.stdout)
        else:
            output_stream = Path(args.output).open("w", encoding="utf-8")
            sink = MockJsonLinesSink(output_stream)

        stats = publish_mock_replay(replayer, sink)
        sink.close()
    except (OSError, RuntimeError, TelemetryContractError, ValueError) as exc:
        print(f"Lỗi phát SafeLoop mock: {exc}", file=sys.stderr)
        return 3
    finally:
        if output_stream is not None:
            output_stream.close()

    rate = stats.count / stats.elapsed_s if stats.elapsed_s > 0 else float("inf")
    print(
        f"SafeLoop MOCK: {stats.count} decision, frame "
        f"{stats.first_frame_id}..{stats.last_frame_id}, "
        f"{stats.elapsed_s:.3f}s, {rate:.1f} msg/s, sink={args.sink}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
