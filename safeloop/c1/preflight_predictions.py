"""CLI for strict C1 prediction completeness checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .prediction_preflight import (
    DEFAULT_EXPECTED_FRAMES,
    PredictionPreflightError,
    discover_prediction_csvs,
    preflight_prediction_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m safeloop.c1.preflight_predictions",
        description=(
            "Validate complete C1 prediction CSVs before using the official evaluator. "
            "Input paths may be CSV files or directories containing CSV files."
        ),
    )
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=DEFAULT_EXPECTED_FRAMES,
        help=f"required rows and frame IDs 0..N-1 (default: {DEFAULT_EXPECTED_FRAMES})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_frames <= 0:
        print("C1 prediction preflight failed: --expected-frames must be > 0", file=sys.stderr)
        return 2

    csv_paths = discover_prediction_csvs(args.paths)
    if not csv_paths:
        print("C1 prediction preflight failed: no CSV files found", file=sys.stderr)
        return 2

    reports = []
    try:
        for csv_path in csv_paths:
            reports.append(
                preflight_prediction_csv(
                    csv_path,
                    expected_frames=args.expected_frames,
                ).to_dict()
            )
    except PredictionPreflightError as exc:
        print(f"C1 prediction preflight failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "expected_frames": args.expected_frames,
                "files": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
