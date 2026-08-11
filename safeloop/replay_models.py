"""Replay C1, C2 and C3 in one trip stream.

Example::

    python -m safeloop.replay_models --dataset data --trip T01-Sample \
        --device cuda --limit 100

C1 is sampled at its trained 10-Hz cadence and causally forward-filled onto
the 20-Hz source rows.  C2 processes every driver-camera frame.  The combined
submission CSV therefore contains exactly one row per source frame.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tripkit import TripLoader, TripReplayer
from tripkit.replayer import MODES

from .combined_replay import CombinedFramePrediction, CombinedModelReplay


SUBMISSION_FIELDS = (
    "frame_id",
    "timestamp",
    "predicted_ttc",
    "predicted_driver_state",
    "predicted_risk_score",
)
C1_DIAGNOSTIC_FIELDS = (
    "c1_collision_probability",
    "c1_ttc_raw_s",
    "c1_ttc_display_s",
    "c1_warning",
    "c1_model_updated",
    "c1_model_frame_id",
    "c1_latency_ms",
)
C3_DIAGNOSTIC_FIELDS = (
    "c3_safe_score_estimate",
    "c3_risk_score_pct",
    "c3_grade",
    "c3_total_penalty",
    "c3_processed_frames",
    "c3_near_miss_frames",
    "c3_harsh_brake_frames",
    "c3_harsh_accel_frames",
    "c3_harsh_corner_frames",
    "c3_speeding_frames",
    "c3_speeding_pct_time",
    "c3_is_near_miss",
    "c3_is_harsh_brake",
    "c3_is_harsh_accel",
    "c3_is_harsh_corner",
    "c3_is_speeding",
    "c3_trip_complete",
    "c3_tailgating_penalty_omitted",
)
DRIVE_QUALITY_DIAGNOSTIC_FIELDS = (
    "drive_quality_score_available",
    "drive_quality_score_pct",
    "drive_quality_grade",
    "drive_quality_total_penalty",
    "drive_quality_scope",
    "drive_quality_observed_seconds",
    "drive_quality_window_ready",
    "drive_quality_near_miss_events_window",
    "drive_quality_harsh_brake_events_window",
    "drive_quality_harsh_accel_events_window",
    "drive_quality_harsh_corner_events_window",
    "drive_quality_speeding_pct_window",
    "drive_quality_new_near_miss_event",
    "drive_quality_new_harsh_brake_event",
    "drive_quality_new_harsh_accel_event",
    "drive_quality_new_harsh_corner_event",
    "drive_quality_trip_complete",
)
CONTEXTUAL_RISK_DIAGNOSTIC_FIELDS = (
    "contextual_risk_score_pct",
    "contextual_risk_level",
    "contextual_risk_action",
    "contextual_risk_brake_request_pct",
    "contextual_risk_reasons",
)


def _finite_or_inf(value: float) -> float | str:
    return round(value, 6) if math.isfinite(value) else "inf"


def diagnostic_row(
    frame: CombinedFramePrediction,
    *,
    dms_diagnostic_row: Any,
) -> dict[str, object]:
    c1 = frame.c1
    c2_values = dms_diagnostic_row(frame.c2)
    return {
        **frame.submission_row(),
        "c1_collision_probability": round(c1.collision_probability, 6),
        "c1_ttc_raw_s": _finite_or_inf(c1.predicted_ttc_s),
        "c1_ttc_display_s": _finite_or_inf(c1.display_ttc_s),
        "c1_warning": c1.is_warning,
        "c1_model_updated": c1.model_updated,
        "c1_model_frame_id": c1.model_frame_id,
        "c1_latency_ms": round(c1.latency_ms, 3),
        **frame.c3.diagnostic_row(),
        **frame.drive_quality.diagnostic_row(),
        **frame.contextual_risk.diagnostic_row(),
        **{
            key: value
            for key, value in c2_values.items()
            if key not in {"frame_id", "timestamp", "predicted_driver_state"}
        },
    }


class _AtomicCsv:
    def __init__(self, destination: Path, fields: Sequence[str]) -> None:
        self.destination = destination
        self.temporary = destination.with_suffix(destination.suffix + ".tmp")
        self.fields = tuple(fields)
        self._handle: Any = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "_AtomicCsv":
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.fields, extrasaction="ignore"
        )
        self._writer.writeheader()
        return self

    def writerow(self, row: Mapping[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("CSV writer is not open")
        self._writer.writerow(row)

    def __exit__(self, exc_type: object, _exc: object, _traceback: object) -> None:
        if self._handle is not None:
            self._handle.close()
        if exc_type is None:
            self.temporary.replace(self.destination)
        else:
            self.temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _partial_output_path(path: Path) -> Path:
    return path.parent / "partial" / path.name


def _archive_existing_outputs(
    output_dir: Path,
    *,
    archive_id: str,
    paths: Sequence[Path],
) -> list[Path]:
    """Move stale outputs out of submission locations before a new run.

    The operation is recoverable: files are retained under
    ``<output-dir>/archive/<archive-id>/`` with their relative layout.
    """

    archived: list[Path] = []
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        try:
            relative = path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError(f"replay output is outside output_dir: {path}") from exc
        destination = output_dir / "archive" / archive_id / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)
        archived.append(destination)
    return archived


def run_trip(
    trip_path: Path,
    *,
    c1_checkpoint: Path,
    c2_bundle: Path,
    output_dir: Path,
    device: str,
    mode: str,
    speed: float,
    limit: int | None,
    c1_stride: int,
    c1_ema: float,
    c1_warning_on: float,
    c1_warning_off: float,
    show: bool,
    write_video: bool,
    video_fourcc: str,
    frame_sink: Callable[[CombinedFramePrediction], None] | None = None,
) -> dict[str, object]:
    # Heavy modules are lazy so tripkit/base tests do not require the ML stack.
    from C1.runtime import StudentTTCRuntime
    from .c3 import Challenge3Accumulator
    from .contextual_risk import ContextualRiskPolicy
    from .drive_quality import DriveQualityAccumulator
    from drive_state.phase_2.cli.replay import (
        DIAGNOSTIC_FIELDS as DMS_DIAGNOSTIC_FIELDS,
        diagnostic_row as dms_diagnostic_row,
    )
    from drive_state.phase_2.replay import DMSBundle, GeneralDMS
    from .replay_video import (
        ReplayVideoWriter,
        WINDOW_NAME,
        render_combined_dashboard,
    )

    import cv2

    loader = TripLoader(trip_path)
    end = loader.n_frames if limit is None else min(loader.n_frames, limit)
    replayer = TripReplayer(loader, mode=mode, speed=speed, end=end)
    final_submission_path = output_dir / f"{loader.trip_id}.csv"
    final_diagnostic_path = output_dir / "diagnostics" / f"{loader.trip_id}.csv"
    partial_submission_path = _partial_output_path(final_submission_path)
    partial_diagnostic_path = _partial_output_path(final_diagnostic_path)
    diagnostic_fields = (
        *SUBMISSION_FIELDS,
        *C1_DIAGNOSTIC_FIELDS,
        *C3_DIAGNOSTIC_FIELDS,
        *DRIVE_QUALITY_DIAGNOSTIC_FIELDS,
        *CONTEXTUAL_RISK_DIAGNOSTIC_FIELDS,
        *(
            field
            for field in DMS_DIAGNOSTIC_FIELDS
            if field not in {"frame_id", "timestamp", "predicted_driver_state"}
        ),
    )

    frame_count = 0
    c1_updates = 0
    c1_warnings = 0
    c1_latencies: list[float] = []
    c2_latencies: list[float] = []
    states: Counter[str] = Counter()
    contextual_levels: Counter[str] = Counter()
    contextual_actions: Counter[str] = Counter()
    contextual_scores: list[float] = []
    bundle = DMSBundle.at(c2_bundle)
    bundle.validate()
    final_video_path = output_dir / "videos" / f"{loader.trip_id}.mp4"
    partial_video_path = _partial_output_path(final_video_path)
    archived_outputs = _archive_existing_outputs(
        output_dir,
        archive_id=f"{loader.trip_id}-{time.time_ns()}",
        paths=(
            final_submission_path,
            partial_submission_path,
            final_diagnostic_path,
            partial_diagnostic_path,
            final_video_path,
            partial_video_path,
        ),
    )
    video_writer = (
        ReplayVideoWriter(
            partial_video_path, fps=loader.fps, fourcc=video_fourcc
        )
        if write_video
        else None
    )
    stopped_by_user = False
    c3 = Challenge3Accumulator(
        speed_limit_kmh=loader.metadata.get("speed_limit_kmh"),
        expected_frames=loader.n_frames,
    )
    drive_quality = DriveQualityAccumulator()
    contextual_risk = ContextualRiskPolicy()

    # Fresh instances per trip prevent temporal/face-detector state leakage.
    try:
        with StudentTTCRuntime(
            c1_checkpoint,
            metadata=loader.metadata,
            source_fps=loader.fps,
            stride=c1_stride,
            device=device,
            ema=c1_ema,
            warning_on=c1_warning_on,
            warning_off=c1_warning_off,
        ) as c1, GeneralDMS(
            bundle,
            device=device,
            fps=loader.fps,
        ) as c2, _AtomicCsv(
            partial_submission_path, SUBMISSION_FIELDS
        ) as submission, _AtomicCsv(
            partial_diagnostic_path, diagnostic_fields
        ) as diagnostics:
            combined = CombinedModelReplay(
                replayer,
                c1=c1,
                c2=c2,
                c3=c3,
                drive_quality=drive_quality,
                contextual_risk=contextual_risk,
            )
            for frame_count, prediction in enumerate(combined, start=1):
                if frame_sink is not None:
                    frame_sink(prediction)
                submission.writerow(prediction.submission_row())
                diagnostics.writerow(
                    diagnostic_row(
                        prediction, dms_diagnostic_row=dms_diagnostic_row
                    )
                )
                c1_updates += int(prediction.c1.model_updated)
                c1_warnings += int(prediction.c1.is_warning)
                if prediction.c1.model_updated:
                    c1_latencies.append(float(prediction.c1.latency_ms))
                c2_latencies.append(float(prediction.c2.latency_ms))
                states[str(prediction.c2.state)] += 1
                contextual_levels[prediction.contextual_risk.level] += 1
                contextual_actions[prediction.contextual_risk.action] += 1
                contextual_scores.append(
                    float(prediction.contextual_risk.score_pct)
                )

                if show or video_writer is not None:
                    dashboard = render_combined_dashboard(prediction)
                    if video_writer is not None:
                        video_writer.write(dashboard)
                    if show:
                        cv2.imshow(WINDOW_NAME, dashboard)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            stopped_by_user = True
                            break

                if frame_count % 300 == 0:
                    ttc = prediction.c1.predicted_ttc_s
                    ttc_text = f"{ttc:.2f}s" if math.isfinite(ttc) else "inf"
                    print(
                        f"{loader.trip_id}: {frame_count}/{end} "
                        f"C1={ttc_text} C2={prediction.c2.state}",
                        flush=True,
                    )
            expected_updates = (
                0 if frame_count == 0 else (frame_count - 1) // c1_stride + 1
            )
            if c1_updates != expected_updates:
                raise RuntimeError(
                    f"C1 cadence mismatch: {c1_updates} updates, "
                    f"expected {expected_updates}"
                )
    finally:
        if video_writer is not None:
            video_writer.close()
        if show:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass
    trip_complete = frame_count == loader.n_frames and not stopped_by_user
    if trip_complete:
        partial_submission_path.replace(final_submission_path)
        partial_diagnostic_path.replace(final_diagnostic_path)
        submission_path = final_submission_path
        diagnostic_path = final_diagnostic_path
        if video_writer is not None and video_writer.frames > 0:
            partial_video_path.replace(final_video_path)
            video_path: Path | None = final_video_path
        else:
            video_path = None
    else:
        submission_path = partial_submission_path
        diagnostic_path = partial_diagnostic_path
        video_path = (
            partial_video_path
            if video_writer is not None and video_writer.frames > 0
            else None
        )
    return {
        "trip_id": loader.trip_id,
        "frames": frame_count,
        "source_fps": loader.fps,
        "trip_complete": trip_complete,
        "output_kind": "submission" if trip_complete else "partial_debug",
        "archived_previous_outputs": [str(path) for path in archived_outputs],
        "c1": {
            "source": str(c1_checkpoint),
            "model_hz": loader.fps / c1_stride,
            "updates": c1_updates,
            "forward_filled_frames": frame_count - c1_updates,
            "warning_frames": c1_warnings,
            "latency_mean_ms_on_update": (
                round(mean(c1_latencies), 3) if c1_latencies else 0.0
            ),
            "latency_p95_ms_on_update": (
                round(float(np.quantile(c1_latencies, 0.95)), 3)
                if c1_latencies
                else 0.0
            ),
        },
        "c2": {
            "source": "drive_state.phase_2 v13",
            "model_hz": loader.fps,
            "state_counts": dict(states),
            "latency_mean_ms": (
                round(mean(c2_latencies), 3) if c2_latencies else 0.0
            ),
            "latency_p95_ms": (
                round(float(np.quantile(c2_latencies, 0.95)), 3)
                if c2_latencies
                else 0.0
            ),
        },
        "challenge3_estimate": c3.summary(),
        "product_drive_quality": drive_quality.summary(),
        "product_contextual_risk": {
            "semantics": "instantaneous C1+C2 product risk",
            "score_direction": "higher_is_more_dangerous",
            "policy_version": contextual_risk.version,
            "max_score_pct": (
                round(max(contextual_scores), 3) if contextual_scores else 0.0
            ),
            "mean_score_pct": (
                round(mean(contextual_scores), 3)
                if contextual_scores
                else 0.0
            ),
            "level_counts": dict(contextual_levels),
            "action_counts": dict(contextual_actions),
        },
        "submission_csv": str(submission_path),
        "diagnostic_csv": str(diagnostic_path),
        "video": str(video_path) if video_path is not None else None,
        "stopped_by_user": stopped_by_user,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safeloop.replay_models",
        description="Chạy C1, C2 và C3 trong cùng một TripReplayer.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--trip", action="append", help="trip ID; lặp option để chạy nhiều trip"
    )
    parser.add_argument(
        "--c1-checkpoint", type=Path, default=Path("C1/student_ttc.pth")
    )
    parser.add_argument(
        "--c2-bundle", type=Path, default=Path("models/driver_state_phase_2_v13")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("predictions/safeloop_models")
    )
    parser.add_argument("--mode", choices=MODES, default="fast")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--c1-stride", type=int, default=2)
    parser.add_argument("--c1-ema", type=float, default=0.9)
    parser.add_argument("--c1-warning-on", type=float, default=0.85)
    parser.add_argument("--c1-warning-off", type=float, default=0.5)
    parser.add_argument(
        "--show",
        action="store_true",
        help="hiển thị dashboard C1+C2+C3 trực tiếp; q/ESC để dừng",
    )
    parser.add_argument(
        "--write-video",
        action="store_true",
        help="ghi dashboard MP4 vào <output-dir>/videos/<trip>.mp4",
    )
    parser.add_argument("--video-fourcc", default="mp4v")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        print("Lỗi: --limit phải >= 0", file=sys.stderr)
        return 2
    if args.show and os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "Lỗi: --show cần desktop display; trên server headless hãy dùng "
            "--write-video",
            file=sys.stderr,
        )
        return 2
    trip_ids = args.trip or [f"T{index:02d}d" for index in range(1, 11)]
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for trip_id in trip_ids:
        try:
            summary = run_trip(
                args.dataset / trip_id,
                c1_checkpoint=args.c1_checkpoint,
                c2_bundle=args.c2_bundle,
                output_dir=args.output_dir,
                device=args.device,
                mode=args.mode,
                speed=args.speed,
                limit=args.limit,
                c1_stride=args.c1_stride,
                c1_ema=args.c1_ema,
                c1_warning_on=args.c1_warning_on,
                c1_warning_off=args.c1_warning_off,
                show=args.show,
                write_video=args.write_video,
                video_fourcc=args.video_fourcc,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {"trip_id": trip_id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)

    report: dict[str, object] = {
        "pipeline": (
            "C1 StudentTTC + C2 drive_state.phase_2 + C3 safe score "
            "+ Drive Quality"
        ),
        "replay_source": "one shared tripkit.TripReplayer per trip",
        "c1_checkpoint": str(args.c1_checkpoint),
        "c2_bundle": str(args.c2_bundle),
        "completed_trips": len(summaries),
        "full_trips": sum(bool(item["trip_complete"]) for item in summaries),
        "processed_frames": sum(int(item["frames"]) for item in summaries),
        "summaries": summaries,
        "failures": failures,
    }
    _write_json_atomic(args.output_dir / "replay_report.json", report)
    print(f"REPORT={args.output_dir / 'replay_report.json'}", flush=True)
    stopped = any(bool(item["stopped_by_user"]) for item in summaries)
    return 0 if not failures and not stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C1_DIAGNOSTIC_FIELDS",
    "C3_DIAGNOSTIC_FIELDS",
    "CONTEXTUAL_RISK_DIAGNOSTIC_FIELDS",
    "DRIVE_QUALITY_DIAGNOSTIC_FIELDS",
    "SUBMISSION_FIELDS",
    "build_parser",
    "diagnostic_row",
    "main",
    "run_trip",
]
