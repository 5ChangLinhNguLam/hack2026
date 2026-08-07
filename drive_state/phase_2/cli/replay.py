"""Run the phase-2 DMD model over one or more native tripkit trips."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np
import torch

from ..replay import DMSBundle, DMSFramePrediction, GeneralDMS


SUBMISSION_FIELDS = ("frame_id", "timestamp", "predicted_driver_state")
DIAGNOSTIC_FIELDS = (
    *SUBMISSION_FIELDS,
    "state_id",
    "confidence",
    "p_alert",
    "p_drowsy",
    "p_microsleep",
    "p_yawning",
    "p_distraction",
    "closed_probability",
    "ocular_reliability",
    "closure_duration_seconds",
    "perclos_10s",
    "perclos_30s",
    "perclos_60s",
    "slow_perclos_10s",
    "slow_perclos_30s",
    "slow_perclos_60s",
    "nod_probability",
    "microsleep_active",
    "drowsy_episode_active",
    "latency_ms",
    "Vehicle.Driver.AttentiveProbability",
    "Vehicle.Driver.DistractionLevel",
    "Vehicle.Driver.FatigueLevel",
    "Vehicle.Driver.IsEyesOnRoad",
    "Vehicle.ADAS.DMS.IsWarning",
)


def diagnostic_row(frame: DMSFramePrediction) -> dict[str, object]:
    alert, drowsy, microsleep, yawning, distraction = frame.probabilities
    vss = frame.vss_signals().as_vss_dict()
    return {
        **frame.submission_row(),
        "state_id": frame.state_id,
        "confidence": round(frame.confidence, 6),
        "p_alert": round(alert, 6),
        "p_drowsy": round(drowsy, 6),
        "p_microsleep": round(microsleep, 6),
        "p_yawning": round(yawning, 6),
        "p_distraction": round(distraction, 6),
        "closed_probability": round(frame.closed_probability, 6),
        "ocular_reliability": round(frame.ocular_reliability, 6),
        "closure_duration_seconds": round(frame.closure_duration_seconds, 3),
        "perclos_10s": round(frame.perclos[0], 6),
        "perclos_30s": round(frame.perclos[1], 6),
        "perclos_60s": round(frame.perclos[2], 6),
        "slow_perclos_10s": round(frame.slow_perclos[0], 6),
        "slow_perclos_30s": round(frame.slow_perclos[1], 6),
        "slow_perclos_60s": round(frame.slow_perclos[2], 6),
        "nod_probability": round(frame.nod_probability, 6),
        "microsleep_active": frame.microsleep_active,
        "drowsy_episode_active": frame.drowsy_episode_active,
        "latency_ms": round(frame.latency_ms, 3),
        **vss,
    }


def _atomic_csv(path: Path, fields: Sequence[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_trip(
    trip_path: Path,
    *,
    bundle: DMSBundle,
    output_dir: Path,
    device: str,
    mode: str,
    speed: float,
    limit: int | None,
) -> dict[str, object]:
    frames: list[DMSFramePrediction] = []
    with GeneralDMS(bundle, device=device, fps=20.0) as dms:
        for index, frame in enumerate(
            dms.replay_trip(trip_path, mode=mode, speed=speed, limit=limit), start=1
        ):
            frames.append(frame)
            if index % 300 == 0:
                print(
                    f"{trip_path.name}: {index} frames "
                    f"state={frame.state} confidence={frame.confidence:.3f}",
                    flush=True,
                )

    submissions = [frame.submission_row() for frame in frames]
    diagnostics = [diagnostic_row(frame) for frame in frames]
    submission_path = output_dir / f"{trip_path.name}.csv"
    diagnostic_path = output_dir / "diagnostics" / f"{trip_path.name}.csv"
    _atomic_csv(submission_path, SUBMISSION_FIELDS, submissions)
    _atomic_csv(diagnostic_path, DIAGNOSTIC_FIELDS, diagnostics)

    states = [frame.state for frame in frames]
    latencies = [frame.latency_ms for frame in frames]
    confidences = np.asarray([frame.confidence for frame in frames])
    return {
        "trip_id": trip_path.name,
        "frames": len(frames),
        "state_counts": dict(Counter(states)),
        "state_transitions": sum(a != b for a, b in zip(states, states[1:])),
        "mean_confidence": round(float(confidences.mean()), 4) if frames else 0.0,
        "low_confidence_frames_lt_0_5": int((confidences < 0.5).sum()),
        "latency_mean_ms": round(mean(latencies), 3) if latencies else 0.0,
        "latency_p95_ms": (
            round(float(np.quantile(latencies, 0.95)), 3) if latencies else 0.0
        ),
        "capacity_fps": round(1000.0 / mean(latencies), 2) if latencies else 0.0,
        "submission_csv": str(submission_path),
        "diagnostic_csv": str(diagnostic_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay hackathon trips through drive_state.phase_2"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, default=Path("models/driver_state_phase_2_v13")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("predictions/driver_state_phase_2_v13")
    )
    parser.add_argument(
        "--trip", action="append", help="trip ID; repeat for multiple trips"
    )
    parser.add_argument("--mode", choices=("fast", "realtime"), default="fast")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = DMSBundle.at(args.bundle)
    bundle.validate()
    trip_ids = args.trip or [f"T{index:02d}d" for index in range(1, 11)]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for trip_id in trip_ids:
        try:
            summary = run_trip(
                args.dataset / trip_id,
                bundle=bundle,
                output_dir=args.output_dir,
                device=args.device,
                mode=args.mode,
                speed=args.speed,
                limit=args.limit,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {"trip_id": trip_id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "dms": "drive_state.phase_2 MobileNetV3 + dual ocular/temporal LSTM v13",
        "bundle": str(args.bundle),
        "replay_source": "tripkit.TripReplayer",
        "completed_trips": len(summaries),
        "processed_frames": sum(int(item["frames"]) for item in summaries),
        "summaries": summaries,
        "failures": failures,
    }
    if torch.cuda.is_available():
        report["peak_cuda_memory_mib"] = round(
            torch.cuda.max_memory_allocated() / (1024 * 1024), 2
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "replay_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"REPORT={report_path}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
