from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Sequence

from PIL import Image
import torch

from .demo import create_lstm_pipeline, create_pipeline
from .precompute_embeddings import checkpoint_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end T4 latency")
    parser.add_argument("--lstm-visual-checkpoint", type=Path)
    parser.add_argument("--lstm-ocular-checkpoint", type=Path)
    parser.add_argument("--lstm-gate-ocular-checkpoint", type=Path)
    parser.add_argument("--lstm-temporal-checkpoint", type=Path)
    parser.add_argument("--eye-checkpoint", type=Path)
    parser.add_argument("--primitive-checkpoint", type=Path)
    parser.add_argument("--mouth-checkpoint", type=Path)
    parser.add_argument("--cabin-temporal-checkpoint", type=Path)
    parser.add_argument("--face-mouth-temporal-checkpoint", type=Path)
    parser.add_argument("--ultralight-model", type=Path, required=True)
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--detector-interval", type=int, default=5)
    parser.add_argument("--primitive-interval", type=int, default=2)
    return parser


def _benchmark_mode(args: argparse.Namespace) -> str:
    lstm_paths = (
        args.lstm_visual_checkpoint,
        args.lstm_ocular_checkpoint,
        args.lstm_temporal_checkpoint,
    )
    legacy_paths = (
        args.eye_checkpoint,
        args.primitive_checkpoint,
        args.mouth_checkpoint,
        args.cabin_temporal_checkpoint,
        args.face_mouth_temporal_checkpoint,
    )
    if any(path is not None for path in lstm_paths) and not all(
        path is not None for path in lstm_paths
    ):
        raise ValueError(
            "visual, ocular, and temporal LSTM checkpoints must be supplied "
            "together"
        )
    if args.lstm_gate_ocular_checkpoint is not None and not all(
        path is not None for path in lstm_paths
    ):
        raise ValueError(
            "the microsleep-gate ocular checkpoint requires all three LSTM "
            "checkpoints"
        )
    if all(path is not None for path in lstm_paths):
        if any(path is not None for path in legacy_paths):
            raise ValueError(
                "the LSTM pipeline cannot be combined with legacy checkpoints"
            )
        return "mobilenet_ocular_lstm"
    if args.eye_checkpoint is None or args.primitive_checkpoint is None:
        raise ValueError(
            "supply all three LSTM checkpoints or both legacy eye and "
            "primitive checkpoints"
        )
    return "specialist"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = _benchmark_mode(args)
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError("measured frames must be positive and warm-up non-negative")
    requested_frames = args.warmup + args.frames
    paths = sorted(args.frames_dir.glob("*.jpg"))[:requested_frames]
    if len(paths) < requested_frames:
        raise SystemExit(
            "not enough JPEG frames for requested benchmark and warm-up"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy_landmarker = None
    if mode == "mobilenet_ocular_lstm":
        assert args.lstm_visual_checkpoint is not None
        assert args.lstm_ocular_checkpoint is not None
        assert args.lstm_temporal_checkpoint is not None
        pipeline = create_lstm_pipeline(
            visual_checkpoint=args.lstm_visual_checkpoint,
            ocular_checkpoint=args.lstm_ocular_checkpoint,
            temporal_checkpoint=args.lstm_temporal_checkpoint,
            ultralight_model=args.ultralight_model,
            landmark_model=args.landmark_model,
            device=device,
            fps=args.fps,
            detector_interval=args.detector_interval,
            gate_ocular_checkpoint=args.lstm_gate_ocular_checkpoint,
        )
        checkpoint_fingerprints = {
            "visual": checkpoint_fingerprint(args.lstm_visual_checkpoint),
            "ocular": checkpoint_fingerprint(args.lstm_ocular_checkpoint),
            "temporal": checkpoint_fingerprint(args.lstm_temporal_checkpoint),
        }
        if args.lstm_gate_ocular_checkpoint is not None:
            checkpoint_fingerprints["microsleep_gate_ocular"] = (
                checkpoint_fingerprint(args.lstm_gate_ocular_checkpoint)
            )
        model_input_resolution = list(pipeline.runtime.image_size)
        temporal_mode = True
    else:
        assert args.eye_checkpoint is not None
        assert args.primitive_checkpoint is not None
        pipeline, legacy_landmarker = create_pipeline(
            eye_checkpoint=args.eye_checkpoint,
            primitive_checkpoint=args.primitive_checkpoint,
            ultralight_model=args.ultralight_model,
            landmark_model=args.landmark_model,
            device=device,
            fps=args.fps,
            detector_interval=args.detector_interval,
            primitive_interval=args.primitive_interval,
            mouth_checkpoint=args.mouth_checkpoint,
            cabin_temporal_checkpoint=args.cabin_temporal_checkpoint,
            face_mouth_temporal_checkpoint=args.face_mouth_temporal_checkpoint,
        )
        checkpoint_fingerprints = {
            "eye": checkpoint_fingerprint(args.eye_checkpoint),
            "primitive": checkpoint_fingerprint(args.primitive_checkpoint),
        }
        model_input_resolution = None
        temporal_mode = pipeline.temporal_enabled
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latencies: list[float] = []
    source_resolution: list[int] | None = None
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            if source_resolution is None:
                source_resolution = list(image.size)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pipeline.process(image, timestamp=index / args.fps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                latencies.append(elapsed)
    finally:
        pipeline.close()
        if legacy_landmarker is not None:
            legacy_landmarker.close()
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    throughput = len(latencies) / sum(latencies)
    report = {
        "device": str(device),
        "pipeline_mode": mode,
        "temporal_mode": temporal_mode,
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "source_resolution": source_resolution,
        "model_input_resolution": model_input_resolution,
        "warmup_frames": args.warmup,
        "frames": len(latencies),
        "throughput_fps": throughput,
        "median_latency_ms": statistics.median(latencies) * 1000.0,
        "mean_latency_ms": statistics.mean(latencies) * 1000.0,
        "p95_latency_ms": p95 * 1000.0,
        "peak_allocated_vram_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        ),
        "passes_20fps": throughput >= 20.0 and p95 <= 0.05,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passes_20fps"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
