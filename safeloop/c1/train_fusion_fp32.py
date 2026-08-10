"""Three-seed FP32 sanity protocol for the P1 fusion research head.

This runner intentionally does not train or evaluate the temporal-only
ablation.  It repeats the unchanged P1 nested leave-one-trip-out selection for
the fusion head with three predeclared seeds and AMP disabled.  The three seed
results are averaged; no seed is selected for deployment or reporting.

All checkpoints, TensorBoard logs and predictions default to ignored paths.
The physics pipeline remains the authoritative default and is gated at 55.2
before the first optimizer is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .lean_loader import C1LeanTripLoader
from .prepare_manifest import prepare_c1_manifest
from .temporal_features import (
    TemporalTripFeatures,
    build_trip_features,
    load_camera_geometry,
    load_detection_cache,
    locate_detection_cache,
)
from .temporal_model import (
    DEFAULT_CANDIDATES,
    CausalTemporalTTC,
    TemporalModelConfig,
    calibrated_ttc,
    checkpoint_payload,
)
from .train_temporal import (
    P1_TRIP_IDS,
    InnerFold,
    OuterFold,
    SupervisedTrip,
    TTCSupervision,
    ThresholdChoice,
    TrainSettings,
    _ablation_report,
    _atomic_json,
    _atomic_torch_save,
    _device,
    _read_prediction_csv,
    _require_exact_oof_csv_set,
    _write_prediction_csv,
    load_training_ttc_labels,
    nested_lopo_splits,
    outer_training_trips,
    predict_raw,
    select_threshold,
    train_final_model,
    train_inner_model,
)


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_EVALUATOR = ROOT / "team_kit/evaluation.py"
EXPECTED_EVALUATOR_SHA256 = (
    "674320460240797c2cb3a2b60815629c1cc1dcdcd9cc17ad52988c3dd353065b"
)

# This tuple is part of the protocol, not a tuning surface.  Do not add a CLI
# option that permits cherry-picking or omitting seeds from the aggregate.
FP32_SEEDS: tuple[int, int, int] = (2026, 2027, 2028)
FP32_THRESHOLDS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
PHYSICS_REFERENCE_SCORE = 55.2
FUSION_VARIANT = "fusion"


def _evaluator_sha256() -> str:
    return hashlib.sha256(OFFICIAL_EVALUATOR.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FusionCandidateResult:
    """Inner-validation result for one fusion architecture only."""

    config: TemporalModelConfig
    choice: ThresholdChoice
    best_epochs: tuple[int, ...]
    validation_trip_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedProtocolData:
    """Runtime inputs and separately held supervision for the fixed six trips."""

    supervised: Mapping[str, SupervisedTrip]
    labels: Mapping[str, TTCSupervision]
    physics_score: float
    physics_report: Mapping[str, object]


def _assert_fp32_finite(model: CausalTemporalTTC) -> None:
    """Fail a seed rather than silently accepting non-FP32/non-finite weights."""

    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                f"Fusion FP32 protocol produced {name} with dtype {parameter.dtype}"
            )
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                f"Fusion FP32 protocol produced non-finite parameter {name}"
            )


def evaluate_fusion_candidate_inner(
    supervised: Mapping[str, SupervisedTrip],
    outer: OuterFold,
    inner_folds: Sequence[InnerFold],
    config: TemporalModelConfig,
    settings: TrainSettings,
    *,
    thresholds: Sequence[float],
    device: torch.device,
    log_root: Path,
    seed: int,
) -> FusionCandidateResult:
    """Tune one candidate using only fusion fits on inner validation trips."""

    if settings.amp:
        raise ValueError("The FP32 sanity protocol forbids AMP")
    raw_by_trip = {}
    labels_by_trip: dict[str, TTCSupervision] = {}
    best_epochs: list[int] = []
    for inner_index, inner in enumerate(inner_folds):
        if outer.test_trip_id in inner.train_trip_ids:
            raise AssertionError("Outer test trip leaked into inner training")
        if inner.validation_trip_id == outer.test_trip_id:
            raise AssertionError("Outer test trip leaked into inner validation")
        model, normalizer, best_epoch = train_inner_model(
            [supervised[trip_id] for trip_id in inner.train_trip_ids],
            [supervised[inner.validation_trip_id]],
            config,
            settings,
            device=device,
            log_dir=(
                log_root
                / FUSION_VARIANT
                / config.tag
                / f"inner-{inner.validation_trip_id}"
            ),
            # Preserve the fusion seed offset used by the accepted P1 runner.
            seed=seed + 10_000 + inner_index,
            variant=FUSION_VARIANT,
        )
        _assert_fp32_finite(model)
        raw_by_trip[inner.validation_trip_id] = predict_raw(
            model,
            normalizer,
            supervised[inner.validation_trip_id].runtime,
            device=device,
        )
        labels_by_trip[inner.validation_trip_id] = supervised[
            inner.validation_trip_id
        ].labels
        best_epochs.append(best_epoch)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_validation = {fold.validation_trip_id for fold in inner_folds}
    if set(raw_by_trip) != expected_validation:
        raise AssertionError("Inner fusion prediction provenance is incomplete")
    if outer.test_trip_id in raw_by_trip:
        raise AssertionError("Outer predictions cannot participate in model selection")
    return FusionCandidateResult(
        config=config,
        choice=select_threshold(
            raw_by_trip,
            labels_by_trip,
            variant=FUSION_VARIANT,
            thresholds=thresholds,
        ),
        best_epochs=tuple(best_epochs),
        validation_trip_ids=tuple(sorted(raw_by_trip)),
    )


def select_fusion_candidate(
    results: Sequence[FusionCandidateResult],
    *,
    outer_train_trip_ids: Sequence[str],
    outer_test_trip_id: str,
) -> FusionCandidateResult:
    """Select by inner fusion score with the unchanged P1 tie-break."""

    if not results:
        raise ValueError("At least one fusion candidate result is required")
    allowed = set(outer_train_trip_ids)
    for result in results:
        validation = set(result.validation_trip_ids)
        if not validation <= allowed or outer_test_trip_id in validation:
            raise ValueError("Candidate result contains outer-test provenance")
    return max(
        results,
        key=lambda result: (
            result.choice.macro_composite,
            -result.config.gru_layers,
            -result.config.hidden_size,
            -result.config.sequence_length,
        ),
    )


def _prepare_protocol_data(args: argparse.Namespace) -> PreparedProtocolData:
    """Build label-free runtime features, then attach training-only labels."""

    cache_runtime: dict[str, TemporalTripFeatures] = {}
    labels: dict[str, TTCSupervision] = {}
    for trip_id in P1_TRIP_IDS:
        trip_dir = args.data_dir / trip_id
        manifest_path = args.manifest_dir / f"{trip_id}.json.gz"
        if not manifest_path.is_file():
            prepare_c1_manifest(trip_dir, manifest_path)
        lean_loader = C1LeanTripLoader(trip_dir, manifest_path)
        lean_loader.check_image_2_integrity().require_ok()
        if lean_loader.n_frames != 600:
            raise ValueError(
                f"FP32 protocol expects 600 frames in {trip_id}, "
                f"got {lean_loader.n_frames}"
            )
        cache_path = locate_detection_cache(
            args.detection_cache_dir,
            trip_id,
            pattern=args.cache_pattern,
        )
        cache_runtime[trip_id] = build_trip_features(
            lean_loader,
            load_detection_cache(cache_path),
            load_camera_geometry(trip_dir),
            confidence_threshold=args.detection_confidence,
        )
        labels[trip_id] = load_training_ttc_labels(trip_dir)
        print(f"Prepared {trip_id}: 600 causal runtime rows", flush=True)

    authoritative_physics: dict[str, np.ndarray] = {}
    for trip_id in P1_TRIP_IDS:
        authoritative_physics[trip_id] = _read_prediction_csv(
            args.baseline_dir / f"{trip_id}.csv",
            runtime=cache_runtime[trip_id],
        )
    physics_report = _ablation_report(authoritative_physics, labels)
    physics_score = float(
        physics_report["macro"]["composite_score"]  # type: ignore[index]
    )
    if abs(physics_score - PHYSICS_REFERENCE_SCORE) > 0.05:
        raise RuntimeError(
            f"physics baseline gate failed: expected {PHYSICS_REFERENCE_SCORE:.1f}, "
            f"recreated {physics_score:.1f}"
        )

    supervised: dict[str, SupervisedTrip] = {}
    for trip_id in P1_TRIP_IDS:
        cached = cache_runtime[trip_id]
        runtime = TemporalTripFeatures(
            trip_id=cached.trip_id,
            frame_ids=cached.frame_ids,
            timestamps=cached.timestamps,
            features=cached.features,
            # Use the same frozen, frame-gated physics input as accepted P1.
            physics_ttc_s=authoritative_physics[trip_id].astype(np.float32),
        )
        supervised[trip_id] = SupervisedTrip(runtime, labels[trip_id])
    return PreparedProtocolData(
        supervised=supervised,
        labels=labels,
        physics_score=physics_score,
        physics_report=physics_report,
    )


def run_fp32_seed(
    seed: int,
    prepared: PreparedProtocolData,
    *,
    candidates: Sequence[TemporalModelConfig],
    thresholds: Sequence[float],
    settings: TrainSettings,
    device: torch.device,
    output_root: Path,
    run_root: Path,
) -> dict[str, object]:
    """Run all six outer folds for one declared seed, fusion only."""

    if seed not in FP32_SEEDS:
        raise ValueError(f"seed {seed} is not in the declared FP32 protocol")
    if settings.seed != seed:
        raise ValueError("TrainSettings seed must match the current protocol seed")
    if settings.amp:
        raise ValueError("The FP32 sanity protocol forbids AMP")

    seed_output = output_root / f"seed-{seed}" / FUSION_VARIANT
    seed_run = run_root / f"seed-{seed}"
    split_plan = nested_lopo_splits(P1_TRIP_IDS, inner_folds=5)
    fold_summaries: list[dict[str, object]] = []
    for outer_index, (outer, inner_folds) in enumerate(split_plan):
        print(
            f"FP32 seed={seed} outer={outer.test_trip_id}: fusion-only inner LOPO",
            flush=True,
        )
        candidate_results: list[FusionCandidateResult] = []
        for config in candidates:
            result = evaluate_fusion_candidate_inner(
                prepared.supervised,
                outer,
                inner_folds,
                config,
                settings,
                thresholds=thresholds,
                device=device,
                log_root=(
                    seed_run / "tensorboard" / f"outer-{outer.test_trip_id}"
                ),
                seed=seed + outer_index * 100,
            )
            candidate_results.append(result)
            print(
                f"  {config.tag}: inner fusion={result.choice.macro_composite:.1f}",
                flush=True,
            )

        selected = select_fusion_candidate(
            candidate_results,
            outer_train_trip_ids=outer.train_trip_ids,
            outer_test_trip_id=outer.test_trip_id,
        )
        final_epochs = max(1, int(round(statistics.median(selected.best_epochs))))
        model, normalizer = train_final_model(
            outer_training_trips(prepared.supervised, outer),
            selected.config,
            settings,
            epochs=final_epochs,
            device=device,
            log_dir=(
                seed_run
                / "tensorboard"
                / f"outer-{outer.test_trip_id}"
                / FUSION_VARIANT
                / "final"
            ),
            seed=seed + 10_000 + outer_index,
            variant=FUSION_VARIANT,
        )
        _assert_fp32_finite(model)

        outer_raw = predict_raw(
            model,
            normalizer,
            prepared.supervised[outer.test_trip_id].runtime,
            device=device,
        )
        predictions = calibrated_ttc(
            outer_raw.fusion_inverse,
            outer_raw.fusion_danger_logit,
            danger_threshold=selected.choice.threshold,
        )
        prediction_path = seed_output / f"{outer.test_trip_id}.csv"
        _write_prediction_csv(
            prediction_path,
            prepared.supervised[outer.test_trip_id].runtime,
            predictions,
        )

        checkpoint = checkpoint_payload(
            model,
            normalizer,
            # The temporal head is deliberately untrained and unused.
            temporal_threshold=0.5,
            fusion_threshold=selected.choice.threshold,
            train_trip_ids=outer.train_trip_ids,
            outer_test_trip_id=outer.test_trip_id,
            seed=seed + 10_000 + outer_index,
        )
        checkpoint.update(
            {
                "ablation_variant": FUSION_VARIANT,
                "temporal_head_trained": False,
                "precision": "float32",
                "amp": False,
            }
        )
        checkpoint_path = (
            seed_run / "checkpoints" / f"{outer.test_trip_id}.fusion.pt"
        )
        _atomic_torch_save(checkpoint_path, checkpoint)

        fold_manifest: dict[str, object] = {
            "outer_test_trip_id": outer.test_trip_id,
            "outer_train_trip_ids": list(outer.train_trip_ids),
            "protocol_seed": seed,
            "precision": "float32",
            "amp": False,
            "variant": FUSION_VARIANT,
            "temporal_only_trained": False,
            "selected_config": selected.config.to_dict(),
            "parameter_count": model.parameter_count,
            "final_epochs": final_epochs,
            "threshold": selected.choice.threshold,
            "validation_trip_ids": list(selected.validation_trip_ids),
            "inner_results": [
                {
                    "config": result.config.to_dict(),
                    "fusion": asdict(result.choice),
                    "best_epochs": list(result.best_epochs),
                    "validation_trip_ids": list(result.validation_trip_ids),
                }
                for result in candidate_results
            ],
            "prediction": str(prediction_path),
            "checkpoint": str(checkpoint_path),
        }
        _atomic_json(
            seed_run / "folds" / f"{outer.test_trip_id}.json",
            fold_manifest,
        )
        fold_summaries.append(fold_manifest)
        print(
            f"FP32 seed={seed} outer={outer.test_trip_id}: complete "
            f"({selected.config.tag}, {final_epochs} epochs, "
            f"threshold={selected.choice.threshold:.2f})",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _require_exact_oof_csv_set(seed_output)
    seed_predictions = {
        trip_id: _read_prediction_csv(seed_output / f"{trip_id}.csv")
        for trip_id in P1_TRIP_IDS
    }
    fusion_report = _ablation_report(seed_predictions, prepared.labels)
    if fusion_report["frames"] != 3600:
        raise AssertionError(
            f"seed {seed} emitted {fusion_report['frames']} rows instead of 3600"
        )
    report: dict[str, object] = {
        "seed": seed,
        "precision": "float32",
        "amp": False,
        "variant": FUSION_VARIANT,
        "temporal_only_trained": False,
        "outer_folds": 6,
        "inner_folds": 5,
        "outer_test_selection": False,
        "frames": 3600,
        "fusion": fusion_report,
        "folds": fold_summaries,
    }
    _atomic_json(output_root / f"seed-{seed}" / "report.json", report)
    return report


def aggregate_seed_reports(
    reports: Mapping[int, Mapping[str, object]],
    *,
    physics_score: float,
) -> dict[str, object]:
    """Aggregate all declared seeds without selecting the best realization."""

    if tuple(sorted(reports)) != FP32_SEEDS:
        raise ValueError(
            "FP32 aggregate requires exactly the declared seeds "
            f"{FP32_SEEDS}; got {tuple(sorted(reports))}"
        )
    macro_keys = ("mae_critical", "f1", "inv_ttc_mae", "composite_score")
    values: dict[str, list[float]] = {key: [] for key in macro_keys}
    per_seed: list[dict[str, object]] = []
    for seed in FP32_SEEDS:
        report = reports[seed]
        if int(report["frames"]) != 3600:
            raise ValueError(f"seed {seed} does not contain exactly 3,600 frames")
        macro = report["fusion"]["macro"]  # type: ignore[index]
        copied_macro: dict[str, float] = {}
        for key in macro_keys:
            value = float(macro[key])  # type: ignore[index]
            if not math.isfinite(value):
                raise ValueError(f"seed {seed} has non-finite macro {key}")
            values[key].append(value)
            copied_macro[key] = value
        per_seed.append({"seed": seed, "frames": 3600, "macro": copied_macro})

    aggregate: dict[str, dict[str, object]] = {}
    for key in macro_keys:
        aggregate[key] = {
            "values_in_seed_order": values[key],
            "mean": round(statistics.mean(values[key]), 6),
            "sample_std_ddof_1": round(statistics.stdev(values[key]), 6),
        }
    mean_composite = float(aggregate["composite_score"]["mean"])
    p1_closed = mean_composite <= physics_score
    return {
        "declared_seeds": list(FP32_SEEDS),
        "seed_selection": False,
        "aggregation": "arithmetic mean +/- sample standard deviation (ddof=1)",
        "per_seed": per_seed,
        "macro_mean_std": aggregate,
        "physics_reference": physics_score,
        "fusion_mean_delta": round(mean_composite - physics_score, 6),
        "p1_closed": p1_closed,
        "decision": (
            "close_p1_keep_physics_default"
            if p1_closed
            else "fusion_mean_exceeds_physics_requires_review"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safeloop.c1.train_fusion_fp32",
        description=(
            "Fusion-only, strict-FP32, three-seed nested-LOPO sanity check. "
            "Seeds and optimizer protocol are intentionally fixed."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--detection-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "predictions/c1_baseline_recheck/six_samples",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "predictions/c1_fusion_fp32"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs/c1/fusion_fp32"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "runs/c1/input_manifests",
    )
    parser.add_argument(
        "--cache-pattern", default="{trip_id}.stride3.conf020.json.gz"
    )
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = _device(args.device)
        evaluator_sha_before = _evaluator_sha256()
        if evaluator_sha_before != EXPECTED_EVALUATOR_SHA256:
            raise RuntimeError(
                "official evaluator SHA-256 gate failed: "
                f"{evaluator_sha_before} != {EXPECTED_EVALUATOR_SHA256}"
            )
    except RuntimeError as exc:
        print(f"C1 fusion FP32 configuration error: {exc}", file=sys.stderr)
        return 2

    # AMP and TensorFloat-32 are both disabled for this sanity protocol.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(device)
        print(
            f"Fusion FP32 GPU={properties.name}, "
            f"VRAM={properties.total_memory / 2**30:.2f} GiB, "
            "AMP=False, TF32=False",
            flush=True,
        )
    else:
        print("Fusion FP32 device=cpu, AMP=False", flush=True)

    try:
        prepared = _prepare_protocol_data(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"C1 fusion FP32 data/baseline error: {exc}", file=sys.stderr)
        return 3
    print(
        f"Physics baseline gate PASS: macro composite={prepared.physics_score:.1f}",
        flush=True,
    )

    seed_reports: dict[int, Mapping[str, object]] = {}
    try:
        for seed in FP32_SEEDS:
            settings = TrainSettings(
                max_epochs=80,
                patience=15,
                batch_size=128,
                learning_rate=1e-3,
                weight_decay=1e-4,
                gradient_clip_norm=1.0,
                seed=seed,
                amp=False,
                num_workers=0,
            )
            seed_reports[seed] = run_fp32_seed(
                seed,
                prepared,
                candidates=DEFAULT_CANDIDATES,
                thresholds=FP32_THRESHOLDS,
                settings=settings,
                device=device,
                output_root=args.output_dir,
                run_root=args.run_dir,
            )
    except (FloatingPointError, OSError, RuntimeError, ValueError) as exc:
        print(f"C1 fusion FP32 training error: {exc}", file=sys.stderr)
        return 4

    evaluator_sha_after = _evaluator_sha256()
    if evaluator_sha_after != evaluator_sha_before:
        print(
            "C1 fusion FP32 evaluator changed during training: "
            f"{evaluator_sha_before} -> {evaluator_sha_after}",
            file=sys.stderr,
        )
        return 5
    summary = aggregate_seed_reports(seed_reports, physics_score=prepared.physics_score)
    report: dict[str, object] = {
        "protocol": {
            "name": "fusion_fp32_three_seed_nested_leave_one_trip_out",
            "development_cross_validation": True,
            "external_held_out_claim": False,
            "declared_seeds": list(FP32_SEEDS),
            "seed_selection": False,
            "variant": FUSION_VARIANT,
            "temporal_only_rerun": False,
            "precision": "float32",
            "amp": False,
            "tf32": False,
            "outer_folds": 6,
            "inner_folds": 5,
            "outer_test_selection": False,
            "threshold_calibration": "inner validation only",
            "frames_per_seed": 3600,
            "physics_pipeline_default": True,
            "official_evaluator_sha256_before": evaluator_sha_before,
            "official_evaluator_sha256_after": evaluator_sha_after,
        },
        "device": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "training": {
            "settings_except_seed": {
                "max_epochs": 80,
                "patience": 15,
                "batch_size": 128,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "amp": False,
                "num_workers": 0,
            },
            "candidates": [config.to_dict() for config in DEFAULT_CANDIDATES],
            "danger_threshold_grid": list(FP32_THRESHOLDS),
            "finite_fp32_parameter_guard": True,
        },
        "physics": prepared.physics_report,
        "fusion_fp32": summary,
        "artifacts": {
            "predictions": str(args.output_dir),
            "checkpoints": str(args.run_dir / "seed-*/checkpoints"),
            "tensorboard": str(args.run_dir / "seed-*/tensorboard"),
            "onnx_exported": False,
        },
    }
    _atomic_json(args.output_dir / "report.json", report)
    _atomic_json(
        args.run_dir / "protocol.json",
        report["protocol"],  # type: ignore[arg-type]
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
