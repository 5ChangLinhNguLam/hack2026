"""Nested leave-one-trip-out training for the causal C1 temporal TTC head.

The module keeps runtime inputs and supervision structurally separate:

* :mod:`safeloop.c1.temporal_features` builds label-free features using the
  explicit :class:`~safeloop.c1.C1LeanTripLoader` contract.
* ``load_training_ttc_labels`` below is training-only and returns a separate
  object which is joined only after feature extraction.

The outer trip is never passed to normalization, optimization, early stopping,
candidate selection, or danger-threshold calibration.  It is evaluated once,
after all choices for that fold have been frozen by inner LOPO validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import shutil
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from team_kit.evaluation import TripMetrics, compute_trip_metrics

from .lean_loader import C1LeanTripLoader
from .prepare_manifest import prepare_c1_manifest
from .prediction_preflight import preflight_prediction_csv
from .temporal_features import (
    FEATURE_DIM,
    TemporalTripFeatures,
    build_trip_features,
    inverse_ttc,
    load_camera_geometry,
    load_detection_cache,
    locate_detection_cache,
)
from .temporal_model import (
    DEFAULT_CANDIDATES,
    CausalTemporalTTC,
    FeatureNormalizer,
    TemporalModelConfig,
    build_causal_windows,
    calibrated_ttc,
    checkpoint_payload,
    decode_current_outputs,
    export_temporal_onnx,
    gather_current,
)


ROOT = Path(__file__).resolve().parents[2]
P1_TRIP_IDS: tuple[str, ...] = tuple(f"T{index:02d}-Sample" for index in range(1, 7))
ABLATIONS = ("physics", "temporal", "fusion")


@dataclass(frozen=True)
class TTCSupervision:
    """Real practice-set TTC labels; never used by an inference object."""

    trip_id: str
    frame_ids: np.ndarray
    ttc_s: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.frame_ids)
        if self.frame_ids.shape != (n,) or self.ttc_s.shape != (n,):
            raise ValueError("TTC supervision arrays must be one-dimensional")
        if not np.array_equal(self.frame_ids, np.arange(n, dtype=np.int64)):
            raise ValueError(
                f"TTC labels for {self.trip_id} must have contiguous frame IDs 0..{n - 1}"
            )
        invalid = np.isnan(self.ttc_s) | (self.ttc_s <= 0.0)
        if np.any(invalid):
            bad = np.flatnonzero(invalid)[:10].tolist()
            raise ValueError(f"Invalid TTC labels in {self.trip_id} at frames {bad}")


@dataclass(frozen=True)
class SupervisedTrip:
    runtime: TemporalTripFeatures
    labels: TTCSupervision

    def __post_init__(self) -> None:
        if self.runtime.trip_id != self.labels.trip_id:
            raise ValueError("Runtime/label trip IDs do not match")
        if not np.array_equal(self.runtime.frame_ids, self.labels.frame_ids):
            raise ValueError(f"Runtime/label frame mismatch for {self.runtime.trip_id}")

    @property
    def trip_id(self) -> str:
        return self.runtime.trip_id


@dataclass(frozen=True)
class OuterFold:
    train_trip_ids: tuple[str, ...]
    test_trip_id: str


@dataclass(frozen=True)
class InnerFold:
    train_trip_ids: tuple[str, ...]
    validation_trip_id: str


@dataclass(frozen=True)
class TrainSettings:
    max_epochs: int = 80
    patience: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 2026
    amp: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("batch_size and learning_rate must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True)
class MaterializedArrays:
    windows: np.ndarray
    lengths: np.ndarray
    physics_inverse: np.ndarray
    target_inverse: np.ndarray
    target_ttc: np.ndarray
    danger: np.ndarray
    trip_indices: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.lengths)


@dataclass(frozen=True)
class RawModelPredictions:
    temporal_inverse: np.ndarray
    fusion_inverse: np.ndarray
    temporal_danger_logit: np.ndarray
    fusion_danger_logit: np.ndarray


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    macro_composite: float
    macro_f1: float
    macro_mae_critical: float


@dataclass(frozen=True)
class CandidateResult:
    config: TemporalModelConfig
    temporal_choice: ThresholdChoice
    fusion_choice: ThresholdChoice
    temporal_best_epochs: tuple[int, ...]
    fusion_best_epochs: tuple[int, ...]
    validation_trip_ids: tuple[str, ...]


def _resolve_trip_json(trip_dir: Path) -> Path:
    """Training-only resolver that ignores a directory named ``*.json``."""

    trip_id = trip_dir.name
    candidates = (
        trip_dir / f"{trip_id}.json",
        trip_dir / f"{trip_id}.json.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fallback = sorted(
        path
        for pattern in ("*.json", "*.json.gz")
        for path in trip_dir.glob(pattern)
        if path.is_file()
    )
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"No trip JSON file in {trip_dir}")


def load_training_ttc_labels(trip_dir: str | Path) -> TTCSupervision:
    """Load only real TTC labels for supervised practice-set training.

    This function is intentionally defined in the training module.  It is not
    imported by feature extraction or inference.  Redacted trips and pseudo
    labels are rejected rather than silently treated as ground truth.
    """

    directory = Path(trip_dir)
    path = _resolve_trip_json(directory)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    trip_id = str(document.get("trip_id") or directory.name)
    if trip_id not in P1_TRIP_IDS:
        raise ValueError(
            f"P1 supervision is restricted to the six labelled practice trips; got {trip_id}"
        )
    raw_frames = document.get("frames") or []
    frame_ids = np.empty(len(raw_frames), dtype=np.int64)
    # Keep evaluator-facing labels at source precision. Materialization casts
    # optimization tensors to float32 separately, so this does not change the
    # training numerics while preserving exact official-metric parity.
    ttc = np.empty(len(raw_frames), dtype=np.float64)
    for position, raw in enumerate(raw_frames):
        if "min_ttc" not in raw or raw["min_ttc"] is None:
            raise ValueError(
                f"{trip_id} frame {position} has no real TTC label; redacted/pseudo "
                "data cannot be used as P1 supervision"
            )
        frame_ids[position] = int(raw["frame_id"])
        ttc[position] = float(raw["min_ttc"])
    # Do not retain the source document, which contains other annotations.
    del raw_frames, document
    return TTCSupervision(trip_id=trip_id, frame_ids=frame_ids, ttc_s=ttc)


def nested_lopo_splits(
    trip_ids: Sequence[str],
    *,
    inner_folds: int | None = None,
) -> tuple[tuple[OuterFold, tuple[InnerFold, ...]], ...]:
    """Build deterministic nested trip-level splits with leakage assertions."""

    ordered = tuple(trip_ids)
    if len(ordered) < 3 or len(set(ordered)) != len(ordered):
        raise ValueError("Need at least three unique trip IDs for nested LOPO")
    inner_count = len(ordered) - 1 if inner_folds is None else inner_folds
    if not 1 <= inner_count <= len(ordered) - 1:
        raise ValueError(f"inner_folds must be in [1, {len(ordered) - 1}]")

    all_splits: list[tuple[OuterFold, tuple[InnerFold, ...]]] = []
    for outer_index, test_trip in enumerate(ordered):
        outer_train = tuple(trip for trip in ordered if trip != test_trip)
        # Rotate before truncation so a reduced diagnostic run does not always
        # validate on the same trip.  Production default uses all five.
        rotation = outer_index % len(outer_train)
        validation_order = outer_train[rotation:] + outer_train[:rotation]
        inner: list[InnerFold] = []
        for validation_trip in validation_order[:inner_count]:
            inner_train = tuple(
                trip for trip in outer_train if trip != validation_trip
            )
            if test_trip in inner_train or test_trip == validation_trip:
                raise AssertionError("Outer-test leakage into an inner fold")
            inner.append(InnerFold(inner_train, validation_trip))
        outer = OuterFold(outer_train, test_trip)
        all_splits.append((outer, tuple(inner)))
    return tuple(all_splits)


def outer_training_trips(
    supervised: Mapping[str, SupervisedTrip],
    outer: OuterFold,
) -> tuple[SupervisedTrip, ...]:
    """Resolve final-fit data and fail closed on outer-test provenance."""

    if outer.test_trip_id in outer.train_trip_ids:
        raise ValueError("Outer-test trip appears in outer training IDs")
    selected = tuple(supervised[trip_id] for trip_id in outer.train_trip_ids)
    selected_ids = {trip.trip_id for trip in selected}
    if selected_ids != set(outer.train_trip_ids):
        raise ValueError("Resolved outer training trips do not match the split")
    if outer.test_trip_id in selected_ids:
        raise ValueError("Outer-test supervision cannot enter final fitting")
    return selected


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _materialize(
    trips: Sequence[SupervisedTrip],
    normalizer: FeatureNormalizer,
    sequence_length: int,
) -> MaterializedArrays:
    if not trips:
        raise ValueError("At least one supervised trip is required")
    windows: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    physics: list[np.ndarray] = []
    target_inverse: list[np.ndarray] = []
    target_ttc: list[np.ndarray] = []
    dangers: list[np.ndarray] = []
    trip_indices: list[np.ndarray] = []
    for trip_index, trip in enumerate(trips):
        normalized = normalizer.transform(trip.runtime.features)
        trip_windows, trip_lengths = build_causal_windows(normalized, sequence_length)
        windows.append(trip_windows)
        lengths.append(trip_lengths)
        physics.append(inverse_ttc(trip.runtime.physics_ttc_s))
        target_inverse.append(inverse_ttc(trip.labels.ttc_s))
        target_ttc.append(trip.labels.ttc_s.astype(np.float32, copy=False))
        dangers.append((trip.labels.ttc_s < 2.0).astype(np.float32))
        trip_indices.append(np.full(trip.runtime.n_frames, trip_index, dtype=np.int64))
    return MaterializedArrays(
        windows=np.concatenate(windows),
        lengths=np.concatenate(lengths),
        physics_inverse=np.concatenate(physics),
        target_inverse=np.concatenate(target_inverse),
        target_ttc=np.concatenate(target_ttc),
        danger=np.concatenate(dangers),
        trip_indices=np.concatenate(trip_indices),
    )


def _tensor_dataset(arrays: MaterializedArrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.windows),
        torch.from_numpy(arrays.lengths),
        torch.from_numpy(arrays.physics_inverse),
        torch.from_numpy(arrays.target_inverse),
        torch.from_numpy(arrays.target_ttc),
        torch.from_numpy(arrays.danger),
    )


def _trip_balanced_sampler(
    trip_indices: np.ndarray,
    *,
    seed: int,
) -> WeightedRandomSampler:
    """Equal expected sampling mass per trip, independent of trip length."""

    counts = np.bincount(trip_indices)
    if np.any(counts <= 0):
        raise ValueError("Every sampler trip index must have samples")
    weights = 1.0 / counts[trip_indices]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _positive_weight(danger: np.ndarray, device: torch.device) -> Tensor:
    positives = float(np.sum(danger > 0.5))
    negatives = float(len(danger) - positives)
    ratio = negatives / positives if positives > 0 else 1.0
    return torch.tensor(min(50.0, max(1.0, ratio)), device=device)


def _loss(
    current: Tensor,
    physics_inverse: Tensor,
    target_inverse: Tensor,
    target_ttc: Tensor,
    danger: Tensor,
    *,
    config: TemporalModelConfig,
    positive_weight: Tensor,
    variant: str,
) -> Tensor:
    temporal_inverse, fusion_inverse, temporal_logit, fusion_logit = (
        decode_current_outputs(current, physics_inverse, config)
    )
    if variant == "temporal":
        predicted_inverse = temporal_inverse
        predicted_logit = temporal_logit
    elif variant == "fusion":
        predicted_inverse = fusion_inverse
        predicted_logit = fusion_logit
    else:
        raise ValueError("variant must be temporal or fusion")
    inverse_loss = F.smooth_l1_loss(predicted_inverse, target_inverse)
    danger_loss = F.binary_cross_entropy_with_logits(
        predicted_logit, danger, pos_weight=positive_weight
    )
    critical = target_ttc < 3.0
    if torch.any(critical):
        predicted_ttc = torch.reciprocal(
            predicted_inverse.clamp(min=1.0 / 99.0)
        )
        critical_loss = F.smooth_l1_loss(
            predicted_ttc[critical], target_ttc[critical]
        )
    else:
        critical_loss = inverse_loss.new_zeros(())
    return 0.40 * critical_loss + 0.30 * danger_loss + 0.30 * inverse_loss


def _batch_loss(
    model: CausalTemporalTTC,
    batch: Sequence[Tensor],
    *,
    device: torch.device,
    positive_weight: Tensor,
    amp_enabled: bool,
    variant: str,
) -> Tensor:
    windows, lengths, physics, target_inverse, target_ttc, danger = (
        item.to(device, non_blocking=True) for item in batch
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        current = gather_current(model(windows), lengths)
        return _loss(
            current,
            physics,
            target_inverse,
            target_ttc,
            danger,
            config=model.config,
            positive_weight=positive_weight,
            variant=variant,
        )


@torch.inference_mode()
def _validation_loss(
    model: CausalTemporalTTC,
    loader: DataLoader,
    *,
    device: torch.device,
    positive_weight: Tensor,
    amp_enabled: bool,
    variant: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        loss = _batch_loss(
            model,
            batch,
            device=device,
            positive_weight=positive_weight,
            amp_enabled=amp_enabled,
            variant=variant,
        )
        batch_size = len(batch[0])
        total += float(loss.item()) * batch_size
        count += batch_size
    return total / max(count, 1)


def train_inner_model(
    train_trips: Sequence[SupervisedTrip],
    validation_trips: Sequence[SupervisedTrip],
    config: TemporalModelConfig,
    settings: TrainSettings,
    *,
    device: torch.device,
    log_dir: Path,
    seed: int,
    variant: str,
) -> tuple[CausalTemporalTTC, FeatureNormalizer, int]:
    """Fit with train-only normalization and validation-only early stopping."""

    train_ids = {trip.trip_id for trip in train_trips}
    validation_ids = {trip.trip_id for trip in validation_trips}
    if train_ids & validation_ids:
        raise ValueError("Training/validation trip leakage")
    _seed_everything(seed)
    normalizer = FeatureNormalizer.fit(trip.runtime.features for trip in train_trips)
    train_arrays = _materialize(train_trips, normalizer, config.sequence_length)
    validation_arrays = _materialize(
        validation_trips, normalizer, config.sequence_length
    )
    sampler = _trip_balanced_sampler(train_arrays.trip_indices, seed=seed)
    train_loader = DataLoader(
        _tensor_dataset(train_arrays),
        batch_size=settings.batch_size,
        sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        _tensor_dataset(validation_arrays),
        batch_size=max(settings.batch_size, 256),
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CausalTemporalTTC(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    amp_enabled = settings.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    positive_weight = _positive_weight(train_arrays.danger, device)
    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        writer.add_text("split/train_trips", ",".join(sorted(train_ids)))
        writer.add_text("split/validation_trips", ",".join(sorted(validation_ids)))
        writer.add_text("model/config", json.dumps(config.to_dict(), sort_keys=True))
        writer.add_text("ablation/variant", variant)
        writer.add_scalar("model/parameters", model.parameter_count, 0)
        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            train_total = 0.0
            train_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(
                    model,
                    batch,
                    device=device,
                    positive_weight=positive_weight,
                    amp_enabled=amp_enabled,
                    variant=variant,
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                batch_size = len(batch[0])
                train_total += float(loss.item()) * batch_size
                train_count += batch_size
            validation_loss = _validation_loss(
                model,
                validation_loader,
                device=device,
                positive_weight=positive_weight,
                amp_enabled=amp_enabled,
                variant=variant,
            )
            train_loss = train_total / max(train_count, 1)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", validation_loss, epoch)
            writer.add_scalar("optimization/gradient_norm", float(gradient_norm), epoch)
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= settings.patience:
                    break
    finally:
        writer.close()
    if best_state is None:  # pragma: no cover - one epoch always records a state
        raise RuntimeError("Inner training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, normalizer, best_epoch


def train_final_model(
    train_trips: Sequence[SupervisedTrip],
    config: TemporalModelConfig,
    settings: TrainSettings,
    *,
    epochs: int,
    device: torch.device,
    log_dir: Path,
    seed: int,
    variant: str,
) -> tuple[CausalTemporalTTC, FeatureNormalizer]:
    """Retrain on all five outer-train trips for inner-selected epoch count."""

    if epochs < 1:
        raise ValueError("Final epoch count must be positive")
    _seed_everything(seed)
    normalizer = FeatureNormalizer.fit(trip.runtime.features for trip in train_trips)
    arrays = _materialize(train_trips, normalizer, config.sequence_length)
    sampler = _trip_balanced_sampler(arrays.trip_indices, seed=seed)
    loader = DataLoader(
        _tensor_dataset(arrays),
        batch_size=settings.batch_size,
        sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = CausalTemporalTTC(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    amp_enabled = settings.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    positive_weight = _positive_weight(arrays.danger, device)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        writer.add_text(
            "split/train_trips", ",".join(sorted(trip.trip_id for trip in train_trips))
        )
        writer.add_text("model/config", json.dumps(config.to_dict(), sort_keys=True))
        writer.add_text("ablation/variant", variant)
        writer.add_scalar("model/parameters", model.parameter_count, 0)
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(
                    model,
                    batch,
                    device=device,
                    positive_weight=positive_weight,
                    amp_enabled=amp_enabled,
                    variant=variant,
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                batch_size = len(batch[0])
                total += float(loss.item()) * batch_size
                count += batch_size
            writer.add_scalar("loss/train", total / max(count, 1), epoch)
            writer.add_scalar("optimization/gradient_norm", float(gradient_norm), epoch)
    finally:
        writer.close()
    return model.eval(), normalizer


@torch.inference_mode()
def predict_raw(
    model: CausalTemporalTTC,
    normalizer: FeatureNormalizer,
    runtime: TemporalTripFeatures,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> RawModelPredictions:
    normalized = normalizer.transform(runtime.features)
    windows, lengths = build_causal_windows(normalized, model.config.sequence_length)
    physics = inverse_ttc(runtime.physics_ttc_s)
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, runtime.n_frames, batch_size):
        end = min(runtime.n_frames, start + batch_size)
        window_tensor = torch.from_numpy(windows[start:end]).to(device)
        length_tensor = torch.from_numpy(lengths[start:end]).to(device)
        physics_tensor = torch.from_numpy(physics[start:end]).to(device)
        current = gather_current(model(window_tensor), length_tensor)
        temporal_inv, fusion_inv, temporal_logit, fusion_logit = (
            decode_current_outputs(current, physics_tensor, model.config)
        )
        outputs.append(
            torch.stack(
                (temporal_inv, fusion_inv, temporal_logit, fusion_logit), dim=1
            ).cpu().numpy()
        )
    combined = np.concatenate(outputs, axis=0)
    return RawModelPredictions(
        temporal_inverse=combined[:, 0],
        fusion_inverse=combined[:, 1],
        temporal_danger_logit=combined[:, 2],
        fusion_danger_logit=combined[:, 3],
    )


def _metrics_for_predictions(
    trip_id: str,
    predictions: np.ndarray,
    labels: TTCSupervision,
) -> TripMetrics:
    if len(predictions) != len(labels.ttc_s):
        raise ValueError(f"Prediction/label count mismatch for {trip_id}")
    pairs = {
        int(frame_id): (float(predictions[index]), float(labels.ttc_s[index]))
        for index, frame_id in enumerate(labels.frame_ids)
    }
    return compute_trip_metrics(trip_id, pairs)


def _macro(metrics: Sequence[TripMetrics]) -> dict[str, float]:
    if not metrics:
        raise ValueError("Cannot macro-average an empty metric list")
    finite_mae = [metric.mae_critical for metric in metrics if metric.mae_critical >= 0]
    return {
        "mae_critical": round(float(np.mean(finite_mae)), 3) if finite_mae else -1.0,
        "f1": round(float(np.mean([metric.f1 for metric in metrics])), 3),
        "inv_ttc_mae": round(
            float(np.mean([metric.inv_ttc_mae for metric in metrics])), 4
        ),
        "composite_score": round(
            float(np.mean([metric.composite_score for metric in metrics])), 1
        ),
    }


def select_threshold(
    raw_by_trip: Mapping[str, RawModelPredictions],
    labels_by_trip: Mapping[str, TTCSupervision],
    *,
    variant: str,
    thresholds: Sequence[float],
) -> ThresholdChoice:
    """Calibrate solely from inner-validation predictions."""

    if variant not in {"temporal", "fusion"}:
        raise ValueError("variant must be temporal or fusion")
    if set(raw_by_trip) != set(labels_by_trip):
        raise ValueError("Threshold calibration trip sets do not match")
    choices: list[ThresholdChoice] = []
    for threshold in thresholds:
        metrics: list[TripMetrics] = []
        for trip_id in sorted(raw_by_trip):
            raw = raw_by_trip[trip_id]
            if variant == "temporal":
                inverse_values = raw.temporal_inverse
                logits = raw.temporal_danger_logit
            else:
                inverse_values = raw.fusion_inverse
                logits = raw.fusion_danger_logit
            decoded = calibrated_ttc(
                inverse_values, logits, danger_threshold=float(threshold)
            )
            metrics.append(
                _metrics_for_predictions(trip_id, decoded, labels_by_trip[trip_id])
            )
        macro = _macro(metrics)
        choices.append(
            ThresholdChoice(
                threshold=float(threshold),
                macro_composite=macro["composite_score"],
                macro_f1=macro["f1"],
                macro_mae_critical=macro["mae_critical"],
            )
        )
    # Deterministic tie-break: score, F1, lower MAE, then threshold closest to
    # 0.5 (and lower threshold if still tied).  No outer data appears here.
    return max(
        choices,
        key=lambda choice: (
            choice.macro_composite,
            choice.macro_f1,
            -choice.macro_mae_critical,
            -abs(choice.threshold - 0.5),
            -choice.threshold,
        ),
    )


def evaluate_candidate_inner(
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
) -> CandidateResult:
    """Evaluate one architecture using inner validation trips only."""

    temporal_raw_by_trip: dict[str, RawModelPredictions] = {}
    fusion_raw_by_trip: dict[str, RawModelPredictions] = {}
    labels_by_trip: dict[str, TTCSupervision] = {}
    temporal_best_epochs: list[int] = []
    fusion_best_epochs: list[int] = []
    for inner_index, inner in enumerate(inner_folds):
        if outer.test_trip_id in inner.train_trip_ids:
            raise AssertionError("Outer test trip leaked into inner training")
        if inner.validation_trip_id == outer.test_trip_id:
            raise AssertionError("Outer test trip leaked into inner validation")
        for variant, raw_destination, epoch_destination, seed_offset in (
            ("temporal", temporal_raw_by_trip, temporal_best_epochs, 0),
            ("fusion", fusion_raw_by_trip, fusion_best_epochs, 10_000),
        ):
            # Independent model instances are essential for a clean ablation:
            # temporal-only never receives a gradient from the fusion loss.
            model, normalizer, best_epoch = train_inner_model(
                [supervised[trip] for trip in inner.train_trip_ids],
                [supervised[inner.validation_trip_id]],
                config,
                settings,
                device=device,
                log_dir=log_root
                / variant
                / config.tag
                / f"inner-{inner.validation_trip_id}",
                seed=seed + seed_offset + inner_index,
                variant=variant,
            )
            raw_destination[inner.validation_trip_id] = predict_raw(
                model,
                normalizer,
                supervised[inner.validation_trip_id].runtime,
                device=device,
            )
            epoch_destination.append(best_epoch)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        labels_by_trip[inner.validation_trip_id] = supervised[
            inner.validation_trip_id
        ].labels
    expected_validation = {fold.validation_trip_id for fold in inner_folds}
    if (
        set(temporal_raw_by_trip) != expected_validation
        or set(fusion_raw_by_trip) != expected_validation
    ):
        raise AssertionError("Inner prediction provenance is incomplete")
    if (
        outer.test_trip_id in temporal_raw_by_trip
        or outer.test_trip_id in fusion_raw_by_trip
    ):
        raise AssertionError("Outer predictions cannot participate in model selection")
    return CandidateResult(
        config=config,
        temporal_choice=select_threshold(
            temporal_raw_by_trip,
            labels_by_trip,
            variant="temporal",
            thresholds=thresholds,
        ),
        fusion_choice=select_threshold(
            fusion_raw_by_trip,
            labels_by_trip,
            variant="fusion",
            thresholds=thresholds,
        ),
        temporal_best_epochs=tuple(temporal_best_epochs),
        fusion_best_epochs=tuple(fusion_best_epochs),
        validation_trip_ids=tuple(sorted(temporal_raw_by_trip)),
    )


def select_candidate(
    results: Sequence[CandidateResult],
    *,
    outer_train_trip_ids: Sequence[str],
    outer_test_trip_id: str,
    variant: str,
) -> CandidateResult:
    """Select one standalone ablation architecture without outer-test access."""

    allowed = set(outer_train_trip_ids)
    for result in results:
        validation = set(result.validation_trip_ids)
        if not validation <= allowed or outer_test_trip_id in validation:
            raise ValueError("Candidate result contains outer-test provenance")
    if variant not in {"temporal", "fusion"}:
        raise ValueError("variant must be temporal or fusion")
    return max(
        results,
        key=lambda result: (
            (
                result.temporal_choice.macro_composite
                if variant == "temporal"
                else result.fusion_choice.macro_composite
            ),
            -result.config.gru_layers,
            -result.config.hidden_size,
            -result.config.sequence_length,
        ),
    )


def _read_prediction_csv(
    path: Path,
    *,
    expected_frames: int = 600,
    runtime: TemporalTripFeatures | None = None,
) -> np.ndarray:
    preflight_prediction_csv(path, expected_frames=expected_frames)
    values = np.full(expected_frames, np.inf, dtype=np.float64)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame_id = int(row["frame_id"])
            if runtime is not None:
                if frame_id >= runtime.n_frames:
                    raise ValueError(f"{path}: frame {frame_id} exceeds runtime data")
                timestamp = float(row["timestamp"])
                if not math.isclose(
                    timestamp,
                    float(runtime.timestamps[frame_id]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        f"{path}: timestamp mismatch at frame {frame_id}: "
                        f"{timestamp} != {runtime.timestamps[frame_id]}"
                    )
            values[frame_id] = float(row["predicted_ttc"])
    return values


def _write_prediction_csv(
    path: Path,
    runtime: TemporalTripFeatures,
    predictions: np.ndarray,
) -> None:
    if len(predictions) != runtime.n_frames:
        raise ValueError(f"Expected {runtime.n_frames} predictions for {runtime.trip_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["frame_id", "timestamp", "predicted_ttc"]
        )
        writer.writeheader()
        for frame_id, timestamp, ttc in zip(
            runtime.frame_ids, runtime.timestamps, predictions
        ):
            writer.writerow(
                {
                    "frame_id": int(frame_id),
                    "timestamp": round(float(timestamp), 3),
                    "predicted_ttc": round(float(ttc), 6)
                    if math.isfinite(float(ttc))
                    else "inf",
                }
            )
    temporary.replace(path)
    preflight_prediction_csv(path, expected_frames=runtime.n_frames)


def _require_exact_oof_csv_set(directory: Path) -> tuple[Path, ...]:
    """Reject missing or stale extra CSVs before claiming exactly 3,600 OOF rows."""

    expected_names = {f"{trip_id}.csv" for trip_id in P1_TRIP_IDS}
    actual_paths = tuple(sorted(directory.glob("*.csv")))
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise ValueError(
            f"{directory}: OOF CSV set mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    return actual_paths


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _ablation_report(
    predictions: Mapping[str, np.ndarray],
    labels: Mapping[str, TTCSupervision],
) -> dict[str, object]:
    metrics = [
        _metrics_for_predictions(trip_id, predictions[trip_id], labels[trip_id])
        for trip_id in P1_TRIP_IDS
    ]
    return {
        "frames": int(sum(len(predictions[trip]) for trip in P1_TRIP_IDS)),
        "macro": _macro(metrics),
        "per_trip": [asdict(metric) for metric in metrics],
    }


def _physics_cache_parity(
    authoritative: Mapping[str, np.ndarray],
    cache_runtime: Mapping[str, TemporalTripFeatures],
) -> dict[str, object]:
    per_trip: list[dict[str, object]] = []
    all_abs: list[np.ndarray] = []
    finite_disagreement = 0
    total = 0
    for trip_id in P1_TRIP_IDS:
        frozen = authoritative[trip_id]
        replayed = cache_runtime[trip_id].physics_ttc_s.astype(np.float64)
        both = np.isfinite(frozen) & np.isfinite(replayed)
        differences = np.abs(frozen[both] - replayed[both])
        all_abs.append(differences)
        disagreement = int(np.sum(np.isfinite(frozen) != np.isfinite(replayed)))
        finite_disagreement += disagreement
        total += len(frozen)
        per_trip.append(
            {
                "trip_id": trip_id,
                "both_finite": int(np.sum(both)),
                "finite_mask_disagreement": disagreement,
                "mean_abs_ttc_delta_s": round(float(np.mean(differences)), 6)
                if len(differences)
                else 0.0,
                "max_abs_ttc_delta_s": round(float(np.max(differences)), 6)
                if len(differences)
                else 0.0,
            }
        )
    combined = np.concatenate(all_abs) if all_abs else np.asarray([])
    return {
        "note": (
            "physics and fusion use frozen authoritative baseline CSV values; "
            "cache-replayed tracker physics is reported here only as a rounding "
            "parity diagnostic"
        ),
        "finite_mask_disagreement": finite_disagreement,
        "frames": total,
        "mean_abs_ttc_delta_s_on_both_finite": round(float(np.mean(combined)), 6)
        if len(combined)
        else 0.0,
        "max_abs_ttc_delta_s_on_both_finite": round(float(np.max(combined)), 6)
        if len(combined)
        else 0.0,
        "per_trip": per_trip,
    }


def _parse_candidates(value: str) -> tuple[TemporalModelConfig, ...]:
    candidates: list[TemporalModelConfig] = []
    for raw in value.split(","):
        try:
            sequence, hidden, layers = (int(part) for part in raw.split(":"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Candidates must use seq:hidden:layers, comma-separated"
            ) from exc
        candidates.append(
            TemporalModelConfig(
                sequence_length=sequence, hidden_size=hidden, gru_layers=layers
            )
        )
    if {candidate.sequence_length for candidate in candidates} != {8, 16, 24}:
        raise argparse.ArgumentTypeError(
            "P1 protocol requires at least one candidate for each sequence length 8/16/24"
        )
    return tuple(candidates)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated floats") from exc
    if not thresholds or any(not 0.0 < item < 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("Every threshold must be in (0, 1)")
    return thresholds


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safeloop.c1.train_temporal",
        description="Nested LOPO training for the causal monocular C1 TTC model.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--detection-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "predictions/c1_baseline_recheck/six_samples",
        help="frozen authoritative physics-v2 CSVs (55.2 baseline)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "predictions/c1_temporal_oof",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/c1/temporal_oof")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "runs/c1/input_manifests",
        help="sanitized runtime manifests (outside Git; never contains labels)",
    )
    parser.add_argument(
        "--cache-pattern", default="{trip_id}.stride3.conf020.json.gz"
    )
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    parser.add_argument(
        "--candidates",
        type=_parse_candidates,
        default=DEFAULT_CANDIDATES,
        help="comma-separated seq:hidden:layers presets",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=(0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-physics-score", type=float, default=55.2)
    parser.add_argument("--physics-score-tolerance", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = _device(args.device)
        settings = TrainSettings(
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip,
            seed=args.seed,
            amp=args.amp,
            num_workers=args.num_workers,
        )
        split_plan = nested_lopo_splits(P1_TRIP_IDS, inner_folds=args.inner_folds)
    except (RuntimeError, ValueError) as exc:
        print(f"C1 temporal configuration error: {exc}", file=sys.stderr)
        return 2

    print(
        f"C1 temporal device={device}, AMP={settings.amp and device.type == 'cuda'}, "
        f"candidates={','.join(config.tag for config in args.candidates)}",
        flush=True,
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(
            f"GPU={properties.name}, VRAM={properties.total_memory / 2**30:.2f} GiB",
            flush=True,
        )

    cache_runtime: dict[str, TemporalTripFeatures] = {}
    labels: dict[str, TTCSupervision] = {}
    try:
        for trip_id in P1_TRIP_IDS:
            trip_dir = args.data_dir / trip_id
            # Explicit lean-loader use is a protocol boundary: runtime feature
            # code has no accessor for depth, targets, events or TTC labels.
            manifest_path = args.manifest_dir / f"{trip_id}.json.gz"
            if not manifest_path.is_file():
                prepare_c1_manifest(trip_dir, manifest_path)
            lean_loader = C1LeanTripLoader(trip_dir, manifest_path)
            integrity = lean_loader.check_image_2_integrity()
            integrity.require_ok()
            if lean_loader.n_frames != 600:
                raise ValueError(
                    f"P1 expects 600 frames in {trip_id}, got {lean_loader.n_frames}"
                )
            cache_path = locate_detection_cache(
                args.detection_cache_dir,
                trip_id,
                pattern=args.cache_pattern,
            )
            runtime = build_trip_features(
                lean_loader,
                load_detection_cache(cache_path),
                load_camera_geometry(trip_dir),
                confidence_threshold=args.detection_confidence,
            )
            supervision = load_training_ttc_labels(trip_dir)
            labels[trip_id] = supervision
            cache_runtime[trip_id] = runtime
            print(f"Prepared {trip_id}: {runtime.n_frames} causal runtime rows", flush=True)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"C1 temporal data error: {exc}", file=sys.stderr)
        return 3

    # Baseline gate happens before the first optimizer/model is constructed.
    authoritative_physics: dict[str, np.ndarray] = {}
    try:
        for trip_id in P1_TRIP_IDS:
            source = args.baseline_dir / f"{trip_id}.csv"
            authoritative_physics[trip_id] = _read_prediction_csv(
                source, runtime=cache_runtime[trip_id]
            )
            destination = args.output_dir / "physics" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            preflight_prediction_csv(destination, expected_frames=600)
        physics_report = _ablation_report(authoritative_physics, labels)
        physics_score = float(physics_report["macro"]["composite_score"])  # type: ignore[index]
        if abs(physics_score - args.expected_physics_score) > args.physics_score_tolerance:
            raise RuntimeError(
                f"physics baseline gate failed: expected {args.expected_physics_score:.1f}, "
                f"recreated {physics_score:.1f}"
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"C1 temporal baseline gate error: {exc}", file=sys.stderr)
        return 4
    print(f"Physics baseline gate PASS: macro composite={physics_score:.1f}", flush=True)

    # For controlled offline fusion, use the exact frozen physics baseline
    # values rather than the rounded-box cache replay.  These values contain
    # no labels and are fixed before any outer/inner selection.
    supervised: dict[str, SupervisedTrip] = {}
    for trip_id in P1_TRIP_IDS:
        cached = cache_runtime[trip_id]
        runtime = TemporalTripFeatures(
            trip_id=cached.trip_id,
            frame_ids=cached.frame_ids,
            timestamps=cached.timestamps,
            features=cached.features,
            physics_ttc_s=authoritative_physics[trip_id].astype(np.float32),
        )
        supervised[trip_id] = SupervisedTrip(runtime, labels[trip_id])

    fold_summaries: list[dict[str, object]] = []
    for outer_index, (outer, inner_folds) in enumerate(split_plan):
        manifest_path = args.run_dir / "folds" / f"{outer.test_trip_id}.json"
        output_paths = {
            variant: args.output_dir / variant / f"{outer.test_trip_id}.csv"
            for variant in ("temporal", "fusion")
        }
        print(
            f"Outer {outer.test_trip_id}: inner selection on "
            f"{','.join(outer.train_trip_ids)}",
            flush=True,
        )
        candidate_results: list[CandidateResult] = []
        for config in args.candidates:
            result = evaluate_candidate_inner(
                supervised,
                outer,
                inner_folds,
                config,
                settings,
                thresholds=args.thresholds,
                device=device,
                log_root=args.run_dir
                / "tensorboard"
                / f"outer-{outer.test_trip_id}",
                seed=settings.seed + outer_index * 100,
            )
            candidate_results.append(result)
            print(
                f"  {config.tag}: inner temporal={result.temporal_choice.macro_composite:.1f}, "
                f"fusion={result.fusion_choice.macro_composite:.1f}",
                flush=True,
            )
        selected_temporal = select_candidate(
            candidate_results,
            outer_train_trip_ids=outer.train_trip_ids,
            outer_test_trip_id=outer.test_trip_id,
            variant="temporal",
        )
        selected_fusion = select_candidate(
            candidate_results,
            outer_train_trip_ids=outer.train_trip_ids,
            outer_test_trip_id=outer.test_trip_id,
            variant="fusion",
        )
        temporal_epochs = max(
            1,
            int(round(statistics.median(selected_temporal.temporal_best_epochs))),
        )
        fusion_epochs = max(
            1,
            int(round(statistics.median(selected_fusion.fusion_best_epochs))),
        )
        final_train_trips = outer_training_trips(supervised, outer)
        temporal_model, temporal_normalizer = train_final_model(
            final_train_trips,
            selected_temporal.config,
            settings,
            epochs=temporal_epochs,
            device=device,
            log_dir=args.run_dir
            / "tensorboard"
            / f"outer-{outer.test_trip_id}"
            / "temporal"
            / "final",
            seed=settings.seed + outer_index,
            variant="temporal",
        )
        fusion_model, fusion_normalizer = train_final_model(
            final_train_trips,
            selected_fusion.config,
            settings,
            epochs=fusion_epochs,
            device=device,
            log_dir=args.run_dir
            / "tensorboard"
            / f"outer-{outer.test_trip_id}"
            / "fusion"
            / "final",
            seed=settings.seed + 10_000 + outer_index,
            variant="fusion",
        )
        # Each standalone ablation touches the outer test once, after its
        # architecture, epoch count, normalization and threshold are frozen.
        outer_temporal_raw = predict_raw(
            temporal_model,
            temporal_normalizer,
            supervised[outer.test_trip_id].runtime,
            device=device,
        )
        outer_fusion_raw = predict_raw(
            fusion_model,
            fusion_normalizer,
            supervised[outer.test_trip_id].runtime,
            device=device,
        )
        temporal_predictions = calibrated_ttc(
            outer_temporal_raw.temporal_inverse,
            outer_temporal_raw.temporal_danger_logit,
            danger_threshold=selected_temporal.temporal_choice.threshold,
        )
        fusion_predictions = calibrated_ttc(
            outer_fusion_raw.fusion_inverse,
            outer_fusion_raw.fusion_danger_logit,
            danger_threshold=selected_fusion.fusion_choice.threshold,
        )
        _write_prediction_csv(
            output_paths["temporal"],
            supervised[outer.test_trip_id].runtime,
            temporal_predictions,
        )
        _write_prediction_csv(
            output_paths["fusion"],
            supervised[outer.test_trip_id].runtime,
            fusion_predictions,
        )

        onnx_artifacts: dict[str, dict[str, object]] = {}
        for variant, model, normalizer, selected, seed_offset in (
            (
                "temporal",
                temporal_model,
                temporal_normalizer,
                selected_temporal,
                0,
            ),
            ("fusion", fusion_model, fusion_normalizer, selected_fusion, 10_000),
        ):
            payload = checkpoint_payload(
                model,
                normalizer,
                temporal_threshold=selected_temporal.temporal_choice.threshold,
                fusion_threshold=selected_fusion.fusion_choice.threshold,
                train_trip_ids=outer.train_trip_ids,
                outer_test_trip_id=outer.test_trip_id,
                seed=settings.seed + seed_offset + outer_index,
            )
            payload["ablation_variant"] = variant
            _atomic_torch_save(
                args.run_dir
                / "checkpoints"
                / f"{outer.test_trip_id}.{variant}.pt",
                payload,
            )
            onnx_path = export_temporal_onnx(
                model,
                args.run_dir
                / "onnx"
                / f"{outer.test_trip_id}.{variant}.onnx",
            )
            import onnx

            onnx_document = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_document)
            onnx_artifacts[variant] = {
                "path": str(onnx_path),
                "ir_version": int(onnx_document.ir_version),
                "opset": int(onnx_document.opset_import[0].version),
                "input_shape": [
                    int(dimension.dim_value)
                    for dimension in onnx_document.graph.input[0]
                    .type.tensor_type.shape.dim
                ],
            }
        manifest: dict[str, object] = {
            "outer_test_trip_id": outer.test_trip_id,
            "outer_train_trip_ids": list(outer.train_trip_ids),
            "base_seed": settings.seed,
            "selected_temporal_config": selected_temporal.config.to_dict(),
            "selected_fusion_config": selected_fusion.config.to_dict(),
            "temporal_parameter_count": temporal_model.parameter_count,
            "fusion_parameter_count": fusion_model.parameter_count,
            "temporal_final_epochs": temporal_epochs,
            "fusion_final_epochs": fusion_epochs,
            "temporal_threshold": selected_temporal.temporal_choice.threshold,
            "fusion_threshold": selected_fusion.fusion_choice.threshold,
            "onnx": onnx_artifacts,
            "inner_results": [
                {
                    "config": result.config.to_dict(),
                    "temporal": asdict(result.temporal_choice),
                    "fusion": asdict(result.fusion_choice),
                    "temporal_best_epochs": list(result.temporal_best_epochs),
                    "fusion_best_epochs": list(result.fusion_best_epochs),
                    "validation_trip_ids": list(result.validation_trip_ids),
                }
                for result in candidate_results
            ],
        }
        _atomic_json(manifest_path, manifest)
        fold_summaries.append(manifest)
        print(
            f"Outer {outer.test_trip_id}: complete (temporal="
            f"{selected_temporal.config.tag}/{temporal_epochs} epochs, fusion="
            f"{selected_fusion.config.tag}/{fusion_epochs} epochs); outer labels "
            "were not used for selection",
            flush=True,
        )
        del temporal_model, fusion_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions_by_ablation: dict[str, dict[str, np.ndarray]] = {
        "physics": authoritative_physics,
        "temporal": {},
        "fusion": {},
    }
    try:
        for variant in ABLATIONS:
            _require_exact_oof_csv_set(args.output_dir / variant)
        for variant in ("temporal", "fusion"):
            for trip_id in P1_TRIP_IDS:
                predictions_by_ablation[variant][trip_id] = _read_prediction_csv(
                    args.output_dir / variant / f"{trip_id}.csv"
                )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"C1 temporal OOF completeness error: {exc}", file=sys.stderr)
        return 5

    reports = {
        variant: _ablation_report(predictions_by_ablation[variant], labels)
        for variant in ABLATIONS
    }
    report: dict[str, object] = {
        "protocol": {
            "name": "nested_leave_one_trip_out",
            "trips": list(P1_TRIP_IDS),
            "frames_per_trip": 600,
            "outer_folds": 6,
            "inner_folds": args.inner_folds,
            "outer_test_selection": False,
            "normalization": "fit on current training trips only",
            "threshold_calibration": "inner validation only",
            "ablation_training": (
                "independent temporal and fusion model instances; no shared gradients"
            ),
            "fusion_physics_input": (
                "frozen authoritative physics-v2 baseline CSV, frame/timestamp gated"
            ),
            "pseudo_labels_used": False,
            "unlabelled_T01d_T10d_used": False,
            "temporal_runtime_inputs": (
                "image_2 detector bbox/conf/class + tracker association + "
                "timestamp + ego kinematics"
            ),
            "forbidden_runtime_inputs": ["depth", "targets", "events", "TTC labels"],
        },
        "device": {
            "torch": torch.__version__,
            "device": str(device),
            "amp": settings.amp and device.type == "cuda",
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "training": {
            "settings": asdict(settings),
            "candidates": [config.to_dict() for config in args.candidates],
            "danger_threshold_grid": list(args.thresholds),
            "detection_confidence": args.detection_confidence,
            "detection_cache_pattern": args.cache_pattern,
            "expected_physics_score": args.expected_physics_score,
            "physics_score_tolerance": args.physics_score_tolerance,
        },
        "ablations": reports,
        "comparison": {
            "physics_reference": reports["physics"]["macro"]["composite_score"],  # type: ignore[index]
            "temporal_oof": reports["temporal"]["macro"]["composite_score"],  # type: ignore[index]
            "fusion_oof": reports["fusion"]["macro"]["composite_score"],  # type: ignore[index]
            "temporal_delta": round(
                float(reports["temporal"]["macro"]["composite_score"])  # type: ignore[index]
                - float(reports["physics"]["macro"]["composite_score"]),  # type: ignore[index]
                1,
            ),
            "fusion_delta": round(
                float(reports["fusion"]["macro"]["composite_score"])  # type: ignore[index]
                - float(reports["physics"]["macro"]["composite_score"]),  # type: ignore[index]
                1,
            ),
        },
        "physics_cache_parity": _physics_cache_parity(
            authoritative_physics, cache_runtime
        ),
        "folds": fold_summaries,
        "artifacts": {
            "checkpoints": str(args.run_dir / "checkpoints"),
            "tensorboard": str(args.run_dir / "tensorboard"),
            "predictions": str(args.output_dir),
        },
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report["comparison"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
