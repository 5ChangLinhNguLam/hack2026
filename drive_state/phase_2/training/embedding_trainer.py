"""Optimization, calibration, natural metrics, and checkpoints for embedding models."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..data.primitive_labels import IGNORE_INDEX
from ..distraction import action_distraction_probability, pool_distraction_probabilities
from ..losses.embedding_temporal_loss import (
    CabinTemporalLoss,
    FaceMouthTemporalLoss,
)
from ..models.embedding_temporal import CabinTemporalOutput, FaceMouthTemporalOutput
from .eye_trainer import SubjectSplitMetadata


def _runs(values: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(values) - 1):
            end = index if value and index == len(values) - 1 else index - 1
            runs.append((start, end))
            start = None
    return runs


def binary_event_metrics(
    *,
    targets: Sequence[int],
    predictions: Sequence[int],
    timestamps: Sequence[float],
) -> dict[str, float | int]:
    """Greedily match positive runs one-to-one when time intervals overlap."""
    if len(targets) != len(predictions) or len(targets) != len(timestamps):
        raise ValueError("event targets, predictions, and timestamps must align")
    target_events = _runs(targets)
    predicted_events = _runs(predictions)
    used: set[int] = set()
    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for target_event in target_events:
        candidates = [
            (index, event)
            for index, event in enumerate(predicted_events)
            if index not in used
            and max(target_event[0], event[0]) <= min(target_event[1], event[1])
        ]
        if not candidates:
            continue
        chosen_index, chosen_event = max(
            candidates,
            key=lambda item: min(target_event[1], item[1][1])
            - max(target_event[0], item[1][0])
            + 1,
        )
        used.add(chosen_index)
        matches.append((target_event, chosen_event))
    matched = len(matches)
    precision = matched / len(predicted_events) if predicted_events else float(not target_events)
    recall = matched / len(target_events) if target_events else float(not predicted_events)
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    onset_delays = [
        float(timestamps[predicted[0]]) - float(timestamps[target[0]])
        for target, predicted in matches
    ]
    offset_delays = [
        float(timestamps[predicted[1]]) - float(timestamps[target[1]])
        for target, predicted in matches
    ]
    return {
        "target_events": len(target_events),
        "predicted_events": len(predicted_events),
        "matched_events": matched,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": event_f1,
        "mean_onset_delay_seconds": sum(onset_delays) / matched if matched else 0.0,
        "mean_offset_delay_seconds": sum(offset_delays) / matched if matched else 0.0,
    }


def _classification_summary(
    targets: Sequence[int], predictions: Sequence[int], classes: int
) -> dict[str, object]:
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target)][int(prediction)] += 1
    per_class_f1: list[float] = []
    for index in range(classes):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(2 * true_positive / denominator if denominator else 0.0)
    support = len(targets)
    return {
        "support": support,
        "accuracy": sum(confusion[index][index] for index in range(classes)) / support
        if support
        else 0.0,
        "macro_f1": sum(per_class_f1) / classes,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion,
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def calibrate_binary_logits(
    *, logits: Sequence[float], targets: Sequence[int]
) -> dict[str, float]:
    """Fit bounded scalar temperature and macro-F1 threshold deterministically."""
    if len(logits) != len(targets) or not logits:
        raise ValueError("calibration requires aligned non-empty logits and targets")
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("binary calibration targets must be 0 or 1")
    temperatures = np.geomspace(0.5, 3.0, 51)
    temperature = min(
        temperatures,
        key=lambda current: float(
            np.mean(
                np.maximum(values / current, 0.0)
                - labels * (values / current)
                + np.log1p(np.exp(-np.abs(values / current)))
            )
        ),
    )
    probabilities = np.asarray([_sigmoid(float(value / temperature)) for value in values])
    candidates: list[tuple[float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(np.int64).tolist()
        macro_f1 = float(_classification_summary(labels.tolist(), predictions, 2)["macro_f1"])
        candidates.append((macro_f1, float(threshold)))
    _, threshold = max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5), -item[1]))
    return {"temperature": float(temperature), "threshold": threshold}


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def train_embedding_epoch(
    model: nn.Module,
    criterion: CabinTemporalLoss | FaceMouthTemporalLoss,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    specialist: str,
    device: torch.device,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    if specialist not in {"cabin", "face_mouth"}:
        raise ValueError("specialist must be cabin or face_mouth")
    model.train()
    use_amp = amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    totals: dict[str, float] = {"loss": 0.0}
    batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["cabin" if specialist == "cabin" else "face_mouth"])
            breakdown = criterion(output, batch)
        if use_amp:
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        totals["loss"] += float(breakdown.total.detach())
        for name, value in breakdown.components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        batches += 1
    if batches == 0:
        raise ValueError("embedding training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def predict_embedding_model(
    model: nn.Module,
    criterion: CabinTemporalLoss | FaceMouthTemporalLoss,
    loader: Iterable[Mapping[str, object]],
    *,
    specialist: str,
    device: torch.device,
    amp: bool = True,
) -> dict[str, object]:
    if specialist not in {"cabin", "face_mouth"}:
        raise ValueError("specialist must be cabin or face_mouth")
    model.eval()
    use_amp = amp and device.type == "cuda"
    rows: list[dict[str, object]] = []
    losses: list[float] = []
    seen: set[tuple[str, int]] = set()
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch["cabin" if specialist == "cabin" else "face_mouth"])
            breakdown = criterion(output, batch)
        losses.append(float(breakdown.total))
        sessions = list(raw_batch["session"])
        frame_ids = raw_batch["target_frame_id"].tolist()
        timestamps = raw_batch["target_timestamp"].tolist()
        target_indices = raw_batch["target_index"].tolist()
        for index, session in enumerate(sessions):
            identity = (str(session), int(frame_ids[index]))
            if identity in seen:
                raise ValueError(f"natural evaluation reference repeated: {identity}")
            seen.add(identity)
            row: dict[str, object] = {
                "session": str(session),
                "frame_id": int(frame_ids[index]),
                "timestamp": float(timestamps[index]),
                "target_index": int(target_indices[index]),
            }
            if specialist == "cabin":
                assert isinstance(output, CabinTemporalOutput)
                action_logits = output.action_logits[index, -1].float()
                row.update(
                    {
                        "distraction": int(batch["distraction"][index]),
                        "driver_action": int(batch["driver_action"][index]),
                        "road_gaze": int(batch["road_gaze"][index]),
                        "direct_distraction_logit": float(
                            output.distraction_logits[index, -1].float()
                        ),
                        "action_logits": action_logits.cpu().tolist(),
                        "action_distraction_logit": float(
                            torch.logit(
                                action_distraction_probability(action_logits)
                                .float()
                                .clamp(1e-6, 1.0 - 1e-6)
                            )
                        ),
                        "road_gaze_logit": float(output.road_gaze_logits[index, -1].float()),
                    }
                )
            else:
                assert isinstance(output, FaceMouthTemporalOutput)
                type_logits = output.yawn_type_logits[index, -1].float()
                derived = type_logits.softmax(dim=-1)[1:].sum()
                row.update(
                    {
                        "yawn": int(batch["yawn"][index]),
                        "mouth_visibility": float(batch["mouth_visibility"][index, -1]),
                        "binary_yawn_logit": float(output.binary_yawn_logits[index, -1].float()),
                        "yawn_type_logits": type_logits.cpu().tolist(),
                        "type_derived_yawn_logit": float(
                            torch.logit(derived.clamp(1e-6, 1.0 - 1e-6))
                        ),
                    }
                )
            rows.append(row)
    if not losses:
        raise ValueError("embedding evaluation loader produced no batches")
    rows.sort(key=lambda row: (str(row["session"]), int(row["target_index"])))
    return {"loss": sum(losses) / len(losses), "rows": rows}


def _config(
    calibration: Mapping[str, Mapping[str, float]] | None, name: str
) -> tuple[float, float]:
    values = (calibration or {}).get(name, {})
    return float(values.get("temperature", 1.0)), float(values.get("threshold", 0.5))


def _event_summary(rows: Sequence[dict[str, object]], target_name: str, prediction_name: str) -> dict[str, object]:
    by_session: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if int(row[target_name]) == IGNORE_INDEX:
            continue
        by_session.setdefault(str(row["session"]), []).append(row)
    summaries = []
    for session_rows in by_session.values():
        session_rows.sort(key=lambda row: int(row["target_index"]))
        summaries.append(
            binary_event_metrics(
                targets=[int(row[target_name]) for row in session_rows],
                predictions=[int(row[prediction_name]) for row in session_rows],
                timestamps=[float(row["timestamp"]) for row in session_rows],
            )
        )
    target_events = sum(int(value["target_events"]) for value in summaries)
    predicted_events = sum(int(value["predicted_events"]) for value in summaries)
    matched = sum(int(value["matched_events"]) for value in summaries)
    precision = matched / predicted_events if predicted_events else float(target_events == 0)
    recall = matched / target_events if target_events else float(predicted_events == 0)
    return {
        "target_events": target_events,
        "predicted_events": predicted_events,
        "matched_events": matched,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_onset_delay_seconds": (
            sum(float(value["mean_onset_delay_seconds"]) * int(value["matched_events"]) for value in summaries)
            / matched
            if matched
            else 0.0
        ),
        "mean_offset_delay_seconds": (
            sum(float(value["mean_offset_delay_seconds"]) * int(value["matched_events"]) for value in summaries)
            / matched
            if matched
            else 0.0
        ),
    }


def embedding_metrics(
    predictions: Mapping[str, object],
    *,
    specialist: str,
    calibration: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    rows = [dict(row) for row in predictions["rows"]]
    if specialist == "cabin":
        for row in rows:
            direct_temperature, direct_threshold = _config(calibration, "direct_distraction")
            action_temperature, action_threshold = _config(calibration, "action_distraction")
            road_temperature, road_threshold = _config(calibration, "road_gaze")
            fused_temperature, fused_threshold = _config(calibration, "fused_distraction")
            direct_probability = _sigmoid(float(row["direct_distraction_logit"]) / direct_temperature)
            action_probability = _sigmoid(float(row["action_distraction_logit"]) / action_temperature)
            road_probability = _sigmoid(float(row["road_gaze_logit"]) / road_temperature)
            fused_probability = float(
                pool_distraction_probabilities(
                    direct_probability, action_probability, road_probability
                )
            )
            fused_logit = math.log(max(fused_probability, 1e-6) / max(1.0 - fused_probability, 1e-6))
            row["direct_prediction"] = int(direct_probability >= direct_threshold)
            row["action_prediction"] = int(action_probability >= action_threshold)
            row["road_prediction"] = int(road_probability >= road_threshold)
            row["fused_prediction"] = int(_sigmoid(fused_logit / fused_temperature) >= fused_threshold)
            row["action_class_prediction"] = int(np.argmax(row["action_logits"]))
        valid_distraction = [row for row in rows if int(row["distraction"]) != IGNORE_INDEX]
        valid_action = [row for row in rows if int(row["driver_action"]) != IGNORE_INDEX]
        valid_road = [row for row in rows if int(row["road_gaze"]) != IGNORE_INDEX]
        metrics = {
            "loss": float(predictions["loss"]),
            "direct_distraction": _classification_summary(
                [int(row["distraction"]) for row in valid_distraction],
                [int(row["direct_prediction"]) for row in valid_distraction],
                2,
            ),
            "action_distraction": _classification_summary(
                [int(row["distraction"]) for row in valid_distraction],
                [int(row["action_prediction"]) for row in valid_distraction],
                2,
            ),
            "fused_distraction": _classification_summary(
                [int(row["distraction"]) for row in valid_distraction],
                [int(row["fused_prediction"]) for row in valid_distraction],
                2,
            ),
            "road_gaze": _classification_summary(
                [int(row["road_gaze"]) for row in valid_road],
                [int(row["road_prediction"]) for row in valid_road],
                2,
            ),
            "driver_action": _classification_summary(
                [int(row["driver_action"]) for row in valid_action],
                [int(row["action_class_prediction"]) for row in valid_action],
                13,
            ),
            "events": _event_summary(valid_distraction, "distraction", "fused_prediction"),
        }
        metrics["score"] = 0.6 * float(metrics["fused_distraction"]["macro_f1"]) + 0.4 * float(metrics["road_gaze"]["macro_f1"])
        return metrics
    if specialist == "face_mouth":
        for row in rows:
            direct_temperature, direct_threshold = _config(calibration, "binary_yawn")
            derived_temperature, derived_threshold = _config(calibration, "type_derived_yawn")
            row["binary_prediction"] = int(
                _sigmoid(float(row["binary_yawn_logit"]) / direct_temperature) >= direct_threshold
            )
            row["derived_prediction"] = int(
                _sigmoid(float(row["type_derived_yawn_logit"]) / derived_temperature) >= derived_threshold
            )
            row["type_prediction"] = int(np.argmax(row["yawn_type_logits"]))
            row["binary_target"] = int(int(row["yawn"]) > 0)
        valid = [
            row
            for row in rows
            if int(row["yawn"]) != IGNORE_INDEX and float(row["mouth_visibility"]) >= 0.2
        ]
        metrics = {
            "loss": float(predictions["loss"]),
            "binary_yawn": _classification_summary(
                [int(row["binary_target"]) for row in valid],
                [int(row["binary_prediction"]) for row in valid],
                2,
            ),
            "type_derived_yawn": _classification_summary(
                [int(row["binary_target"]) for row in valid],
                [int(row["derived_prediction"]) for row in valid],
                2,
            ),
            "yawn_type": _classification_summary(
                [int(row["yawn"]) for row in valid],
                [int(row["type_prediction"]) for row in valid],
                3,
            ),
            "events": _event_summary(valid, "binary_target", "binary_prediction"),
        }
        metrics["score"] = float(metrics["binary_yawn"]["macro_f1"])
        return metrics
    raise ValueError("specialist must be cabin or face_mouth")


def evaluate_embedding_model(
    model: nn.Module,
    criterion: CabinTemporalLoss | FaceMouthTemporalLoss,
    loader: Iterable[Mapping[str, object]],
    *,
    specialist: str,
    device: torch.device,
    amp: bool = True,
    calibration: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    predictions = predict_embedding_model(
        model, criterion, loader, specialist=specialist, device=device, amp=amp
    )
    return embedding_metrics(predictions, specialist=specialist, calibration=calibration)


def save_embedding_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_kind: str,
    specialist: str,
    input_dim: int,
    channels: int,
    split: SubjectSplitMetadata,
    source_fingerprints: Mapping[str, str],
    epoch: int,
    metrics: Mapping[str, object],
    calibration: Mapping[str, object],
) -> None:
    if model_kind not in {"mlp", "tcn"} or specialist not in {"cabin", "face_mouth"}:
        raise ValueError("invalid embedding checkpoint model/specialist kind")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model_kind": model_kind,
            "specialist": specialist,
            "input_dim": int(input_dim),
            "channels": int(channels),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "split": asdict(split),
            "source_fingerprints": dict(source_fingerprints),
            "epoch": int(epoch),
            "validation_metrics": dict(metrics),
            "calibration": dict(calibration),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_embedding_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported embedding checkpoint schema: {payload.get('schema_version')}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
