"""Event-level and LOSO aggregate metrics for exclusive driver states."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import fmean, pstdev
from typing import Mapping, Sequence

from .data.five_state_labels import FiveState
from .data.primitive_labels import IGNORE_INDEX


def _aligned_lengths(**values: Sequence[object]) -> int:
    lengths = {name: len(value) for name, value in values.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(
            f"{name}={length}" for name, length in lengths.items()
        )
        raise ValueError(f"event metric inputs must align: {detail}")
    return next(iter(lengths.values()), 0)


def five_state_event_metrics(
    *,
    targets: Sequence[int],
    predictions: Sequence[int],
    event_ids: Sequence[str],
    session_ids: Sequence[str],
    frame_ids: Sequence[int],
    fps: float,
) -> dict[str, object]:
    """Measure event detection, causal onset delay, and false alarm runs."""

    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError("event metric FPS must be positive and finite")
    _aligned_lengths(
        targets=targets,
        predictions=predictions,
        event_ids=event_ids,
        session_ids=session_ids,
        frame_ids=frame_ids,
    )
    valid_rows: list[tuple[int, int, str, str, int]] = []
    for target, prediction, event_id, session, frame_id in zip(
        targets,
        predictions,
        event_ids,
        session_ids,
        frame_ids,
        strict=True,
    ):
        target = int(target)
        if target == IGNORE_INDEX:
            continue
        valid_rows.append(
            (
                int(FiveState(target)),
                int(FiveState(int(prediction))),
                str(event_id),
                str(session),
                int(frame_id),
            )
        )

    events: dict[
        tuple[str, str, int],
        list[tuple[int, int]],
    ] = defaultdict(list)
    anonymous_run = 0
    previous_anonymous: tuple[str, int, int] | None = None
    for target, prediction, event_id, session, frame_id in valid_rows:
        resolved_event_id = event_id
        if not resolved_event_id:
            current = (session, target, frame_id)
            if (
                previous_anonymous is None
                or previous_anonymous[0] != session
                or previous_anonymous[1] != target
                or frame_id != previous_anonymous[2] + 1
            ):
                anonymous_run += 1
            resolved_event_id = f"__anonymous_{anonymous_run}"
            previous_anonymous = current
        else:
            previous_anonymous = None
        events[(session, resolved_event_id, target)].append(
            (frame_id, prediction)
        )

    false_events = {state: 0 for state in FiveState}
    active_false = {state: False for state in FiveState}
    previous_session: str | None = None
    previous_frame: int | None = None
    for target, prediction, _, session, frame_id in valid_rows:
        contiguous = (
            previous_session == session
            and previous_frame is not None
            and frame_id == previous_frame + 1
        )
        if not contiguous:
            active_false = {state: False for state in FiveState}
        for state in FiveState:
            is_false = prediction == int(state) and target != int(state)
            if is_false and not active_false[state]:
                false_events[state] += 1
            active_false[state] = is_false
        previous_session = session
        previous_frame = frame_id

    evaluated_minutes = len(valid_rows) / fps / 60.0
    per_class: dict[str, dict[str, float | int | None]] = {}
    for state in FiveState:
        state_events = [
            frames
            for (_, _, target), frames in events.items()
            if target == int(state)
        ]
        delays: list[float] = []
        for frames in state_events:
            ordered = sorted(frames)
            event_start = ordered[0][0]
            hits = [
                frame_id
                for frame_id, prediction in ordered
                if prediction == int(state)
            ]
            if hits:
                delays.append((hits[0] - event_start) / fps)
        event_count = len(state_events)
        detected = len(delays)
        false_count = false_events[state]
        per_class[state.name.lower()] = {
            "events": event_count,
            "detected_events": detected,
            "event_recall": (
                detected / event_count if event_count else None
            ),
            "event_precision": (
                detected / (detected + false_count)
                if event_count and detected + false_count
                else None
            ),
            "mean_onset_delay_seconds": (
                fmean(delays) if delays else None
            ),
            "median_onset_delay_seconds": (
                _median(delays) if delays else None
            ),
            "false_events": false_count,
            "false_events_per_minute": (
                false_count / evaluated_minutes
                if evaluated_minutes
                else 0.0
            ),
        }
    return {
        "evaluated_frames": len(valid_rows),
        "evaluated_minutes": evaluated_minutes,
        "per_class": per_class,
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _metrics_from_confusion(
    confusion: Sequence[Sequence[int]],
) -> dict[str, object]:
    classes = len(FiveState)
    if len(confusion) != classes or any(
        len(row) != classes for row in confusion
    ):
        raise ValueError("LOSO confusion matrices must be 5x5")
    matrix = [
        [int(value) for value in row]
        for row in confusion
    ]
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for state in FiveState:
        index = int(state)
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        support = sum(matrix[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[state.name.lower()] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    support = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(classes))
    return {
        "support": support,
        "accuracy": correct / support if support else 0.0,
        "macro_f1": fmean(f1_values),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _aggregate_loso_events(
    fold_metrics: Mapping[str, Mapping[str, object]],
    subjects: Sequence[str],
) -> dict[str, object] | None:
    blocks = [fold_metrics[subject].get("events") for subject in subjects]
    if all(block is None for block in blocks):
        return None
    if any(not isinstance(block, Mapping) for block in blocks):
        raise ValueError(
            "LOSO event aggregation requires event metrics for every fold"
        )

    evaluated_frames = 0
    evaluated_minutes = 0.0
    for subject, block in zip(subjects, blocks, strict=True):
        assert isinstance(block, Mapping)
        try:
            evaluated_frames += int(block["evaluated_frames"])
            evaluated_minutes += float(block["evaluated_minutes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"fold {subject} has invalid event evaluation totals"
            ) from error

    per_class: dict[str, dict[str, float | int | None]] = {}
    for state in FiveState:
        state_name = state.name.lower()
        event_count = 0
        detected = 0
        false_count = 0
        delay_sum = 0.0
        delay_count = 0
        for subject, block in zip(subjects, blocks, strict=True):
            assert isinstance(block, Mapping)
            fold_per_class = block.get("per_class")
            if not isinstance(fold_per_class, Mapping):
                raise ValueError(
                    f"fold {subject} is missing per-class event metrics"
                )
            state_metrics = fold_per_class.get(state_name)
            if not isinstance(state_metrics, Mapping):
                raise ValueError(
                    f"fold {subject} is missing {state_name} event metrics"
                )
            try:
                fold_events = int(state_metrics["events"])
                fold_detected = int(state_metrics["detected_events"])
                fold_false = int(state_metrics["false_events"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"fold {subject} has invalid {state_name} event counts"
                ) from error
            if (
                fold_events < 0
                or fold_detected < 0
                or fold_detected > fold_events
                or fold_false < 0
            ):
                raise ValueError(
                    f"fold {subject} has invalid {state_name} event counts"
                )
            mean_delay = state_metrics.get("mean_onset_delay_seconds")
            if fold_detected:
                if mean_delay is None:
                    raise ValueError(
                        f"fold {subject} is missing {state_name} onset delay"
                    )
                try:
                    delay = float(mean_delay)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"fold {subject} has invalid {state_name} onset delay"
                    ) from error
                if not math.isfinite(delay):
                    raise ValueError(
                        f"fold {subject} has invalid {state_name} onset delay"
                    )
                delay_sum += delay * fold_detected
                delay_count += fold_detected
            event_count += fold_events
            detected += fold_detected
            false_count += fold_false

        per_class[state_name] = {
            "events": event_count,
            "detected_events": detected,
            "event_recall": (
                detected / event_count if event_count else None
            ),
            "event_precision": (
                detected / (detected + false_count)
                if event_count and detected + false_count
                else None
            ),
            "mean_onset_delay_seconds": (
                delay_sum / delay_count if delay_count else None
            ),
            # Fold medians cannot be pooled without the individual delays.
            "median_onset_delay_seconds": None,
            "false_events": false_count,
            "false_events_per_minute": (
                false_count / evaluated_minutes
                if evaluated_minutes
                else 0.0
            ),
        }
    return {
        "evaluated_frames": evaluated_frames,
        "evaluated_minutes": evaluated_minutes,
        "per_class": per_class,
    }


def aggregate_loso_metrics(
    fold_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate subject-disjoint outer folds without hiding fold variance."""

    if not fold_metrics:
        raise ValueError("LOSO aggregation requires at least one fold")
    subjects = sorted(str(subject) for subject in fold_metrics)
    fields = ("accuracy", "macro_f1", "supported_macro_f1")
    distribution: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [
            float(fold_metrics[subject][field])
            for subject in subjects
        ]
        distribution[field] = {
            "mean": fmean(values),
            "std": pstdev(values),
            "min": min(values),
            "max": max(values),
        }

    classes = len(FiveState)
    pooled_confusion = [
        [0 for _ in range(classes)]
        for _ in range(classes)
    ]
    for subject in subjects:
        confusion = fold_metrics[subject]["confusion_matrix"]
        if not isinstance(confusion, Sequence):
            raise ValueError(
                f"fold {subject} is missing its confusion matrix"
            )
        if len(confusion) != classes:
            raise ValueError(f"fold {subject} confusion matrix is not 5x5")
        for row_index, row in enumerate(confusion):
            if not isinstance(row, Sequence) or len(row) != classes:
                raise ValueError(
                    f"fold {subject} confusion matrix is not 5x5"
                )
            for column_index, value in enumerate(row):
                pooled_confusion[row_index][column_index] += int(value)
    summary = {
        "folds": len(subjects),
        "fold_subjects": subjects,
        "distribution": distribution,
        "pooled": _metrics_from_confusion(pooled_confusion),
    }
    event_summary = _aggregate_loso_events(fold_metrics, subjects)
    if event_summary is not None:
        summary["events"] = event_summary
    return summary


__all__ = [
    "aggregate_loso_metrics",
    "five_state_event_metrics",
]
