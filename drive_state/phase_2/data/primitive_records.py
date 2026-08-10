"""Frame records for protocol-masked DMD spatial primitive training."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .manifest import SessionRecord
from .primitive_labels import IGNORE_INDEX, encode_primitive_row


@dataclass(frozen=True)
class PrimitiveFrameRecord:
    session_name: str
    subject_id: str
    protocol: str
    frame_id: int
    src_frame_id: int
    timestamp: float
    frame_path: Path
    targets: Mapping[str, int]


def load_primitive_records(
    sessions: Iterable[SessionRecord],
) -> tuple[PrimitiveFrameRecord, ...]:
    records: list[PrimitiveFrameRecord] = []
    for session in sessions:
        dataset_root = session.labels_csv.parent.parent
        session_count = 0
        with session.labels_csv.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                records.append(
                    PrimitiveFrameRecord(
                        session_name=session.name,
                        subject_id=session.subject_id,
                        protocol=session.protocol,
                        frame_id=int(row["frame_id"]),
                        src_frame_id=int(row["src_frame_id"]),
                        timestamp=float(row["timestamp"]),
                        frame_path=dataset_root / row["frame_path"],
                        targets=encode_primitive_row(session.protocol, row),
                    )
                )
                session_count += 1
        if session_count != session.n_frames:
            raise ValueError(
                f"{session.name}: manifest has {session.n_frames} frames, CSV has {session_count}"
            )
    return tuple(records)


def _protocol_group(protocol: str) -> str:
    if protocol in {"s1", "s2", "s3"}:
        return "action"
    if protocol == "s5":
        return "eye"
    if protocol == "s6":
        return "gaze"
    raise ValueError(f"unsupported DMD protocol: {protocol}")


def _event_multiplier(record: PrimitiveFrameRecord, boost: float) -> float:
    targets = record.targets
    rare_event = any(
        (
            targets["distraction"] == 1,
            targets["road_gaze"] == 1,
            targets["yawn"] > 0,
            targets["blink"] == 1,
            targets["moving_hands"] == 1,
        )
    )
    return boost if rare_event else 1.0


def build_protocol_sample_weights(
    records: Sequence[PrimitiveFrameRecord], *, event_boost: float = 3.0
) -> tuple[float, ...]:
    """Balance action/s5/s6 protocol mass, then boost rare events within each."""
    if event_boost < 1.0:
        raise ValueError("event_boost must be at least 1")
    raw = [_event_multiplier(record, event_boost) for record in records]
    group_totals = {"action": 0.0, "eye": 0.0, "gaze": 0.0}
    for record, weight in zip(records, raw, strict=True):
        group_totals[_protocol_group(record.protocol)] += weight
    missing = [group for group, total in group_totals.items() if total == 0.0]
    if missing:
        raise ValueError(f"cannot balance missing protocol groups: {missing}")
    return tuple(
        weight / group_totals[_protocol_group(record.protocol)]
        for record, weight in zip(records, raw, strict=True)
    )


def _primary_sampling_stratum(record: PrimitiveFrameRecord) -> tuple[str, int]:
    """Choose one stable, protocol-native label for sampling contiguous events."""
    targets = record.targets
    group = _protocol_group(record.protocol)
    if group == "action":
        if targets["driver_action"] != IGNORE_INDEX:
            return "driver_action", targets["driver_action"]
        return "road_gaze", targets["road_gaze"]
    if group == "eye":
        return "yawn", targets["yawn"]
    if targets["gaze_zone"] != IGNORE_INDEX:
        return "gaze_zone", targets["gaze_zone"]
    return "moving_hands", targets["moving_hands"]


def _contiguous_event_lengths(
    records: Sequence[PrimitiveFrameRecord],
) -> tuple[int, ...]:
    lengths = [1] * len(records)
    by_session: dict[str, list[tuple[int, PrimitiveFrameRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_session[record.session_name].append((index, record))
    for indexed_records in by_session.values():
        indexed_records.sort(key=lambda item: item[1].frame_id)
        start = 0
        while start < len(indexed_records):
            end = start + 1
            previous = indexed_records[start][1]
            stratum = _primary_sampling_stratum(previous)
            while end < len(indexed_records):
                current = indexed_records[end][1]
                if (
                    current.frame_id != previous.frame_id + 1
                    or _primary_sampling_stratum(current) != stratum
                ):
                    break
                previous = current
                end += 1
            event_length = end - start
            for index, _ in indexed_records[start:end]:
                lengths[index] = event_length
            start = end
    return tuple(lengths)


def build_balanced_sample_weights(
    records: Sequence[PrimitiveFrameRecord],
    *,
    max_class_boost: float = 5.0,
    event_power: float = 0.5,
) -> tuple[float, ...]:
    """Balance protocol mass while limiting duplicated-frame dominance.

    Square-root inverse class frequency is deliberately milder than fully
    uniform class sampling.  A second square-root correction reduces each
    frame's weight inside long contiguous events without allowing one-frame
    annotation noise to dominate.  Validation and test loaders must not use
    these weights.
    """
    if not records:
        raise ValueError("cannot build sampling weights for no records")
    if max_class_boost < 1.0:
        raise ValueError("max_class_boost must be at least 1")
    if not 0.0 <= event_power <= 1.0:
        raise ValueError("event_power must be between 0 and 1")

    strata = tuple(_primary_sampling_stratum(record) for record in records)
    counts_by_group: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
    for record, stratum in zip(records, strata, strict=True):
        counts_by_group[_protocol_group(record.protocol)][stratum] += 1
    largest_class = {
        group: max(counts.values()) for group, counts in counts_by_group.items()
    }
    event_lengths = _contiguous_event_lengths(records)
    raw: list[float] = []
    for record, stratum, event_length in zip(
        records, strata, event_lengths, strict=True
    ):
        group = _protocol_group(record.protocol)
        class_count = counts_by_group[group][stratum]
        class_boost = min(
            math.sqrt(largest_class[group] / class_count), max_class_boost
        )
        raw.append(class_boost / (event_length**event_power))

    group_totals = {"action": 0.0, "eye": 0.0, "gaze": 0.0}
    for record, weight in zip(records, raw, strict=True):
        group_totals[_protocol_group(record.protocol)] += weight
    missing = [group for group, total in group_totals.items() if total == 0.0]
    if missing:
        raise ValueError(f"cannot balance missing protocol groups: {missing}")
    return tuple(
        weight / group_totals[_protocol_group(record.protocol)]
        for record, weight in zip(records, raw, strict=True)
    )
