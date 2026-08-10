"""Dense s5 mouth-crop dataset and event-balanced sampling weights."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .mouth_cache import MouthCropStore
from .primitive_labels import IGNORE_INDEX
from .primitive_records import PrimitiveFrameRecord


class MouthFrameDataset(Dataset[dict[str, Tensor | str | int]]):
    """One cached 96x64 mouth crop per protocol-s5 frame."""

    def __init__(
        self,
        records: Sequence[PrimitiveFrameRecord],
        *,
        cache_dir: Path | str,
    ) -> None:
        self.records = tuple(records)
        invalid = sorted({record.protocol for record in self.records if record.protocol != "s5"})
        if invalid:
            raise ValueError(f"mouth training accepts only s5 records, got {invalid}")
        cache_dir = Path(cache_dir)
        self.stores: dict[str, MouthCropStore] = {}
        self.rows: dict[tuple[str, int], int] = {}
        by_session: dict[str, list[int]] = defaultdict(list)
        for record in self.records:
            by_session[record.session_name].append(record.frame_id)
        for session_name, frame_ids in by_session.items():
            store = MouthCropStore(cache_dir, session_name)
            rows = store.rows_for(frame_ids)
            self.stores[session_name] = store
            self.rows.update(
                ((session_name, frame_id), int(row))
                for frame_id, row in zip(frame_ids, rows, strict=True)
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        record = self.records[index]
        store = self.stores[record.session_name]
        row = self.rows[(record.session_name, record.frame_id)]
        array = np.asarray(store.mouths[row], dtype=np.float32).copy()
        mouth = torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)
        return {
            "mouth": mouth,
            "mouth_visibility": torch.tensor(
                float(store.visibility[row]), dtype=torch.float32
            ),
            "yawn": torch.tensor(record.targets.get("yawn", IGNORE_INDEX), dtype=torch.long),
            "session": record.session_name,
            "subject": record.subject_id,
            "frame_id": record.frame_id,
        }


def build_yawn_event_sample_weights(
    records: Sequence[PrimitiveFrameRecord],
) -> tuple[float, ...]:
    """Give equal mass to positive events, and split total mass 50/50.

    Each contiguous positive run is sampled uniformly as an event and then
    uniformly within that event. Annotated no-yawn frames share the other half
    of the probability mass. Unlabelled rows receive zero mass.
    """
    if not records:
        raise ValueError("cannot sample an empty mouth dataset")
    negative = [
        index for index, record in enumerate(records) if record.targets.get("yawn") == 0
    ]
    indexed_by_session: dict[str, list[tuple[int, PrimitiveFrameRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.protocol != "s5":
            raise ValueError("yawn event sampling accepts only s5 records")
        indexed_by_session[record.session_name].append((index, record))

    events: list[list[int]] = []
    for indexed in indexed_by_session.values():
        indexed.sort(key=lambda item: item[1].frame_id)
        current: list[int] = []
        previous_frame: int | None = None
        for index, record in indexed:
            positive = record.targets.get("yawn", IGNORE_INDEX) > 0
            if positive and (previous_frame is None or record.frame_id == previous_frame + 1):
                current.append(index)
            elif positive:
                if current:
                    events.append(current)
                current = [index]
            else:
                if current:
                    events.append(current)
                    current = []
            previous_frame = record.frame_id
        if current:
            events.append(current)

    if not events or not negative:
        raise ValueError("event-balanced yawn sampling needs positive events and negatives")
    weights = [0.0] * len(records)
    negative_weight = 0.5 / len(negative)
    for index in negative:
        weights[index] = negative_weight
    event_weight = 0.5 / len(events)
    for event in events:
        frame_weight = event_weight / len(event)
        for index in event:
            weights[index] = frame_weight
    return tuple(weights)
