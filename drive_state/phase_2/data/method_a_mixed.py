"""Coverage-first sampling for Method A mixed primitive training."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator, Mapping, Sequence

from torch.utils.data import Sampler

from .evidence_regions import EvidenceRegionStore
from .five_state_labels import IGNORE_INDEX, FiveState
from .five_state_visual import (
    EYE_PHASE_CLASSES,
    FiveStateVisualRecord,
)
from .nitymed import (
    NitymedFrameRecord,
    NitymedTeacherStore,
)


class CoverageFirstBatchSampler(Sampler[list[int]]):
    """Emit every eligible pool occurrence once in deterministic round-robin."""

    def __init__(
        self,
        *,
        pools: Mapping[str, Sequence[int]],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("coverage batch size must be positive")
        normalized = {
            str(name): tuple(int(value) for value in values)
            for name, values in pools.items()
            if values
        }
        if not normalized:
            raise ValueError("coverage sampling requires a nonempty pool")
        if any(value < 0 for values in normalized.values() for value in values):
            raise ValueError("coverage indices must be non-negative")
        self.pools = normalized
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.occurrences = sum(len(values) for values in normalized.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        queues: dict[str, list[int]] = {}
        for name, values in sorted(self.pools.items()):
            queue = list(values)
            rng.shuffle(queue)
            queues[name] = queue
        names = list(queues)
        offsets = {name: 0 for name in names}
        batch: list[int] = []
        while names:
            next_names: list[str] = []
            for name in names:
                offset = offsets[name]
                queue = queues[name]
                if offset < len(queue):
                    batch.append(queue[offset])
                    offsets[name] = offset + 1
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                if offsets[name] < len(queue):
                    next_names.append(name)
            names = next_names
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.occurrences / self.batch_size)


def _thin_run(run: Sequence[int], stride: int, *, keep_all: bool) -> tuple[int, ...]:
    if keep_all or len(run) <= 2:
        return tuple(run)
    selected = {run[0], run[-1]}
    selected.update(run[offset] for offset in range(0, len(run), stride))
    return tuple(index for index in run if index in selected)


def _group_runs(
    records: Sequence[FiveStateVisualRecord],
    values: Mapping[int, int],
) -> list[tuple[int, list[int]]]:
    runs: list[tuple[int, list[int]]] = []
    active_value: int | None = None
    active_session = ""
    previous_frame: int | None = None
    active: list[int] = []
    for index, record in enumerate(records):
        value = values.get(index)
        contiguous = (
            value is not None
            and value == active_value
            and record.session == active_session
            and previous_frame is not None
            and record.frame_id == previous_frame + 1
        )
        if not contiguous:
            if active_value is not None and active:
                runs.append((active_value, active))
            active = []
            active_value = value
            active_session = record.session if value is not None else ""
        if value is not None:
            active.append(index)
            previous_frame = record.frame_id
        else:
            previous_frame = None
    if active_value is not None and active:
        runs.append((active_value, active))
    return runs


def build_dmd_method_a_pools(
    records: Sequence[FiveStateVisualRecord],
    *,
    region_stores: Mapping[str, EvidenceRegionStore],
    target_fps: float,
    decorrelated_fps: float = 5.0,
) -> Mapping[str, tuple[int, ...]]:
    """Build primitive pools without balancing or labeling by dataset source."""

    if target_fps <= 0.0 or decorrelated_fps <= 0.0:
        raise ValueError("DMD coverage FPS values must be positive")
    if decorrelated_fps > target_fps:
        raise ValueError("DMD decorrelated FPS cannot exceed target FPS")
    stride = max(1, round(target_fps / decorrelated_fps))
    phase_names = {
        EYE_PHASE_CLASSES["open"]: "open",
        EYE_PHASE_CLASSES["closing"]: "closing",
        EYE_PHASE_CLASSES["close"]: "closed",
        EYE_PHASE_CLASSES["opening"]: "opening",
    }
    pools: dict[str, list[int]] = defaultdict(list)
    phases = {
        index: int(record.eye_phase_target)
        for index, record in enumerate(records)
        if record.eye_phase_target != IGNORE_INDEX
    }
    for phase, run in _group_runs(records, phases):
        pools[f"dmd_eye_{phase_names[phase]}"].extend(
            _thin_run(
                run,
                stride,
                keep_all=phase
                in {
                    EYE_PHASE_CLASSES["closing"],
                    EYE_PHASE_CLASSES["opening"],
                },
            )
        )

    yawn_values = {
        index: int(record.target == FiveState.YAWNING)
        for index, record in enumerate(records)
        if record.protocol == "s5" and record.target != IGNORE_INDEX
    }
    for value, run in _group_runs(records, yawn_values):
        pools[f"dmd_yawn_{'yes' if value else 'no'}"].extend(
            _thin_run(run, stride, keep_all=False)
        )

    distraction_values = {
        index: int(record.target == FiveState.DISTRACTION)
        for index, record in enumerate(records)
        if record.protocol in {"s1", "s2", "s3"}
        and record.target != IGNORE_INDEX
    }
    for value, run in _group_runs(records, distraction_values):
        pools[f"dmd_distraction_{'yes' if value else 'no'}"].extend(
            _thin_run(run, stride, keep_all=False)
        )

    pose_values: dict[int, int] = {}
    for index, record in enumerate(records):
        store = region_stores.get(record.session)
        if store is not None and store.get(record.frame_id).visibility[0] > 0.0:
            pose_values[index] = 1
    for _, run in _group_runs(records, pose_values):
        pools["dmd_pose_visibility"].extend(
            _thin_run(run, stride, keep_all=False)
        )
    return {
        name: tuple(values)
        for name, values in sorted(pools.items())
        if values
    }


def build_nitymed_method_a_pools(
    records: Sequence[NitymedFrameRecord],
    *,
    teacher_stores: Mapping[str, NitymedTeacherStore],
    dmd_offset: int,
) -> Mapping[str, tuple[int, ...]]:
    if dmd_offset < 0:
        raise ValueError("DMD offset cannot be negative")
    pools: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        try:
            teacher = teacher_stores[record.video_id].get(record.frame_id)
        except KeyError as error:
            raise ValueError(
                f"teacher cache is missing {record.video_id} frame "
                f"{record.frame_id}"
            ) from error
        global_index = dmd_offset + index
        if teacher.eye_mask:
            name = "open" if teacher.eye_target == 0 else "closed"
            pools[f"nity_eye_{name}"].append(global_index)
        if teacher.yawn_mask:
            name = "yes" if teacher.yawn_target == 1 else "no"
            pools[f"nity_yawn_{name}"].append(global_index)
        if teacher.region_visibility[0] > 0.0:
            pools["nity_consistency"].append(global_index)
    return {
        name: tuple(values)
        for name, values in sorted(pools.items())
        if values
    }


def select_method_a_adaptation_pools(
    pools: Mapping[str, Sequence[int]],
    *,
    adaptation: str,
) -> Mapping[str, tuple[int, ...]]:
    if adaptation == "all":
        selected = pools
    elif adaptation == "eye-yawn":
        prefixes = ("dmd_eye_", "dmd_yawn_", "nity_yawn_")
        selected = {
            name: values
            for name, values in pools.items()
            if name.startswith(prefixes)
        }
    elif adaptation == "yawn":
        prefixes = ("dmd_yawn_", "nity_yawn_")
        selected = {
            name: values
            for name, values in pools.items()
            if name.startswith(prefixes)
        }
    else:
        raise ValueError(f"unknown Method A adaptation: {adaptation}")
    return {
        name: tuple(values)
        for name, values in sorted(selected.items())
        if values
    }


__all__ = [
    "CoverageFirstBatchSampler",
    "build_dmd_method_a_pools",
    "build_nitymed_method_a_pools",
    "select_method_a_adaptation_pools",
]
