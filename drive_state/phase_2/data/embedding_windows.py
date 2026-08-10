"""Causal embedding windows and event-aware target timestamp samplers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .five_state_labels import FiveState, WeakStateTargets
from .primitive_labels import (
    DISTRACTING_ACTIONS,
    DRIVER_ACTIONS,
    IGNORE_INDEX,
)


@dataclass(frozen=True)
class EmbeddingTargetRef:
    session_index: int
    target_index: int


class EmbeddingWindowDataset(Dataset[dict[str, Tensor | str | int | float]]):
    """Return fixed-length history ending at one referenced real timestamp."""

    def __init__(
        self,
        stores: Sequence[object],
        references: Sequence[EmbeddingTargetRef],
        *,
        length: int = 20,
        five_state_targets: Mapping[str, WeakStateTargets] | None = None,
    ) -> None:
        if length <= 0:
            raise ValueError("window length must be positive")
        self.stores = tuple(stores)
        self.references = tuple(references)
        self.length = length
        self.five_state_targets = (
            None if five_state_targets is None else dict(five_state_targets)
        )
        for reference in self.references:
            if not 0 <= reference.session_index < len(self.stores):
                raise IndexError("embedding reference has invalid session index")
            if not 0 <= reference.target_index < len(self.stores[reference.session_index].frame_ids):
                raise IndexError("embedding reference has invalid target index")
        if self.five_state_targets is not None:
            for store in self.stores:
                session_name = getattr(store, "session_name", "")
                if session_name not in self.five_state_targets:
                    raise ValueError(f"missing five-state targets for {session_name}")
                if len(self.five_state_targets[session_name].targets) != len(store.frame_ids):
                    raise ValueError(f"five-state targets do not align with {session_name}")

    def __len__(self) -> int:
        return len(self.references)

    @staticmethod
    def _padded(array: np.ndarray, start: int, stop: int, pad: int) -> Tensor:
        values = torch.from_numpy(np.asarray(array[start:stop], dtype=np.float32).copy())
        if pad:
            shape = (pad, *values.shape[1:])
            values = torch.cat((torch.zeros(shape, dtype=torch.float32), values), dim=0)
        return values

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int | float]:
        reference = self.references[index]
        store = self.stores[reference.session_index]
        target = reference.target_index
        start = max(0, target - self.length + 1)
        stop = target + 1
        pad = self.length - (stop - start)
        cabin = self._padded(store.cabin, start, stop, pad)
        face = self._padded(store.face, start, stop, pad)
        mouth = self._padded(store.mouth, start, stop, pad)
        face_visibility = self._padded(store.face_visibility, start, stop, pad)
        mouth_visibility = self._padded(store.mouth_visibility, start, stop, pad)
        head_pose = self._padded(store.head_pose, start, stop, pad)
        normalized_pose = head_pose.clamp(-90.0, 90.0).div(90.0)
        face_mouth = torch.cat(
            (
                face,
                mouth,
                face_visibility.unsqueeze(-1),
                mouth_visibility.unsqueeze(-1),
                normalized_pose,
            ),
            dim=-1,
        )
        sample: dict[str, Tensor | str | int | float] = {
            "cabin": cabin,
            "face": face,
            "mouth": mouth,
            "face_mouth": face_mouth,
            "face_visibility": face_visibility,
            "mouth_visibility": mouth_visibility,
            "head_pose": head_pose,
            "context_valid": torch.tensor(
                [False] * pad + [True] * (stop - start), dtype=torch.bool
            ),
            "session": getattr(store, "session_name", ""),
            "subject": store.subject_id,
            "protocol": store.protocol,
            "target_frame_id": int(store.frame_ids[target]),
            "target_timestamp": float(store.timestamps[target]),
            "target_index": target,
            "session_index": reference.session_index,
        }
        if hasattr(store, "eye") and getattr(store, "eye") is not None:
            eye = self._padded(store.eye, start, stop, pad)
            eye_visibility = self._padded(store.eye_visibility, start, stop, pad)
            eye_evidence = self._padded(store.eye_evidence, start, stop, pad)
            fused = torch.cat(
                (
                    cabin,
                    face,
                    mouth,
                    eye,
                    face_visibility.unsqueeze(-1),
                    mouth_visibility.unsqueeze(-1),
                    eye_visibility,
                    normalized_pose,
                    eye_evidence,
                ),
                dim=-1,
            )
            sample.update(
                {
                    "eye": eye,
                    "eye_visibility": eye_visibility,
                    "eye_evidence": eye_evidence,
                    "fused": fused,
                }
            )
        if self.five_state_targets is not None:
            state_targets = self.five_state_targets[getattr(store, "session_name", "")]
            sample.update(
                {
                    "five_state_target": torch.tensor(
                        int(state_targets.targets[target]), dtype=torch.long
                    ),
                    "five_state_confidence": torch.tensor(
                        state_targets.confidence[target], dtype=torch.float32
                    ),
                    "five_state_reason": state_targets.reasons[target],
                }
            )
        for column, task in enumerate(store.task_names):
            sample[task] = torch.tensor(int(store.targets[target, column]), dtype=torch.long)
        return sample


def chronological_target_index(stores: Sequence[object]) -> tuple[EmbeddingTargetRef, ...]:
    return tuple(
        EmbeddingTargetRef(session_index, target_index)
        for session_index, store in enumerate(stores)
        for target_index in range(len(store.frame_ids))
    )


def build_five_state_target_index(
    stores: Sequence[object],
    targets_by_session: Mapping[str, WeakStateTargets],
    *,
    samples: int,
    seed: int,
) -> tuple[EmbeddingTargetRef, ...]:
    """Sample state events before timestamps from fold-training subjects only."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    session_indices = _training_session_indices(stores)
    state_events: dict[FiveState, list[list[EmbeddingTargetRef]]] = {
        state: [] for state in FiveState
    }
    for session_index in session_indices:
        store = stores[session_index]
        session_name = getattr(store, "session_name", "")
        if session_name not in targets_by_session:
            raise ValueError(f"missing five-state targets for {session_name}")
        targets = targets_by_session[session_name].targets
        if len(targets) != len(store.frame_ids):
            raise ValueError(f"five-state targets do not align with {session_name}")
        start = 0
        while start < len(targets):
            target = targets[start]
            end = start + 1
            while end < len(targets) and targets[end] == target:
                end += 1
            if target != IGNORE_INDEX:
                state = FiveState(int(target))
                state_events[state].append(
                    [
                        EmbeddingTargetRef(session_index, target_index)
                        for target_index in range(start, end)
                    ]
                )
            start = end

    ordered_states = (
        FiveState.MICROSLEEP,
        FiveState.DROWSY,
        FiveState.YAWNING,
        FiveState.DISTRACTION,
        FiveState.ALERT,
    )
    requested = (0.25, 0.25, 0.20, 0.20, 0.10)
    present = [
        (state, proportion)
        for state, proportion in zip(ordered_states, requested, strict=True)
        if state_events[state]
    ]
    if not present:
        raise ValueError("no usable five-state training events are available")
    total = sum(proportion for _, proportion in present)
    proportions = tuple(proportion / total for _, proportion in present)
    rng = random.Random(seed)
    references: list[EmbeddingTargetRef] = []
    for category in _category_schedule(samples, proportions, rng):
        state = present[category][0]
        references.append(rng.choice(rng.choice(state_events[state])))
    return tuple(references)


def _column(store: object, task: str) -> int | None:
    try:
        return tuple(store.task_names).index(task)
    except ValueError:
        return None


def _value(store: object, task: str, index: int) -> int:
    column = _column(store, task)
    return IGNORE_INDEX if column is None else int(store.targets[index, column])


def _training_session_indices(stores: Sequence[object]) -> tuple[int, ...]:
    if not stores:
        raise ValueError("cannot sample without embedding stores")
    split = stores[0].split
    if any(store.split != split for store in stores):
        raise ValueError("embedding stores have inconsistent split metadata")
    allowed = set(split.train_subjects)
    selected = tuple(
        index for index, store in enumerate(stores) if store.subject_id in allowed
    )
    if not selected:
        raise ValueError("no fold-training embedding stores are available")
    return selected


def _events(
    stores: Sequence[object],
    session_indices: Sequence[int],
    predicate,
    *,
    include_boundaries: bool = True,
) -> list[list[EmbeddingTargetRef]]:
    events: list[list[EmbeddingTargetRef]] = []
    for session_index in session_indices:
        store = stores[session_index]
        current: list[int] = []
        for target_index in range(len(store.frame_ids)):
            if predicate(store, target_index):
                current.append(target_index)
            elif current:
                indices = current
                if include_boundaries:
                    indices = list(
                        range(max(0, current[0] - 1), min(len(store.frame_ids), current[-1] + 2))
                    )
                events.append(
                    [EmbeddingTargetRef(session_index, value) for value in indices]
                )
                current = []
        if current:
            indices = current
            if include_boundaries:
                indices = list(range(max(0, current[0] - 1), len(store.frame_ids)))
            events.append([EmbeddingTargetRef(session_index, value) for value in indices])
    return events


def _draw_event(
    rng: random.Random,
    events: Sequence[Sequence[EmbeddingTargetRef]],
    fallback: Sequence[EmbeddingTargetRef],
) -> EmbeddingTargetRef:
    if not events:
        return rng.choice(fallback)
    return rng.choice(rng.choice(events))


def _category_schedule(samples: int, proportions: Sequence[float], rng: random.Random) -> list[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    counts = [int(samples * value) for value in proportions[:-1]]
    counts.append(samples - sum(counts))
    categories = [category for category, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(categories)
    return categories


def build_cabin_target_index(
    stores: Sequence[object], *, samples: int, seed: int
) -> tuple[EmbeddingTargetRef, ...]:
    """Draw the 40/20/20/20 cabin event mixture from training subjects only."""
    session_indices = _training_session_indices(stores)
    uniform = [
        EmbeddingTargetRef(session_index, target_index)
        for session_index in session_indices
        for target_index in range(len(stores[session_index].frame_ids))
    ]
    distracting_indices = {
        index for index, name in enumerate(DRIVER_ACTIONS) if name in DISTRACTING_ACTIONS
    }
    distraction_events = _events(
        stores,
        session_indices,
        lambda store, index: _value(store, "distraction", index) == 1
        or _value(store, "driver_action", index) in distracting_indices,
    )
    road_events = _events(
        stores,
        session_indices,
        lambda store, index: _value(store, "road_gaze", index) == 1,
    )
    hard_action_indices = {
        DRIVER_ACTIONS.index("change_gear"),
        DRIVER_ACTIONS.index("standstill_or_waiting"),
    }
    hard_negatives = [
        EmbeddingTargetRef(session_index, index)
        for session_index in session_indices
        for index in range(len(stores[session_index].frame_ids))
        if _value(stores[session_index], "distraction", index) == 0
        and (
            _value(stores[session_index], "driver_action", index) in hard_action_indices
            or _value(stores[session_index], "moving_hands", index) == 1
            or _value(stores[session_index], "talking", index) == 1
        )
    ]
    rng = random.Random(seed)
    references: list[EmbeddingTargetRef] = []
    for category in _category_schedule(samples, (0.4, 0.2, 0.2, 0.2), rng):
        if category == 0:
            references.append(_draw_event(rng, distraction_events, uniform))
        elif category == 1:
            references.append(_draw_event(rng, road_events, uniform))
        elif category == 2:
            references.append(rng.choice(hard_negatives or uniform))
        else:
            references.append(rng.choice(uniform))
    return tuple(references)


def build_face_mouth_target_index(
    stores: Sequence[object],
    *,
    samples: int,
    seed: int,
    hard_negative_scores: Mapping[str, np.ndarray] | None = None,
) -> tuple[EmbeddingTargetRef, ...]:
    """Draw the 50/25/25 yawn event/hard-negative/natural mixture."""
    session_indices = tuple(
        index
        for index in _training_session_indices(stores)
        if stores[index].protocol == "s5"
    )
    if not session_indices:
        raise ValueError("no fold-training s5 embedding stores are available")
    uniform = [
        EmbeddingTargetRef(session_index, target_index)
        for session_index in session_indices
        for target_index in range(len(stores[session_index].frame_ids))
        if _value(stores[session_index], "yawn", target_index) != IGNORE_INDEX
    ]
    yawn_events = _events(
        stores,
        session_indices,
        lambda store, index: _value(store, "yawn", index) > 0,
    )
    negatives = [
        EmbeddingTargetRef(session_index, target_index)
        for session_index in session_indices
        for target_index in range(len(stores[session_index].frame_ids))
        if _value(stores[session_index], "yawn", target_index) == 0
    ]
    if not uniform or not negatives:
        raise ValueError("face-mouth sampling needs annotated s5 negatives")
    hard_negatives = negatives
    if hard_negative_scores is not None:
        scored: list[tuple[float, EmbeddingTargetRef]] = []
        for reference in negatives:
            store = stores[reference.session_index]
            scores = hard_negative_scores.get(store.session_name)
            if scores is None or scores.shape != (len(store.frame_ids),):
                continue
            scored.append((float(scores[reference.target_index]), reference))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            hard_negatives = [
                reference for _, reference in scored[: max(1, len(scored) // 4)]
            ]
    rng = random.Random(seed)
    references: list[EmbeddingTargetRef] = []
    for category in _category_schedule(samples, (0.5, 0.25, 0.25), rng):
        if category == 0:
            references.append(_draw_event(rng, yawn_events, uniform))
        elif category == 1:
            references.append(rng.choice(hard_negatives))
        else:
            references.append(rng.choice(uniform))
    return tuple(references)
