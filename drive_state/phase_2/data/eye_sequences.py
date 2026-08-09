"""Temporal eye-label sequences and transition-aware clip sampling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Sequence

from .manifest import SessionRecord


PHASE_NAMES = ("open", "closing", "close", "opening")
PHASE_TO_INDEX = {name: index for index, name in enumerate(PHASE_NAMES)}
IGNORE_INDEX = -100
_TRANSITION_INDICES = {PHASE_TO_INDEX["closing"], PHASE_TO_INDEX["opening"]}


@dataclass(frozen=True)
class EyeSequence:
    session_name: str
    subject_id: str
    protocol: str
    fps: float
    frame_ids: tuple[int, ...]
    timestamps: tuple[float, ...]
    frame_paths: tuple[Path, ...]
    phase_targets: tuple[int, ...]
    blink_targets: tuple[bool | None, ...]
    yawn_targets: tuple[bool | None, ...]

    def __len__(self) -> int:
        return len(self.frame_paths)


@dataclass(frozen=True)
class EyeClipRef:
    sequence_index: int
    start: int


def _phase_index(value: str | None) -> int:
    normalized = (value or "").strip().lower()
    return PHASE_TO_INDEX.get(normalized, IGNORE_INDEX)


def _load_sequences(sessions: Iterable[SessionRecord]) -> tuple[EyeSequence, ...]:
    sequences: list[EyeSequence] = []
    for session in sessions:
        frame_ids: list[int] = []
        timestamps: list[float] = []
        frame_paths: list[Path] = []
        phases: list[int] = []
        blinks: list[bool | None] = []
        yawns: list[bool | None] = []
        dataset_root = session.labels_csv.parent.parent
        with session.labels_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            required = {"frame_id", "timestamp", "frame_path"}
            missing = required.difference(fieldnames)
            if missing:
                raise ValueError(f"{session.labels_csv} is missing columns: {sorted(missing)}")
            for row in reader:
                frame_ids.append(int(row["frame_id"]))
                timestamps.append(float(row["timestamp"]))
                frame_paths.append(dataset_root / row["frame_path"])
                phases.append(
                    _phase_index(row.get("eyes_state"))
                    if "eyes_state" in fieldnames
                    else IGNORE_INDEX
                )
                blinks.append(
                    bool((row.get("blinks") or "").strip())
                    if "blinks" in fieldnames
                    else None
                )
                yawns.append(
                    bool((row.get("yawning") or "").strip())
                    if "yawning" in fieldnames
                    else None
                )

        if len(frame_paths) != session.n_frames:
            raise ValueError(
                f"{session.name}: manifest has {session.n_frames} frames, "
                f"CSV has {len(frame_paths)}"
            )
        sequences.append(
            EyeSequence(
                session_name=session.name,
                subject_id=session.subject_id,
                protocol=session.protocol,
                fps=session.fps,
                frame_ids=tuple(frame_ids),
                timestamps=tuple(timestamps),
                frame_paths=tuple(frame_paths),
                phase_targets=tuple(phases),
                blink_targets=tuple(blinks),
                yawn_targets=tuple(yawns),
            )
        )
    return tuple(sequences)


def load_eye_sequences(sessions: Iterable[SessionRecord]) -> tuple[EyeSequence, ...]:
    """Load eye-phase records from DMD protocol s5 sessions only."""

    selected = (
        session
        for session in sessions
        if session.protocol == "s5" and "eyes_state" in session.label_groups
    )
    return _load_sequences(selected)


def load_frame_sequences(sessions: Iterable[SessionRecord]) -> tuple[EyeSequence, ...]:
    """Load chronological rows for ROI caching without inventing absent labels."""

    return _load_sequences(sessions)


def build_balanced_clip_starts(
    phase_targets: Sequence[int],
    *,
    sequence_length: int,
    transition_fraction: float = 0.5,
    seed: int = 0,
    samples: int | None = None,
) -> tuple[int, ...]:
    """Sample causal clip starts with guaranteed transition representation.

    The returned value has ``samples`` elements (all valid starts by default).
    Sampling is with replacement so short ``closing``/``opening`` runs are not
    drowned out by the much longer ``open`` and ``close`` runs.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if not 0.0 <= transition_fraction <= 1.0:
        raise ValueError("transition_fraction must be between 0 and 1")
    maximum_start = len(phase_targets) - sequence_length
    if maximum_start < 0:
        return ()

    valid_starts = tuple(range(maximum_start + 1))
    output_count = len(valid_starts) if samples is None else samples
    if output_count < 0:
        raise ValueError("samples cannot be negative")
    if output_count == 0:
        return ()

    half_window = sequence_length // 2
    transition_starts = tuple(
        min(max(index - half_window, 0), maximum_start)
        for index, phase in enumerate(phase_targets)
        if phase in _TRANSITION_INDICES
    )
    rng = random.Random(seed)
    priority_count = round(output_count * transition_fraction) if transition_starts else 0
    starts = list(rng.choices(transition_starts, k=priority_count))
    starts.extend(rng.choices(valid_starts, k=output_count - priority_count))
    rng.shuffle(starts)
    return tuple(starts)


def build_eye_clip_index(
    sequences: Sequence[EyeSequence],
    *,
    sequence_length: int,
    transition_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[EyeClipRef, ...]:
    """Build deterministic clip references without crossing session boundaries."""
    references: list[EyeClipRef] = []
    for sequence_index, sequence in enumerate(sequences):
        starts = build_balanced_clip_starts(
            sequence.phase_targets,
            sequence_length=sequence_length,
            transition_fraction=transition_fraction,
            seed=seed + sequence_index,
        )
        references.extend(EyeClipRef(sequence_index, start) for start in starts)
    return tuple(references)


def eye_auxiliary_targets(
    phase_targets: Sequence[int],
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...]]:
    """Derive closedness and eye-motion supervision from phase labels.

    Closedness is supervised only at the unambiguous endpoints.  Assigning a
    binary open/closed target to ``closing`` or ``opening`` would inject label
    noise, exactly where the previous frame-only eye model was weakest.
    """
    open_phase = PHASE_TO_INDEX["open"]
    close_phase = PHASE_TO_INDEX["close"]
    closedness: list[float] = []
    closedness_mask: list[bool] = []
    moving: list[float] = []
    for phase in phase_targets:
        is_valid = phase != IGNORE_INDEX
        closedness.append(1.0 if phase == close_phase else 0.0)
        closedness_mask.append(phase in {open_phase, close_phase})
        moving.append(1.0 if phase in _TRANSITION_INDICES and is_valid else 0.0)
    return tuple(closedness), tuple(closedness_mask), tuple(moving)
