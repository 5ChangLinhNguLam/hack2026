"""Causal ocular windows and event-balanced DMD s5 sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..models.ocular_lstm import assemble_ocular_features
from .five_state_labels import IGNORE_INDEX
from .five_state_visual import EYE_PHASE_CLASSES
from .five_state_visual_cache import VisualSessionCache


OPEN = EYE_PHASE_CLASSES["open"]
CLOSING = EYE_PHASE_CLASSES["closing"]
CLOSED = EYE_PHASE_CLASSES["close"]
OPENING = EYE_PHASE_CLASSES["opening"]

_BUCKET_ORDER = (
    "microsleep",
    "slow_closure",
    "transition",
    "ordinary_blink",
    "open_background",
    "hard_invalid",
)


@dataclass(frozen=True)
class OcularTargetRef:
    session: str
    target_index: int
    bucket: str
    event_id: str


@dataclass(frozen=True)
class OcularSamplingAudit:
    requested: int
    actual: int
    bucket_counts: Mapping[str, int]
    subject_counts: Mapping[str, int]
    windows_per_event: Mapping[str, int]


def causal_progress_targets(
    phase_targets: np.ndarray,
    *,
    fps: float,
    microsleep_seconds: float = 2.0,
) -> np.ndarray:
    """Return past-only progress through the current reliable closed run."""

    if fps <= 0.0 or microsleep_seconds <= 0.0:
        raise ValueError("FPS and microsleep duration must be positive")
    values = np.asarray(phase_targets, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("phase targets must be one-dimensional")
    progress = np.zeros(len(values), dtype=np.float32)
    closed_frames = 0
    boundary = fps * microsleep_seconds
    for index, phase in enumerate(values):
        if phase == CLOSED:
            closed_frames += 1
            progress[index] = min(closed_frames / boundary, 1.0)
        else:
            closed_frames = 0
    return progress


def _session_features(
    cache: VisualSessionCache,
    *,
    embedding_dim: int,
    region_dim: int,
) -> Tensor:
    expected = embedding_dim + 4 * region_dim
    if cache.visual_embedding.shape[1] != expected:
        raise ValueError(
            f"visual embedding dimension mismatch for {cache.session}"
        )
    visual = torch.from_numpy(
        np.asarray(cache.visual_embedding, dtype=np.float32)
    )
    regions = visual[:, embedding_dim:].reshape(-1, 4, region_dim)
    return assemble_ocular_features(
        region_embeddings=regions,
        raw_eye_probabilities=torch.from_numpy(
            np.asarray(cache.raw_eye_probabilities, dtype=np.float32)
        ),
        eye_visibility=torch.from_numpy(
            np.asarray(cache.region_visibility[:, 1:3], dtype=np.float32)
        ),
    )


class OcularWindowDataset(
    Dataset[dict[str, Tensor | str | int | float]]
):
    """Return fixed causal eye histories ending at referenced s5 frames."""

    def __init__(
        self,
        caches: Sequence[VisualSessionCache],
        references: Sequence[OcularTargetRef],
        *,
        length: int = 60,
        embedding_dim: int,
        region_dim: int,
    ) -> None:
        if min(length, embedding_dim, region_dim) <= 0:
            raise ValueError("ocular window dimensions must be positive")
        if any(cache.protocol != "s5" for cache in caches):
            raise ValueError("ocular phase windows accept only DMD s5 caches")
        by_session = {cache.session: cache for cache in caches}
        if len(by_session) != len(caches):
            raise ValueError("ocular caches must have unique session names")
        for reference in references:
            cache = by_session.get(reference.session)
            if cache is None:
                raise ValueError(
                    f"ocular reference has no cache: {reference.session}"
                )
            if not 0 <= reference.target_index < len(cache.frame_ids):
                raise ValueError("ocular reference target is outside its session")
        self.caches = by_session
        self.references = tuple(references)
        self.length = int(length)
        self.features = {
            cache.session: _session_features(
                cache,
                embedding_dim=embedding_dim,
                region_dim=region_dim,
            )
            for cache in caches
        }
        self.progress = {
            cache.session: causal_progress_targets(
                cache.eye_phase_targets,
                fps=cache.fps,
            )
            for cache in caches
        }

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor | str | int | float]:
        reference = self.references[index]
        cache = self.caches[reference.session]
        target = reference.target_index
        start = max(0, target - self.length + 1)
        stop = target + 1
        real_count = stop - start
        pad = self.length - real_count
        feature_values = self.features[cache.session][start:stop]
        feature_dim = feature_values.shape[-1]
        features = torch.cat(
            (
                torch.zeros((pad, feature_dim), dtype=torch.float32),
                feature_values,
            )
        )
        phases = torch.full(
            (self.length,),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        real_phases = torch.from_numpy(
            np.asarray(cache.eye_phase_targets[start:stop], dtype=np.int64)
        )
        phases[pad:] = real_phases
        closed = torch.zeros(self.length, dtype=torch.float32)
        closed[pad:] = real_phases.eq(CLOSED).to(dtype=torch.float32)
        progress = torch.zeros(self.length, dtype=torch.float32)
        progress[pad:] = torch.from_numpy(
            self.progress[cache.session][start:stop].copy()
        )
        reliability = torch.zeros(self.length, dtype=torch.float32)
        reliability[pad:] = torch.from_numpy(
            np.asarray(
                cache.region_visibility[start:stop, 1:3].mean(axis=1),
                dtype=np.float32,
            )
        )
        context_valid = torch.cat(
            (
                torch.zeros(pad, dtype=torch.float32),
                torch.ones(real_count, dtype=torch.float32),
            )
        )
        return {
            "ocular_features": features,
            "phase_targets": phases,
            "closed_targets": closed,
            "progress_targets": progress,
            "reliability_targets": reliability,
            "context_valid": context_valid,
            "session": cache.session,
            "subject": cache.subject,
            "bucket": reference.bucket,
            "event_id": reference.event_id,
            "target_frame_id": int(cache.frame_ids[target]),
            "target_timestamp": float(target / cache.fps),
        }


def _bucket_references(
    cache: VisualSessionCache,
) -> tuple[OcularTargetRef, ...]:
    if cache.protocol != "s5":
        return ()
    references: list[OcularTargetRef] = []
    close_event = 0
    open_event = 0
    transition_event = 0
    invalid_event = 0
    closed_frames = 0
    previous_kind = ""
    current_close_id = ""
    current_open_id = ""
    current_transition_id = ""
    current_invalid_id = ""
    for index, phase in enumerate(cache.eye_phase_targets):
        if int(phase) == CLOSED:
            if previous_kind != "close":
                close_event += 1
                current_close_id = f"{cache.session}:close:{close_event}"
                closed_frames = 0
            closed_frames += 1
            duration = closed_frames / cache.fps
            if duration >= 2.0:
                bucket = "microsleep"
            elif duration >= 0.5:
                bucket = "slow_closure"
            else:
                bucket = "ordinary_blink"
            event_id = current_close_id
            kind = "close"
        elif int(phase) in (CLOSING, OPENING):
            if previous_kind != "transition":
                transition_event += 1
                current_transition_id = (
                    f"{cache.session}:transition:{transition_event}"
                )
            bucket = "transition"
            event_id = current_transition_id
            closed_frames = 0
            kind = "transition"
        elif int(phase) == OPEN:
            if previous_kind != "open":
                open_event += 1
                current_open_id = f"{cache.session}:open:{open_event}"
            bucket = "open_background"
            event_id = current_open_id
            closed_frames = 0
            kind = "open"
        else:
            if previous_kind != "invalid":
                invalid_event += 1
                current_invalid_id = f"{cache.session}:invalid:{invalid_event}"
            bucket = "hard_invalid"
            event_id = current_invalid_id
            closed_frames = 0
            kind = "invalid"
        references.append(
            OcularTargetRef(cache.session, index, bucket, event_id)
        )
        previous_kind = kind
    return tuple(references)


def build_chronological_ocular_references(
    caches: Sequence[VisualSessionCache],
) -> tuple[OcularTargetRef, ...]:
    """Return each DMD s5 frame once in cache/session order."""

    return tuple(
        reference
        for cache in caches
        for reference in _bucket_references(cache)
    )


def build_ocular_epoch_references(
    caches: Sequence[VisualSessionCache],
    *,
    samples: int,
    max_windows_per_event: int,
    seed: int,
) -> tuple[tuple[OcularTargetRef, ...], OcularSamplingAudit]:
    """Balance bucket, subject, and event coverage with a hard event cap."""

    if samples <= 0 or max_windows_per_event <= 0:
        raise ValueError("ocular samples and event cap must be positive")
    rng = random.Random(seed)
    subject_by_session = {cache.session: cache.subject for cache in caches}
    pool: dict[
        str,
        dict[str, dict[str, list[OcularTargetRef]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cache in caches:
        for reference in _bucket_references(cache):
            pool[reference.bucket][cache.subject][reference.event_id].append(
                reference
            )
    if not pool:
        raise ValueError("no DMD s5 ocular references are available")

    selected: list[OcularTargetRef] = []
    event_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    while len(selected) < samples:
        made_progress = False
        for bucket in _BUCKET_ORDER:
            subjects = pool.get(bucket, {})
            eligible_subjects = []
            for subject, events in subjects.items():
                if any(
                    event_counts[event_id] < max_windows_per_event
                    for event_id in events
                ):
                    eligible_subjects.append(subject)
            if not eligible_subjects:
                continue
            minimum_subject = min(
                subject_counts[subject] for subject in eligible_subjects
            )
            subject_choices = [
                subject
                for subject in eligible_subjects
                if subject_counts[subject] == minimum_subject
            ]
            subject = rng.choice(sorted(subject_choices))
            events = subjects[subject]
            eligible_events = [
                event_id
                for event_id in events
                if event_counts[event_id] < max_windows_per_event
            ]
            minimum_event = min(event_counts[event_id] for event_id in eligible_events)
            event_choices = [
                event_id
                for event_id in eligible_events
                if event_counts[event_id] == minimum_event
            ]
            event_id = rng.choice(sorted(event_choices))
            reference = rng.choice(events[event_id])
            selected.append(reference)
            event_counts[event_id] += 1
            subject_counts[subject_by_session[reference.session]] += 1
            bucket_counts[bucket] += 1
            made_progress = True
            if len(selected) >= samples:
                break
        if not made_progress:
            break
    audit = OcularSamplingAudit(
        requested=int(samples),
        actual=len(selected),
        bucket_counts=dict(sorted(bucket_counts.items())),
        subject_counts=dict(sorted(subject_counts.items())),
        windows_per_event=dict(sorted(event_counts.items())),
    )
    return tuple(selected), audit


__all__ = [
    "OcularSamplingAudit",
    "OcularTargetRef",
    "OcularWindowDataset",
    "build_chronological_ocular_references",
    "build_ocular_epoch_references",
    "causal_progress_targets",
]
