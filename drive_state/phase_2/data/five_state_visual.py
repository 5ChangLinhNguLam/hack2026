"""Visual frame records joined to standardized five-state supervision."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .evidence_regions import (
    EvidenceRegionRecord,
    EvidenceRegionStore,
    project_normalized_boxes_to_letterbox,
)
from .primitive_dataset import letterbox, normalized_image_tensor
from .five_state_events import DEFAULT_STATE_MASS, SamplingAudit
from .five_state_labels import IGNORE_INDEX, FiveState
from .five_state_standardization import (
    load_standardized_index,
    load_standardized_session,
)
from .manifest import load_sessions

EYE_PHASE_CLASSES: Mapping[str, int] = {
    "open": 0,
    "closing": 1,
    "close": 2,
    "opening": 3,
}
EYE_APERTURE_CLASSES: Mapping[str, int] = {
    "open": 0,
    "closing": 1,
    "opening": 1,
    "close": 2,
}
_EYE_PHASE_TO_APERTURE: Mapping[int, int] = {
    EYE_PHASE_CLASSES["open"]: EYE_APERTURE_CLASSES["open"],
    EYE_PHASE_CLASSES["closing"]: EYE_APERTURE_CLASSES["closing"],
    EYE_PHASE_CLASSES["close"]: EYE_APERTURE_CLASSES["close"],
    EYE_PHASE_CLASSES["opening"]: EYE_APERTURE_CLASSES["opening"],
}


@dataclass(frozen=True)
class FiveStateVisualRecord:
    session: str
    subject: str
    protocol: str
    frame_id: int
    frame_path: Path
    target: int
    confidence: float
    event_id: str
    eye_phase_target: int
    source_frame_id: int | None = None


@dataclass(frozen=True)
class VisualTrainingAudit:
    final_state: SamplingAudit
    eye_samples: int
    eye_phase_counts: Mapping[int, int]


class FiveStateFrameDataset(Dataset[dict[str, Tensor | str | int]]):
    """Return one high-resolution full cabin frame and standardized targets."""

    def __init__(
        self,
        records: Sequence[FiveStateVisualRecord],
        *,
        image_size: tuple[int, int] = (640, 384),
        training: bool = False,
        region_stores: Mapping[str, EvidenceRegionStore] | None = None,
        teacher_embedding_dim: int = 0,
    ) -> None:
        if any(value <= 0 for value in image_size):
            raise ValueError("image dimensions must be positive")
        self.records = tuple(records)
        self.image_size = image_size
        self.training = training
        self.region_stores = dict(region_stores or {})
        if teacher_embedding_dim < 0:
            raise ValueError("teacher embedding dimension cannot be negative")
        self.teacher_embedding_dim = int(teacher_embedding_dim)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        record = self.records[index]
        with Image.open(record.frame_path) as source:
            full_frame = source.convert("RGB")
        source_size = full_frame.size
        if self.training:
            full_frame = _augment_training_frame(full_frame)
        image = normalized_image_tensor(
            letterbox(full_frame, self.image_size),
            self.image_size,
        )
        store = self.region_stores.get(record.session)
        regions = (
            EvidenceRegionRecord.missing(record.frame_id)
            if store is None
            else store.get(record.frame_id)
        )
        projected_boxes = project_normalized_boxes_to_letterbox(
            regions.boxes,
            source_size=source_size,
            target_size=self.image_size,
        )
        valid_target = record.target != IGNORE_INDEX
        eye_aperture = _EYE_PHASE_TO_APERTURE.get(
            record.eye_phase_target,
            IGNORE_INDEX,
        )
        return {
            "image": image,
            "region_boxes": torch.from_numpy(projected_boxes.copy()),
            "region_visibility": torch.from_numpy(
                regions.visibility.astype(np.float32, copy=True)
            ),
            "head_pose_target": torch.from_numpy(
                regions.head_pose.astype(np.float32, copy=True)
            ),
            "five_state_target": torch.tensor(record.target, dtype=torch.long),
            "confidence": torch.tensor(record.confidence, dtype=torch.float32),
            "eye_phase_target": torch.tensor(
                record.eye_phase_target,
                dtype=torch.long,
            ),
            "eye_aperture_target": torch.tensor(
                eye_aperture,
                dtype=torch.long,
            ),
            "yawn_target": torch.tensor(
                int(record.target == FiveState.YAWNING),
                dtype=torch.long,
            ),
            "yawn_mask": torch.tensor(
                valid_target and record.protocol == "s5",
                dtype=torch.bool,
            ),
            "distraction_target": torch.tensor(
                int(record.target == FiveState.DISTRACTION),
                dtype=torch.long,
            ),
            "distraction_mask": torch.tensor(
                valid_target and record.protocol in {"s1", "s2", "s3"},
                dtype=torch.bool,
            ),
            "pose_mask": torch.tensor(
                bool(regions.visibility[0] > 0.0),
                dtype=torch.bool,
            ),
            "source": "dmd",
            "supervision_weight": torch.tensor(1.0, dtype=torch.float32),
            "teacher_embedding": torch.zeros(
                self.teacher_embedding_dim,
                dtype=torch.float32,
            ),
            "consistency_mask": torch.tensor(False, dtype=torch.bool),
            "session": record.session,
            "subject": record.subject,
            "protocol": record.protocol,
            "frame_id": record.frame_id,
            "event_id": record.event_id,
        }


def _augment_training_frame(image: Image.Image) -> Image.Image:
    """Apply mild label-preserving variation for cross-driver generalization."""

    augmented = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.80, 1.20)
    )
    augmented = ImageEnhance.Contrast(augmented).enhance(
        random.uniform(0.80, 1.20)
    )
    return ImageEnhance.Color(augmented).enhance(
        random.uniform(0.85, 1.15)
    )


def build_event_balanced_record_indices(
    records: Sequence[FiveStateVisualRecord],
    *,
    samples: int,
    seed: int,
) -> tuple[tuple[int, ...], SamplingAudit]:
    """Select visual frames by state, event, then timestamp."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    events: dict[FiveState, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        if record.target == IGNORE_INDEX:
            continue
        if not record.event_id:
            raise ValueError("valid visual record is missing an event ID")
        events[FiveState(record.target)][record.event_id].append(index)
    present = tuple(state for state in FiveState if events[state])
    if not present:
        raise ValueError("visual balancing requires valid events")
    total_mass = sum(DEFAULT_STATE_MASS[state] for state in present)
    raw = {
        state: samples * DEFAULT_STATE_MASS[state] / total_mass
        for state in present
    }
    counts = {state: math.floor(raw[state]) for state in present}
    remaining = samples - sum(counts.values())
    order = sorted(
        present,
        key=lambda state: (-(raw[state] - counts[state]), state.value),
    )
    for state in order[:remaining]:
        counts[state] += 1

    rng = random.Random(seed)
    selected: list[int] = []
    for state in present:
        event_items = list(events[state].items())
        rng.shuffle(event_items)
        for sample_index in range(counts[state]):
            if sample_index and sample_index % len(event_items) == 0:
                rng.shuffle(event_items)
            _, candidates = event_items[sample_index % len(event_items)]
            selected.append(rng.choice(candidates))
    rng.shuffle(selected)
    return (
        tuple(selected),
        SamplingAudit(
            requested=samples,
            counts={state: counts.get(state, 0) for state in FiveState},
            missing_states=tuple(state for state in FiveState if not events[state]),
        ),
    )


def build_visual_training_indices(
    records: Sequence[FiveStateVisualRecord],
    *,
    samples: int,
    eye_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], VisualTrainingAudit]:
    """Mix final-state event samples with balanced native eye-phase runs."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 <= eye_fraction < 1.0:
        raise ValueError("eye fraction must be in [0, 1)")
    eye_samples = round(samples * eye_fraction)
    final_samples = samples - eye_samples
    final_indices, final_audit = build_event_balanced_record_indices(
        records,
        samples=final_samples,
        seed=seed,
    )

    eye_runs: dict[int, list[list[int]]] = defaultdict(list)
    active_session = ""
    active_phase = IGNORE_INDEX
    active_run: list[int] | None = None
    for index, record in enumerate(records):
        phase = int(record.eye_phase_target)
        if phase == IGNORE_INDEX:
            active_session = ""
            active_phase = IGNORE_INDEX
            active_run = None
            continue
        if record.session != active_session or phase != active_phase:
            active_run = []
            eye_runs[phase].append(active_run)
            active_session = record.session
            active_phase = phase
        assert active_run is not None
        active_run.append(index)
    present_phases = tuple(phase for phase in range(4) if eye_runs[phase])
    if eye_samples and not present_phases:
        raise ValueError("eye-balanced sampling requires native eye phases")
    phase_counts = {phase: 0 for phase in range(4)}
    if present_phases:
        base, remainder = divmod(eye_samples, len(present_phases))
        for offset, phase in enumerate(present_phases):
            phase_counts[phase] = base + int(offset < remainder)

    rng = random.Random(seed + 1)
    selected_eye: list[int] = []
    for phase in present_phases:
        runs = list(eye_runs[phase])
        rng.shuffle(runs)
        for sample_index in range(phase_counts[phase]):
            if sample_index and sample_index % len(runs) == 0:
                rng.shuffle(runs)
            run = runs[sample_index % len(runs)]
            selected_eye.append(rng.choice(run))
    selected = [*final_indices, *selected_eye]
    rng.shuffle(selected)
    return (
        tuple(selected),
        VisualTrainingAudit(
            final_state=final_audit,
            eye_samples=eye_samples,
            eye_phase_counts=phase_counts,
        ),
    )


def _native_rows(labels_csv: Path) -> dict[int, Mapping[str, str]]:
    with labels_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    by_frame: dict[int, Mapping[str, str]] = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        if frame_id in by_frame:
            raise ValueError(f"duplicate native label frame ID: {frame_id}")
        by_frame[frame_id] = row
    return by_frame


def _eye_phase_target(protocol: str, row: Mapping[str, str]) -> int:
    if protocol != "s5":
        return IGNORE_INDEX
    phase = str(row.get("eyes_state", row.get("phase", ""))).strip().lower()
    return EYE_PHASE_CLASSES.get(phase, IGNORE_INDEX)


def eye_aperture_target(phase: str) -> int:
    """Map frame-observable eye phases to open/transition/closed."""

    return EYE_APERTURE_CLASSES.get(str(phase).strip().lower(), IGNORE_INDEX)


def load_visual_records(
    *,
    root: Path,
    standardized_index: Path,
    fold_manifest: Path,
    partition: str,
) -> tuple[FiveStateVisualRecord, ...]:
    """Join one LOSO partition across manifest, standardized labels, and images."""

    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    root = root.resolve()
    index_path = standardized_index.resolve()
    index = load_standardized_index(index_path)
    target_fps = float(index["target_fps"])
    with fold_manifest.open("r", encoding="utf-8") as stream:
        fold = json.load(stream)
    session_key = f"{partition}_sessions"
    requested = tuple(str(value) for value in fold.get(session_key, ()))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError(f"{partition} fold sessions must be nonempty and unique")

    if partition == "train":
        expected_subjects = {str(value) for value in fold.get("train_subjects", ())}
    elif partition == "validation":
        expected_subjects = {str(fold.get("validation_subject", ""))}
    else:
        expected_subjects = {str(fold.get("test_subject", ""))}
    if not expected_subjects or "" in expected_subjects:
        raise ValueError(f"{partition} fold subjects are missing")

    native_by_name = {
        session.name: session
        for session in load_sessions(root / "labels_20fps" / "manifest_20fps.json")
    }
    entries = index["sessions"]
    assert isinstance(entries, list)
    index_by_name = {str(entry["session"]): entry for entry in entries}

    records: list[FiveStateVisualRecord] = []
    for session_name in requested:
        native = native_by_name.get(session_name)
        entry = index_by_name.get(session_name)
        if native is None or entry is None:
            raise ValueError(f"fold session is missing from joined data: {session_name}")
        if native.subject_id not in expected_subjects:
            raise ValueError(
                f"subject leakage in {partition}: {session_name} belongs to "
                f"{native.subject_id}"
            )
        if native.fps != target_fps:
            raise ValueError(f"FPS mismatch for session {session_name}")
        standardized = load_standardized_session(
            index_path.parent / str(entry["csv"]),
            fps=target_fps,
            expected_session=session_name,
        )
        if (
            standardized.subject != native.subject_id
            or standardized.protocol != native.protocol
        ):
            raise ValueError(f"standardized metadata mismatch for {session_name}")
        native_rows = _native_rows(native.labels_csv)
        for frame_id, target, confidence, event_id in zip(
            standardized.frame_ids,
            standardized.targets.targets,
            standardized.targets.confidence,
            standardized.targets.event_ids,
            strict=True,
        ):
            row = native_rows.get(frame_id)
            if row is None:
                raise ValueError(
                    f"standardized frame {frame_id} is missing native labels "
                    f"for {session_name}"
                )
            frame_path = native.frames_dir / f"frame_{frame_id:06d}.jpg"
            if not frame_path.is_file():
                raise FileNotFoundError(
                    f"standardized frame image is missing: {frame_path}"
                )
            records.append(
                FiveStateVisualRecord(
                    session=session_name,
                    subject=native.subject_id,
                    protocol=native.protocol,
                    frame_id=frame_id,
                    frame_path=frame_path,
                    target=target,
                    confidence=confidence,
                    event_id=event_id,
                    eye_phase_target=_eye_phase_target(native.protocol, row),
                    source_frame_id=int(row.get("src_frame_id", frame_id)),
                )
            )
    return tuple(records)


__all__ = [
    "EYE_APERTURE_CLASSES",
    "EYE_PHASE_CLASSES",
    "FiveStateFrameDataset",
    "FiveStateVisualRecord",
    "VisualTrainingAudit",
    "build_event_balanced_record_indices",
    "build_visual_training_indices",
    "eye_aperture_target",
    "load_visual_records",
]
