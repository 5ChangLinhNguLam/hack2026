"""Typed access to the local 20 FPS DMD manifest.

The split unit is the full DMD subject identifier (for example ``gF_23``),
never a recording name.  This keeps repeat recordings such as ``_r2`` and
``_r3`` on the same side of a LOSO split.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


_PROTOCOL_RE = re.compile(r"^s\d+$")


@dataclass(frozen=True)
class SessionRecord:
    name: str
    subject_id: str
    protocol: str
    fps: float
    n_frames: int
    duration_sec: float
    label_groups: tuple[str, ...]
    labels_csv: Path
    frames_dir: Path
    subject_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class LosoFold:
    held_out_subject: str
    train: tuple[SessionRecord, ...]
    test: tuple[SessionRecord, ...]


@dataclass(frozen=True)
class NestedLosoFold:
    held_out_subject: str
    validation_subject: str
    train: tuple[SessionRecord, ...]
    validation: tuple[SessionRecord, ...]
    test: tuple[SessionRecord, ...]


def subject_from_session(session_name: str) -> str:
    """Return the full cohort-plus-person identifier from a DMD session."""
    parts = session_name.split("_")
    if len(parts) < 3 or not parts[0].startswith("g"):
        raise ValueError(f"invalid DMD session name: {session_name!r}")
    return "_".join(parts[:2])


def protocol_from_session(session_name: str) -> str:
    """Extract protocol name (s1, s2, ...) while ignoring repeat suffixes."""
    for part in session_name.split("_")[2:]:
        if _PROTOCOL_RE.fullmatch(part):
            return part
    raise ValueError(f"session has no DMD protocol token: {session_name!r}")


def load_sessions(manifest_path: Path | str) -> tuple[SessionRecord, ...]:
    """Load a generated manifest and resolve all paths to absolute paths."""
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list):
        raise ValueError("manifest root must be a list")

    labels_root = manifest_path.parent
    dataset_root = labels_root.parent
    sessions: list[SessionRecord] = []
    for entry in raw:
        name = str(entry["session"])
        labels_csv = labels_root / str(entry["labels_csv"])
        frames_dir = dataset_root / str(entry["frames_dir"])
        if not labels_csv.is_file():
            raise FileNotFoundError(f"labels CSV missing for {name}: {labels_csv}")
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"frame directory missing for {name}: {frames_dir}")

        sessions.append(
            SessionRecord(
                name=name,
                subject_id=subject_from_session(name),
                protocol=protocol_from_session(name),
                fps=float(entry["fps"]),
                n_frames=int(entry["n_frames"]),
                duration_sec=float(entry["duration_sec"]),
                label_groups=tuple(str(group) for group in entry.get("label_groups", ())),
                labels_csv=labels_csv,
                frames_dir=frames_dir,
                subject_metadata=dict(entry.get("subject", {})),
            )
        )
    return tuple(sessions)


def make_loso_folds(sessions: Iterable[SessionRecord]) -> tuple[LosoFold, ...]:
    """Create deterministic leave-one-subject-out folds."""
    records = tuple(sessions)
    subjects = sorted({record.subject_id for record in records})
    return tuple(
        LosoFold(
            held_out_subject=subject,
            train=tuple(record for record in records if record.subject_id != subject),
            test=tuple(record for record in records if record.subject_id == subject),
        )
        for subject in subjects
    )


def make_nested_loso_folds(
    sessions: Iterable[SessionRecord],
) -> tuple[NestedLosoFold, ...]:
    """LOSO outer folds with a different whole subject for model selection."""
    records = tuple(sessions)
    subjects = sorted({record.subject_id for record in records})
    if len(subjects) < 3:
        raise ValueError("nested LOSO requires at least three subjects")
    folds: list[NestedLosoFold] = []
    for index, held_out in enumerate(subjects):
        validation = subjects[(index + 1) % len(subjects)]
        folds.append(
            NestedLosoFold(
                held_out_subject=held_out,
                validation_subject=validation,
                train=tuple(
                    record
                    for record in records
                    if record.subject_id not in {held_out, validation}
                ),
                validation=tuple(
                    record for record in records if record.subject_id == validation
                ),
                test=tuple(record for record in records if record.subject_id == held_out),
            )
        )
    return tuple(folds)
