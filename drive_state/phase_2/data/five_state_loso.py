"""Deterministic subject-disjoint nested LOSO manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .five_state_labels import FiveState

_RARE_STATES = (
    FiveState.DROWSY,
    FiveState.MICROSLEEP,
    FiveState.YAWNING,
)


@dataclass(frozen=True)
class FiveStateFold:
    test_subject: str
    validation_subject: str
    train_subjects: tuple[str, ...]
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def _sessions(index: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = index.get("sessions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("standardized index must contain sessions")
    sessions = tuple(raw)
    if any(not isinstance(entry, Mapping) for entry in sessions):
        raise ValueError("standardized session entries must be mappings")
    names = [str(entry.get("session", "")) for entry in sessions]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("standardized sessions must have unique names")
    return sessions


def _subject_support(
    index: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    summary = index.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("standardized index summary is missing")
    support = summary.get("by_subject")
    if not isinstance(support, Mapping):
        raise ValueError("standardized subject support is missing")
    return support


def select_validation_subject(
    index: Mapping[str, object],
    *,
    excluded_subject: str,
) -> str:
    """Choose the best rare-event validation subject without outer leakage."""

    subjects = sorted({str(entry["subject"]) for entry in _sessions(index)})
    candidates = [subject for subject in subjects if subject != excluded_subject]
    if not candidates:
        raise ValueError("no validation subject remains after exclusion")
    support = _subject_support(index)

    def score(subject: str) -> tuple[int, int, int, str]:
        subject_support = support.get(subject)
        if not isinstance(subject_support, Mapping):
            raise ValueError(f"missing support for subject {subject}")
        events = subject_support.get("events")
        if not isinstance(events, Mapping):
            raise ValueError(f"missing event support for subject {subject}")
        counts = [
            int(events.get(state.name.lower(), 0)) for state in _RARE_STATES
        ]
        positive = [count for count in counts if count > 0]
        return (
            -len(positive),
            -min(positive) if positive else 0,
            -sum(counts),
            subject,
        )

    return min(candidates, key=score)


def build_nested_loso_folds(
    index: Mapping[str, object],
) -> tuple[FiveStateFold, ...]:
    """Create one outer fold per complete DMD subject."""

    sessions = _sessions(index)
    subjects = sorted({str(entry["subject"]) for entry in sessions})
    if len(subjects) < 3:
        raise ValueError("nested LOSO requires at least three subjects")
    sessions_by_subject: dict[str, list[str]] = {subject: [] for subject in subjects}
    for entry in sessions:
        subject = str(entry["subject"])
        sessions_by_subject[subject].append(str(entry["session"]))

    folds: list[FiveStateFold] = []
    for test_subject in subjects:
        validation_subject = select_validation_subject(
            index,
            excluded_subject=test_subject,
        )
        train_subjects = tuple(
            subject
            for subject in subjects
            if subject not in {test_subject, validation_subject}
        )
        folds.append(
            FiveStateFold(
                test_subject=test_subject,
                validation_subject=validation_subject,
                train_subjects=train_subjects,
                train_sessions=tuple(
                    session
                    for subject in train_subjects
                    for session in sessions_by_subject[subject]
                ),
                validation_sessions=tuple(sessions_by_subject[validation_subject]),
                test_sessions=tuple(sessions_by_subject[test_subject]),
            )
        )
    return tuple(folds)


def _empty_support() -> dict[str, object]:
    return {
        "frames": {state.name.lower(): 0 for state in FiveState},
        "events": {state.name.lower(): 0 for state in FiveState},
        "ignored": 0,
        "reasons": {},
    }


def _partition_support(
    session_names: Sequence[str],
    entries_by_name: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    total = _empty_support()
    for session_name in session_names:
        entry = entries_by_name[session_name]
        stats = entry.get("stats")
        if not isinstance(stats, Mapping):
            raise ValueError(f"session {session_name} has no support statistics")
        for category in ("frames", "events"):
            destination = total[category]
            source = stats.get(category)
            assert isinstance(destination, dict)
            if not isinstance(source, Mapping):
                raise ValueError(f"session {session_name} has invalid {category}")
            for state in FiveState:
                key = state.name.lower()
                destination[key] += int(source.get(key, 0))
        total["ignored"] = int(total["ignored"]) + int(stats.get("ignored", 0))
        destination_reasons = total["reasons"]
        source_reasons = stats.get("reasons", {})
        assert isinstance(destination_reasons, dict)
        if not isinstance(source_reasons, Mapping):
            raise ValueError(f"session {session_name} has invalid reasons")
        for reason, count in source_reasons.items():
            destination_reasons[str(reason)] = (
                int(destination_reasons.get(str(reason), 0)) + int(count)
            )
    return total


def write_nested_loso_manifests(
    index: Mapping[str, object],
    output_dir: Path,
) -> Path:
    """Write one deterministic JSON manifest per outer subject."""

    sessions = _sessions(index)
    entries_by_name = {str(entry["session"]): entry for entry in sessions}
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in build_nested_loso_folds(index):
        payload = {
            **asdict(fold),
            "support": {
                "train": _partition_support(
                    fold.train_sessions,
                    entries_by_name,
                ),
                "validation": _partition_support(
                    fold.validation_sessions,
                    entries_by_name,
                ),
                "test": _partition_support(
                    fold.test_sessions,
                    entries_by_name,
                ),
            },
        }
        path = output_dir / f"{fold.test_subject}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return output_dir


__all__ = [
    "FiveStateFold",
    "build_nested_loso_folds",
    "select_validation_subject",
    "write_nested_loso_manifests",
]
