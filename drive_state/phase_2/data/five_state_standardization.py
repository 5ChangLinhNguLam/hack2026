"""Materialized, target-rate five-state labels for DMD sessions."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .five_state_labels import (
    IGNORE_INDEX,
    FiveState,
    WeakStateConfig,
    WeakStateTargets,
    build_weak_state_targets,
)

STANDARDIZED_SCHEMA_VERSION = 2
STANDARDIZED_COLUMNS = (
    "session",
    "subject",
    "protocol",
    "frame_id",
    "timestamp_seconds",
    "target_id",
    "state",
    "valid",
    "confidence",
    "reason",
    "event_id",
)


@dataclass(frozen=True)
class StandardizedSession:
    session: str
    subject: str
    protocol: str
    fps: float
    frame_ids: tuple[int, ...]
    targets: WeakStateTargets

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError("standardized session FPS must be positive")
        if len(self.frame_ids) != len(self.targets.targets):
            raise ValueError("standardized frames and targets must align")


def select_target_rate_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_fps: float,
    target_fps: float,
) -> tuple[Mapping[str, str], ...]:
    """Select rows at an integer FPS divisor before temporal labels are built."""

    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source and target FPS must be positive")
    ratio = source_fps / target_fps
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, abs_tol=1e-9):
        raise ValueError("source FPS must be an integer multiple of target FPS")
    frame_ids = tuple(int(row["frame_id"]) for row in rows)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("duplicate frame IDs")
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("frame IDs must be strictly increasing")
    return tuple(rows[::stride])


def standardize_session(
    *,
    session: str,
    subject: str,
    protocol: str,
    rows: Sequence[Mapping[str, str]],
    source_fps: float,
    target_fps: float,
    config: WeakStateConfig | None = None,
) -> StandardizedSession:
    """Build session-namespaced targets at the actual model sampling rate."""

    selected = select_target_rate_rows(
        rows,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    native_targets = build_weak_state_targets(
        protocol,
        selected,
        fps=target_fps,
        config=config,
    )
    targets = WeakStateTargets(
        native_targets.targets,
        native_targets.confidence,
        native_targets.reasons,
        tuple(
            f"{session}:{event_id}" if event_id else ""
            for event_id in native_targets.event_ids
        ),
    )
    return StandardizedSession(
        session=session,
        subject=subject,
        protocol=protocol,
        fps=target_fps,
        frame_ids=tuple(int(row["frame_id"]) for row in selected),
        targets=targets,
    )


def write_standardized_session(
    session: StandardizedSession,
    output_csv: Path,
) -> dict[str, object]:
    """Atomically write one standardized session and return support counts."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    frame_counts: Counter[str] = Counter(
        {state.name.lower(): 0 for state in FiveState}
    )
    event_ids: dict[str, set[str]] = {
        state.name.lower(): set() for state in FiveState
    }
    reason_counts: Counter[str] = Counter()
    ignored = 0
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STANDARDIZED_COLUMNS)
        writer.writeheader()
        for index, (frame_id, target, confidence, reason, event_id) in enumerate(
            zip(
                session.frame_ids,
                session.targets.targets,
                session.targets.confidence,
                session.targets.reasons,
                session.targets.event_ids,
                strict=True,
            )
        ):
            valid = target != IGNORE_INDEX
            state = FiveState(target).name.lower() if valid else ""
            if valid:
                frame_counts[state] += 1
                event_ids[state].add(event_id)
            else:
                ignored += 1
            reason_counts[reason] += 1
            writer.writerow(
                {
                    "session": session.session,
                    "subject": session.subject,
                    "protocol": session.protocol,
                    "frame_id": frame_id,
                    "timestamp_seconds": f"{index / session.fps:.6f}",
                    "target_id": target,
                    "state": state,
                    "valid": int(valid),
                    "confidence": f"{confidence:.6f}",
                    "reason": reason,
                    "event_id": event_id,
                }
            )
    temporary.replace(output_csv)
    return {
        "frames": dict(frame_counts),
        "events": {name: len(values) for name, values in event_ids.items()},
        "ignored": ignored,
        "reasons": dict(reason_counts),
    }


def load_standardized_session(
    csv_path: Path,
    *,
    fps: float,
    expected_session: str | None = None,
) -> StandardizedSession:
    """Load and validate a materialized standardized session."""

    if fps <= 0.0:
        raise ValueError("FPS must be positive")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != STANDARDIZED_COLUMNS:
            raise ValueError("invalid standardized session schema")
        rows = tuple(reader)
    if not rows:
        raise ValueError("standardized session cannot be empty")
    session_names = {row["session"] for row in rows}
    subjects = {row["subject"] for row in rows}
    protocols = {row["protocol"] for row in rows}
    if len(session_names) != 1 or len(subjects) != 1 or len(protocols) != 1:
        raise ValueError("standardized session metadata is inconsistent")
    session_name = next(iter(session_names))
    if expected_session is not None and session_name != expected_session:
        raise ValueError("standardized session name mismatch")
    frame_ids = tuple(int(row["frame_id"]) for row in rows)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("duplicate standardized frame IDs")
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("standardized frame IDs must be strictly increasing")
    targets = tuple(int(row["target_id"]) for row in rows)
    confidence = tuple(float(row["confidence"]) for row in rows)
    reasons = tuple(row["reason"] for row in rows)
    event_ids = tuple(row["event_id"] for row in rows)
    for index, (row, target, weight, event_id) in enumerate(
        zip(rows, targets, confidence, event_ids, strict=True)
    ):
        if not math.isclose(
            float(row["timestamp_seconds"]),
            index / fps,
            abs_tol=1e-6,
        ):
            raise ValueError("standardized timestamp does not match FPS")
        valid = target != IGNORE_INDEX
        expected_state = FiveState(target).name.lower() if valid else ""
        if row["valid"] != str(int(valid)) or row["state"] != expected_state:
            raise ValueError("standardized target metadata is inconsistent")
        if valid and (weight <= 0.0 or not event_id):
            raise ValueError("valid standardized target needs confidence and event ID")
        if not valid and (weight != 0.0 or event_id):
            raise ValueError("ignored standardized target carries supervision")
    return StandardizedSession(
        session=session_name,
        subject=next(iter(subjects)),
        protocol=next(iter(protocols)),
        fps=fps,
        frame_ids=frame_ids,
        targets=WeakStateTargets(targets, confidence, reasons, event_ids),
    )


def _empty_support() -> dict[str, object]:
    return {
        "frames": {state.name.lower(): 0 for state in FiveState},
        "events": {state.name.lower(): 0 for state in FiveState},
        "ignored": 0,
        "reasons": {},
    }


def _add_support(destination: dict[str, object], source: Mapping[str, object]) -> None:
    for category in ("frames", "events"):
        destination_counts = destination[category]
        source_counts = source[category]
        assert isinstance(destination_counts, dict)
        assert isinstance(source_counts, Mapping)
        for state in FiveState:
            name = state.name.lower()
            destination_counts[name] += int(source_counts.get(name, 0))
    destination["ignored"] = int(destination["ignored"]) + int(source["ignored"])
    destination_reasons = destination["reasons"]
    source_reasons = source["reasons"]
    assert isinstance(destination_reasons, dict)
    assert isinstance(source_reasons, Mapping)
    for reason, count in source_reasons.items():
        destination_reasons[str(reason)] = (
            int(destination_reasons.get(str(reason), 0)) + int(count)
        )


def summarize_standardized_entries(
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reduce per-session support into global, subject, and protocol groups."""

    global_support = _empty_support()
    by_subject: dict[str, dict[str, object]] = {}
    by_protocol: dict[str, dict[str, object]] = {}
    for entry in entries:
        stats = entry["stats"]
        assert isinstance(stats, Mapping)
        _add_support(global_support, stats)
        subject = str(entry["subject"])
        protocol = str(entry["protocol"])
        _add_support(by_subject.setdefault(subject, _empty_support()), stats)
        _add_support(by_protocol.setdefault(protocol, _empty_support()), stats)
    return {
        "global": global_support,
        "by_subject": by_subject,
        "by_protocol": by_protocol,
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_standardized_index(
    *,
    output_dir: Path,
    target_fps: float,
    config: WeakStateConfig,
    source_manifest: Path,
    sessions: Sequence[StandardizedSession],
) -> Path:
    """Write all session tables plus a versioned dataset index and summary."""

    if target_fps <= 0.0:
        raise ValueError("target FPS must be positive")
    names = [session.session for session in sessions]
    if len(set(names)) != len(names):
        raise ValueError("duplicate standardized session")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for session in sessions:
        if not math.isclose(session.fps, target_fps):
            raise ValueError("standardized session target FPS mismatch")
        relative_csv = Path("sessions") / f"{session.session}.csv"
        stats = write_standardized_session(session, output_dir / relative_csv)
        entries.append(
            {
                "session": session.session,
                "subject": session.subject,
                "protocol": session.protocol,
                "csv": relative_csv.as_posix(),
                "stats": stats,
            }
        )
    summary = summarize_standardized_entries(entries)
    payload = {
        "schema_version": STANDARDIZED_SCHEMA_VERSION,
        "target_fps": target_fps,
        "config": asdict(config),
        "source_manifest": str(source_manifest.resolve()),
        "sessions": entries,
        "summary": summary,
    }
    index_path = output_dir / "index.json"
    _atomic_json(index_path, payload)
    _atomic_json(output_dir / "summary.json", summary)
    return index_path


def load_standardized_index(
    index_path: Path,
    *,
    expected_fps: float | None = None,
) -> dict[str, object]:
    """Load an index and reject incompatible or internally invalid metadata."""

    with index_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != STANDARDIZED_SCHEMA_VERSION:
        raise ValueError("standardized index schema mismatch")
    config_payload = payload.get("config")
    expected_config_fields = set(asdict(WeakStateConfig()))
    if (
        not isinstance(config_payload, dict)
        or set(config_payload) != expected_config_fields
    ):
        raise ValueError("standardized index weak-label configuration mismatch")
    try:
        WeakStateConfig(**config_payload)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "standardized index weak-label configuration is invalid"
        ) from error
    target_fps = float(payload.get("target_fps", 0.0))
    if target_fps <= 0.0:
        raise ValueError("standardized index target FPS is invalid")
    if expected_fps is not None and not math.isclose(target_fps, expected_fps):
        raise ValueError("standardized index target FPS mismatch")
    entries = payload.get("sessions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("standardized index must contain sessions")
    names = [str(entry.get("session", "")) for entry in entries]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("standardized index has duplicate or empty sessions")
    for entry in entries:
        csv_path = index_path.parent / str(entry.get("csv", ""))
        if not csv_path.is_file():
            raise ValueError(f"standardized session CSV is missing: {csv_path}")
    return payload


__all__ = [
    "STANDARDIZED_COLUMNS",
    "STANDARDIZED_SCHEMA_VERSION",
    "StandardizedSession",
    "load_standardized_index",
    "load_standardized_session",
    "select_target_rate_rows",
    "standardize_session",
    "summarize_standardized_entries",
    "write_standardized_index",
    "write_standardized_session",
]
