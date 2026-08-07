"""Reproducible label-only EDA for the local DMD 20 FPS subset."""

from __future__ import annotations

from collections import Counter
import csv
from typing import Iterable

from .data.manifest import SessionRecord


def analyze_dmd_labels(sessions: Iterable[SessionRecord]) -> dict[str, object]:
    sessions = tuple(sessions)
    protocol_report: dict[str, dict[str, int]] = {}
    for protocol in sorted({session.protocol for session in sessions}):
        selected = [session for session in sessions if session.protocol == protocol]
        protocol_report[protocol] = {
            "sessions": len(selected),
            "frames": sum(session.n_frames for session in selected),
            "subjects": len({session.subject_id for session in selected}),
        }

    eye_phases: Counter[str] = Counter()
    action_labels: Counter[str] = Counter()
    gaze_zones: Counter[str] = Counter()
    s5_frames: Counter[str] = Counter()
    s5_phases: dict[str, Counter[str]] = {}
    s5_yawn_events: Counter[str] = Counter()
    s5_close_events: Counter[str] = Counter()
    s5_microsleeps: Counter[str] = Counter()
    yawn_events = 0
    close_runs: list[int] = []
    for session in sessions:
        if session.protocol == "s5":
            s5_frames[session.subject_id] += session.n_frames
            s5_phases.setdefault(session.subject_id, Counter())
        previous_yawn = ""
        close_length = 0
        with session.labels_csv.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                action = (row.get("driver_actions") or "").strip()
                if action:
                    action_labels[action] += 1
                gaze_zone = (row.get("gaze_zone") or "").strip()
                if gaze_zone:
                    gaze_zones[gaze_zone] += 1

                if session.protocol != "s5":
                    continue
                phase = (row.get("eyes_state") or "undefined").strip().lower()
                eye_phases[phase] += 1
                s5_phases[session.subject_id][phase] += 1
                if phase == "close":
                    close_length += 1
                elif close_length:
                    close_runs.append(close_length)
                    s5_close_events[session.subject_id] += 1
                    s5_microsleeps[session.subject_id] += int(close_length >= 40)
                    close_length = 0

                yawn = (row.get("yawning") or "").strip()
                if yawn and yawn != previous_yawn:
                    yawn_events += 1
                    s5_yawn_events[session.subject_id] += 1
                previous_yawn = yawn
        if close_length:
            close_runs.append(close_length)
            s5_close_events[session.subject_id] += 1
            s5_microsleeps[session.subject_id] += int(close_length >= 40)

    subject_report = {
        subject: {
            "frames": s5_frames[subject],
            "eye_phases": dict(s5_phases[subject]),
            "yawn_events": s5_yawn_events[subject],
            "close_events": s5_close_events[subject],
            "microsleep_candidates_2s": s5_microsleeps[subject],
        }
        for subject in sorted(s5_frames)
    }

    return {
        "frames": sum(session.n_frames for session in sessions),
        "sessions": len(sessions),
        "subjects": len({session.subject_id for session in sessions}),
        "protocols": protocol_report,
        "eye_phases": dict(eye_phases),
        "yawn_events": yawn_events,
        "close_events": len(close_runs),
        "microsleep_candidates_2s": sum(length >= 40 for length in close_runs),
        "s5_by_subject": subject_report,
        "driver_actions": dict(action_labels.most_common()),
        "gaze_zones": dict(gaze_zones.most_common()),
    }
