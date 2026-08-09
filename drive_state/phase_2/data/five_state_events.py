"""Event-first references for balanced five-state temporal training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from .five_state_labels import IGNORE_INDEX, FiveState
from .five_state_standardization import StandardizedSession

DEFAULT_STATE_MASS: Mapping[FiveState, float] = {
    FiveState.ALERT: 0.10,
    FiveState.DROWSY: 0.25,
    FiveState.MICROSLEEP: 0.25,
    FiveState.YAWNING: 0.20,
    FiveState.DISTRACTION: 0.20,
}


@dataclass(frozen=True)
class StateEvent:
    session_index: int
    target: int
    event_id: str
    start_index: int
    stop_index: int


@dataclass(frozen=True)
class StateTargetRef:
    session_index: int
    target_index: int
    event_id: str


@dataclass(frozen=True)
class SamplingAudit:
    requested: int
    counts: Mapping[FiveState, int]
    missing_states: tuple[FiveState, ...]


@dataclass(frozen=True)
class SubjectEventSamplingAudit:
    requested: int
    actual: int
    counts: Mapping[FiveState, int]
    missing_states: tuple[FiveState, ...]
    windows_per_event: Mapping[str, int]
    subject_counts: Mapping[str, int]


def extract_state_events(
    sessions: Sequence[StandardizedSession],
) -> tuple[StateEvent, ...]:
    """Extract contiguous, globally unique valid events."""

    events: list[StateEvent] = []
    seen: set[str] = set()
    for session_index, session in enumerate(sessions):
        active_id = ""
        active_target = IGNORE_INDEX
        active_start = 0

        def finish(stop_index: int) -> None:
            nonlocal active_id
            if not active_id:
                return
            events.append(
                StateEvent(
                    session_index=session_index,
                    target=active_target,
                    event_id=active_id,
                    start_index=active_start,
                    stop_index=stop_index,
                )
            )
            active_id = ""

        for index, (target, event_id) in enumerate(
            zip(
                session.targets.targets,
                session.targets.event_ids,
                strict=True,
            )
        ):
            if target == IGNORE_INDEX:
                if event_id:
                    raise ValueError("ignored target cannot carry an event ID")
                finish(index)
                continue
            if not event_id:
                raise ValueError("valid target is missing an event ID")
            if event_id == active_id:
                if target != active_target:
                    raise ValueError("one event ID cannot contain multiple states")
                continue
            finish(index)
            if event_id in seen:
                raise ValueError("event ID is reused across disjoint runs")
            seen.add(event_id)
            active_id = event_id
            active_target = target
            active_start = index
        finish(len(session.targets.targets))
    return tuple(events)


def build_chronological_references(
    sessions: Sequence[StandardizedSession],
) -> tuple[StateTargetRef, ...]:
    """Return every valid target once in session/time order."""

    references: list[StateTargetRef] = []
    for session_index, session in enumerate(sessions):
        for target_index, (target, event_id) in enumerate(
            zip(
                session.targets.targets,
                session.targets.event_ids,
                strict=True,
            )
        ):
            if target == IGNORE_INDEX:
                continue
            if not event_id:
                raise ValueError("valid target is missing an event ID")
            references.append(
                StateTargetRef(session_index, target_index, event_id)
            )
    return tuple(references)


def _allocated_counts(
    *,
    samples: int,
    present_states: Sequence[FiveState],
    state_mass: Mapping[FiveState, float],
) -> dict[FiveState, int]:
    total_mass = sum(float(state_mass[state]) for state in present_states)
    if total_mass <= 0.0:
        raise ValueError("present state sampling mass must be positive")
    raw = {
        state: samples * float(state_mass[state]) / total_mass
        for state in present_states
    }
    counts = {state: math.floor(value) for state, value in raw.items()}
    remaining = samples - sum(counts.values())
    order = sorted(
        present_states,
        key=lambda state: (-(raw[state] - counts[state]), state.value),
    )
    for state in order[:remaining]:
        counts[state] += 1
    return counts


def build_event_balanced_references(
    sessions: Sequence[StandardizedSession],
    *,
    samples: int,
    seed: int,
    state_mass: Mapping[FiveState, float] = DEFAULT_STATE_MASS,
) -> tuple[tuple[StateTargetRef, ...], SamplingAudit]:
    """Sample state, then event, then timestamp with deterministic balancing."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if set(state_mass) != set(FiveState):
        raise ValueError("state mass must define all five states")
    if any(float(value) < 0.0 for value in state_mass.values()):
        raise ValueError("state sampling mass cannot be negative")
    events_by_state: dict[FiveState, list[StateEvent]] = defaultdict(list)
    for event in extract_state_events(sessions):
        events_by_state[FiveState(event.target)].append(event)
    present_states = tuple(state for state in FiveState if events_by_state[state])
    if not present_states:
        raise ValueError("event balancing requires at least one valid event")
    missing_states = tuple(state for state in FiveState if not events_by_state[state])
    counts = _allocated_counts(
        samples=samples,
        present_states=present_states,
        state_mass=state_mass,
    )

    rng = random.Random(seed)
    references: list[StateTargetRef] = []
    for state in present_states:
        events = list(events_by_state[state])
        rng.shuffle(events)
        for sample_index in range(counts[state]):
            if sample_index and sample_index % len(events) == 0:
                rng.shuffle(events)
            event = events[sample_index % len(events)]
            target_index = rng.randrange(event.start_index, event.stop_index)
            references.append(
                StateTargetRef(
                    session_index=event.session_index,
                    target_index=target_index,
                    event_id=event.event_id,
                )
            )
    rng.shuffle(references)
    audit = SamplingAudit(
        requested=samples,
        counts={state: counts.get(state, 0) for state in FiveState},
        missing_states=missing_states,
    )
    return tuple(references), audit


def build_subject_event_epoch_references(
    sessions: Sequence[StandardizedSession],
    *,
    samples: int,
    max_windows_per_event: int,
    seed: int,
    state_mass: Mapping[FiveState, float] = DEFAULT_STATE_MASS,
) -> tuple[tuple[StateTargetRef, ...], SubjectEventSamplingAudit]:
    """Balance subjects and states without repeatedly cloning rare events."""

    if samples <= 0 or max_windows_per_event <= 0:
        raise ValueError("samples and max windows per event must be positive")
    if set(state_mass) != set(FiveState):
        raise ValueError("state mass must define all five states")
    events = extract_state_events(sessions)
    events_by_state: dict[FiveState, list[StateEvent]] = defaultdict(list)
    for event in events:
        events_by_state[FiveState(event.target)].append(event)
    present = tuple(state for state in FiveState if events_by_state[state])
    if not present:
        raise ValueError("event sampling requires at least one valid event")
    desired = _allocated_counts(
        samples=samples,
        present_states=present,
        state_mass=state_mass,
    )
    rng = random.Random(seed)

    candidates: dict[FiveState, list[StateTargetRef]] = {}
    for state in present:
        by_subject: dict[str, list[StateTargetRef]] = defaultdict(list)
        for event in events_by_state[state]:
            offsets = list(range(event.start_index, event.stop_index))
            rng.shuffle(offsets)
            subject = sessions[event.session_index].subject
            by_subject[subject].extend(
                StateTargetRef(
                    session_index=event.session_index,
                    target_index=target_index,
                    event_id=event.event_id,
                )
                for target_index in offsets[:max_windows_per_event]
            )
        subject_order = sorted(by_subject)
        rng.shuffle(subject_order)
        for subject in subject_order:
            rng.shuffle(by_subject[subject])
        interleaved: list[StateTargetRef] = []
        while any(by_subject[subject] for subject in subject_order):
            for subject in subject_order:
                if by_subject[subject]:
                    interleaved.append(by_subject[subject].pop())
        candidates[state] = interleaved

    selected_by_state: dict[FiveState, list[StateTargetRef]] = {}
    offsets: dict[FiveState, int] = {}
    for state in present:
        count = min(desired[state], len(candidates[state]))
        selected_by_state[state] = candidates[state][:count]
        offsets[state] = count

    remaining = samples - sum(len(values) for values in selected_by_state.values())
    while remaining > 0:
        progressed = False
        for state in present:
            offset = offsets[state]
            if offset >= len(candidates[state]):
                continue
            selected_by_state[state].append(candidates[state][offset])
            offsets[state] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    references = [
        reference
        for state in present
        for reference in selected_by_state[state]
    ]
    rng.shuffle(references)
    windows_per_event: dict[str, int] = defaultdict(int)
    subject_counts: dict[str, int] = defaultdict(int)
    for reference in references:
        windows_per_event[reference.event_id] += 1
        subject_counts[sessions[reference.session_index].subject] += 1
    counts = {
        state: len(selected_by_state.get(state, ()))
        for state in FiveState
    }
    return (
        tuple(references),
        SubjectEventSamplingAudit(
            requested=samples,
            actual=len(references),
            counts=counts,
            missing_states=tuple(
                state for state in FiveState if state not in present
            ),
            windows_per_event=dict(windows_per_event),
            subject_counts=dict(subject_counts),
        ),
    )


__all__ = [
    "DEFAULT_STATE_MASS",
    "SamplingAudit",
    "StateEvent",
    "StateTargetRef",
    "SubjectEventSamplingAudit",
    "build_chronological_references",
    "build_event_balanced_references",
    "build_subject_event_epoch_references",
    "extract_state_events",
]
