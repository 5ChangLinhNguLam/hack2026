"""Reproducible real-video validation for all five C2 live states.

The unit tests validate temporal transitions with synthetic signals.  This
script adds a stronger integration gate: it decodes annotated DMD RGB video,
runs the same pretrained models and temporal state machine as the live demo,
and checks that curated behaviour segments produce every C2 state.  This is a
functional regression gate on one unrestricted DMD subject, not a claim of
cross-subject accuracy.  It never reads leaderboard labels or retrieval hashes.

Usage:
    python c2/validate_live_demo.py --dmd-root C:\\DMD\\dmd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from .live_detector import (
        CellPhoneDetector,
        FaceSignalExtractor,
        LiveConfig,
        TemporalStateMachine,
    )
except ImportError:
    from live_detector import (
        CellPhoneDetector,
        FaceSignalExtractor,
        LiveConfig,
        TemporalStateMachine,
    )


@dataclass(frozen=True, slots=True)
class SegmentCase:
    name: str
    relative_session: str
    start_seconds: float
    duration_seconds: float
    expected_min_frames: dict[str, int]
    phone_enabled: bool = False


@dataclass(slots=True)
class CaseResult:
    name: str
    passed: bool
    video: str
    start_seconds: float
    frames: int
    wall_fps: float
    face_found: float
    face_mean_ms: float
    phone_mean_ms: float | None
    state_frames: dict[str, int]
    events: list[tuple[float, str, str]]
    failures: list[str]


# These intervals are taken from the public OpenLABEL files for unrestricted
# DMD subject gA/1.  The case starts include two neutral seconds for per-person
# calibration before the target action.
CASES = (
    SegmentCase(
        name="safe-alert",
        relative_session="gA/1/s2",
        start_seconds=0.0,
        duration_seconds=6.0,
        expected_min_frames={"alert": 60},
    ),
    SegmentCase(
        name="phone-distraction",
        relative_session="gA/1/s2",
        start_seconds=48.0,
        duration_seconds=9.0,
        expected_min_frames={"distracted": 30},
        phone_enabled=True,
    ),
    SegmentCase(
        name="yawn",
        relative_session="gA/1/s5",
        start_seconds=43.0,
        duration_seconds=8.0,
        expected_min_frames={"yawning": 30},
    ),
    SegmentCase(
        name="long-eye-closure",
        relative_session="gA/1/s5",
        start_seconds=66.0,
        duration_seconds=8.5,
        expected_min_frames={"microsleep": 10, "drowsy": 20},
    ),
)


def _video_for(root: Path, relative_session: str) -> Path:
    session = root.joinpath(*relative_session.split("/"))
    matches = sorted(session.glob("*_rgb_face.mp4"))
    if not matches:
        raise FileNotFoundError(f"no RGB face video under {session}")
    return matches[0]


def _run_case(root: Path, case: SegmentCase, phone_every: int) -> CaseResult:
    video = _video_for(root, case.relative_session)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    capture.set(cv2.CAP_PROP_POS_MSEC, case.start_seconds * 1000.0)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1.0:
        fps = 29.76
    requested_frames = int(round(case.duration_seconds * fps))

    config = LiveConfig(calibration_seconds=2.0)
    machine = TemporalStateMachine(config)
    face_model = FaceSignalExtractor()
    phone_model = CellPhoneDetector() if case.phone_enabled else None
    states: Counter[str] = Counter()
    events: list[tuple[float, str, str]] = []
    last_state: str | None = None
    face_frames = 0
    face_ms = 0.0
    phone_runs = 0
    phone_ms = 0.0
    processed = 0
    started = time.perf_counter()

    try:
        for frame_index in range(requested_frames):
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            face = face_model.process(frame, int(round(timestamp * 1000.0)))
            phone_score = None
            if phone_model is not None and frame_index % max(1, phone_every) == 0:
                phone = phone_model.process(frame)
                phone_score = phone.score
                phone_runs += 1
                phone_ms += phone.inference_ms
            snapshot = machine.update(timestamp, face, phone_score)
            processed += 1
            face_frames += int(face.face_found)
            face_ms += face.inference_ms
            if snapshot.calibrated:
                states[snapshot.state] += 1
                if snapshot.state != last_state:
                    events.append(
                        (round(timestamp, 2), snapshot.state, snapshot.reason)
                    )
                    last_state = snapshot.state
    finally:
        capture.release()
        face_model.close()
        if phone_model is not None:
            phone_model.close()

    elapsed = max(time.perf_counter() - started, 1e-9)
    failures = [
        f"{state}: observed {states[state]} < required {minimum}"
        for state, minimum in case.expected_min_frames.items()
        if states[state] < minimum
    ]
    if processed < requested_frames * 0.95:
        failures.append(
            f"decoded only {processed}/{requested_frames} requested frames"
        )
    if processed and face_frames / processed < 0.90:
        failures.append(
            f"face coverage {face_frames / processed:.1%} is below 90%"
        )
    return CaseResult(
        name=case.name,
        passed=not failures,
        video=str(video),
        start_seconds=case.start_seconds,
        frames=processed,
        wall_fps=round(processed / elapsed, 1),
        face_found=round(face_frames / processed, 4) if processed else 0.0,
        face_mean_ms=round(face_ms / processed, 2) if processed else 0.0,
        phone_mean_ms=(
            round(phone_ms / phone_runs, 2) if phone_runs else None
        ),
        state_frames=dict(states),
        events=events,
        failures=failures,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dmd-root",
        type=Path,
        default=Path(r"C:\DMD\dmd"),
        help=r"extracted DMD root (default C:\DMD\dmd)",
    )
    parser.add_argument(
        "--phone-every",
        type=int,
        default=5,
        help="run phone detector every N frames",
    )
    parser.add_argument("--json-out", type=Path, help="optional JSON report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.dmd_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: DMD root does not exist: {root}", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for case in CASES:
        print(f"[RUN] {case.name}", flush=True)
        try:
            result = _run_case(root, case, args.phone_every)
        except Exception as exc:
            print(f"[FAIL] {case.name}: {exc}", file=sys.stderr)
            return 2
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {case.name}: states={result.state_frames} "
            f"fps={result.wall_fps} face={result.face_found:.1%}",
            flush=True,
        )
        for event_time, state, reason in result.events:
            print(f"       {event_time:5.2f}s {state:12s} {reason}")
        for failure in result.failures:
            print(f"       ERROR: {failure}", file=sys.stderr)

    report = {
        "passed": all(result.passed for result in results),
        "states_observed": sorted(
            {
                state
                for result in results
                for state, count in result.state_frames.items()
                if count
            }
        ),
        "cases": [asdict(result) for result in results],
    }
    if args.json_out:
        destination = args.json_out.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report: {destination}")
    print(
        f"OVERALL: {'PASS' if report['passed'] else 'FAIL'} "
        f"states={report['states_observed']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
