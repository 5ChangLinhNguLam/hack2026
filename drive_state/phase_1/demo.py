"""Live replay demo: cabin video in, predicted driver state on screen.

This is the streaming counterpart to `predict.py`. The difference is not
cosmetic — `predict.py` decides frame `i` using a window centred on `i`, which
needs 3.75 s of future and is therefore only valid offline. Here the window is
**trailing**: it ends at the current frame, so nothing is used that a live feed
would not already have.

That costs accuracy: re-tuned for trailing windows the pipeline scores 88.2
against 96.2 centred. The loss splits into two distinct causes, and on the
worst trip the smaller one is the transition:

* **Warm-up, ~3.5 points.** The first `window_frames` frames are decided over a
  partial buffer. T01 opens in `distracted` and every one of its 90 warm-up
  frames is wrong; T04 loses 91 the same way. Dropping warm-up frames from
  scoring moves 88.2 to 91.7.
* **Transition lag, ~4.5 points.** After the driver's state changes the window
  still holds the old evidence, so the prediction catches up 29-68 frames
  later (T06 29, T01 64, T04 68).

Together those account for every misclassified frame on T01 and T04 — 182 and
159 respectively, with no scattered error in between. The single-state trips
lose almost nothing (T02 even gains).

Frame sourcing goes through `tripkit.TripReplayer`, which ships in this repo,
so the demo runs on the same replay path as the rest of the kit and inherits
its drift-free realtime pacing. The built-in loader remains as a fallback for
the one case tripkit cannot handle here -- see `_check_tripkit_layout`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from drive_state.phase_1.classifier import (
    ClassifierConfig,
    WindowFeatures,
    classify_window,
)
from drive_state.phase_1.features import FaceFeatureExtractor, FrameFeatures
from drive_state.phase_1.phone import DEFAULT_MODEL, create_phone_detector
from drive_state.phase_1.practice import BGRImage, imread_bgr, load_trip
from drive_state.vendor.models import (
    DetectionEvent,
    DriverState,
    FramePacket,
    ProcessedFrame,
    Severity,
)
from drive_state.vendor.risk import RiskScorer
from drive_state.vendor.overlay import draw_overlay

#: Thresholds re-fitted for a trailing window, by the same grid as `tuning.py`.
#: They differ from the offline defaults in the direction you would expect: a
#: shorter window (91 vs 151) to cut transition lag, and a stricter
#: `perclos_microsleep` (0.95 vs 0.80) because a trailing window reaches full
#: PERCLOS more slowly and a lower bar would fire during ordinary long blinks.
STREAMING_CONFIG = ClassifierConfig(
    window_frames=91,
    blink_closed=0.35,
    perclos_microsleep=0.90,
    perclos_drowsy=0.06,
    mar_yawning=0.35,
    mar_talking=0.20,
    phone_confidence=0.20,
    phone_distracted=0.50,
)

STATE_COLOURS: dict[str, tuple[int, int, int]] = {
    "alert": (120, 220, 120),
    "distracted": (80, 200, 250),
    "drowsy": (60, 160, 255),
    "yawning": (240, 190, 90),
    "microsleep": (80, 80, 255),
}


class StreamingClassifier:
    """Trailing-window classifier over a frame stream.

    Holds the last `window_frames` feature rows in a ring buffer and recomputes
    the window statistics on each update. At 91 frames that is a few
    microseconds per frame — far below the cost of landmarking — so there is no
    reason to complicate it with incremental updates.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or STREAMING_CONFIG
        self._buffer: deque[FrameFeatures] = deque(maxlen=self.config.window_frames)

    @property
    def filled(self) -> int:
        """Frames currently in the buffer, out of `window_frames`."""
        return len(self._buffer)

    @property
    def warming_up(self) -> bool:
        """True until the buffer holds a full window.

        Predictions before that are made over a short window and are markedly
        less reliable: the rates the rules threshold on (PERCLOS, MAR p75) are
        computed from few samples, and every trip whose opening state is not
        `alert` is misread for a while. Across the six Practice trips, dropping
        the warm-up frames moves the streaming score from 88.2 to 91.7, so this
        accounts for 3.5 of the 8 points between streaming and offline.

        It is not corrected for. A vehicle that has just started its camera
        genuinely does not know yet, and the submission needs a state for every
        frame regardless -- but a demo should say which frames these are rather
        than present a warm-up guess as a settled answer.
        """
        return len(self._buffer) < self.config.window_frames

    def update(self, features: FrameFeatures) -> tuple[str, WindowFeatures]:
        self._buffer.append(features)
        window = self._window()
        return classify_window(window, self.config), window

    def _window(self) -> WindowFeatures:
        rows = list(self._buffer)
        found = np.array([r.face_found for r in rows], dtype=float)
        n_seen = float(found.sum())
        current = rows[-1]
        if n_seen == 0:
            return WindowFeatures(current.frame_id, 0.0, 0.0, 0.0, 0.0)

        mask = found > 0
        blink = np.array([r.blink for r in rows])[mask]
        mar = np.array([r.mar for r in rows])[mask]
        jaw = np.array([r.jaw_open for r in rows])[mask]
        # Phone is counted over every frame in the buffer, face or not, matching
        # compute_window_features -- the driver looking down at a handset is
        # exactly when the face is lost.
        phone = np.array([r.phone_conf for r in rows])
        return WindowFeatures(
            frame_id=current.frame_id,
            perclos=float((blink > self.config.blink_closed).sum() / n_seen),
            mar_p75=float(np.percentile(mar, 75)),
            mouth_open_frac=float((jaw > 0.08).sum() / n_seen),
            face_ratio=n_seen / len(rows),
            phone_frac=float((phone >= self.config.phone_confidence).sum() / len(rows)),
        )


@dataclass(slots=True)
class DemoFrame:
    """One replayed frame, after inference."""

    frame_id: int
    timestamp: float
    image: npt.NDArray[np.uint8]  # cabin camera, BGR
    predicted: str
    truth: str | None
    window: WindowFeatures
    features: FrameFeatures
    latency_ms: float  # this frame only; spikes on the frames the detector runs
    mean_latency_ms: float = 0.0  # rolling mean, the number worth displaying
    warming_up: bool = False  # buffer not yet a full window; see StreamingClassifier
    buffer_filled: int = 0
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    face_bbox: tuple[int, int, int, int] | None = None


class TripkitLayoutError(RuntimeError):
    """tripkit cannot open this trip directory, for a reason we can name."""


def _check_tripkit_layout(trip_root: Path) -> None:
    """Refuse the one dataset layout tripkit mishandles.

    `TripLoader._resolve_json_path` prefers `<trip>.json` over `<trip>.json.gz`
    and tests it with `Path.exists()`, which is also true for a directory. Some
    unzip runs leave exactly that: a `T01-Sample.json/` directory holding a
    nested `T01-Sample.json`. tripkit then tries to open the directory and dies
    with a bare PermissionError several frames later.
    """
    stray = trip_root / f"{trip_root.name}.json"
    if stray.is_dir():
        raise TripkitLayoutError(
            f"{stray} is a directory, not a file -- an unzip artifact. tripkit resolves "
            f"<trip>.json before <trip>.json.gz and would try to open it. Fix it by "
            f"deleting that directory ({trip_root.name}.json.gz holds the same content), "
            f"or by moving {stray / trip_root.name}.json up one level."
        )


def _iter_tripkit(
    trip_root: Path, mode: str, speed: float, limit: int | None
) -> Iterator[tuple[int, float, npt.NDArray[np.uint8], str | None]]:
    from tripkit import TripLoader, TripReplayer

    _check_tripkit_layout(trip_root)
    loader = TripLoader(str(trip_root))
    end = loader.n_frames if limit is None else min(loader.n_frames, limit)
    for bundle in TripReplayer(loader, mode=mode, speed=speed, end=end):
        truth = ((bundle.gt or {}).get("driver") or {}).get("state")
        yield bundle.frame_id, bundle.timestamp, cast(BGRImage, bundle.driver()), truth


def _iter_builtin(
    trip_root: Path, mode: str, speed: float, limit: int | None
) -> Iterator[tuple[int, float, npt.NDArray[np.uint8], str | None]]:
    """Fallback source. Paces on an absolute schedule so a slow frame does not
    push every later frame back -- the same drift-free approach tripkit uses."""
    trip = load_trip(trip_root)
    truths = trip.states()
    paths = trip.frame_paths()
    if limit is not None:
        paths = paths[:limit]

    started = monotonic()
    for index, path in enumerate(paths):
        frame_id = int(path.stem.split("_")[-1])
        timestamp = frame_id / trip.fps
        if mode == "realtime":
            deadline = started + (index / trip.fps) / speed
            delay = deadline - monotonic()
            if delay > 0:
                from time import sleep

                sleep(delay)
        yield frame_id, timestamp, imread_bgr(path), truths.get(frame_id)


def iter_demo_frames(
    trip_root: str | Path,
    *,
    config: ClassifierConfig | None = None,
    mode: str = "realtime",
    speed: float = 1.0,
    limit: int | None = None,
    model_path: str | Path = "models/face_landmarker.task",
    phone_model: str | Path | None = DEFAULT_MODEL,
    phone_stride: int = 5,
    use_tripkit: bool | None = None,
) -> Iterator[DemoFrame]:
    """Replay a trip, landmarking and classifying each frame as it arrives.

    `use_tripkit=None` means "use it if it imports". Set it True to make a
    missing tripkit an error rather than a silent downgrade to the driver-only
    source. `phone_model=None` turns off phone detection, which is what makes
    distraction detectable at all -- see `phone.py`.
    """
    trip_root = Path(trip_root)
    resolved = config or STREAMING_CONFIG
    classifier = StreamingClassifier(resolved)
    # Per-frame latency alternates between ~8 ms and ~130 ms depending on
    # whether the strided detector ran, so displaying it raw would show a
    # number that is true of no typical frame. A short rolling mean covers at
    # least one full stride and reads as the rate the pipeline actually
    # sustains.
    recent_latency: deque[float] = deque(maxlen=max(2 * phone_stride, 20))
    phone_detector = create_phone_detector(
        phone_model, stride=phone_stride, confidence=resolved.phone_confidence
    )

    source = _iter_builtin
    if use_tripkit is not False:
        try:
            import tripkit  # noqa: F401

            _check_tripkit_layout(trip_root)
            source = _iter_tripkit
        except (ImportError, TripkitLayoutError):
            # Auto mode degrades to the built-in loader, which reads the .gz
            # directly and is unaffected by either problem. `--tripkit` means
            # the caller wants to know that tripkit specifically works, so let
            # it fail there rather than quietly succeed by another route.
            if use_tripkit:
                raise

    with FaceFeatureExtractor(model_path) as extractor:
        for frame_id, timestamp, image, truth in source(trip_root, mode, speed, limit):
            started = perf_counter()
            phone = (
                phone_detector.detect(image, frame_id, timestamp)
                if phone_detector is not None
                else None
            )
            features = extractor.extract(image, frame_id, timestamp, phone)
            predicted, window = classifier.update(features)
            latency_ms = (perf_counter() - started) * 1000
            recent_latency.append(latency_ms)
            yield DemoFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
                predicted=predicted,
                truth=truth,
                window=window,
                features=features,
                latency_ms=latency_ms,
                mean_latency_ms=sum(recent_latency) / len(recent_latency),
                warming_up=classifier.warming_up,
                buffer_filled=classifier.filled,
                landmarks=extractor.last_points,
                face_bbox=extractor.last_bbox,
            )


def _signals(frame: DemoFrame, config: ClassifierConfig) -> dict[str, float]:
    """Map window evidence onto the five bars the shared overlay draws.

    Each bar is the evidence scaled by the threshold that would fire it, so a
    full bar means "this rule is at its trigger point" rather than an abstract
    0-1 score. That makes the meters readable as *why* the state is what it is.
    """
    window = frame.window
    return {
        "eyes_closed": window.perclos,
        "drowsy": min(1.0, window.perclos / max(config.perclos_drowsy, 1e-6)),
        "yawning": min(1.0, window.mar_p75 / max(config.mar_yawning, 1e-6)),
        "distracted": min(1.0, window.phone_frac / max(config.phone_distracted, 1e-6)),
        "phone_use": frame.features.phone_conf,
    }


def _messages(frame: DemoFrame, config: ClassifierConfig) -> list[str]:
    """The bottom-left lines: which rule fired, in the words of the rule."""
    window = frame.window
    out: list[str] = []
    if frame.warming_up:
        out.append(
            f"Warming up: {frame.buffer_filled}/{config.window_frames} frames "
            "- decision made over a partial window"
        )
    if window.face_ratio < config.min_face_ratio:
        out.append("Face not trackable for most of the window")
    if window.perclos >= config.perclos_microsleep:
        out.append(f"Eyes closed for {window.perclos:.0%} of the window")
    elif window.perclos >= config.perclos_drowsy:
        out.append(f"PERCLOS {window.perclos:.0%} above drowsy threshold")
    if window.mar_p75 >= config.mar_yawning:
        out.append("Sustained wide mouth - yawn")
    if window.phone_frac >= config.phone_distracted:
        out.append(f"Phone visible in {window.phone_frac:.0%} of the window")
    elif frame.features.phone_conf > 0:
        out.append(f"Phone detected ({frame.features.phone_conf:.2f})")
    return out[:3]


def build_processed_frame(
    frame: DemoFrame,
    config: ClassifierConfig,
    *,
    landmarks: list[tuple[float, float]] | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> ProcessedFrame:
    """Adapt a challenge prediction to the shape the shared overlay renders.

    `DriverState` carries both vocabularies (`alert`/`microsleep` alongside the
    realtime pipeline's `attentive`/`eyes_closed`), so the challenge state goes
    through unmapped -- the banner says exactly what would be submitted, rather
    than a near-synonym.
    """
    objects: list[dict[str, Any]] = []
    if frame.features.phone_conf > 0:
        objects.append(
            {
                "label": "cell phone",
                "confidence": round(frame.features.phone_conf, 3),
                "bbox": (
                    int(frame.features.phone_x),
                    int(frame.features.phone_y),
                    int(frame.features.phone_w),
                    int(frame.features.phone_h),
                ),
                "provider": "onnx",
            }
        )

    signals = _signals(frame, config)
    events = [
        DetectionEvent(
            timestamp=frame.timestamp,
            frame_index=frame.frame_id,
            signal=frame.predicted,
            state=DriverState(frame.predicted),
            score=1.0,
            severity=Severity.WARNING,
            message=message,
        )
        for message in _messages(frame, config)
    ]

    return ProcessedFrame(
        packet=FramePacket(
            frame=frame.image, timestamp=frame.timestamp, frame_index=frame.frame_id
        ),
        state=DriverState(frame.predicted),
        # Risk is the shared scorer over the same signals the bars show, so the
        # number on the panel is consistent with the meters beside it.
        risk_score=RiskScorer().score(signals),
        signals=signals,
        events=events,
        latency_ms=frame.mean_latency_ms or frame.latency_ms,
        face_bbox=face_bbox,
        landmarks=landmarks or [],
        objects=objects,
    )


def draw_hud(
    frame: DemoFrame,
    *,
    config: ClassifierConfig | None = None,
    fps: float | None = None,
    landmarks: list[tuple[float, float]] | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    """The Inferensys overlay, plus the two things only this demo can show:
    the ground-truth state next to the prediction, and the warm-up state."""
    config = config or STREAMING_CONFIG
    processed = build_processed_frame(frame, config, landmarks=landmarks, face_bbox=face_bbox)
    canvas: npt.NDArray[np.uint8] = draw_overlay(processed)
    height, width = canvas.shape[:2]

    stamp = f"t={frame.timestamp:6.2f}s  #{frame.frame_id}"
    if fps is not None:
        stamp += f"   {fps:.0f} FPS"
    cv2.putText(
        canvas,
        stamp,
        (width - 232, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (208, 214, 214),
        1,
        cv2.LINE_AA,
    )

    if frame.truth is not None:
        correct = frame.truth == frame.predicted
        cv2.putText(
            canvas,
            f"truth  {frame.truth}",
            (28, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (98, 210, 130) if correct else (58, 58, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas
