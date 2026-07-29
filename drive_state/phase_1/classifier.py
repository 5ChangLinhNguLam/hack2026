"""Frame features -> one of the five Challenge 2 driver states.

Design notes, because the obvious approach does not work here:

**Distraction is the phone, seen directly.** The labels call every `distracted`
frame ``head_pose = "side"``, but measured yaw does not separate the classes at
all — inside T01 the *alert* half has the wider yaw spread (median -5.5 deg)
than the *distracted* half (+1.5 deg), and T04 is the same shape. The frames
come from DMD phone-call recordings where the driver keeps their eyes on the
road, so the `side` label describes the scenario, not anything the camera sees.
What the camera *does* see is the handset: a COCO `cell phone` detection covers
83-96 % of `distracted` frames against 0-20 % of the rest (see `phone.py`).
Mouth activity — the driver talking — is kept only as a fallback for when the
detector is off or misses, because on its own it manages roughly a 3:1 median
ratio where the phone box manages 20:1. Against a mouth-only pipeline retuned
from scratch, adding the phone is worth +3.2 composite offline (96.2 -> 99.4)
and +6.6 streaming (88.2 -> 94.8); per trip it takes T01 from 88.7 to 100.0
and T04 from 93.0 to 100.0.

**Decisions are made over a window, not a frame.** The organiser holds each
state for contiguous 300-frame (15 s) blocks, and the underlying evidence is
rate-shaped anyway — PERCLOS is a fraction of closed frames, a yawn is a
sustained wide mouth. A centred window turns noisy per-frame landmarks into
features that are flat across a block.

**Eye closure comes from the blendshape, not EAR.** `eyeBlink*` is a trained
output and holds up under head rotation; raw EAR on the same frames overlaps
badly between drowsy and alert (median 0.286 vs 0.343).

Rule order is by risk, so a frame that trips several conditions resolves to the
one that matters most.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from drive_state.phase_1.features import FrameFeatures


@dataclass(slots=True)
class ClassifierConfig:
    """Every tunable in one place.

    Defaults come from the `tuning.py` grid over the six Practice trips, with
    one adjustment: where the objective is flat across a knob's range, the
    default is the centre of the plateau rather than whichever endpoint the
    search happened to visit first. `perclos_microsleep` scores identically
    anywhere in 0.6-0.95 (eyes are shut for the whole of T05, so any threshold
    in that band is equivalent) and `mar_yawning` anywhere in 0.32-0.45.
    Taking the tie-break value would have read as a fitted number when it is
    really an artifact of iteration order.

    `blink_closed` is the fragile one: 96.2 at 0.25 against 86.9 at 0.20 and
    91.7 at 0.30. It is the threshold most in need of more subjects.
    """

    window_frames: int = 181  # ~9 s at 20 fps, centred

    # Eye closure: `blink` blendshape above this counts the frame as closed.
    blink_closed: float = 0.25
    # Fraction of closed frames in the window (PERCLOS).
    perclos_microsleep: float = 0.80
    perclos_drowsy: float = 0.06

    # Mouth: 75th-percentile MAR over the window.
    mar_yawning: float = 0.35
    mar_talking: float = 0.20

    # Phone: a detection at or above `phone_confidence` marks the frame, and
    # `phone_distracted` is the fraction of the window that must be marked.
    phone_confidence: float = 0.05
    phone_distracted: float = 0.50

    # Frames with no face are ignored inside a window rather than treated as
    # closed eyes; a long blackout falls back to `no_face_state`.
    min_face_ratio: float = 0.4
    no_face_state: str = "distracted"


#: The knobs the tuner sweeps and the CLI serialises. `min_face_ratio` and
#: `no_face_state` are excluded on purpose: they govern the tracking-failure
#: fallback, which fires on 0.6 % of Practice frames -- far too few to fit on.
TUNABLE_KNOBS: tuple[str, ...] = (
    "window_frames",
    "blink_closed",
    "perclos_microsleep",
    "perclos_drowsy",
    "mar_yawning",
    "mar_talking",
    "phone_confidence",
    "phone_distracted",
)


@dataclass(slots=True)
class WindowFeatures:
    """What the rules actually read, one row per frame."""

    frame_id: int
    perclos: float  # fraction of window with eyes closed
    mar_p75: float  # 75th-percentile mouth aspect ratio
    mouth_open_frac: float  # fraction of window with jaw clearly open
    face_ratio: float  # fraction of window where a face was found
    phone_frac: float = 0.0  # fraction of window with a phone detected


def compute_window_features(
    rows: Sequence[FrameFeatures],
    config: ClassifierConfig | None = None,
    *,
    trailing: bool = False,
) -> list[WindowFeatures]:
    """Aggregate per-frame features over a window.

    Centred by default: frame `i` is decided using the `window_frames // 2`
    frames on either side. That is right for scoring a complete trip, and
    impossible live — it needs 3.75 s of future.

    `trailing=True` uses only frames up to and including `i`, which is what a
    stream can supply. It costs accuracy at every state transition, because the
    window still holds the previous state for `window_frames` frames after the
    change. See `demo.py`.

    Windows are truncated at the sequence start rather than padded, so the first
    frames are computed over fewer samples. That is preferable to padding with a
    fabricated value, which would bias the very frames where a trip's state is
    established.
    """
    config = config or ClassifierConfig()
    if not rows:
        return []

    found = np.array([r.face_found for r in rows], dtype=float)
    blink = np.array([r.blink for r in rows], dtype=float)
    mar = np.array([r.mar for r in rows], dtype=float)
    jaw = np.array([r.jaw_open for r in rows], dtype=float)

    closed = ((blink > config.blink_closed) & (found > 0)).astype(float)
    open_mouth = ((jaw > 0.08) & (found > 0)).astype(float)
    # Phone presence is independent of face tracking on purpose: looking down at
    # a handset is exactly when the face is lost.
    phone = np.array([r.phone_conf for r in rows], dtype=float)
    has_phone = (phone >= config.phone_confidence).astype(float)

    n = len(rows)
    if trailing:
        lo = np.maximum(0, np.arange(n) - config.window_frames + 1)
        hi = np.arange(n) + 1
    else:
        half = config.window_frames // 2
        lo = np.maximum(0, np.arange(n) - half)
        hi = np.minimum(n, np.arange(n) + half + 1)

    # Rate features are window sums, so a prefix sum gives all of them in one
    # pass instead of re-summing each window -- this runs inside the tuner's
    # grid search, where it is the hot loop.
    def windowed_sum(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        result: npt.NDArray[np.float64] = cumulative[hi] - cumulative[lo]
        return result

    n_seen = windowed_sum(found)
    safe_seen = np.where(n_seen > 0, n_seen, 1.0)
    perclos = windowed_sum(closed) / safe_seen
    mouth_open_frac = windowed_sum(open_mouth) / safe_seen
    face_ratio = n_seen / (hi - lo).astype(float)
    # Denominator is every frame in the window, not just the ones with a face.
    phone_frac = windowed_sum(has_phone) / (hi - lo).astype(float)

    # A percentile has no prefix-sum shortcut; frames with no face are dropped
    # rather than contributing a zero MAR that would drag the quantile down.
    mar_masked = np.where(found > 0, mar, np.nan)
    mar_p75 = np.empty(n)
    for index in range(n):
        window = mar_masked[lo[index] : hi[index]]
        mar_p75[index] = 0.0 if np.all(np.isnan(window)) else np.nanpercentile(window, 75)

    return [
        WindowFeatures(
            frame_id=row.frame_id,
            perclos=float(perclos[i]) if n_seen[i] else 0.0,
            mar_p75=float(mar_p75[i]) if n_seen[i] else 0.0,
            mouth_open_frac=float(mouth_open_frac[i]) if n_seen[i] else 0.0,
            face_ratio=float(face_ratio[i]),
            phone_frac=float(phone_frac[i]),
        )
        for i, row in enumerate(rows)
    ]


def classify_window(window: WindowFeatures, config: ClassifierConfig | None = None) -> str:
    """One window -> one state. Ordered by risk; first match wins."""
    config = config or ClassifierConfig()

    if window.face_ratio < config.min_face_ratio:
        return config.no_face_state
    # Eyes shut for essentially the whole window: nothing else can outrank this.
    if window.perclos >= config.perclos_microsleep:
        return "microsleep"
    # A yawn is checked before drowsiness because yawning also squints the eyes
    # (median PERCLOS 0.36 on T03), which would otherwise read as drowsy.
    if window.mar_p75 >= config.mar_yawning:
        return "yawning"
    # A visible phone is checked before drowsiness. It is the only direct
    # evidence of distraction in the pipeline, and a driver on a handset who is
    # also blinking heavily is still, first, on a handset.
    if window.phone_frac >= config.phone_distracted:
        return "distracted"
    if window.perclos >= config.perclos_drowsy:
        return "drowsy"
    # Fallback when the detector is off or missed it: mouth moving, eyes open,
    # no yawn -> talking. Much weaker than the phone box, and the reason
    # `mar_talking` sits where it does.
    if window.mar_p75 >= config.mar_talking:
        return "distracted"
    return "alert"


def state_confidences(
    window: WindowFeatures, config: ClassifierConfig | None = None
) -> dict[str, float]:
    """How strongly each of the five states' evidence is satisfied, 0-1.

    These are **not** probabilities. The classifier is a rule cascade, not a
    model with a softmax, so there is no distribution to report. Each value is
    that state's evidence as a fraction of the threshold that would fire it,
    clipped at 1.

    Two consequences worth knowing before reading the bars:

    * several can sit at 1.0 at once — a yawn also closes the eyes, so
      `yawning` and `drowsy` both saturate across T03 — and which one wins is
      the risk ordering in :func:`classify_window`, not the largest value. The
      reported state is therefore not always the tallest bar.
    * `alert` is the leftover. It is high exactly when nothing else has
      evidence, which is what the rule cascade means by falling through.
    """
    config = config or ClassifierConfig()

    def ratio(value: float, threshold: float) -> float:
        return min(1.0, value / threshold) if threshold > 0 else 0.0

    microsleep = ratio(window.perclos, config.perclos_microsleep)
    yawning = ratio(window.mar_p75, config.mar_yawning)
    distracted = max(
        ratio(window.phone_frac, config.phone_distracted),
        ratio(window.mar_p75, config.mar_talking),
    )
    drowsy = ratio(window.perclos, config.perclos_drowsy)
    return {
        "alert": max(0.0, 1.0 - max(microsleep, yawning, distracted, drowsy)),
        "drowsy": drowsy,
        "yawning": yawning,
        "distracted": distracted,
        "microsleep": microsleep,
    }


def classify_windows(
    windows: Sequence[WindowFeatures], config: ClassifierConfig | None = None
) -> dict[int, str]:
    """Decide over windows that were computed earlier.

    Split out from :func:`classify_sequence` so the tuner can compute windows
    once per (`window_frames`, `blink_closed`) pair and reuse them across every
    combination of the four decision thresholds.
    """
    config = config or ClassifierConfig()
    return {w.frame_id: classify_window(w, config) for w in windows}


def classify_sequence(
    rows: Sequence[FrameFeatures], config: ClassifierConfig | None = None
) -> list[str]:
    """Full trip: per-frame features -> one state per frame."""
    config = config or ClassifierConfig()
    return [classify_window(w, config) for w in compute_window_features(rows, config)]


def predict_trip(
    rows: Sequence[FrameFeatures], config: ClassifierConfig | None = None
) -> dict[int, str]:
    states = classify_sequence(rows, config)
    return {row.frame_id: state for row, state in zip(rows, states, strict=True)}
