"""Truth-free, transport-agnostic live inference orchestration.

This module is the boundary between a future CarSky camera/telemetry adapter
and the existing C1, C2, and C3 runtimes.  It deliberately contains no
``TripLoader``, dataset paths, prediction files, or CarSky VIDEO SDK shim.
Upstream code must first synchronize one road frame, one cabin frame, and one
ego sample; :class:`LiveInferenceSession` then processes that pair exactly
once.

The C1 runtime owns its trained cadence (currently stride 2 at a 20 Hz source).
The orchestrator therefore calls C1 for every accepted source frame and never
implements an additional stride or cache of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from .combined_replay import CombinedFramePrediction


EGO_FIELDS = (
    "speed_kmh",
    "longitudinal_accel",
    "lateral_accel",
)


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validated_session_id(value: object, *, field: str = "session_id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > 128:
        raise ValueError(f"{field} must be at most 128 characters")
    return value


def _source_timestamp(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("source_timestamp must be numeric")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_timestamp must be numeric") from exc
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("source_timestamp must be finite and non-negative")
    return timestamp


def _truth_free_ego(values: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError("ego must be a mapping")
    normalized: dict[str, float] = {}
    for field in EGO_FIELDS:
        if field not in values:
            raise ValueError(f"ego.{field} is required")
        try:
            number = float(values[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ego.{field} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"ego.{field} must be finite")
        normalized[field] = number
    # Whitelisting prevents unrelated dataset metadata or privileged labels
    # from crossing the live inference boundary by accident.
    return MappingProxyType(normalized)


def _owned_image(image: Any, *, field: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{field} must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError(f"{field} must use uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{field} must be a BGR HxWx3 image")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{field} dimensions must be positive")
    if not image.flags.c_contiguous:
        raise ValueError(f"{field} must be C-contiguous")
    # VIDEO transports commonly expose a view onto a reusable shared-memory
    # slot. Own the pixels before returning from the ingest call so a later
    # publisher frame cannot mutate an image while inference is reading it.
    owned = np.array(image, dtype=np.uint8, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _owned_images(
    road_bgr: Any, cabin_bgr: Any
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _owned_image(road_bgr, field="road_bgr"),
        _owned_image(cabin_bgr, field="cabin_bgr"),
    )


@dataclass(frozen=True)
class PairedCameraSample:
    """One already-synchronized pair produced by a transport adapter.

    ``source_sequence`` is the camera/broker sequence.  The latest-only slot
    records gaps caused by upstream backpressure, while the inference session
    rejects such a discontinuity until its temporal models are reset.  The
    adapter is responsible for pairing road and cabin images from the same
    source tick before constructing this object.
    """

    session_id: str
    source_sequence: int
    source_timestamp: float
    road_bgr: Any
    cabin_bgr: Any
    ego: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _validated_session_id(self.session_id),
        )
        object.__setattr__(
            self,
            "source_sequence",
            _non_negative_integer(
                self.source_sequence, field="source_sequence"
            ),
        )
        object.__setattr__(
            self, "source_timestamp", _source_timestamp(self.source_timestamp)
        )
        road, cabin = _owned_images(self.road_bgr, self.cabin_bgr)
        object.__setattr__(self, "road_bgr", road)
        object.__setattr__(self, "cabin_bgr", cabin)
        object.__setattr__(self, "ego", _truth_free_ego(self.ego))


@dataclass(frozen=True)
class LiveFrameInput:
    """Minimal truth-free input for one accepted live inference tick."""

    session_id: str
    frame_id: int
    source_timestamp: float
    road_bgr: Any
    cabin_bgr: Any
    ego: Mapping[str, float]
    source_sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _validated_session_id(self.session_id),
        )
        frame_id = _non_negative_integer(self.frame_id, field="frame_id")
        source_sequence = (
            frame_id
            if self.source_sequence is None
            else _non_negative_integer(
                self.source_sequence, field="source_sequence"
            )
        )
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "source_sequence", source_sequence)
        object.__setattr__(
            self, "source_timestamp", _source_timestamp(self.source_timestamp)
        )
        road, cabin = _owned_images(self.road_bgr, self.cabin_bgr)
        object.__setattr__(self, "road_bgr", road)
        object.__setattr__(self, "cabin_bgr", cabin)
        object.__setattr__(self, "ego", _truth_free_ego(self.ego))

    @property
    def timestamp(self) -> float:
        """Compatibility name consumed by C3 and combined predictions."""

        return self.source_timestamp

    def left(self) -> Any:
        """Return the in-memory road image without touching a dataset."""

        return self.road_bgr

    def driver(self) -> Any:
        """Return the in-memory cabin image without touching a dataset."""

        return self.cabin_bgr


@dataclass(frozen=True)
class LatestSlotStats:
    session_id: str | None
    offered: int
    emitted: int
    overwritten: int
    source_gaps: int
    pending: bool


class LatestPairedFrameSlot:
    """Bound live-source latency with a single latest-pair slot.

    Offering a newer pair while one is pending overwrites the stale pair.
    Emitted frames are re-indexed contiguously from zero while preserving the
    original ``source_sequence`` for gap/drop diagnostics.  This class does
    not pretend to pair independent VIDEO pins; the actual transport adapter
    must do that before calling :meth:`offer`.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._lock = Lock()
        self.reset(session_id)

    def reset(self, session_id: str | None = None) -> None:
        validated = (
            None
            if session_id is None
            else _validated_session_id(session_id)
        )
        with self._lock:
            self._session_id = validated
            self._pending: PairedCameraSample | None = None
            self._last_offered_sequence: int | None = None
            self._last_offered_timestamp: float | None = None
            self._next_frame_id = 0
            self._offered = 0
            self._emitted = 0
            self._overwritten = 0
            self._source_gaps = 0

    def offer(self, sample: PairedCameraSample) -> None:
        if not isinstance(sample, PairedCameraSample):
            raise TypeError("latest slot accepts PairedCameraSample")
        with self._lock:
            if self._session_id is None:
                raise RuntimeError("latest slot has no active session")
            if sample.session_id != self._session_id:
                raise ValueError("sample belongs to a stale or foreign session")
            if (
                self._last_offered_sequence is not None
                and sample.source_sequence <= self._last_offered_sequence
            ):
                raise ValueError("source_sequence must be strictly increasing")
            if (
                self._last_offered_timestamp is not None
                and sample.source_timestamp <= self._last_offered_timestamp
            ):
                raise ValueError("source timestamps must be strictly increasing")
            if self._last_offered_sequence is not None:
                self._source_gaps += max(
                    0,
                    sample.source_sequence - self._last_offered_sequence - 1,
                )
            if self._pending is not None:
                self._overwritten += 1
            self._pending = sample
            self._last_offered_sequence = sample.source_sequence
            self._last_offered_timestamp = sample.source_timestamp
            self._offered += 1

    def pop_latest(self) -> LiveFrameInput | None:
        with self._lock:
            sample = self._pending
            if sample is None:
                return None
            self._pending = None
            frame = LiveFrameInput(
                session_id=sample.session_id,
                frame_id=self._next_frame_id,
                source_sequence=sample.source_sequence,
                source_timestamp=sample.source_timestamp,
                road_bgr=sample.road_bgr,
                cabin_bgr=sample.cabin_bgr,
                ego=sample.ego,
            )
            self._next_frame_id += 1
            self._emitted += 1
            return frame

    @property
    def stats(self) -> LatestSlotStats:
        with self._lock:
            return LatestSlotStats(
                session_id=self._session_id,
                offered=self._offered,
                emitted=self._emitted,
                overwritten=self._overwritten,
                source_gaps=self._source_gaps,
                pending=self._pending is not None,
            )


class C1LiveProcessor(Protocol):
    def process_bgr(
        self,
        image: Any,
        *,
        frame_id: int,
        timestamp: float,
        ego: Mapping[str, Any] | None,
        target_count: int,
    ) -> Any: ...

    def reset(self) -> None: ...


class C2LiveProcessor(Protocol):
    def process_bgr(
        self,
        image: Any,
        *,
        frame_id: int,
        timestamp: float,
    ) -> Any: ...

    def reset(self) -> None: ...


class C3LiveProcessor(Protocol):
    def update(self, bundle: Any, *, predicted_ttc_s: object) -> Any: ...

    def reset(self) -> None: ...


class DriveQualityLiveProcessor(Protocol):
    def update(self, c3_frame: Any) -> Any: ...

    def reset(self) -> None: ...


class ContextualRiskLiveProcessor(Protocol):
    def evaluate(self, c1: Any, c2: Any) -> Any: ...


@dataclass(frozen=True)
class LiveSessionSnapshot:
    session_id: str | None
    active: bool
    faulted: bool
    processed_frames: int
    next_frame_id: int
    last_source_sequence: int | None
    last_source_timestamp: float | None


class LiveInferenceSession:
    """Run C1 + C2 + C3 once per synchronized live source frame."""

    def __init__(
        self,
        *,
        c1: C1LiveProcessor,
        c2: C2LiveProcessor,
        c3: C3LiveProcessor,
        drive_quality: DriveQualityLiveProcessor,
        contextual_risk: ContextualRiskLiveProcessor,
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.drive_quality = drive_quality
        self.contextual_risk = contextual_risk
        self._process_lock = Lock()
        self._session_id: str | None = None
        self._active = False
        self._faulted = False
        self._processed_frames = 0
        self._next_frame_id = 0
        self._last_source_sequence: int | None = None
        self._last_source_timestamp: float | None = None

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        return _validated_session_id(session_id)

    def _reset_processors(self) -> None:
        for processor in (
            self.c1,
            self.c2,
            self.c3,
            self.drive_quality,
            self.contextual_risk,
        ):
            reset = getattr(processor, "reset", None)
            if callable(reset):
                reset()

    def _activate(self, session_id: str) -> None:
        validated_session_id = self._validate_session_id(session_id)
        self._active = False
        self._faulted = False
        try:
            self._reset_processors()
        except Exception:
            self._faulted = True
            raise
        self._session_id = validated_session_id
        self._processed_frames = 0
        self._next_frame_id = 0
        self._last_source_sequence = None
        self._last_source_timestamp = None
        self._active = True

    def start_session(self, session_id: str) -> None:
        """Start a fresh session; all stateful processors are reset."""

        with self._process_lock:
            if self._active:
                raise RuntimeError("a live inference session is already active")
            self._activate(session_id)

    def reset_session(self, session_id: str) -> None:
        """Abandon current state and start a fresh session at frame zero."""

        with self._process_lock:
            self._activate(session_id)

    def end_session(self) -> LiveSessionSnapshot:
        """Stop accepting input without closing model resources."""

        with self._process_lock:
            self._active = False
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> LiveSessionSnapshot:
        return LiveSessionSnapshot(
            session_id=self._session_id,
            active=self._active,
            faulted=self._faulted,
            processed_frames=self._processed_frames,
            next_frame_id=self._next_frame_id,
            last_source_sequence=self._last_source_sequence,
            last_source_timestamp=self._last_source_timestamp,
        )

    @property
    def snapshot(self) -> LiveSessionSnapshot:
        with self._process_lock:
            return self._snapshot_unlocked()

    def _validate_frame(self, frame: LiveFrameInput) -> None:
        if not isinstance(frame, LiveFrameInput):
            raise TypeError("process accepts LiveFrameInput")
        if frame.session_id != self._session_id:
            raise ValueError("live frame belongs to a stale or foreign session")
        if frame.frame_id != self._next_frame_id:
            raise ValueError(
                "live frame IDs must be contiguous and start at zero: "
                f"received {frame.frame_id}, expected {self._next_frame_id}"
            )
        source_sequence = int(frame.source_sequence)
        if (
            self._last_source_sequence is not None
            and source_sequence != self._last_source_sequence + 1
        ):
            raise ValueError(
                "live source_sequence must be contiguous; a source gap "
                "requires a fresh session"
            )
        if (
            self._last_source_timestamp is not None
            and frame.source_timestamp <= self._last_source_timestamp
        ):
            raise ValueError("live source timestamps must be strictly increasing")

    def process(self, frame: LiveFrameInput) -> CombinedFramePrediction:
        """Run all challenge processors on one paired source tick.

        Any inference-time exception faults the session because one or more
        stateful models may already have advanced.  Call :meth:`reset_session`
        with a new source session before accepting further frames.
        """

        if not self._process_lock.acquire(blocking=False):
            raise RuntimeError("concurrent live inference is not supported")
        try:
            if not self._active:
                raise RuntimeError("no live inference session is active")
            if self._faulted:
                raise RuntimeError("live inference session is faulted; reset it")
            self._validate_frame(frame)
            try:
                collision = self.c1.process_bgr(
                    frame.road_bgr,
                    frame_id=frame.frame_id,
                    timestamp=frame.source_timestamp,
                    ego=frame.ego,
                    # No labels/targets cross the live boundary.  This value
                    # is unused by the checkpoint's selected feature order.
                    target_count=0,
                )
                driver = self.c2.process_bgr(
                    frame.cabin_bgr,
                    frame_id=frame.frame_id,
                    timestamp=frame.source_timestamp,
                )
                safe_score = self.c3.update(
                    frame, predicted_ttc_s=collision.predicted_ttc_s
                )
                drive_quality = self.drive_quality.update(safe_score)
                risk = self.contextual_risk.evaluate(collision, driver)
                prediction = CombinedFramePrediction(
                    frame_id=frame.frame_id,
                    timestamp=frame.source_timestamp,
                    c1=collision,
                    c2=driver,
                    c3=safe_score,
                    drive_quality=drive_quality,
                    contextual_risk=risk,
                    source_bundle=frame,
                )
            except Exception:
                self._faulted = True
                raise
            self._processed_frames += 1
            self._next_frame_id += 1
            self._last_source_sequence = int(frame.source_sequence)
            self._last_source_timestamp = frame.source_timestamp
            return prediction
        finally:
            self._process_lock.release()


class LiveInferenceController:
    """Fence ingress and model state with one shared session generation.

    The short controller lock only protects slot/session cutover. Model
    inference still runs without blocking producers from replacing the
    pending latest frame. If reset wins a race after a frame is popped, the
    session-id check rejects that old frame before any model advances.
    """

    def __init__(
        self,
        session: LiveInferenceSession,
        *,
        slot: LatestPairedFrameSlot | None = None,
    ) -> None:
        if not isinstance(session, LiveInferenceSession):
            raise TypeError("session must be a LiveInferenceSession")
        self.session = session
        self.slot = slot or LatestPairedFrameSlot()
        self._control_lock = Lock()

    def start_session(self, session_id: str) -> None:
        validated = _validated_session_id(session_id)
        with self._control_lock:
            self.slot.reset(None)
            self.session.start_session(validated)
            self.slot.reset(validated)

    def reset_session(self, session_id: str) -> None:
        validated = _validated_session_id(session_id)
        with self._control_lock:
            self.slot.reset(None)
            self.session.reset_session(validated)
            self.slot.reset(validated)

    def end_session(self) -> LiveSessionSnapshot:
        with self._control_lock:
            self.slot.reset(None)
            return self.session.end_session()

    def offer(self, sample: PairedCameraSample) -> None:
        with self._control_lock:
            self.slot.offer(sample)

    def process_latest(self) -> CombinedFramePrediction | None:
        with self._control_lock:
            frame = self.slot.pop_latest()
        if frame is None:
            return None
        return self.session.process(frame)


__all__ = [
    "C1LiveProcessor",
    "C2LiveProcessor",
    "C3LiveProcessor",
    "ContextualRiskLiveProcessor",
    "DriveQualityLiveProcessor",
    "EGO_FIELDS",
    "LatestPairedFrameSlot",
    "LatestSlotStats",
    "LiveFrameInput",
    "LiveInferenceController",
    "LiveInferenceSession",
    "LiveSessionSnapshot",
    "PairedCameraSample",
]
