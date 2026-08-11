"""Truth-free, transport-agnostic live inference orchestration.

This module is the boundary between a future CarSky camera/telemetry adapter
and the existing C1, C2, and C3 runtimes.  It deliberately contains no
``TripLoader``, dataset paths, prediction files, or CarSky VIDEO SDK shim.
Road, cabin, and ego adapters submit independently keyed envelopes to
:class:`LiveInputSynchronizer`; only an exact-key, bounded-skew tick can reach
:class:`LiveInferenceSession`. The older already-paired API remains available
for room-local adapters.

The C1 runtime owns its trained cadence (currently stride 2 at a 20 Hz source).
The orchestrator therefore calls C1 for every accepted source frame and never
implements an additional stride or cache of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

MAX_SOURCE_SKEW_MS = 25
DEFAULT_COHERENT_DWELL_TICKS = 2
MAX_RTP_TIMESTAMP = (1 << 32) - 1


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


def _validate_image(image: Any, *, field: str) -> np.ndarray:
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
    return image


def _owned_image(image: Any, *, field: str) -> np.ndarray:
    image = _validate_image(image, field=field)
    # VIDEO transports commonly expose a view onto a reusable shared-memory
    # slot. Own the pixels before returning from the ingest call so a later
    # publisher frame cannot mutate an image while inference is reading it.
    owned = np.array(image, dtype=np.uint8, order="C", copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class _OwnedImageReference:
    """Internal proof that pixels were copied at the decoder boundary."""

    value: np.ndarray


def _input_image(image: Any, *, field: str) -> np.ndarray:
    """Copy legacy inputs, but preserve an explicitly owned ingest reference."""

    if isinstance(image, _OwnedImageReference):
        referenced = _validate_image(image.value, field=field)
        if referenced.flags.writeable:
            raise ValueError(f"{field} owned reference must be read-only")
        return referenced
    return _owned_image(image, field=field)


def _owned_images(
    road_bgr: Any, cabin_bgr: Any
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _owned_image(road_bgr, field="road_bgr"),
        _owned_image(cabin_bgr, field="cabin_bgr"),
    )


def _input_images(
    road_bgr: Any, cabin_bgr: Any
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _input_image(road_bgr, field="road_bgr"),
        _input_image(cabin_bgr, field="cabin_bgr"),
    )


@dataclass(frozen=True)
class SourceTickKey:
    """Exact identity shared by road, cabin, and ego for one source tick."""

    session_id: str
    generation: int
    source_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _validated_session_id(self.session_id))
        object.__setattr__(
            self,
            "generation",
            _non_negative_integer(self.generation, field="generation"),
        )
        object.__setattr__(
            self,
            "source_sequence",
            _non_negative_integer(
                self.source_sequence, field="source_sequence"
            ),
        )


@dataclass(frozen=True)
class RtpFrameIdentity:
    """Explicit RTP-to-metadata mapping supplied by the media adapter.

    Decoder callback order is deliberately absent from this contract.  The
    adapter must look up the RTP timestamp in ingress metadata and attach the
    resulting :class:`SourceTickKey`; the synchronizer only trusts that key.
    """

    rtp_timestamp: int
    metadata_key: SourceTickKey

    def __post_init__(self) -> None:
        timestamp = _non_negative_integer(
            self.rtp_timestamp, field="rtp_timestamp"
        )
        if timestamp > MAX_RTP_TIMESTAMP:
            raise ValueError("rtp_timestamp must fit an unsigned 32-bit value")
        if not isinstance(self.metadata_key, SourceTickKey):
            raise TypeError("metadata_key must be a SourceTickKey")
        object.__setattr__(self, "rtp_timestamp", timestamp)


def _capture_timestamp_ms(value: object) -> int:
    return _non_negative_integer(value, field="capture_timestamp_ms")


@dataclass(frozen=True)
class RoadFrameEnvelope:
    """One decoded road frame copied once from its transport-owned buffer."""

    identity: RtpFrameIdentity
    capture_timestamp_ms: int
    bgr: Any

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RtpFrameIdentity):
            raise TypeError("identity must be an RtpFrameIdentity")
        object.__setattr__(
            self,
            "capture_timestamp_ms",
            _capture_timestamp_ms(self.capture_timestamp_ms),
        )
        object.__setattr__(self, "bgr", _owned_image(self.bgr, field="road_bgr"))

    @property
    def key(self) -> SourceTickKey:
        return self.identity.metadata_key


@dataclass(frozen=True)
class CabinFrameEnvelope:
    """One decoded cabin frame copied once from its transport-owned buffer."""

    identity: RtpFrameIdentity
    capture_timestamp_ms: int
    bgr: Any

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RtpFrameIdentity):
            raise TypeError("identity must be an RtpFrameIdentity")
        object.__setattr__(
            self,
            "capture_timestamp_ms",
            _capture_timestamp_ms(self.capture_timestamp_ms),
        )
        object.__setattr__(
            self, "bgr", _owned_image(self.bgr, field="cabin_bgr")
        )

    @property
    def key(self) -> SourceTickKey:
        return self.identity.metadata_key


@dataclass(frozen=True)
class EgoTelemetryEnvelope:
    """Truth-free ego telemetry explicitly keyed to one source tick."""

    key: SourceTickKey
    capture_timestamp_ms: int
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceTickKey):
            raise TypeError("key must be a SourceTickKey")
        object.__setattr__(
            self,
            "capture_timestamp_ms",
            _capture_timestamp_ms(self.capture_timestamp_ms),
        )
        object.__setattr__(self, "values", _truth_free_ego(self.values))


@dataclass(frozen=True)
class SynchronizedSourceTick:
    """One exact-key, bounded-skew road/cabin/ego source tick."""

    key: SourceTickKey
    road: RoadFrameEnvelope
    cabin: CabinFrameEnvelope
    ego: EgoTelemetryEnvelope
    capture_timestamp_ms: int = field(init=False)
    component_capture_timestamp_ms: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceTickKey):
            raise TypeError("key must be a SourceTickKey")
        for name, envelope in (
            ("road", self.road),
            ("cabin", self.cabin),
            ("ego", self.ego),
        ):
            if getattr(envelope, "key", None) != self.key:
                raise ValueError(f"{name} key does not match synchronized tick")
        timestamps = {
            "road": self.road.capture_timestamp_ms,
            "cabin": self.cabin.capture_timestamp_ms,
            "ego": self.ego.capture_timestamp_ms,
        }
        skew = max(timestamps.values()) - min(timestamps.values())
        if skew > MAX_SOURCE_SKEW_MS:
            raise ValueError(
                f"source capture skew {skew} ms exceeds {MAX_SOURCE_SKEW_MS} ms"
            )
        object.__setattr__(self, "capture_timestamp_ms", max(timestamps.values()))
        object.__setattr__(
            self,
            "component_capture_timestamp_ms",
            MappingProxyType(timestamps),
        )

    @property
    def source_timestamp(self) -> float:
        return self.capture_timestamp_ms / 1_000.0

    def to_live_frame(self, *, frame_id: int) -> "LiveFrameInput":
        """Create a model input using the already-owned pixel references."""

        return LiveFrameInput(
            session_id=self.key.session_id,
            generation=self.key.generation,
            frame_id=frame_id,
            source_sequence=self.key.source_sequence,
            source_timestamp=self.source_timestamp,
            road_bgr=_OwnedImageReference(self.road.bgr),
            cabin_bgr=_OwnedImageReference(self.cabin.bgr),
            ego=self.ego.values,
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
    generation: int = 0

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
            self,
            "generation",
            _non_negative_integer(self.generation, field="generation"),
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
    generation: int = 0

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
            self,
            "generation",
            _non_negative_integer(self.generation, field="generation"),
        )
        object.__setattr__(
            self, "source_timestamp", _source_timestamp(self.source_timestamp)
        )
        road, cabin = _input_images(self.road_bgr, self.cabin_bgr)
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
class SynchronizerOfferResult:
    disposition: str
    accepted: bool
    reset_required: bool
    tick: SynchronizedSourceTick | None = None


@dataclass(frozen=True)
class LiveSynchronizerStats:
    session_id: str
    generation: int
    offered: int
    accepted_components: int
    paired_ticks: int
    dropped_duplicate: int
    dropped_reordered: int
    dropped_stale: int
    dropped_future: int
    discontinuities: int
    pending_key: SourceTickKey | None
    faulted: bool


class LiveInputSynchronizer:
    """Capacity-one exact-key synchronizer for independent live inputs.

    Only one incomplete source key may be resident. A newer component before
    completion is a backlog overwrite, not an invitation to accumulate a
    second key. Duplicate/reordered/foreign inputs are dropped. A gap,
    overwrite, capture-time regression, or skew violation requests a model
    generation reset before any later tick can reach inference.
    """

    _KINDS = ("road", "cabin", "ego")

    def __init__(
        self,
        session_id: str,
        generation: int,
        *,
        max_skew_ms: int = MAX_SOURCE_SKEW_MS,
    ) -> None:
        skew = _non_negative_integer(max_skew_ms, field="max_skew_ms")
        if skew > MAX_SOURCE_SKEW_MS:
            raise ValueError(
                f"max_skew_ms cannot exceed {MAX_SOURCE_SKEW_MS}"
            )
        self.max_skew_ms = skew
        self._lock = Lock()
        self._offered = 0
        self._accepted_components = 0
        self._paired_ticks = 0
        self._dropped_duplicate = 0
        self._dropped_reordered = 0
        self._dropped_stale = 0
        self._dropped_future = 0
        self._discontinuities = 0
        self.reset(session_id, generation)

    def reset(self, session_id: str, generation: int) -> None:
        validated_session = _validated_session_id(session_id)
        validated_generation = _non_negative_integer(
            generation, field="generation"
        )
        with self._lock:
            self._session_id = validated_session
            self._generation = validated_generation
            self._pending_key: SourceTickKey | None = None
            self._pending: dict[str, Any] = {}
            self._last_seen: dict[str, int | None] = {
                kind: None for kind in self._KINDS
            }
            self._last_complete_sequence: int | None = None
            self._last_capture_timestamp_ms: int | None = None
            self._faulted = False

    def _drop(self, disposition: str) -> SynchronizerOfferResult:
        if disposition == "DUPLICATE":
            self._dropped_duplicate += 1
        elif disposition == "REORDERED":
            self._dropped_reordered += 1
        elif disposition in {"STALE_SESSION", "STALE_GENERATION"}:
            self._dropped_stale += 1
        elif disposition == "FUTURE_GENERATION":
            self._dropped_future += 1
        return SynchronizerOfferResult(disposition, False, False)

    def _discontinuity(self, disposition: str) -> SynchronizerOfferResult:
        self._discontinuities += 1
        self._faulted = True
        self._pending_key = None
        self._pending.clear()
        return SynchronizerOfferResult(disposition, False, True)

    def _offer(self, kind: str, envelope: Any) -> SynchronizerOfferResult:
        key = envelope.key
        with self._lock:
            self._offered += 1
            if self._faulted:
                return SynchronizerOfferResult(
                    "RESET_REQUIRED", False, True
                )
            if key.session_id != self._session_id:
                return self._drop("STALE_SESSION")
            if key.generation < self._generation:
                return self._drop("STALE_GENERATION")
            if key.generation > self._generation:
                return self._drop("FUTURE_GENERATION")

            last_seen = self._last_seen[kind]
            if last_seen is not None and key.source_sequence <= last_seen:
                return self._drop(
                    "DUPLICATE"
                    if key.source_sequence == last_seen
                    else "REORDERED"
                )

            if self._pending_key is None:
                if self._last_complete_sequence is not None:
                    if key.source_sequence <= self._last_complete_sequence:
                        return self._drop("REORDERED")
                    if key.source_sequence != self._last_complete_sequence + 1:
                        return self._discontinuity("SOURCE_GAP")
                self._pending_key = key
            elif key != self._pending_key:
                if key.source_sequence < self._pending_key.source_sequence:
                    return self._drop("REORDERED")
                return self._discontinuity("BACKLOG_OVERWRITE")

            if kind in self._pending:
                return self._drop("DUPLICATE")
            self._pending[kind] = envelope
            self._last_seen[kind] = key.source_sequence
            self._accepted_components += 1
            if len(self._pending) != len(self._KINDS):
                return SynchronizerOfferResult("PENDING", True, False)

            road = self._pending["road"]
            cabin = self._pending["cabin"]
            ego = self._pending["ego"]
            timestamps = (
                road.capture_timestamp_ms,
                cabin.capture_timestamp_ms,
                ego.capture_timestamp_ms,
            )
            if max(timestamps) - min(timestamps) > self.max_skew_ms:
                return self._discontinuity("CAPTURE_SKEW")
            capture_timestamp_ms = max(timestamps)
            if (
                self._last_capture_timestamp_ms is not None
                and capture_timestamp_ms <= self._last_capture_timestamp_ms
            ):
                return self._discontinuity("CAPTURE_REORDER")

            tick = SynchronizedSourceTick(
                key=key,
                road=road,
                cabin=cabin,
                ego=ego,
            )
            self._pending_key = None
            self._pending.clear()
            self._last_complete_sequence = key.source_sequence
            self._last_capture_timestamp_ms = capture_timestamp_ms
            self._paired_ticks += 1
            return SynchronizerOfferResult("READY", True, False, tick)

    def offer_road(self, envelope: RoadFrameEnvelope) -> SynchronizerOfferResult:
        if not isinstance(envelope, RoadFrameEnvelope):
            raise TypeError("offer_road requires RoadFrameEnvelope")
        return self._offer("road", envelope)

    def offer_cabin(self, envelope: CabinFrameEnvelope) -> SynchronizerOfferResult:
        if not isinstance(envelope, CabinFrameEnvelope):
            raise TypeError("offer_cabin requires CabinFrameEnvelope")
        return self._offer("cabin", envelope)

    def offer_ego(self, envelope: EgoTelemetryEnvelope) -> SynchronizerOfferResult:
        if not isinstance(envelope, EgoTelemetryEnvelope):
            raise TypeError("offer_ego requires EgoTelemetryEnvelope")
        return self._offer("ego", envelope)

    @property
    def stats(self) -> LiveSynchronizerStats:
        with self._lock:
            return LiveSynchronizerStats(
                session_id=self._session_id,
                generation=self._generation,
                offered=self._offered,
                accepted_components=self._accepted_components,
                paired_ticks=self._paired_ticks,
                dropped_duplicate=self._dropped_duplicate,
                dropped_reordered=self._dropped_reordered,
                dropped_stale=self._dropped_stale,
                dropped_future=self._dropped_future,
                discontinuities=self._discontinuities,
                pending_key=self._pending_key,
                faulted=self._faulted,
            )


@dataclass(frozen=True)
class LatestSlotStats:
    session_id: str | None
    generation: int
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

    def __init__(
        self, session_id: str | None = None, *, generation: int = 0
    ) -> None:
        self._lock = Lock()
        self.reset(session_id, generation=generation)

    def reset(
        self, session_id: str | None = None, *, generation: int = 0
    ) -> None:
        validated = (
            None
            if session_id is None
            else _validated_session_id(session_id)
        )
        validated_generation = _non_negative_integer(
            generation, field="generation"
        )
        with self._lock:
            self._session_id = validated
            self._generation = validated_generation
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
            if sample.generation != self._generation:
                raise ValueError("sample belongs to a stale or future generation")
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
                generation=sample.generation,
                frame_id=self._next_frame_id,
                source_sequence=sample.source_sequence,
                source_timestamp=sample.source_timestamp,
                road_bgr=_OwnedImageReference(sample.road_bgr),
                cabin_bgr=_OwnedImageReference(sample.cabin_bgr),
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
                generation=self._generation,
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

    def reset(self) -> None: ...


@dataclass(frozen=True)
class WarningOnlyRiskDecision:
    """Live-cloud risk projection that cannot request vehicle actuation."""

    score_pct: float
    level: str
    action: str
    brake_request_pct: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        score = float(self.score_pct)
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise ValueError("live risk score must be finite in [0, 100]")
        if self.action not in {
            "MONITOR",
            "VISUAL_WARNING",
            "VISUAL_AUDIO_HAPTIC_WARNING",
        }:
            raise ValueError("live risk action must be warning-only")
        if float(self.brake_request_pct) != 0.0:
            raise ValueError("live risk brake request must be zero")
        if any(not isinstance(reason, str) or not reason for reason in self.reasons):
            raise ValueError("live risk reasons must be non-empty strings")
        object.__setattr__(self, "score_pct", score)
        object.__setattr__(self, "brake_request_pct", 0.0)

    def diagnostic_row(self) -> dict[str, object]:
        return {
            "contextual_risk_score_pct": round(self.score_pct, 3),
            "contextual_risk_level": self.level,
            "contextual_risk_action": self.action,
            "contextual_risk_brake_request_pct": 0.0,
            "contextual_risk_reasons": "|".join(self.reasons),
        }


def _warning_only_risk(value: Any) -> WarningOnlyRiskDecision:
    try:
        score = float(value.score_pct)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("contextual risk must expose a finite score_pct") from exc
    level = str(getattr(value, "level", "SAFE"))
    action = str(getattr(value, "action", "MONITOR"))
    try:
        brake = float(getattr(value, "brake_request_pct", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("contextual risk brake request must be numeric") from exc
    if not math.isfinite(brake) or brake < 0.0:
        raise ValueError(
            "contextual risk brake request must be finite and non-negative"
        )
    raw_reasons = getattr(value, "reasons", ("NORMAL",))
    try:
        reasons = tuple(str(item) for item in raw_reasons)
    except TypeError as exc:
        raise ValueError("contextual risk reasons must be iterable") from exc
    if not reasons:
        reasons = ("NORMAL",)
    if action == "EMERGENCY_BRAKE_REQUEST" or brake > 0.0:
        action = "VISUAL_AUDIO_HAPTIC_WARNING"
    return WarningOnlyRiskDecision(
        score_pct=score,
        level=level,
        action=action,
        brake_request_pct=0.0,
        reasons=reasons,
    )


@dataclass(frozen=True)
class LiveSessionSnapshot:
    session_id: str | None
    generation: int
    active: bool
    faulted: bool
    processed_frames: int
    next_frame_id: int
    last_source_sequence: int | None
    last_source_timestamp: float | None
    c1_calls: int
    c1_model_updates: int
    c2_calls: int
    c3_calls: int
    drive_quality_calls: int
    contextual_risk_calls: int


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
        self._generation = 0
        self._active = False
        self._faulted = False
        self._processed_frames = 0
        self._next_frame_id = 0
        self._last_source_sequence: int | None = None
        self._last_source_timestamp: float | None = None
        self._c1_calls = 0
        self._c1_model_updates = 0
        self._c2_calls = 0
        self._c3_calls = 0
        self._drive_quality_calls = 0
        self._contextual_risk_calls = 0

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

    def _activate(self, session_id: str, generation: int) -> None:
        validated_session_id = self._validate_session_id(session_id)
        validated_generation = _non_negative_integer(
            generation, field="generation"
        )
        self._active = False
        self._faulted = False
        self._session_id = validated_session_id
        self._generation = validated_generation
        self._processed_frames = 0
        self._next_frame_id = 0
        self._last_source_sequence = None
        self._last_source_timestamp = None
        self._c1_calls = 0
        self._c1_model_updates = 0
        self._c2_calls = 0
        self._c3_calls = 0
        self._drive_quality_calls = 0
        self._contextual_risk_calls = 0
        try:
            self._reset_processors()
        except Exception:
            self._faulted = True
            raise
        self._active = True

    def start_session(self, session_id: str, *, generation: int = 0) -> None:
        """Start a fresh session; all stateful processors are reset."""

        with self._process_lock:
            if self._active:
                raise RuntimeError("a live inference session is already active")
            self._activate(session_id, generation)

    def reset_session(
        self, session_id: str, *, generation: int | None = None
    ) -> None:
        """Abandon current state and start a fresh session at frame zero."""

        with self._process_lock:
            validated = self._validate_session_id(session_id)
            next_generation = (
                self._generation + 1
                if generation is None and validated == self._session_id
                else 0
                if generation is None
                else generation
            )
            self._activate(validated, next_generation)

    def end_session(self) -> LiveSessionSnapshot:
        """Stop accepting input without closing model resources."""

        with self._process_lock:
            self._active = False
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> LiveSessionSnapshot:
        return LiveSessionSnapshot(
            session_id=self._session_id,
            generation=self._generation,
            active=self._active,
            faulted=self._faulted,
            processed_frames=self._processed_frames,
            next_frame_id=self._next_frame_id,
            last_source_sequence=self._last_source_sequence,
            last_source_timestamp=self._last_source_timestamp,
            c1_calls=self._c1_calls,
            c1_model_updates=self._c1_model_updates,
            c2_calls=self._c2_calls,
            c3_calls=self._c3_calls,
            drive_quality_calls=self._drive_quality_calls,
            contextual_risk_calls=self._contextual_risk_calls,
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
        if frame.generation != self._generation:
            raise ValueError("live frame belongs to a stale or future generation")
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
            if self._faulted:
                raise RuntimeError("live inference session is faulted; reset it")
            if not self._active:
                raise RuntimeError("no live inference session is active")
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
                expected_c1_update = frame.frame_id % 2 == 0
                actual_c1_update = getattr(collision, "model_updated", None)
                if not isinstance(actual_c1_update, bool):
                    raise ValueError(
                        "C1 live result must expose boolean model_updated"
                    )
                if actual_c1_update != expected_c1_update:
                    raise RuntimeError(
                        "C1 live cadence must update exactly every second "
                        f"20 Hz tick; frame {frame.frame_id} expected "
                        f"model_updated={expected_c1_update}"
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
                risk = _warning_only_risk(
                    self.contextual_risk.evaluate(collision, driver)
                )
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
            self._c1_calls += 1
            self._c1_model_updates += int(actual_c1_update)
            self._c2_calls += 1
            self._c3_calls += 1
            self._drive_quality_calls += 1
            self._contextual_risk_calls += 1
            self._next_frame_id += 1
            self._last_source_sequence = int(frame.source_sequence)
            self._last_source_timestamp = frame.source_timestamp
            return prediction
        finally:
            self._process_lock.release()


@dataclass(frozen=True)
class LiveRuntimeSnapshot:
    """Transport-neutral, warning-only result or DEGRADED heartbeat."""

    session_id: str
    generation: int
    sequence: int
    health: str
    reason: str
    source_sequence: int | None
    capture_timestamp_ms: int | None
    prediction: CombinedFramePrediction | None
    warning_action: str
    brake_request_pct: float = 0.0
    actuation_authorized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _validated_session_id(self.session_id))
        object.__setattr__(
            self,
            "generation",
            _non_negative_integer(self.generation, field="generation"),
        )
        object.__setattr__(
            self,
            "sequence",
            _non_negative_integer(self.sequence, field="sequence"),
        )
        if self.health not in {"LIVE", "DEGRADED"}:
            raise ValueError("runtime health must be LIVE or DEGRADED")
        if self.warning_action not in {
            "MONITOR",
            "VISUAL_WARNING",
            "VISUAL_AUDIO_HAPTIC_WARNING",
        }:
            raise ValueError("runtime warning action must be warning-only")
        if float(self.brake_request_pct) != 0.0:
            raise ValueError("runtime snapshots cannot request braking")
        if self.actuation_authorized is not False:
            raise ValueError("runtime snapshots cannot authorize actuation")
        if self.source_sequence is not None:
            _non_negative_integer(
                self.source_sequence, field="source_sequence"
            )
        if self.capture_timestamp_ms is not None:
            _capture_timestamp_ms(self.capture_timestamp_ms)
        if self.health == "LIVE" and self.prediction is None:
            raise ValueError("LIVE runtime snapshots require a prediction")
        if self.health == "DEGRADED" and self.warning_action != "MONITOR":
            raise ValueError("DEGRADED runtime snapshots must suppress warnings")


class RuntimeSnapshotHandoff(Protocol):
    """Non-blocking capacity-one handoff used by publisher workers."""

    def offer(self, snapshot: LiveRuntimeSnapshot) -> bool: ...


@dataclass(frozen=True)
class RuntimeHandoffStats:
    offered: int
    emitted: int
    overwritten: int
    pending: bool


class LatestRuntimeSnapshotSlot:
    """Lock-only capacity-one handoff; broker I/O never runs on inference."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: LiveRuntimeSnapshot | None = None
        self._offered = 0
        self._emitted = 0
        self._overwritten = 0

    def offer(self, snapshot: LiveRuntimeSnapshot) -> bool:
        if not isinstance(snapshot, LiveRuntimeSnapshot):
            raise TypeError("handoff accepts LiveRuntimeSnapshot")
        with self._lock:
            replaced = self._pending is not None
            if replaced:
                self._overwritten += 1
            self._pending = snapshot
            self._offered += 1
            return replaced

    def pop_latest(self) -> LiveRuntimeSnapshot | None:
        with self._lock:
            snapshot = self._pending
            if snapshot is not None:
                self._pending = None
                self._emitted += 1
            return snapshot

    def clear(self) -> None:
        with self._lock:
            self._pending = None

    @property
    def stats(self) -> RuntimeHandoffStats:
        with self._lock:
            return RuntimeHandoffStats(
                offered=self._offered,
                emitted=self._emitted,
                overwritten=self._overwritten,
                pending=self._pending is not None,
            )


@dataclass(frozen=True)
class LiveControllerSnapshot:
    session_id: str | None
    generation: int
    health: str
    reason: str
    coherent_ticks: int
    pending_input: bool
    runtime_sequence: int
    handoff_errors: int
    latest_handoff_error: str | None


class LiveInferenceController:
    """Fence ingress and model state with one shared session generation.

    The short controller lock protects ingress/session cutover while model
    inference runs outside it. If a reset or end wins while inference is in
    flight, the session-generation fence suppresses the retired result.
    """

    def __init__(
        self,
        session: LiveInferenceSession,
        *,
        slot: LatestPairedFrameSlot | None = None,
        handoff: RuntimeSnapshotHandoff | None = None,
        coherent_dwell_ticks: int = DEFAULT_COHERENT_DWELL_TICKS,
        max_skew_ms: int = MAX_SOURCE_SKEW_MS,
    ) -> None:
        if not isinstance(session, LiveInferenceSession):
            raise TypeError("session must be a LiveInferenceSession")
        dwell = _non_negative_integer(
            coherent_dwell_ticks, field="coherent_dwell_ticks"
        )
        if dwell < 1:
            raise ValueError("coherent_dwell_ticks must be at least one")
        skew = _non_negative_integer(max_skew_ms, field="max_skew_ms")
        if skew > MAX_SOURCE_SKEW_MS:
            raise ValueError(
                f"max_skew_ms cannot exceed {MAX_SOURCE_SKEW_MS}"
            )
        self.session = session
        self.slot = slot if slot is not None else LatestPairedFrameSlot()
        self.handoff = (
            handoff if handoff is not None else LatestRuntimeSnapshotSlot()
        )
        if not callable(getattr(self.handoff, "offer", None)):
            raise TypeError("handoff.offer must be callable")
        self.coherent_dwell_ticks = dwell
        self.max_skew_ms = skew
        self._control_lock = Lock()
        self._session_id: str | None = None
        self._generation = 0
        self._active = False
        self._health = "DEGRADED"
        self._reason = "NO_SESSION"
        self._coherent_ticks = 0
        self._runtime_sequence = 0
        self._next_frame_id = 0
        self._inference_in_flight = False
        self._pending_tick: SynchronizedSourceTick | None = None
        self._synchronizer: LiveInputSynchronizer | None = None
        self._input_mode: str | None = None
        self._handoff_errors = 0
        self._latest_handoff_error: str | None = None

    def _emit_locked(self, snapshot: LiveRuntimeSnapshot) -> None:
        # The default implementation is a lock-only capacity-one slot. Any
        # injected implementation is required by RuntimeSnapshotHandoff to be
        # non-blocking and must perform broker I/O on its own worker.
        try:
            self.handoff.offer(snapshot)
        except Exception as exc:
            # Publication is an asynchronous side effect. A closed/broken
            # handoff must be observable but can never fault temporal models
            # or change source cadence.
            self._handoff_errors += 1
            self._latest_handoff_error = (
                f"{type(exc).__name__}: {exc}"
            )[:160]

    def _next_runtime_sequence_locked(self) -> int:
        sequence = self._runtime_sequence
        self._runtime_sequence += 1
        return sequence

    def _emit_heartbeat_locked(self, reason: str) -> None:
        assert self._session_id is not None
        self._health = "DEGRADED"
        self._reason = reason
        self._emit_locked(
            LiveRuntimeSnapshot(
                session_id=self._session_id,
                generation=self._generation,
                sequence=self._next_runtime_sequence_locked(),
                health="DEGRADED",
                reason=reason,
                source_sequence=None,
                capture_timestamp_ms=None,
                prediction=None,
                warning_action="MONITOR",
            )
        )

    def _configure_ingress_locked(self) -> None:
        assert self._session_id is not None
        self.slot.reset(None, generation=self._generation)
        self._pending_tick = None
        self._input_mode = None
        if self._synchronizer is None:
            self._synchronizer = LiveInputSynchronizer(
                self._session_id,
                self._generation,
                max_skew_ms=self.max_skew_ms,
            )
        else:
            self._synchronizer.reset(self._session_id, self._generation)

    def start_session(self, session_id: str, *, generation: int = 0) -> None:
        validated = _validated_session_id(session_id)
        validated_generation = _non_negative_integer(
            generation, field="generation"
        )
        with self._control_lock:
            if self._active:
                raise RuntimeError("a live inference session is already active")
            self._session_id = validated
            self._generation = validated_generation
            self._runtime_sequence = 0
            self._next_frame_id = 0
            self._coherent_ticks = 0
            self._configure_ingress_locked()
            try:
                self.session.start_session(
                    validated, generation=validated_generation
                )
            except Exception:
                self._active = False
                self._emit_heartbeat_locked("RESET_FAILED")
                raise
            self._active = True
            self.slot.reset(validated, generation=validated_generation)
            self._emit_heartbeat_locked("COHERENCE_DWELL")

    def _reset_active_locked(
        self, session_id: str, generation: int, reason: str
    ) -> None:
        self._session_id = session_id
        self._generation = generation
        self._runtime_sequence = 0
        self._next_frame_id = 0
        self._coherent_ticks = 0
        self._configure_ingress_locked()
        try:
            self.session.reset_session(session_id, generation=generation)
        except Exception:
            self._active = False
            self._emit_heartbeat_locked(f"{reason}:RESET_FAILED")
            raise
        self._active = True
        self.slot.reset(session_id, generation=generation)
        self._emit_heartbeat_locked(reason)

    def _advance_generation_locked(self, reason: str) -> None:
        if self._session_id is None:
            raise RuntimeError("no live inference session is active")
        self._reset_active_locked(
            self._session_id, self._generation + 1, reason
        )

    def reset_session(self, session_id: str) -> None:
        validated = _validated_session_id(session_id)
        with self._control_lock:
            generation = (
                self._generation + 1
                if validated == self._session_id
                else 0
            )
            self._reset_active_locked(validated, generation, "RESTART")

    def report_state_loss(self, reason: str = "STATE_LOSS") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("state-loss reason must be a non-empty string")
        with self._control_lock:
            self._advance_generation_locked(reason.strip())

    def end_session(self) -> LiveSessionSnapshot:
        with self._control_lock:
            self.slot.reset(None, generation=self._generation)
            self._pending_tick = None
            self._active = False
            self._health = "DEGRADED"
            self._reason = "ENDED"
            if self._session_id is not None:
                self._emit_heartbeat_locked("ENDED")
            return self.session.end_session()

    def offer(self, sample: PairedCameraSample) -> None:
        """Legacy already-paired ingest path retained for local adapters."""

        with self._control_lock:
            if not self._active:
                raise RuntimeError("no live inference session is active")
            if self._input_mode not in {None, "legacy"}:
                raise RuntimeError("cannot mix legacy and synchronized ingress")
            self._input_mode = "legacy"
            self.slot.offer(sample)

    def _offer_independent(
        self, kind: str, envelope: Any
    ) -> SynchronizerOfferResult:
        with self._control_lock:
            if not self._active or self._synchronizer is None:
                raise RuntimeError("no live inference session is active")
            if self._input_mode not in {None, "synchronized"}:
                raise RuntimeError("cannot mix legacy and synchronized ingress")
            self._input_mode = "synchronized"
            if self._pending_tick is not None:
                replacement = SynchronizerOfferResult(
                    "INFERENCE_BACKLOG", False, True
                )
                self._advance_generation_locked(replacement.disposition)
                return replacement
            method = getattr(self._synchronizer, f"offer_{kind}")
            result = method(envelope)
            if result.reset_required:
                self._advance_generation_locked(result.disposition)
                return result
            if result.tick is not None:
                self._pending_tick = result.tick
            return result

    def offer_road(
        self, envelope: RoadFrameEnvelope
    ) -> SynchronizerOfferResult:
        return self._offer_independent("road", envelope)

    def offer_cabin(
        self, envelope: CabinFrameEnvelope
    ) -> SynchronizerOfferResult:
        return self._offer_independent("cabin", envelope)

    def offer_ego(
        self, envelope: EgoTelemetryEnvelope
    ) -> SynchronizerOfferResult:
        return self._offer_independent("ego", envelope)

    def process_latest(self) -> CombinedFramePrediction | None:
        """Process one pending tick and offer its safe runtime snapshot.

        The return value preserves the legacy local API. Cloud publishers must
        consume ``handoff`` snapshots, which suppress predictions throughout
        coherence dwell and carry the active generation fence.
        """

        with self._control_lock:
            if self._inference_in_flight:
                raise RuntimeError("concurrent live inference is not supported")
            if self._input_mode == "synchronized":
                tick = self._pending_tick
                if tick is None:
                    return None
                self._pending_tick = None
                frame = tick.to_live_frame(
                    frame_id=self._next_frame_id
                )
            else:
                tick = None
                frame = self.slot.pop_latest()
            token = (self._session_id, self._generation)
            if frame is not None:
                self._inference_in_flight = True
        if frame is None:
            return None
        try:
            prediction = self.session.process(frame)
        except Exception:
            with self._control_lock:
                self._inference_in_flight = False
                if self._active and token == (
                    self._session_id,
                    self._generation,
                ):
                    self._advance_generation_locked("MODEL_EXCEPTION")
            raise

        with self._control_lock:
            self._inference_in_flight = False
            if not self._active or token != (
                self._session_id,
                self._generation,
            ):
                # A reset won while inference was running. Its processor reset
                # completed after this call released the model lock, so never
                # publish the retired generation's result.
                return None
            self._coherent_ticks += 1
            self._next_frame_id += 1
            live = self._coherent_ticks >= self.coherent_dwell_ticks
            self._health = "LIVE" if live else "DEGRADED"
            self._reason = "" if live else "COHERENCE_DWELL"
            risk = prediction.contextual_risk
            warning_action = risk.action if live else "MONITOR"
            capture_timestamp_ms = (
                tick.capture_timestamp_ms
                if tick is not None
                else int(round(frame.source_timestamp * 1_000.0))
            )
            self._emit_locked(
                LiveRuntimeSnapshot(
                    session_id=frame.session_id,
                    generation=frame.generation,
                    sequence=self._next_runtime_sequence_locked(),
                    health=self._health,
                    reason=self._reason,
                    source_sequence=int(frame.source_sequence),
                    capture_timestamp_ms=capture_timestamp_ms,
                    prediction=prediction if live else None,
                    warning_action=warning_action,
                )
            )
            return prediction

    @property
    def generation(self) -> int:
        with self._control_lock:
            return self._generation

    @property
    def controller_snapshot(self) -> LiveControllerSnapshot:
        with self._control_lock:
            return LiveControllerSnapshot(
                session_id=self._session_id,
                generation=self._generation,
                health=self._health,
                reason=self._reason,
                coherent_ticks=self._coherent_ticks,
                pending_input=(
                    self._pending_tick is not None
                    or self.slot.stats.pending
                ),
                runtime_sequence=self._runtime_sequence,
                handoff_errors=self._handoff_errors,
                latest_handoff_error=self._latest_handoff_error,
            )

    @property
    def synchronizer_stats(self) -> LiveSynchronizerStats | None:
        with self._control_lock:
            if self._synchronizer is None:
                return None
            return self._synchronizer.stats


__all__ = [
    "CabinFrameEnvelope",
    "C1LiveProcessor",
    "C2LiveProcessor",
    "C3LiveProcessor",
    "ContextualRiskLiveProcessor",
    "DEFAULT_COHERENT_DWELL_TICKS",
    "DriveQualityLiveProcessor",
    "EGO_FIELDS",
    "EgoTelemetryEnvelope",
    "LatestRuntimeSnapshotSlot",
    "LatestPairedFrameSlot",
    "LatestSlotStats",
    "LiveControllerSnapshot",
    "LiveFrameInput",
    "LiveInputSynchronizer",
    "LiveInferenceController",
    "LiveInferenceSession",
    "LiveRuntimeSnapshot",
    "LiveSessionSnapshot",
    "LiveSynchronizerStats",
    "MAX_SOURCE_SKEW_MS",
    "PairedCameraSample",
    "RoadFrameEnvelope",
    "RtpFrameIdentity",
    "RuntimeHandoffStats",
    "RuntimeSnapshotHandoff",
    "SourceTickKey",
    "SynchronizedSourceTick",
    "SynchronizerOfferResult",
    "WarningOnlyRiskDecision",
]
