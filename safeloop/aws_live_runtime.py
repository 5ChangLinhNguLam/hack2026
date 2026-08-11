"""Stateful model/session controller for the AWS fast bridge.

One worker owns C1/C2/C3 and all temporal state.  Live ingress has exactly one
pending slot.  A gap, reconnect or live overwrite advances an internal
generation; the worker resets every model before using the replacement tick
and discards any result completed by an older generation.  Recorded demos use
bounded backpressure instead: their source thread waits for the preceding tick
to finish so a slower GPU cannot turn deterministic replay into overflow.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol
import uuid

from C1.features import FEATURES_USED, ScalarFeatureStream
from C1.runtime import StudentTTCRuntime
from drive_state.phase_2.replay import DMSBundle, GeneralDMS

from .aws_live_contract import InputTick, PreparedDemoBundle
from .c3 import Challenge3Accumulator
from .carsky_decision import (
    DecisionEnvelope,
    DecisionEnvelopeBuilder,
    DecisionValidity,
)
from .carsky_live import LiveFrameInput, LiveInferenceSession
from .contextual_risk import ContextualRiskPolicy
from .drive_quality import DriveQualityAccumulator


DECISION_MAX_BYTES = 1_472
SOURCE_HZ = 20.0
SOURCE_PERIOD_NS = 50_000_000
COHERENT_DWELL_TICKS = 5
SOURCE_DEGRADED_SECONDS = 1.0
SOURCE_ERROR_SECONDS = 10.0
LATENCY_WINDOW = 4_096
STATES = frozenset(
    {"STOPPED", "STARTING", "RUNNING", "DEGRADED", "STOPPING", "ERROR"}
)


class FastBridgeError(RuntimeError):
    """The live bridge cannot safely continue the current session."""


class ModelPipeline(Protocol):
    model_versions: Mapping[str, str]
    model_hashes: Mapping[str, str]

    def start_session(
        self,
        session_id: str,
        metadata: Mapping[str, Any],
        *,
        expected_frames: int | None,
    ) -> None: ...

    def process(self, tick: InputTick, *, frame_id: int) -> Any: ...

    def close(self) -> None: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


class RealModelPipeline:
    """Load the real C1/C2 weights once and reconfigure only temporal state."""

    def __init__(
        self,
        *,
        c1_checkpoint: str | Path,
        c2_bundle: str | Path,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.c1_path = Path(c1_checkpoint).resolve()
        self.c2_root = Path(c2_bundle).resolve()
        initial_metadata = {"speed_limit_kmh": 0.0, "weather": {}}
        self.c1 = StudentTTCRuntime(
            self.c1_path,
            metadata=initial_metadata,
            source_fps=SOURCE_HZ,
            stride=2,
            device=device,
        )
        self.c2 = GeneralDMS(
            DMSBundle.at(self.c2_root),
            device=device,
            fps=SOURCE_HZ,
        )
        self.c3 = Challenge3Accumulator(speed_limit_kmh=0.0)
        self.drive_quality = DriveQualityAccumulator()
        self.contextual_risk = ContextualRiskPolicy()
        self.session = LiveInferenceSession(
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            drive_quality=self.drive_quality,
            contextual_risk=self.contextual_risk,
        )
        c2_files = [path for path in self.c2_root.rglob("*") if path.is_file()]
        self.model_hashes = MappingProxyType(
            {
                "c1": _sha256_file(self.c1_path),
                "c2": _tree_digest(c2_files, root=self.c2_root),
            }
        )
        self.model_versions = MappingProxyType(
            {
                "c1": "student-ttc",
                "c2": "dms-v13",
                "c3": "challenge3",
                "dq": "drive-quality",
                "risk": "context-v1",
            }
        )
        self.load_count = 1

    def start_session(
        self,
        session_id: str,
        metadata: Mapping[str, Any],
        *,
        expected_frames: int | None,
    ) -> None:
        # Replace only the causal scalar stream; the C1 model remains resident.
        self.c1.scalar_stream = ScalarFeatureStream(
            metadata=metadata,
            sample_hz=self.c1.sample_hz,
            feature_order=FEATURES_USED,
        )
        speed_limit = float(metadata.get("speed_limit_kmh", 0.0))
        if not math.isfinite(speed_limit) or speed_limit < 0.0:
            raise FastBridgeError("invalid session speed limit")
        self.c3.speed_limit_kmh = speed_limit
        self.c3.expected_frames = expected_frames
        snapshot = self.session.snapshot
        if snapshot.active:
            self.session.reset_session(session_id)
        else:
            self.session.start_session(session_id)

    def process(self, tick: InputTick, *, frame_id: int) -> Any:
        # Recorded inference may be deliberately slower than the bundle's
        # nominal 20 Hz clock.  Temporal models must still see the original
        # media timeline; capture_timestamp_ms remains the fresh wall-clock
        # evidence carried by the outbound decision envelope.
        model_timestamp_ms = (
            tick.source_media_timestamp_ms
            if tick.source_kind == "RECORDED_STREAM"
            and tick.source_media_timestamp_ms is not None
            else tick.capture_timestamp_ms
        )
        frame = LiveFrameInput(
            session_id=self.session.snapshot.session_id or "invalid",
            frame_id=frame_id,
            source_sequence=tick.source_sequence,
            source_timestamp=model_timestamp_ms / 1_000.0,
            road_bgr=tick.road_bgr,
            cabin_bgr=tick.cabin_bgr,
            ego=tick.ego,
        )
        return self.session.process(frame)

    def close(self) -> None:
        self.session.end_session()
        self.c1.close()
        self.c2.close()


def _warning_only(prediction: Any) -> Any:
    risk = prediction.contextual_risk
    action = str(risk.action)
    brake = float(risk.brake_request_pct)
    if action == "EMERGENCY_BRAKE_REQUEST" or brake > 0.0:
        risk = replace(
            risk,
            action="VISUAL_AUDIO_HAPTIC_WARNING",
            brake_request_pct=0.0,
        )
        prediction = replace(prediction, contextual_risk=risk)
    if (
        str(prediction.contextual_risk.action) == "EMERGENCY_BRAKE_REQUEST"
        or float(prediction.contextual_risk.brake_request_pct) != 0.0
    ):
        raise FastBridgeError("cloud decision mapper is not warning-only")
    return prediction


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


@dataclass(frozen=True)
class DecisionPublication:
    envelope: DecisionEnvelope
    payload: bytes
    source_kind: str
    telemetry_source: str
    generation: int
    source_sequence: int
    capture_timestamp_ms: int
    inference_latency_ms: float


@dataclass(frozen=True)
class ControlResult:
    revision: int
    state: str
    session_id: str | None
    demo_id: str | None


class FastBridgeController:
    """Serialize admin control, source identity and one-slot inference."""

    def __init__(
        self,
        models: ModelPipeline,
        *,
        on_decision: Callable[[DecisionPublication], None],
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.models = models
        self.on_decision = on_decision
        self.on_event = on_event or (lambda _event: None)
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic = monotonic or time.monotonic
        self._condition = threading.Condition()
        self._control_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._pending: tuple[int, InputTick] | None = None
        self._inference_in_flight = False
        self._closed = False
        self._state = "STOPPED"
        self._revision = 0
        self._generation = 0
        self._internal_session_id: str | None = None
        self._external_session_id: str | None = None
        self._active_demo_id: str | None = None
        self._source_kind: str | None = None
        self._telemetry_source: str | None = None
        self._metadata: Mapping[str, Any] | None = None
        self._expected_frames: int | None = None
        self._last_source_sequence: int | None = None
        self._last_capture_timestamp_ms: int | None = None
        self._last_input_monotonic: float | None = None
        self._started_monotonic: float | None = None
        self._live_connected = False
        self._live_connection_token = 0
        self._source_run_token = 0
        self._coherent_ticks = 0
        self._local_frame_id = 0
        self._runtime_generation = -1
        self._builder: DecisionEnvelopeBuilder | None = None
        self._last_error = ""
        self._last_publication: DecisionPublication | None = None
        self._max_decision_bytes = 0
        self._counts: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._c1_latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._c2_latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._post_latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._input_times: deque[float] = deque()
        self._decision_times: deque[float] = deque()
        self._recorded_thread: threading.Thread | None = None
        self._recorded_stop = threading.Event()
        self._sse_subscribers = 0
        self._last_sse_write_ms: int | None = None
        self._analysis_running = False
        self._analysis_cache: dict[str, Mapping[str, Any]] = {}
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="safeloop-fast-bridge-inference",
            daemon=True,
        )
        self._worker.start()

    def _emit_event_locked(self, event: str) -> None:
        payload = {
            "event": event,
            "timestamp_ms": self.clock_ms(),
            "revision": self._revision,
            "state": self._state,
            "demo_id": self._active_demo_id,
            "session_id": self._internal_session_id,
            "generation": self._generation,
            "error": self._last_error,
        }
        try:
            self.on_event(MappingProxyType(payload))
        except Exception:
            self._counts["event_callback_errors"] += 1

    @staticmethod
    def _internal_id(external_id: str, generation: int) -> str:
        identity = hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:6]
        suffix = uuid.uuid4().hex[:8]
        return f"sl-{identity}:g{generation & 0xFFFFFFFF:08x}:{suffix}"

    def _clear_source_locked(self) -> None:
        self._pending = None
        self._external_session_id = None
        self._last_source_sequence = None
        self._last_capture_timestamp_ms = None
        self._last_input_monotonic = None
        self._coherent_ticks = 0
        self._local_frame_id = 0
        self._builder = None
        self._runtime_generation = -1

    def _advance_generation_locked(self, reason: str, *, initial: bool = False) -> None:
        self._generation += 1
        external = self._external_session_id or self._active_demo_id or "safeloop"
        self._internal_session_id = self._internal_id(external, self._generation)
        self._pending = None
        self._last_source_sequence = None
        self._last_capture_timestamp_ms = None
        self._coherent_ticks = 0
        self._local_frame_id = 0
        self._builder = None
        self._runtime_generation = -1
        self._state = "STARTING" if initial else "DEGRADED"
        self._last_error = "" if initial else reason
        if not initial:
            self._counts["generation_resets"] += 1
        self._emit_event_locked("SESSION_START" if initial else "GENERATION_RESET")

    def _begin_locked(
        self,
        *,
        demo_id: str,
        source_kind: str,
        telemetry_source: str,
        metadata: Mapping[str, Any] | None,
        expected_frames: int | None,
    ) -> None:
        self._counts.clear()
        self._latencies.clear()
        self._c1_latencies.clear()
        self._c2_latencies.clear()
        self._post_latencies.clear()
        self._input_times.clear()
        self._decision_times.clear()
        self._last_publication = None
        self._max_decision_bytes = 0
        self._source_run_token += 1
        self._active_demo_id = demo_id
        self._source_kind = source_kind
        self._telemetry_source = telemetry_source
        self._metadata = metadata
        self._expected_frames = expected_frames
        self._started_monotonic = self.monotonic()
        self._live_connected = False
        self._clear_source_locked()
        self._advance_generation_locked("START", initial=True)
        self._condition.notify_all()

    def _check_revision_locked(self, expected_revision: int) -> None:
        if expected_revision != self._revision:
            raise FastBridgeError(
                f"revision conflict: expected {expected_revision}, current {self._revision}"
            )

    def start_live(self, *, expected_revision: int) -> ControlResult:
        with self._control_lock, self._condition:
            self._check_revision_locked(expected_revision)
            if self._state not in {"STOPPED", "ERROR"}:
                self._stop_locked("SWITCH")
            self._revision += 1
            self._begin_locked(
                demo_id="LIVE_FULL_SYSTEM",
                source_kind="LIVE_CAMERA",
                telemetry_source="THIRD_PARTY",
                metadata=None,
                expected_frames=None,
            )
            return self._control_result_locked()

    def start_recorded(
        self,
        bundle: PreparedDemoBundle,
        *,
        expected_revision: int,
    ) -> ControlResult:
        with self._control_lock, self._condition:
            self._check_revision_locked(expected_revision)
            if self._state not in {"STOPPED", "ERROR"}:
                self._stop_locked("SWITCH")
            self._revision += 1
            self._begin_locked(
                demo_id=bundle.trip_id,
                source_kind="RECORDED_STREAM",
                telemetry_source="RECORDED_DATA",
                metadata=bundle.metadata,
                expected_frames=len(bundle.frames),
            )
            run_token = self._source_run_token
            self._recorded_stop = threading.Event()
            self._recorded_thread = threading.Thread(
                target=self._recorded_loop,
                args=(bundle, run_token, self._recorded_stop),
                name="safeloop-recorded-source",
                daemon=True,
            )
            self._recorded_thread.start()
            return self._control_result_locked()

    def _recorded_loop(
        self,
        bundle: PreparedDemoBundle,
        run_token: int,
        stop_event: threading.Event,
    ) -> None:
        round_index = 0
        absolute_tick = 0
        started_ns = time.monotonic_ns()
        while not stop_event.is_set():
            session_id = f"{bundle.trip_id}-{round_index}"
            for index in range(len(bundle.frames)):
                if stop_event.is_set():
                    return
                deadline = started_ns + absolute_tick * SOURCE_PERIOD_NS
                remaining = deadline - time.monotonic_ns()
                if remaining > 0:
                    stop_event.wait(remaining / 1_000_000_000.0)
                if stop_event.is_set():
                    return
                try:
                    tick = bundle.tick(
                        index,
                        session_id=session_id,
                        capture_timestamp_ms=self.clock_ms(),
                    )
                    self.ingest(tick, source_run_token=run_token)
                except Exception as exc:
                    with self._condition:
                        self._counts["recorded_source_errors"] += 1
                        self._state = "ERROR"
                        self._last_error = f"{type(exc).__name__}: {exc}"[:200]
                        self._emit_event_locked("RECORDED_SOURCE_ERROR")
                    return
                absolute_tick += 1
            round_index += 1

    def _stop_locked(self, reason: str) -> threading.Thread | None:
        self._state = "STOPPING"
        self._emit_event_locked("STOPPING")
        self._recorded_stop.set()
        self._source_run_token += 1
        thread = self._recorded_thread
        self._recorded_thread = None
        self._generation += 1
        self._active_demo_id = None
        self._source_kind = None
        self._telemetry_source = None
        self._metadata = None
        self._expected_frames = None
        self._live_connected = False
        self._clear_source_locked()
        self._state = "STOPPED"
        self._last_error = "" if reason == "ADMIN_STOP" else reason
        self._emit_event_locked("STOPPED")
        self._condition.notify_all()
        return thread

    def stop(self, *, expected_revision: int) -> ControlResult:
        with self._control_lock:
            with self._condition:
                self._check_revision_locked(expected_revision)
                self._revision += 1
                thread = self._stop_locked("ADMIN_STOP")
                result = self._control_result_locked()
            if thread is not None and thread is not threading.current_thread():
                thread.join(2.0)
            return result

    def _control_result_locked(self) -> ControlResult:
        return ControlResult(
            revision=self._revision,
            state=self._state,
            session_id=self._internal_session_id,
            demo_id=self._active_demo_id,
        )

    def live_connect(self) -> int:
        with self._condition:
            if self._source_kind != "LIVE_CAMERA" or self._state not in {
                "STARTING",
                "RUNNING",
                "DEGRADED",
            }:
                raise FastBridgeError("LIVE_FULL_SYSTEM is not active")
            if self._live_connected:
                raise FastBridgeError("only one live source is allowed")
            reconnect = self._external_session_id is not None or self._last_input_monotonic is not None
            self._live_connected = True
            self._live_connection_token += 1
            if reconnect:
                self._advance_generation_locked("SOURCE_RECONNECT")
            token = self._live_connection_token
            self._emit_event_locked("LIVE_SOURCE_CONNECTED")
            return token

    def live_disconnect(self, token: int) -> None:
        with self._condition:
            if not self._live_connected or token != self._live_connection_token:
                return
            self._live_connected = False
            if self._state in {"STARTING", "RUNNING", "DEGRADED"}:
                self._advance_generation_locked("SOURCE_DISCONNECTED")
            self._condition.notify_all()

    def ingest(self, tick: InputTick, *, source_run_token: int | None = None) -> bool:
        with self._condition:
            # Admin Stop/Switch advances this token before clearing the source.
            # Checking it first makes a recorded producer blocked on
            # backpressure exit quietly instead of racing Stop into ERROR.
            if source_run_token is not None and source_run_token != self._source_run_token:
                return False
            if self._closed or self._state not in {"STARTING", "RUNNING", "DEGRADED"}:
                raise FastBridgeError("no active source accepts input")
            if tick.source_kind != self._source_kind or tick.telemetry_source != self._telemetry_source:
                raise FastBridgeError("input provenance differs from the active demo")
            if tick.source_kind == "LIVE_CAMERA" and not self._live_connected:
                raise FastBridgeError("live input arrived without an active WSS source")

            # Recorded input is finite, ordered evidence.  Never overwrite it
            # merely because inference is slower than its nominal 20 Hz media
            # clock.  Capacity remains bounded (one worker, no queued history),
            # and Stop/Switch wakes this wait by advancing source_run_token.
            while tick.source_kind == "RECORDED_STREAM" and (
                self._pending is not None or self._inference_in_flight
            ):
                self._condition.wait(timeout=0.1)
                if (
                    source_run_token is not None
                    and source_run_token != self._source_run_token
                ):
                    return False
                if self._closed:
                    return False
                if self._state not in {"STARTING", "RUNNING", "DEGRADED"}:
                    raise FastBridgeError("recorded source stopped during backpressure")

            now = self.monotonic()
            if self._external_session_id is None:
                self._external_session_id = tick.session_id
                self._internal_session_id = self._internal_id(
                    tick.session_id, self._generation
                )
            elif tick.session_id != self._external_session_id:
                self._external_session_id = tick.session_id
                self._advance_generation_locked("SOURCE_SESSION_CHANGED")
            if self._last_source_sequence is not None:
                if tick.source_sequence == self._last_source_sequence:
                    self._counts["duplicate_frames"] += 1
                    return False
                if tick.source_sequence < self._last_source_sequence:
                    self._counts["reordered_frames"] += 1
                    return False
                if tick.source_sequence > self._last_source_sequence + 1:
                    self._counts["source_gaps"] += (
                        tick.source_sequence - self._last_source_sequence - 1
                    )
                    self._advance_generation_locked("SOURCE_SEQUENCE_GAP")
            if (
                self._last_capture_timestamp_ms is not None
                and tick.capture_timestamp_ms <= self._last_capture_timestamp_ms
            ):
                self._counts["reordered_frames"] += 1
                return False
            if (
                self._last_input_monotonic is not None
                and now - self._last_input_monotonic > SOURCE_DEGRADED_SECONDS
            ):
                self._advance_generation_locked("SOURCE_RECOVERED_AFTER_GAP")
            if self._pending is not None:
                self._counts["overflow_drops"] += 1
                self._advance_generation_locked("INGRESS_OVERFLOW")
            generation = self._generation
            self._pending = (generation, tick)
            self._metadata = tick.metadata
            self._last_source_sequence = tick.source_sequence
            self._last_capture_timestamp_ms = tick.capture_timestamp_ms
            self._last_input_monotonic = now
            self._input_times.append(now)
            self._trim_rates_locked(now)
            self._counts["accepted_input_ticks"] += 1
            self._condition.notify_all()
            return True

    def _trim_rates_locked(self, now: float) -> None:
        cutoff = now - 5.0
        while self._input_times and self._input_times[0] < cutoff:
            self._input_times.popleft()
        while self._decision_times and self._decision_times[0] < cutoff:
            self._decision_times.popleft()

    def _watchdog_locked(self, now: float) -> None:
        if self._state not in {"STARTING", "RUNNING", "DEGRADED"}:
            return
        baseline = self._last_input_monotonic or self._started_monotonic
        if baseline is None:
            return
        age = now - baseline
        if age > SOURCE_ERROR_SECONDS:
            self._state = "ERROR"
            self._last_error = "SOURCE_TIMEOUT"
            self._pending = None
            self._emit_event_locked("SOURCE_TIMEOUT")
        elif age > SOURCE_DEGRADED_SECONDS and self._state == "RUNNING":
            self._state = "DEGRADED"
            self._last_error = "SOURCE_STALE"
            self._emit_event_locked("SOURCE_STALE")

    def _reset_models_for(self, generation: int, tick: InputTick) -> None:
        internal = self._internal_session_id
        if internal is None:
            raise FastBridgeError("internal session identity is missing")
        expected = self._expected_frames if tick.source_kind == "RECORDED_STREAM" else None
        self.models.start_session(
            internal,
            tick.metadata,
            expected_frames=expected,
        )
        self._builder = DecisionEnvelopeBuilder(
            session_id=internal,
            source_mode="live" if tick.source_kind == "LIVE_CAMERA" else "replay",
            source_fps=SOURCE_HZ,
            ttl_ms=250,
            model_versions=self.models.model_versions,
            clock_ms=self.clock_ms,
        )
        self._runtime_generation = generation
        self._local_frame_id = 0
        self._counts["model_resets"] += 1

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait(timeout=0.1)
                if self._closed:
                    return
                now = self.monotonic()
                self._watchdog_locked(now)
                item = self._pending
                self._pending = None
                if item is not None:
                    self._inference_in_flight = True
                    self._condition.notify_all()
            if item is None:
                continue
            generation, tick = item
            try:
                with self._model_lock:
                    if generation != self._runtime_generation:
                        self._reset_models_for(generation, tick)
                    started = time.perf_counter_ns()
                    prediction = self.models.process(tick, frame_id=self._local_frame_id)
                    prediction = _warning_only(prediction)
                    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                    c1_latency = max(0.0, float(getattr(prediction.c1, "latency_ms", 0.0)))
                    c2_latency = max(0.0, float(getattr(prediction.c2, "latency_ms", 0.0)))
                    with self._condition:
                        if generation != self._generation or self._state in {"STOPPED", "STOPPING", "ERROR"}:
                            self._counts["discarded_generation_results"] += 1
                            continue
                        builder = self._builder
                        if builder is None:
                            raise FastBridgeError("decision builder is unavailable")
                        self._coherent_ticks += 1
                        if self._coherent_ticks >= COHERENT_DWELL_TICKS:
                            self._state = "RUNNING"
                            self._last_error = ""
                        runtime_mode = "NOMINAL" if self._state == "RUNNING" else "DEGRADED"
                    envelope = builder.build(
                        prediction,
                        validity=DecisionValidity(
                            ego=True,
                            front_camera=True,
                            driver_camera=True,
                            c1=True,
                            c2=True,
                            c3=True,
                            drive_quality=True,
                            contextual_risk=True,
                        ),
                        decision_timestamp_ms=self.clock_ms(),
                        runtime_mode=runtime_mode,
                    )
                    payload = envelope.to_json_bytes()
                    if len(payload) > DECISION_MAX_BYTES:
                        raise FastBridgeError(
                            f"decision.v1 exceeds {DECISION_MAX_BYTES} bytes"
                        )
                publication = DecisionPublication(
                    envelope=envelope,
                    payload=payload,
                    source_kind=tick.source_kind,
                    telemetry_source=tick.telemetry_source,
                    generation=generation,
                    source_sequence=tick.source_sequence,
                    capture_timestamp_ms=tick.capture_timestamp_ms,
                    inference_latency_ms=latency_ms,
                )
                with self._condition:
                    if generation != self._generation:
                        self._counts["discarded_generation_results"] += 1
                        continue
                    self._local_frame_id += 1
                    self._last_publication = publication
                    self._max_decision_bytes = max(
                        self._max_decision_bytes, len(publication.payload)
                    )
                    self._counts["decisions"] += 1
                    self._latencies.append(latency_ms)
                    self._c1_latencies.append(c1_latency)
                    self._c2_latencies.append(c2_latency)
                    self._post_latencies.append(
                        max(0.0, latency_ms - c1_latency - c2_latency)
                    )
                    decision_now = self.monotonic()
                    self._decision_times.append(decision_now)
                    self._trim_rates_locked(decision_now)
                    self._emit_event_locked("DECISION")
                self.on_decision(publication)
            except Exception as exc:
                with self._condition:
                    self._counts["model_exceptions"] += 1
                    self._state = "ERROR"
                    self._last_error = f"{type(exc).__name__}: {exc}"[:200]
                    self._pending = None
                    self._emit_event_locked("INFERENCE_ERROR")
            finally:
                with self._condition:
                    self._inference_in_flight = False
                    self._condition.notify_all()

    def update_sse_metrics(self, *, subscribers: int, last_write_ms: int | None) -> None:
        with self._condition:
            self._sse_subscribers = max(0, int(subscribers))
            if last_write_ms is not None:
                self._last_sse_write_ms = int(last_write_ms)

    def status(self) -> Mapping[str, Any]:
        cuda_memory: dict[str, int | str | None] = {
            "device": None,
            "allocated_bytes": None,
            "reserved_bytes": None,
        }
        try:
            import torch

            if torch.cuda.is_available():
                cuda_memory = {
                    "device": torch.cuda.get_device_name(0),
                    "allocated_bytes": int(torch.cuda.memory_allocated(0)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(0)),
                }
        except Exception:
            pass
        with self._condition:
            now_mono = self.monotonic()
            self._trim_rates_locked(now_mono)
            latencies = list(self._latencies)
            c1_latencies = list(self._c1_latencies)
            c2_latencies = list(self._c2_latencies)
            post_latencies = list(self._post_latencies)
            latest = self._last_publication
            input_age_ms = (
                None
                if self._last_input_monotonic is None
                else round(max(0.0, now_mono - self._last_input_monotonic) * 1_000.0, 3)
            )
            input_fps = len(self._input_times) / 5.0
            decision_fps = len(self._decision_times) / 5.0
            status: dict[str, Any] = {
                "revision": self._revision,
                "state": self._state,
                "demo_id": self._active_demo_id,
                "session_id": self._internal_session_id,
                "generation": self._generation,
                "provenance": {
                    "camera_source": self._source_kind,
                    "telemetry_source": self._telemetry_source,
                    "inference_mode": "LIVE_MODEL" if self._active_demo_id else None,
                    "output_transport": "HTTPS_SSE",
                },
                "source_health": {
                    "road_age_ms": input_age_ms,
                    "cabin_age_ms": input_age_ms,
                    "ego_age_ms": input_age_ms,
                    "input_fps_5s": round(input_fps, 3),
                    "dropped_frames": self._counts["overflow_drops"],
                    "duplicate_frames": self._counts["duplicate_frames"],
                    "reordered_frames": self._counts["reordered_frames"],
                    "source_gaps": self._counts["source_gaps"],
                    "current_sequence": self._last_source_sequence,
                    "source_skew_ms": 0 if self._last_source_sequence is not None else None,
                    "last_error": self._last_error,
                },
                "pipeline_health": {
                    "gpu_model_ready": not self._closed,
                    "model_load_count": int(getattr(self.models, "load_count", 1)),
                    "decision_fps_5s": round(decision_fps, 3),
                    "latency_p50_ms": _percentile(latencies, 0.50),
                    "latency_p95_ms": _percentile(latencies, 0.95),
                    "latency_max_ms": round(max(latencies), 3) if latencies else None,
                    "component_latency_p50_ms": {
                        "c1": _percentile(c1_latencies, 0.50),
                        "c2": _percentile(c2_latencies, 0.50),
                        "c3_drive_quality_risk_and_mapping": _percentile(
                            post_latencies, 0.50
                        ),
                    },
                    "component_latency_p95_ms": {
                        "c1": _percentile(c1_latencies, 0.95),
                        "c2": _percentile(c2_latencies, 0.95),
                        "c3_drive_quality_risk_and_mapping": _percentile(
                            post_latencies, 0.95
                        ),
                    },
                    "subscriber_count": self._sse_subscribers,
                    "last_successful_sse_write_ms": self._last_sse_write_ms,
                    "max_decision_bytes": self._max_decision_bytes,
                    "cuda_memory": cuda_memory,
                },
                "counts": dict(self._counts),
                "prediction": None,
                "analysis_running": self._analysis_running,
            }
            if latest is not None:
                envelope = latest.envelope
                status["prediction"] = {
                    "decision_sequence": envelope.sequence,
                    "source_sequence": latest.source_sequence,
                    "capture_timestamp_ms": latest.capture_timestamp_ms,
                    "decision_timestamp_ms": envelope.decision_timestamp_ms,
                    "c1": dict(envelope.c1),
                    "c2": dict(envelope.c2),
                    "c3": dict(envelope.c3),
                    "drive_quality": dict(envelope.drive_quality),
                    "contextual_risk": dict(envelope.contextual_risk),
                    "health": dict(envelope.health),
                }
            return MappingProxyType(status)

    def analyze_bundle(self, bundle: PreparedDemoBundle) -> Mapping[str, Any]:
        with self._control_lock:
            with self._condition:
                if self._state != "STOPPED" or self._analysis_running:
                    raise FastBridgeError("Analyze is allowed only while STOPPED")
                self._analysis_running = True
                self._revision += 1
                self._emit_event_locked("ANALYZE_START")
            model_key = ":".join(
                f"{name}={digest}"
                for name, digest in sorted(self.models.model_hashes.items())
            )
            key = f"{bundle.manifest_sha256}:{model_key}"
            cached = self._analysis_cache.get(key)
            if cached is not None:
                with self._condition:
                    self._analysis_running = False
                    self._emit_event_locked("ANALYZE_CACHE_HIT")
                return cached
            session_id = self._internal_id(f"analyze-{bundle.trip_id}", self._revision)
            latencies: list[float] = []
            ttc_values: list[float] = []
            c1_warnings = 0
            c2_states: Counter[str] = Counter()
            dms_warnings = 0
            max_risk = 0.0
            final: Any = None
            try:
                with self._model_lock:
                    self.models.start_session(
                        session_id,
                        bundle.metadata,
                        expected_frames=len(bundle.frames),
                    )
                    for index in range(len(bundle.frames)):
                        tick = bundle.tick(
                            index,
                            session_id=session_id[:64],
                            capture_timestamp_ms=self.clock_ms(),
                        )
                        started = time.perf_counter_ns()
                        prediction = _warning_only(self.models.process(tick, frame_id=index))
                        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
                        ttc = float(prediction.c1.predicted_ttc_s)
                        if math.isfinite(ttc):
                            ttc_values.append(ttc)
                        c1_warnings += int(bool(prediction.c1.is_warning))
                        c2_states[str(prediction.c2.state)] += 1
                        dms_warnings += int(bool(prediction.c2.vss_signals().is_warning))
                        max_risk = max(max_risk, float(prediction.contextual_risk.score_pct))
                        final = prediction
                if final is None:
                    raise FastBridgeError("analysis produced no frames")
                summary: Mapping[str, Any] = MappingProxyType(
                    {
                        "semantics": "MODEL-PREDICTED SUMMARY — NOT GROUND TRUTH",
                        "demo_id": bundle.trip_id,
                        "source_manifest_sha256": bundle.manifest_sha256,
                        "model_hashes": dict(self.models.model_hashes),
                        "source_completeness_pct": 100.0,
                        "frames": len(bundle.frames),
                        "minimum_predicted_ttc_s": min(ttc_values) if ttc_values else None,
                        "c1_warning_count": c1_warnings,
                        "c2_state_distribution": dict(c2_states),
                        "dms_warning_count": dms_warnings,
                        "c3_final": final.c3.diagnostic_row(),
                        "drive_quality_final": final.drive_quality.diagnostic_row(),
                        "maximum_contextual_risk_pct": round(max_risk, 3),
                        "latency_p50_ms": _percentile(latencies, 0.50),
                        "latency_p95_ms": _percentile(latencies, 0.95),
                    }
                )
                self._analysis_cache[key] = summary
                return summary
            finally:
                with self._condition:
                    self._analysis_running = False
                    self._emit_event_locked("ANALYZE_FINISH")

    def close(self) -> None:
        with self._control_lock:
            with self._condition:
                if self._closed:
                    return
                thread = self._stop_locked("SHUTDOWN")
                self._closed = True
                self._condition.notify_all()
            if thread is not None:
                thread.join(2.0)
        self._worker.join(5.0)
        with self._model_lock:
            self.models.close()


__all__ = [
    "COHERENT_DWELL_TICKS",
    "ControlResult",
    "DECISION_MAX_BYTES",
    "DecisionPublication",
    "FastBridgeController",
    "FastBridgeError",
    "ModelPipeline",
    "RealModelPipeline",
    "SOURCE_HZ",
]
