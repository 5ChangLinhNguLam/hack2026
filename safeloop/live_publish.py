"""Non-blocking, latest-only KUKSA publishing for live inference.

The inference loop must not wait for an observability broker.  This module
therefore keeps at most one not-yet-started snapshot: a newer accepted frame
replaces the pending one while the worker is writing the current snapshot.
There is deliberately no retry queue because replaying stale vehicle state is
worse than dropping it.

The concrete KUKSA implementation is dependency-injected.  Importing this
module does not require ``kuksa-client`` and the wrapped publisher is only
called (and closed) by the worker thread.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Protocol


MAX_RETIRED_SESSIONS = 256
DEFAULT_CLOSE_TIMEOUT_S = 2.0


class LiveKuksaPublisher(Protocol):
    """Minimal contract implemented by ``KuksaSignalPublisher``."""

    def publish(self, frame: Any, envelope: Any) -> Any: ...

    def close(self) -> None: ...


class LivePublishError(RuntimeError):
    """Base error raised before a snapshot enters the worker queue."""


class LivePublishClosedError(LivePublishError):
    """A producer tried to publish after shutdown began."""


class LivePublishOrderError(LivePublishError):
    """A producer supplied a duplicate, stale, or retired-session snapshot."""


class LivePublishCloseTimeout(LivePublishError):
    """The worker did not finish within the configured bounded close wait."""


@dataclass(frozen=True)
class LivePublishReceipt:
    """Immediate acknowledgement that a snapshot entered the latest slot."""

    session_id: str
    generation: int
    sequence: int
    coalesced_previous: bool


@dataclass(frozen=True)
class LivePublishStatus:
    """Thread-safe operational counters for the asynchronous mirror."""

    accepted: int
    coalesced: int
    sent: int
    errors: int
    latest_error: str | None
    pending: bool
    in_flight: bool
    closed: bool
    worker_alive: bool


@dataclass(frozen=True)
class _Snapshot:
    frame: Any
    envelope: Any
    session_id: str
    generation: int
    sequence: int


def _bounded_timeout(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number >= 0")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number >= 0") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite number >= 0")
    return result


def _snapshot(frame: Any, envelope: Any) -> _Snapshot:
    session_id = getattr(envelope, "session_id", None)
    generation = getattr(envelope, "generation", 0)
    sequence = getattr(envelope, "sequence", None)
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("envelope.session_id must be a non-empty string")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("envelope.sequence must be a non-negative integer")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("envelope.generation must be a non-negative integer")
    return _Snapshot(
        frame=frame,
        envelope=envelope,
        session_id=session_id,
        generation=generation,
        sequence=sequence,
    )


class LatestOnlyKuksaMirror:
    """Mirror live snapshots without ever blocking inference on broker I/O.

    ``publish`` only takes a short in-process lock and never invokes the
    wrapped publisher. For one generation, sequences must increase strictly.
    A capacity-one upstream handoff may coalesce the sequence-zero heartbeat,
    so the first observed snapshot of a newer generation or session may have
    any valid sequence. A retired session or generation can never return.
    This prevents races between producers from putting an older snapshot on
    the wire after a newer accepted snapshot without requiring a retry queue.

    ``close`` stops acceptance immediately, drains the newest pending
    snapshot, asks the worker-owned publisher to close, and joins for a
    bounded duration.  It returns ``False`` if a broken publisher remains
    blocked; callers may inspect status and invoke ``close`` again safely.
    """

    def __init__(
        self,
        publisher: LiveKuksaPublisher,
        *,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
        thread_name: str = "safeloop-kuksa-mirror",
    ) -> None:
        if publisher is None:
            raise TypeError("publisher is required")
        if not callable(getattr(publisher, "publish", None)):
            raise TypeError("publisher.publish must be callable")
        if not callable(getattr(publisher, "close", None)):
            raise TypeError("publisher.close must be callable")
        if not isinstance(thread_name, str) or not thread_name.strip():
            raise ValueError("thread_name must be a non-empty string")

        self._publisher = publisher
        self._close_timeout_s = _bounded_timeout(
            close_timeout_s, field="close_timeout_s"
        )
        self._condition = threading.Condition()
        self._pending: _Snapshot | None = None
        self._in_flight = False
        self._accepting = True
        self._closing = False
        self._worker_done = False

        self._accepted = 0
        self._coalesced = 0
        self._sent = 0
        self._errors = 0
        self._latest_error: str | None = None

        self._active_session: str | None = None
        self._active_generation: int | None = None
        self._highest_accepted_sequence: int | None = None
        self._retired_sessions: OrderedDict[str, None] = OrderedDict()

        self._worker = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._worker.start()

    def _retire(self, session_id: str) -> None:
        self._retired_sessions[session_id] = None
        self._retired_sessions.move_to_end(session_id)
        while len(self._retired_sessions) > MAX_RETIRED_SESSIONS:
            self._retired_sessions.popitem(last=False)

    def _accept_order(self, item: _Snapshot) -> None:
        if self._active_session is None:
            # A mirror may attach after inference has already started, so the
            # first observed sequence does not have to be zero.
            self._active_session = item.session_id
            self._active_generation = item.generation
            self._highest_accepted_sequence = item.sequence
            return

        if item.session_id == self._active_session:
            assert self._active_generation is not None
            assert self._highest_accepted_sequence is not None
            if item.generation < self._active_generation:
                raise LivePublishOrderError(
                    "stale generation cannot return to the active session"
                )
            if item.generation > self._active_generation:
                # The upstream latest-only slot may have coalesced the reset
                # heartbeat. Treat this as the first observed sequence for
                # the new generation rather than blocking every later result.
                self._active_generation = item.generation
                self._highest_accepted_sequence = item.sequence
                return
            if item.sequence <= self._highest_accepted_sequence:
                raise LivePublishOrderError(
                    "duplicate or decreasing sequence for active session"
                )
            self._highest_accepted_sequence = item.sequence
            return

        if item.session_id in self._retired_sessions:
            self._retired_sessions.move_to_end(item.session_id)
            raise LivePublishOrderError("retired session cannot become active again")
        self._retire(self._active_session)
        self._active_session = item.session_id
        self._active_generation = item.generation
        self._highest_accepted_sequence = item.sequence

    def publish(self, frame: Any, envelope: Any) -> LivePublishReceipt:
        """Offer one live snapshot without calling or waiting for KUKSA I/O."""

        item = _snapshot(frame, envelope)
        with self._condition:
            if not self._accepting:
                raise LivePublishClosedError("live KUKSA mirror is closed")
            self._accept_order(item)
            replaced = self._pending is not None
            if replaced:
                self._coalesced += 1
            self._pending = item
            self._accepted += 1
            self._condition.notify()
        return LivePublishReceipt(
            session_id=item.session_id,
            generation=item.generation,
            sequence=item.sequence,
            coalesced_previous=replaced,
        )

    @staticmethod
    def _error_text(operation: str, error: Exception) -> str:
        return f"{operation}: {type(error).__name__}: {error}"

    def _record_error(self, operation: str, error: Exception) -> None:
        with self._condition:
            self._errors += 1
            self._latest_error = self._error_text(operation, error)
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._pending is not None or self._closing
                    )
                    if self._pending is None:
                        # No producer can add work after _closing becomes true.
                        break
                    item = self._pending
                    self._pending = None
                    self._in_flight = True

                try:
                    self._publisher.publish(item.frame, item.envelope)
                except Exception as exc:
                    # Broker failures are observability failures, not inference
                    # failures. Do not retry a stale snapshot or kill the worker.
                    self._record_error("publish", exc)
                else:
                    with self._condition:
                        self._sent += 1
                finally:
                    with self._condition:
                        self._in_flight = False
                        self._condition.notify_all()
        finally:
            try:
                self._publisher.close()
            except Exception as exc:
                self._record_error("close", exc)
            finally:
                with self._condition:
                    self._worker_done = True
                    self._condition.notify_all()

    @property
    def status(self) -> LivePublishStatus:
        with self._condition:
            return LivePublishStatus(
                accepted=self._accepted,
                coalesced=self._coalesced,
                sent=self._sent,
                errors=self._errors,
                latest_error=self._latest_error,
                pending=self._pending is not None,
                in_flight=self._in_flight,
                closed=not self._accepting,
                worker_alive=self._worker.is_alive(),
            )

    def summary(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for health and run reports."""

        current = self.status
        return {
            "accepted": current.accepted,
            "coalesced": current.coalesced,
            "sent": current.sent,
            "errors": current.errors,
            "latest_error": current.latest_error,
            "pending": current.pending,
            "in_flight": current.in_flight,
            "closed": current.closed,
            "worker_alive": current.worker_alive,
        }

    def wait_until_idle(self, timeout_s: float) -> bool:
        """Wait only for tests/health checks; producers should never call this."""

        timeout = _bounded_timeout(timeout_s, field="timeout_s")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending is not None or self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, timeout_s: float | None = None) -> bool:
        """Drain the latest snapshot and stop within a bounded join duration."""

        timeout = (
            self._close_timeout_s
            if timeout_s is None
            else _bounded_timeout(timeout_s, field="timeout_s")
        )
        with self._condition:
            self._accepting = False
            self._closing = True
            self._condition.notify_all()

        if threading.current_thread() is self._worker:
            return False
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def __enter__(self) -> "LatestOnlyKuksaMirror":
        return self

    def __exit__(self, *_: object) -> None:
        if not self.close():
            raise LivePublishCloseTimeout(
                "live KUKSA worker did not stop before close timeout"
            )


__all__ = [
    "DEFAULT_CLOSE_TIMEOUT_S",
    "LatestOnlyKuksaMirror",
    "LiveKuksaPublisher",
    "LivePublishClosedError",
    "LivePublishCloseTimeout",
    "LivePublishError",
    "LivePublishOrderError",
    "LivePublishReceipt",
    "LivePublishStatus",
]
