"""UDP latest-snapshot transport and receiver guard for the Android HMI.

UDP is used only for a read-only display/recommendation snapshot.  It is not
an actuator channel.  The sender does not retry or queue old decisions; the
receiver accepts only a fresh, monotonically increasing snapshot per session.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import socket
import time
from typing import Callable, Protocol

from .carsky_decision import DecisionContractError, DecisionEnvelope


# IPv4 over the CarSky Ethernet bridge uses the normal 1,500-byte MTU. Keep
# the complete JSON snapshot at or below 1,472 bytes (MTU - IPv4 - UDP) so a
# safety display update never depends on IP fragmentation.
MAX_SAFE_UDP_PAYLOAD_BYTES = 1_472
DEFAULT_MAX_DATAGRAM_BYTES = MAX_SAFE_UDP_PAYLOAD_BYTES
MAX_UDP_PAYLOAD_BYTES = 65_507
DEFAULT_MAX_FUTURE_SKEW_MS = 1_000
MAX_RETIRED_SESSIONS = 64


class HmiTransportError(RuntimeError):
    """A decision snapshot could not be sent safely."""


class HmiDecisionRejected(ValueError):
    """The Android-side guard rejected a malformed, stale, or old snapshot."""


class DatagramSocket(Protocol):
    def sendto(self, data: bytes, address: tuple[str, int]) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class UdpPublishReceipt:
    session_id: str
    sequence: int
    bytes_sent: int
    destination: tuple[str, int]


def _default_socket_factory() -> DatagramSocket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class UdpDecisionPublisher:
    """Publish one strict-JSON decision envelope per UDP datagram.

    ``socket_factory`` is injectable so tests and a CarSky adapter can verify
    every byte without opening a real network socket.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_datagram_bytes: int = DEFAULT_MAX_DATAGRAM_BYTES,
        max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
        socket_factory: Callable[[], DatagramSocket] = _default_socket_factory,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("UDP HMI host must be a non-empty string")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
            raise ValueError("UDP HMI port must be an integer in [1, 65535]")
        if (
            not isinstance(max_datagram_bytes, int)
            or isinstance(max_datagram_bytes, bool)
            or not 1 <= max_datagram_bytes <= MAX_UDP_PAYLOAD_BYTES
        ):
            raise ValueError(
                "max_datagram_bytes must fit in one UDP payload"
            )
        if (
            not isinstance(max_future_skew_ms, int)
            or isinstance(max_future_skew_ms, bool)
            or max_future_skew_ms < 0
        ):
            raise ValueError("max_future_skew_ms must be a non-negative integer")
        self.destination = (host, port)
        self.max_datagram_bytes = max_datagram_bytes
        self.max_future_skew_ms = max_future_skew_ms
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._socket = socket_factory()
        self._closed = False
        self._last_session_id: str | None = None
        self._last_sequence: int | None = None
        self._retired_sessions: OrderedDict[str, None] = OrderedDict()

    def _is_retired_session(self, session_id: str) -> bool:
        if session_id not in self._retired_sessions:
            return False
        self._retired_sessions.move_to_end(session_id)
        return True

    def _commit_order(self, session_id: str, sequence: int) -> None:
        if self._last_session_id != session_id and self._last_session_id is not None:
            self._retired_sessions[self._last_session_id] = None
            self._retired_sessions.move_to_end(self._last_session_id)
            while len(self._retired_sessions) > MAX_RETIRED_SESSIONS:
                self._retired_sessions.popitem(last=False)
        self._last_session_id = session_id
        self._last_sequence = sequence

    def publish(self, envelope: DecisionEnvelope) -> UdpPublishReceipt:
        if self._closed:
            raise HmiTransportError("UDP HMI publisher is closed")
        if not isinstance(envelope, DecisionEnvelope):
            raise TypeError("publish requires a DecisionEnvelope")
        now_ms = self._clock_ms()
        if not isinstance(now_ms, int) or isinstance(now_ms, bool) or now_ms < 0:
            raise HmiTransportError(
                "UDP HMI clock must return a non-negative integer millisecond value"
            )
        if envelope.decision_timestamp_ms > now_ms + self.max_future_skew_ms:
            raise HmiTransportError(
                "refusing a decision timestamp too far in the future"
            )
        if envelope.is_expired(now_ms):
            raise HmiTransportError("refusing an expired decision snapshot")
        if self._last_session_id == envelope.session_id:
            if (
                self._last_sequence is not None
                and envelope.sequence <= self._last_sequence
            ):
                raise HmiTransportError(
                    "refusing duplicate/out-of-order decision sequence"
                )
        else:
            if self._is_retired_session(envelope.session_id):
                raise HmiTransportError(
                    "refusing a previously retired decision session"
                )

        payload = envelope.to_json_bytes()
        if len(payload) > self.max_datagram_bytes:
            raise HmiTransportError(
                f"decision datagram is {len(payload)} bytes, limit is "
                f"{self.max_datagram_bytes}"
            )
        try:
            sent = self._socket.sendto(payload, self.destination)
        except OSError as exc:
            raise HmiTransportError(
                f"cannot send Android HMI snapshot to {self.destination}"
            ) from exc
        if sent != len(payload):
            raise HmiTransportError(
                f"partial UDP decision send: {sent}/{len(payload)} bytes"
            )
        # Commit ordering only after the datagram was sent. A transient error
        # on the first frame must not poison every later sequence in a session.
        # The first successfully sent sequence is the late-join baseline.
        self._commit_order(envelope.session_id, envelope.sequence)
        return UdpPublishReceipt(
            session_id=envelope.session_id,
            sequence=envelope.sequence,
            bytes_sent=sent,
            destination=self.destination,
        )

    def close(self) -> None:
        if not self._closed:
            self._socket.close()
            self._closed = True

    def __enter__(self) -> "UdpDecisionPublisher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class HmiDecisionGuard:
    """Receiver reference using a local monotonic TTL deadline.

    Producer wall-clock timestamps remain useful diagnostics but are never
    compared with the receiver clock: two CarSky guests need not have
    synchronized wall clocks. The strict envelope parser still validates
    ``expires_at_ms == decision_timestamp_ms + ttl_ms``. On acceptance, the
    receiver starts a fresh local ``ttl_ms`` countdown on its monotonic clock.

    The first packet observed in a previously unknown session establishes its
    ordering baseline, so a receiver may join a live stream after sequence 0.
    Only the 64 most recently retired sessions are replay-protected. If a
    producer process reuses a session identifier, sequence 0 may establish a
    new incarnation only after the current stream's local TTL has expired.
    """

    def __init__(
        self,
        *,
        max_datagram_bytes: int = DEFAULT_MAX_DATAGRAM_BYTES,
        monotonic_ms: Callable[[], int] | None = None,
    ) -> None:
        if (
            not isinstance(max_datagram_bytes, int)
            or isinstance(max_datagram_bytes, bool)
            or not 1 <= max_datagram_bytes <= MAX_UDP_PAYLOAD_BYTES
        ):
            raise ValueError(
                "max_datagram_bytes must fit in one UDP payload"
            )
        self.max_datagram_bytes = max_datagram_bytes
        self._monotonic_ms = monotonic_ms or (
            lambda: time.monotonic_ns() // 1_000_000
        )
        self._last_session_id: str | None = None
        self._last_sequence: int | None = None
        self._retired_sessions: OrderedDict[str, None] = OrderedDict()
        self._dropped_sequences = 0
        self._session_restarts = 0
        self._current: DecisionEnvelope | None = None
        self._current_deadline_ms: int | None = None
        self._last_accepted_deadline_ms: int | None = None

    @property
    def dropped_sequences(self) -> int:
        return self._dropped_sequences

    @property
    def retired_session_count(self) -> int:
        return len(self._retired_sessions)

    @property
    def session_restarts(self) -> int:
        return self._session_restarts

    def _is_retired_session(self, session_id: str) -> bool:
        if session_id not in self._retired_sessions:
            return False
        self._retired_sessions.move_to_end(session_id)
        return True

    def _retire_active_session(self) -> None:
        if self._last_session_id is None:
            return
        self._retired_sessions[self._last_session_id] = None
        self._retired_sessions.move_to_end(self._last_session_id)
        while len(self._retired_sessions) > MAX_RETIRED_SESSIONS:
            self._retired_sessions.popitem(last=False)

    def _can_restart_after_stale(
        self, envelope: DecisionEnvelope, now_ms: int
    ) -> bool:
        return (
            envelope.sequence == 0
            and self._last_accepted_deadline_ms is not None
            and now_ms >= self._last_accepted_deadline_ms
        )

    def accept(
        self,
        datagram: bytes | bytearray | memoryview,
        *,
        received_monotonic_ms: int | None = None,
    ) -> DecisionEnvelope:
        payload = bytes(datagram)
        if len(payload) > self.max_datagram_bytes:
            raise HmiDecisionRejected(
                f"decision datagram exceeds {self.max_datagram_bytes} bytes"
            )
        now_ms = (
            self._monotonic_ms()
            if received_monotonic_ms is None
            else received_monotonic_ms
        )
        if not isinstance(now_ms, int) or isinstance(now_ms, bool) or now_ms < 0:
            raise ValueError(
                "received_monotonic_ms must be a non-negative integer"
            )
        try:
            envelope = DecisionEnvelope.from_json_bytes(payload)
        except DecisionContractError as exc:
            raise HmiDecisionRejected(str(exc)) from exc
        same_session = self._last_session_id == envelope.session_id
        stale_restart = self._can_restart_after_stale(envelope, now_ms)
        if same_session:
            if (
                self._last_sequence is not None
                and envelope.sequence <= self._last_sequence
            ):
                if not stale_restart:
                    raise HmiDecisionRejected(
                        "duplicate/out-of-order decision sequence"
                    )
                self._session_restarts += 1
        else:
            if self._is_retired_session(envelope.session_id):
                if not stale_restart:
                    raise HmiDecisionRejected(
                        "previously retired decision session"
                    )
                # This is a new producer incarnation after an observable quiet
                # dwell, not a fresh replay while another stream is live.
                del self._retired_sessions[envelope.session_id]
            self._retire_active_session()
            if self._last_session_id is not None:
                self._session_restarts += 1

        if (
            same_session
            and not stale_restart
            and self._last_sequence is not None
            and envelope.sequence > self._last_sequence + 1
        ):
            self._dropped_sequences += envelope.sequence - self._last_sequence - 1

        self._last_session_id = envelope.session_id
        self._last_sequence = envelope.sequence
        self._current = envelope
        self._current_deadline_ms = now_ms + envelope.ttl_ms
        self._last_accepted_deadline_ms = self._current_deadline_ms
        return envelope

    def current(
        self, *, now_monotonic_ms: int | None = None
    ) -> DecisionEnvelope | None:
        now = (
            self._monotonic_ms()
            if now_monotonic_ms is None
            else now_monotonic_ms
        )
        if not isinstance(now, int) or isinstance(now, bool) or now < 0:
            raise ValueError("now_monotonic_ms must be a non-negative integer")
        if (
            self._current is not None
            and self._current_deadline_ms is not None
            and now >= self._current_deadline_ms
        ):
            self._current = None
            self._current_deadline_ms = None
        return self._current

    def reset(self) -> None:
        self._last_session_id = None
        self._last_sequence = None
        self._retired_sessions.clear()
        self._dropped_sequences = 0
        self._session_restarts = 0
        self._current = None
        self._current_deadline_ms = None
        self._last_accepted_deadline_ms = None


__all__ = [
    "DEFAULT_MAX_DATAGRAM_BYTES",
    "DEFAULT_MAX_FUTURE_SKEW_MS",
    "DatagramSocket",
    "HmiDecisionGuard",
    "HmiDecisionRejected",
    "HmiTransportError",
    "MAX_SAFE_UDP_PAYLOAD_BYTES",
    "MAX_RETIRED_SESSIONS",
    "MAX_UDP_PAYLOAD_BYTES",
    "UdpDecisionPublisher",
    "UdpPublishReceipt",
]
