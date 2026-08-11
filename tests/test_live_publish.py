from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import pytest

from safeloop.live_publish import (
    LatestOnlyKuksaMirror,
    LivePublishClosedError,
    LivePublishOrderError,
)


@dataclass(frozen=True)
class Envelope:
    session_id: str
    sequence: int


class RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[object, Envelope]] = []
        self.publish_threads: list[int] = []
        self.close_threads: list[int] = []
        self.closed = 0

    def publish(self, frame: object, envelope: Envelope) -> None:
        self.calls.append((frame, envelope))
        self.publish_threads.append(threading.get_ident())

    def close(self) -> None:
        self.closed += 1
        self.close_threads.append(threading.get_ident())


class BlockingPublisher(RecordingPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def publish(self, frame: object, envelope: Envelope) -> None:
        super().publish(frame, envelope)
        self.started.set()
        self.release.wait()


class FailFirstPublisher(RecordingPublisher):
    def publish(self, frame: object, envelope: Envelope) -> None:
        super().publish(frame, envelope)
        if len(self.calls) == 1:
            raise RuntimeError("broker unavailable")


def test_worker_owns_publish_and_close_without_kuksa_dependency() -> None:
    publisher = RecordingPublisher()
    producer_thread = threading.get_ident()
    mirror = LatestOnlyKuksaMirror(publisher)

    receipt = mirror.publish("frame-0", Envelope("drive-a", 0))
    assert receipt.session_id == "drive-a"
    assert receipt.sequence == 0
    assert receipt.coalesced_previous is False
    assert mirror.wait_until_idle(1.0)
    assert mirror.close()

    assert publisher.calls == [("frame-0", Envelope("drive-a", 0))]
    assert publisher.closed == 1
    assert publisher.publish_threads[0] != producer_thread
    assert publisher.close_threads == publisher.publish_threads
    assert mirror.summary() == {
        "accepted": 1,
        "coalesced": 0,
        "sent": 1,
        "errors": 0,
        "latest_error": None,
        "pending": False,
        "in_flight": False,
        "closed": True,
        "worker_alive": False,
    }


def test_pending_slot_coalesces_to_latest_sequence() -> None:
    publisher = BlockingPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)
    try:
        mirror.publish("frame-0", Envelope("drive-a", 0))
        assert publisher.started.wait(1.0)

        first_pending = mirror.publish("frame-1", Envelope("drive-a", 1))
        latest = mirror.publish("frame-2", Envelope("drive-a", 2))
        assert first_pending.coalesced_previous is False
        assert latest.coalesced_previous is True

        publisher.release.set()
        assert mirror.wait_until_idle(1.0)
        assert [envelope.sequence for _, envelope in publisher.calls] == [0, 2]
        status = mirror.status
        assert (status.accepted, status.coalesced, status.sent) == (3, 1, 2)
        assert status.errors == 0
    finally:
        publisher.release.set()
        assert mirror.close(timeout_s=1.0)


def test_publish_does_not_wait_for_blocked_broker_call() -> None:
    publisher = BlockingPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)
    try:
        mirror.publish("frame-0", Envelope("drive-a", 0))
        assert publisher.started.wait(1.0)

        completed = threading.Event()

        def offer_newer() -> None:
            mirror.publish("frame-1", Envelope("drive-a", 1))
            completed.set()

        producer = threading.Thread(target=offer_newer)
        producer.start()
        producer.join(0.25)
        assert completed.is_set(), "producer waited for KUKSA broker I/O"
    finally:
        publisher.release.set()
        assert mirror.close(timeout_s=1.0)


def test_publisher_exception_is_reported_and_worker_continues() -> None:
    publisher = FailFirstPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)

    mirror.publish("frame-0", Envelope("drive-a", 0))
    assert mirror.wait_until_idle(1.0)
    first_status = mirror.status
    assert first_status.accepted == 1
    assert first_status.sent == 0
    assert first_status.errors == 1
    assert first_status.latest_error == (
        "publish: RuntimeError: broker unavailable"
    )

    mirror.publish("frame-1", Envelope("drive-a", 1))
    assert mirror.wait_until_idle(1.0)
    assert mirror.close()
    status = mirror.status
    assert status.accepted == 2
    assert status.sent == 1
    assert status.errors == 1
    # Keep the latest failure for diagnostics even after recovery.
    assert status.latest_error == "publish: RuntimeError: broker unavailable"


def test_order_guard_rejects_duplicates_old_frames_and_retired_sessions() -> None:
    publisher = RecordingPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)
    try:
        # The first snapshot may be a late join to an already running session.
        mirror.publish("a-5", Envelope("drive-a", 5))
        with pytest.raises(LivePublishOrderError, match="duplicate|decreasing"):
            mirror.publish("a-5-again", Envelope("drive-a", 5))
        with pytest.raises(LivePublishOrderError, match="duplicate|decreasing"):
            mirror.publish("a-4", Envelope("drive-a", 4))
        with pytest.raises(LivePublishOrderError, match="sequence zero"):
            mirror.publish("b-7", Envelope("drive-b", 7))

        mirror.publish("b-0", Envelope("drive-b", 0))
        with pytest.raises(LivePublishOrderError, match="retired"):
            mirror.publish("a-6", Envelope("drive-a", 6))

        assert mirror.status.accepted == 2
        assert mirror.wait_until_idle(1.0)
        sent = [(item.session_id, item.sequence) for _, item in publisher.calls]
        assert sent == sorted(sent, key=lambda item: (item[0] == "drive-b", item[1]))
        assert sent[-1] == ("drive-b", 0)
    finally:
        assert mirror.close(timeout_s=1.0)


def test_close_drains_latest_is_idempotent_and_rejects_new_work() -> None:
    publisher = RecordingPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)
    mirror.publish("frame-0", Envelope("drive-a", 0))

    assert mirror.close(timeout_s=1.0)
    assert publisher.calls == [("frame-0", Envelope("drive-a", 0))]
    assert publisher.closed == 1
    assert mirror.close(timeout_s=0.0)
    assert publisher.closed == 1
    with pytest.raises(LivePublishClosedError, match="closed"):
        mirror.publish("frame-1", Envelope("drive-a", 1))


def test_close_is_bounded_when_wrapped_publisher_is_stuck() -> None:
    publisher = BlockingPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)
    mirror.publish("frame-0", Envelope("drive-a", 0))
    assert publisher.started.wait(1.0)

    started = time.monotonic()
    assert mirror.close(timeout_s=0.02) is False
    assert time.monotonic() - started < 0.25
    assert mirror.status.worker_alive is True
    with pytest.raises(LivePublishClosedError):
        mirror.publish("frame-1", Envelope("drive-a", 1))

    publisher.release.set()
    assert mirror.close(timeout_s=1.0)
    assert publisher.closed == 1


def test_close_exception_is_contained_and_counted() -> None:
    class CloseFailure(RecordingPublisher):
        def close(self) -> None:
            super().close()
            raise RuntimeError("close failed")

    publisher = CloseFailure()
    mirror = LatestOnlyKuksaMirror(publisher)
    assert mirror.close(timeout_s=1.0)
    status = mirror.status
    assert status.errors == 1
    assert status.latest_error == "close: RuntimeError: close failed"
    assert status.worker_alive is False


@pytest.mark.parametrize(
    ("session_id", "sequence"),
    [("", 0), ("drive-a", -1), ("drive-a", True), ("drive-a", 1.5)],
)
def test_invalid_order_fields_never_enter_queue(
    session_id: str, sequence: object
) -> None:
    publisher = RecordingPublisher()
    mirror = LatestOnlyKuksaMirror(publisher)
    try:
        with pytest.raises(ValueError):
            mirror.publish("frame", Envelope(session_id, sequence))  # type: ignore[arg-type]
        assert mirror.status.accepted == 0
    finally:
        assert mirror.close(timeout_s=1.0)
