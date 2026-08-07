"""Versioned telemetry contract and sinks for the SafeLoop trip replayer.

The contract deliberately contains only fields available in both practice
and redacted scoring trips. Ground truth, active-event labels, driver state,
and practice-only position data must never leak into this stream.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Protocol, TextIO
from urllib import error, parse, request

from tripkit.types import FrameBundle

SCHEMA_VERSION = "safeloop.telemetry.v1"
SOURCE_NAME = "trip-replayer"


class TelemetryContractError(ValueError):
    """A frame cannot be represented by the SafeLoop telemetry contract."""


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryContractError(f"{field} phải là số, gặp {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise TelemetryContractError(f"{field} phải hữu hạn, gặp {value!r}")
    return result


@dataclass(frozen=True)
class EgoTelemetry:
    """Vehicle signals preserved in every released trip."""

    speed_kmh: float
    longitudinal_accel_mps2: float
    lateral_accel_mps2: float


@dataclass(frozen=True)
class TelemetryMessage:
    """One deterministic source frame plus a runtime emission timestamp."""

    schema_version: str
    message_id: str
    source: str
    trip_id: str
    frame_id: int
    timestamp_ms: int
    emitted_monotonic_ns: int
    ego: EgoTelemetry

    @classmethod
    def from_bundle(
        cls,
        bundle: FrameBundle,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> "TelemetryMessage":
        ego = bundle.ego
        if not isinstance(ego, dict):
            raise TelemetryContractError(
                f"{bundle.trip_id}:{bundle.frame_id} thiếu object ego"
            )

        frame_id = bundle.frame_id
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise TelemetryContractError(f"frame_id không hợp lệ: {frame_id!r}")
        timestamp_s = _finite_number(bundle.timestamp, "timestamp")
        if timestamp_s < 0:
            raise TelemetryContractError(f"timestamp phải >= 0, gặp {timestamp_s}")

        return cls(
            schema_version=SCHEMA_VERSION,
            message_id=f"{bundle.trip_id}:{frame_id}",
            source=SOURCE_NAME,
            trip_id=bundle.trip_id,
            frame_id=frame_id,
            timestamp_ms=round(timestamp_s * 1000),
            emitted_monotonic_ns=clock_ns(),
            ego=EgoTelemetry(
                speed_kmh=_finite_number(ego.get("speed_kmh"), "ego.speed_kmh"),
                longitudinal_accel_mps2=_finite_number(
                    ego.get("longitudinal_accel"), "ego.longitudinal_accel"
                ),
                lateral_accel_mps2=_finite_number(
                    ego.get("lateral_accel"), "ego.lateral_accel"
                ),
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        # allow_nan=False makes the wire contract fail fast instead of emitting
        # non-standard JSON tokens such as NaN/Infinity.
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )


class TelemetrySink(Protocol):
    def publish(self, message: TelemetryMessage) -> None: ...

    def close(self) -> None: ...


class JsonLinesSink:
    """Write one compact JSON message per line; useful locally and in CI."""

    def __init__(self, stream: TextIO | None = None, *, flush: bool = True):
        self.stream = stream if stream is not None else sys.stdout
        self.flush = flush

    def publish(self, message: TelemetryMessage) -> None:
        self.stream.write(message.to_json() + "\n")
        if self.flush:
            self.stream.flush()

    def close(self) -> None:
        # The caller owns stdout or a supplied stream; never close it here.
        if self.flush:
            self.stream.flush()


class CarSkyApiError(RuntimeError):
    """A CarSky REST request failed or returned an invalid response."""


CarSkyScalar = str | int | float | bool | None


class CarSkyRestClient:
    """Small client for the signal routes documented by A8 OpenAPI 1.0.0."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        room_id: str,
        node_key: str,
        *,
        timeout_s: float = 10.0,
        urlopen: Callable | None = None,
    ):
        if not isinstance(base_url, str):
            raise ValueError("A8_URL/base_url CarSky phải là URL http(s) đầy đủ")
        parsed_url = parse.urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("A8_URL/base_url CarSky phải là URL http(s) đầy đủ")
        if not api_key.strip():
            raise ValueError("Thiếu A8_API_KEY")
        if not room_id.strip():
            raise ValueError("Thiếu CarSky roomId")
        if not node_key.strip():
            raise ValueError("Thiếu CarSky nodeKey")
        if timeout_s <= 0:
            raise ValueError("timeout_s phải > 0")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.room_id = room_id
        self.node_key = node_key
        self.timeout_s = timeout_s
        self._urlopen = urlopen if urlopen is not None else request.urlopen

    def _node_path(self, suffix: str = "") -> str:
        room_id = parse.quote(self.room_id, safe="")
        node_key = parse.quote(self.node_key, safe="")
        return f"/api/v1/signals/{room_id}/{node_key}{suffix}"

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        body = None
        headers = {"Accept": "application/json", "X-API-Key": self.api_key}
        if payload is not None:
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"

        http_request = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._urlopen(http_request, timeout=self.timeout_s) as response:
                response_body = response.read()
        except error.HTTPError as exc:
            response_body = exc.read()
            detail = response_body.decode("utf-8", errors="replace")[:1000]
            raise CarSkyApiError(
                f"CarSky HTTP {exc.code} tại {path}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise CarSkyApiError(
                f"Không kết nối được CarSky tại {self.base_url}: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise CarSkyApiError(f"Lỗi I/O khi gọi CarSky: {exc}") from exc

        try:
            decoded = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CarSkyApiError(f"CarSky trả JSON không hợp lệ tại {path}") from exc
        if not isinstance(decoded, dict):
            raise CarSkyApiError(f"CarSky trả response không phải object tại {path}")
        return decoded

    def list_signals(self, pattern: str | None = None) -> list[dict]:
        path = self._node_path()
        if pattern:
            path += "?" + parse.urlencode({"pattern": pattern})
        response = self._request_json("GET", path)
        signals = response.get("signals")
        if not isinstance(signals, list):
            raise CarSkyApiError("Response list signals thiếu array 'signals'")
        return signals

    def publish_signals(
        self, signals: Iterable[tuple[str, CarSkyScalar]]
    ) -> dict:
        items = [{"path": path, "value": value} for path, value in signals]
        if not 1 <= len(items) <= 64:
            raise ValueError("Mỗi batch CarSky phải có 1..64 signal")
        if any(not item["path"].strip() for item in items):
            raise ValueError("CarSky signal path không được rỗng")

        response = self._request_json(
            "POST", self._node_path("/actuate"), {"signals": items}
        )
        if response.get("ok") is not True:
            raise CarSkyApiError("CarSky không xác nhận ok=true khi publish signal")
        sent = response.get("sent")
        if not isinstance(sent, (int, float)) or isinstance(sent, bool):
            raise CarSkyApiError("Response actuate thiếu số lượng 'sent'")
        if sent != len(items):
            raise CarSkyApiError(
                f"CarSky xác nhận sent={sent}, dự kiến {len(items)} signal"
            )
        return response


@dataclass(frozen=True)
class CarSkySignalPaths:
    """Mapping from the SafeLoop envelope to signals present in a live node."""

    speed_kmh: str = "Vehicle.Speed"
    longitudinal_accel_mps2: str = "Vehicle.Acceleration.Longitudinal"
    lateral_accel_mps2: str = "Vehicle.Acceleration.Lateral"
    trip_id: str | None = None
    frame_id: str | None = None
    timestamp_ms: str | None = None

    def required(self) -> tuple[str, ...]:
        paths = (
            self.speed_kmh,
            self.longitudinal_accel_mps2,
            self.lateral_accel_mps2,
        )
        if any(not path.strip() for path in paths):
            raise ValueError("Ba CarSky ego signal path không được rỗng")
        optional = tuple(
            path
            for path in (self.trip_id, self.frame_id, self.timestamp_ms)
            if path is not None
        )
        if any(not path.strip() for path in optional):
            raise ValueError("CarSky metadata signal path không được rỗng")
        combined = paths + optional
        if len(set(combined)) != len(combined):
            raise ValueError("Các CarSky signal path phải khác nhau")
        return combined

    def values(self, message: TelemetryMessage) -> list[tuple[str, CarSkyScalar]]:
        values: list[tuple[str, CarSkyScalar]] = [
            (self.speed_kmh, message.ego.speed_kmh),
            (
                self.longitudinal_accel_mps2,
                message.ego.longitudinal_accel_mps2,
            ),
            (self.lateral_accel_mps2, message.ego.lateral_accel_mps2),
        ]
        if self.trip_id is not None:
            values.append((self.trip_id, message.trip_id))
        if self.frame_id is not None:
            values.append((self.frame_id, message.frame_id))
        if self.timestamp_ms is not None:
            values.append((self.timestamp_ms, message.timestamp_ms))
        return values


class CarSkyRestSink:
    """Publish telemetry as one CarSky REST signal batch per replay frame.

    The request intentionally omits ``actuate``. Per the A8 OpenAPI contract,
    that performs a direct value write for KUKSA sensor/state simulation;
    provider-based actuator commands are outside this replayer's scope.
    """

    def __init__(
        self,
        client: CarSkyRestClient,
        paths: CarSkySignalPaths | None = None,
        *,
        validate_signals: bool = True,
    ):
        self.client = client
        self.paths = paths if paths is not None else CarSkySignalPaths()
        required = set(self.paths.required())
        if validate_signals:
            available = {
                signal.get("path")
                for signal in client.list_signals()
                if isinstance(signal, dict) and isinstance(signal.get("path"), str)
            }
            missing = sorted(required - available)
            if missing:
                raise CarSkyApiError(
                    "Signal path chưa có trên CarSky node: " + ", ".join(missing)
                )

    def publish(self, message: TelemetryMessage) -> None:
        self.client.publish_signals(self.paths.values(message))

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class PublishStats:
    count: int
    first_frame_id: int | None
    last_frame_id: int | None
    elapsed_s: float


def publish_replay(
    bundles: Iterable[FrameBundle],
    sink: TelemetrySink,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> PublishStats:
    """Convert each replay bundle to v1 telemetry and publish in order."""

    started_ns = clock_ns()
    count = 0
    first_frame_id = None
    last_frame_id = None
    previous_frame_id = None
    previous_timestamp_ms = None

    for bundle in bundles:
        message = TelemetryMessage.from_bundle(bundle, clock_ns=clock_ns)
        if previous_frame_id is not None and message.frame_id <= previous_frame_id:
            raise TelemetryContractError(
                f"frame_id không tăng tại {message.message_id}: "
                f"{message.frame_id} <= {previous_frame_id}"
            )
        if previous_timestamp_ms is not None and message.timestamp_ms < previous_timestamp_ms:
            raise TelemetryContractError(
                f"timestamp giảm tại {message.message_id}: "
                f"{message.timestamp_ms} < {previous_timestamp_ms}"
            )
        sink.publish(message)
        if first_frame_id is None:
            first_frame_id = message.frame_id
        last_frame_id = message.frame_id
        previous_frame_id = message.frame_id
        previous_timestamp_ms = message.timestamp_ms
        count += 1

    elapsed_s = max(0, clock_ns() - started_ns) / 1_000_000_000
    return PublishStats(count, first_frame_id, last_frame_id, elapsed_s)
