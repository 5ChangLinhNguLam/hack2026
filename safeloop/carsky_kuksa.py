"""KUKSA signal mirror for the SafeLoop CarSky edge runtime.

The Android decision envelope is the atomic HMI contract.  These VSS values
are deliberately a smaller, standard-path mirror for CarSky Signal Watch and
independent observers.  C1 does not estimate calibrated distance, so this
module must never publish an obstacle-distance value.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

from .carsky_decision import DecisionEnvelope
from .carsky_hmi import MAX_RETIRED_SESSIONS


STANDARD_SIGNAL_TYPES: Mapping[str, str] = {
    "Vehicle.Speed": "FLOAT",
    "Vehicle.Acceleration.Longitudinal": "FLOAT",
    "Vehicle.Acceleration.Lateral": "FLOAT",
    "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap": "UINT32",
    "Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning": "BOOLEAN",
    "Vehicle.Driver.AttentiveProbability": "FLOAT",
    "Vehicle.Driver.DistractionLevel": "FLOAT",
    "Vehicle.Driver.FatigueLevel": "FLOAT",
    "Vehicle.Driver.IsEyesOnRoad": "BOOLEAN",
    "Vehicle.ADAS.DMS.IsWarning": "BOOLEAN",
}

# The stock VSS path is a required uint32 and cannot carry JSON null.  Keeping
# the last finite TTC would be actively misleading, so every no-finite-horizon
# frame refreshes it with the maximum representable interval.  Consumers must
# use IsWarning/the canonical decision envelope for validity and treat this
# value as the documented "no finite TTC" mirror sentinel.
NO_FINITE_TTC_MS = 0xFFFFFFFF


class KuksaPublishError(RuntimeError):
    """A CarSky KUKSA mirror write failed or violated its schema."""


@dataclass(frozen=True)
class SignalValue:
    path: str
    value: float | int | bool | str
    data_type: str


@dataclass(frozen=True)
class KuksaPublishReceipt:
    session_id: str
    sequence: int
    signal_count: int
    paths: tuple[str, ...]


class KuksaBackend(Protocol):
    def connect(self, expected_types: Mapping[str, str]) -> None: ...

    def write(self, values: tuple[SignalValue, ...]) -> None: ...

    def close(self) -> None: ...


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise KuksaPublishError(f"{field} phải là số")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise KuksaPublishError(f"{field} phải là số") from exc
    if not math.isfinite(number):
        raise KuksaPublishError(f"{field} phải hữu hạn")
    if minimum is not None and number < minimum:
        raise KuksaPublishError(f"{field} phải >= {minimum}")
    if maximum is not None and number > maximum:
        raise KuksaPublishError(f"{field} phải <= {maximum}")
    return number


def standard_signal_values(
    frame: Any,
    envelope: DecisionEnvelope,
) -> tuple[SignalValue, ...]:
    """Map one unified prediction to the proved stock VSS paths."""

    if not isinstance(envelope, DecisionEnvelope):
        raise TypeError("envelope phải là DecisionEnvelope")
    bundle = getattr(frame, "source_bundle", None)
    ego = getattr(bundle, "ego", None)
    if not isinstance(ego, Mapping):
        raise KuksaPublishError("source bundle thiếu ego telemetry")

    raw: dict[str, float | int | bool] = {
        "Vehicle.Speed": _finite(
            ego.get("speed_kmh"), field="ego.speed_kmh", minimum=0.0
        ),
        "Vehicle.Acceleration.Longitudinal": _finite(
            ego.get("longitudinal_accel"), field="ego.longitudinal_accel"
        ),
        "Vehicle.Acceleration.Lateral": _finite(
            ego.get("lateral_accel"), field="ego.lateral_accel"
        ),
        "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap": (
            envelope.c1["ttc_ms"]
            if envelope.c1["ttc_valid"]
            else NO_FINITE_TTC_MS
        ),
        "Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning": bool(
            envelope.c1["warning"]
        ),
        "Vehicle.Driver.AttentiveProbability": _finite(
            envelope.c2["attentive_probability_pct"],
            field="c2.attentive_probability_pct",
            minimum=0.0,
            maximum=100.0,
        ),
        "Vehicle.Driver.DistractionLevel": _finite(
            envelope.c2["distraction_level_pct"],
            field="c2.distraction_level_pct",
            minimum=0.0,
            maximum=100.0,
        ),
        "Vehicle.Driver.FatigueLevel": _finite(
            envelope.c2["fatigue_level_pct"],
            field="c2.fatigue_level_pct",
            minimum=0.0,
            maximum=100.0,
        ),
        "Vehicle.Driver.IsEyesOnRoad": bool(envelope.c2["eyes_on_road"]),
        "Vehicle.ADAS.DMS.IsWarning": bool(envelope.c2["warning"]),
    }
    if envelope.c1["ttc_valid"]:
        ttc_ms = envelope.c1["ttc_ms"]
        if not isinstance(ttc_ms, int) or isinstance(ttc_ms, bool) or ttc_ms < 0:
            raise KuksaPublishError("c1.ttc_ms phải là uint32 khi hợp lệ")
        if ttc_ms > 0xFFFFFFFF:
            raise KuksaPublishError("c1.ttc_ms vượt uint32")

    return tuple(
        SignalValue(path=path, value=value, data_type=STANDARD_SIGNAL_TYPES[path])
        for path, value in raw.items()
    )


class KuksaGrpcBackend:
    """Thin lazy wrapper around the official ``kuksa-client`` package."""

    def __init__(
        self,
        host: str = "127.0.0.10",
        port: int = 55_555,
        *,
        timeout_s: float = 2.0,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("KUKSA host không hợp lệ")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
            raise ValueError("KUKSA port phải thuộc [1, 65535]")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("KUKSA timeout phải > 0")
        self.host = host
        self.port = port
        self.timeout_s = float(timeout_s)
        self._client: Any = None
        self._grpc: Any = None

    def connect(self, expected_types: Mapping[str, str]) -> None:
        if self._client is not None:
            raise KuksaPublishError("KUKSA backend đã kết nối")
        try:
            from kuksa_client import grpc as kuksa_grpc
        except ModuleNotFoundError as exc:
            raise KuksaPublishError(
                "Thiếu kuksa-client; cài optional dependency carsky-runtime"
            ) from exc
        client = kuksa_grpc.VSSClient(
            self.host,
            self.port,
            # connect() itself only creates the channel. The bounded metadata
            # preflight below is the actual readiness check; enabling the SDK
            # startup probe here would perform an unbounded RPC first.
            ensure_startup_connection=False,
        )
        try:
            client.connect()
            metadata = client.get_metadata(
                expected_types.keys(), timeout=self.timeout_s
            )
            actual = {
                path: item.data_type.name
                for path, item in metadata.items()
            }
            if actual != dict(expected_types):
                raise KuksaPublishError(
                    f"KUKSA schema không khớp: expected={dict(expected_types)}, actual={actual}"
                )
        except BaseException:
            client.disconnect()
            raise
        self._client = client
        self._grpc = kuksa_grpc

    def write(self, values: tuple[SignalValue, ...]) -> None:
        if self._client is None or self._grpc is None:
            raise KuksaPublishError("KUKSA backend chưa kết nối")
        updates = []
        for item in values:
            try:
                data_type = getattr(self._grpc.DataType, item.data_type)
            except AttributeError as exc:
                raise KuksaPublishError(
                    f"KUKSA data type không hỗ trợ: {item.data_type}"
                ) from exc
            updates.append(
                self._grpc.EntryUpdate(
                    self._grpc.DataEntry(
                        item.path,
                        value=self._grpc.Datapoint(item.value),
                        metadata=self._grpc.Metadata(data_type=data_type),
                    ),
                    (self._grpc.Field.VALUE,),
                )
            )
        try:
            # v1 Set sends the complete mirror in one request.  Metadata above
            # supplies fixed types locally; it is not included in the fields.
            self._client.set(updates, try_v2=False, timeout=self.timeout_s)
        except Exception as exc:
            raise KuksaPublishError("Không ghi được KUKSA signal batch") from exc

    def close(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None
            self._grpc = None


class KuksaSignalPublisher:
    """Publish ordered standard mirrors and an optional custom JSON path."""

    def __init__(
        self,
        backend: KuksaBackend,
        *,
        decision_envelope_path: str | None = None,
    ) -> None:
        if decision_envelope_path is not None and (
            not isinstance(decision_envelope_path, str)
            or not decision_envelope_path.strip()
        ):
            raise ValueError("decision_envelope_path phải rỗng hoặc là VSS path")
        self.backend = backend
        self.decision_envelope_path = decision_envelope_path
        self._connected = False
        self._closed = False
        self._last_session: str | None = None
        self._last_sequence: int | None = None
        self._retired_sessions: OrderedDict[str, None] = OrderedDict()

    def _is_retired_session(self, session_id: str) -> bool:
        if session_id not in self._retired_sessions:
            return False
        self._retired_sessions.move_to_end(session_id)
        return True

    def _commit_order(self, session_id: str, sequence: int) -> None:
        if self._last_session != session_id and self._last_session is not None:
            self._retired_sessions[self._last_session] = None
            self._retired_sessions.move_to_end(self._last_session)
            while len(self._retired_sessions) > MAX_RETIRED_SESSIONS:
                self._retired_sessions.popitem(last=False)
        self._last_session = session_id
        self._last_sequence = sequence

    @property
    def expected_types(self) -> dict[str, str]:
        expected = dict(STANDARD_SIGNAL_TYPES)
        if self.decision_envelope_path is not None:
            expected[self.decision_envelope_path] = "STRING"
        return expected

    def connect(self) -> None:
        if self._closed:
            raise KuksaPublishError("KUKSA publisher đã đóng")
        if self._connected:
            return
        self.backend.connect(self.expected_types)
        self._connected = True

    def publish(self, frame: Any, envelope: DecisionEnvelope) -> KuksaPublishReceipt:
        if not self._connected or self._closed:
            raise KuksaPublishError("KUKSA publisher chưa sẵn sàng")
        if self._last_session == envelope.session_id:
            if self._last_sequence is not None and envelope.sequence <= self._last_sequence:
                raise KuksaPublishError("KUKSA sequence bị trùng hoặc lùi")
        elif self._is_retired_session(envelope.session_id):
            raise KuksaPublishError("KUKSA session đã kết thúc không được quay lại")

        values = list(standard_signal_values(frame, envelope))
        if self.decision_envelope_path is not None:
            values.append(
                SignalValue(
                    path=self.decision_envelope_path,
                    value=envelope.to_json_bytes().decode("utf-8"),
                    data_type="STRING",
                )
            )
        batch = tuple(values)
        self.backend.write(batch)
        # Update ordering only after a successful broker write. This lets a
        # late-joining sequence recover from a transient first-write failure.
        self._commit_order(envelope.session_id, envelope.sequence)
        return KuksaPublishReceipt(
            session_id=envelope.session_id,
            sequence=envelope.sequence,
            signal_count=len(batch),
            paths=tuple(item.path for item in batch),
        )

    def close(self) -> None:
        if not self._closed:
            self.backend.close()
            self._closed = True
            self._connected = False

    def __enter__(self) -> "KuksaSignalPublisher":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "KuksaBackend",
    "KuksaGrpcBackend",
    "KuksaPublishError",
    "KuksaPublishReceipt",
    "KuksaSignalPublisher",
    "NO_FINITE_TTC_MS",
    "STANDARD_SIGNAL_TYPES",
    "SignalValue",
    "standard_signal_values",
]
