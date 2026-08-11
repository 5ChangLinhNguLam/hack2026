#!/usr/bin/env python3
"""Fail-closed CarSky planning and guarded native deployment tooling.

``validate-config``, ``preflight`` and ``plan`` never mutate CarSky.  Mutation
commands are intentionally narrow and independently confirmed:

* ``clone-candidate`` clones only the locked clean base and atomically records
  the verified clone ID in an ignored working config;
* ``patch-gateway`` updates only ``config.scriptContent`` on that inactive,
  editable direct clone; and
* ``deploy`` creates a deployment only after a fresh all-green preflight for
  the exact candidate/device pair.

No command deletes, stops, restarts, unlocks, or modifies an active deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error, parse, request


SCHEMA_VERSION = "safeloop.carsky-native-config.v1"
ADDON_CONTRACT = "safeloop.native-gateway-addon.v1"
KUKSA_CONTRACT = "safeloop.standard-vss.v1"
HMI_TRANSPORT = "udp-ethernet"
DEFAULT_URL = "https://hackathon-1.carsky.io"
DEFAULT_BASE_BLUEPRINT_ID = "rEg5AvGuzcKQHtihPiW0q"
DEFAULT_TARGET_DEVICE_ID = "h5sjc3jtzl8vzl9wyqokv"
DEFAULT_BACKUP_DIR = Path(".carsky-build/carsky-native-backups")
DEFAULT_WORKING_CONFIG_ROOT = Path(__file__).resolve().parents[1] / ".carsky-build"
DEFAULT_OPENAPI = Path(__file__).resolve().parents[1] / "docs/carsky-openapi.json"
DEFAULT_KUKSA_SCHEMA_ATTESTATION = (
    Path(__file__).resolve().parents[1]
    / "reports/evidence/carsky-live-audit-20260810T214419+0700"
    / "deployments/uQQBHr256M5GEdNzq2GuQ/signals/broker-metadata.json"
)
PINNED_KUKSA_ATTESTATION_SHA256 = (
    "a203d81c0bb3a7c03edbe4c2c7e5d1562d76c2f086bee59750d38372f1eaf8c1"
)
PINNED_KUKSA_ARTIFACT_ID = "zsJexrIgGwIeyk3ODyYoh"
PINNED_KUKSA_VERSION_ID = "Cpi8WkMz_bH35uIQFviim"
PINNED_KUKSA_REFERENCE_ROOM_ID = "h5sjc3jtzl8vzl9wyqokv"
PINNED_KUKSA_REFERENCE_NODE_KEY = "central-broker-vss"
MAX_KUKSA_ATTESTATION_BYTES = 1_000_000
ADDON_MARKER = "-- SAFELOOP_NATIVE_GATEWAY_ADDON_V1"
GENERATED_REPLAY_BEGIN = "-- SAFELOOP_NATIVE_REPLAY_ADDON_V1_BEGIN"
GENERATED_REPLAY_END = "-- SAFELOOP_NATIVE_REPLAY_ADDON_V1_END"
SOURCE_MODE_MARKERS = {
    "REAL_MODEL": "-- SAFELOOP_SOURCE_MODE: REAL_MODEL",
    "MODEL_REPLAY": "-- SAFELOOP_SOURCE_MODE: MODEL_REPLAY",
}
BOUNDARY_START = "-- >>> SAFELOOP_NATIVE_GATEWAY_ADDON_V1"
BOUNDARY_END = "-- <<< SAFELOOP_NATIVE_GATEWAY_ADDON_V1"
MAX_ADDON_BYTES = 1_000_000

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

_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "token",
    }
)
_CREDENTIAL_VALUE = re.compile(r"a8k_[A-Za-z0-9._~-]+", re.IGNORECASE)
_URI_CREDENTIAL = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@:]+:[^/@]+@", re.IGNORECASE)
_FORBIDDEN_ADDON_TEXT = (
    "a8_api_key",
    "api_key",
    "authorization",
    "bearer ",
    "ground_truth",
    "events_active",
    "driver_summary",
    "trip_aggregate",
    "mock=true",
    "mock = true",
)
_SAFE_GET_PATHS = (
    re.compile(r"^/api/v1/healthz$"),
    re.compile(r"^/api/v1/config/(?:limits|cpu-status)$"),
    re.compile(r"^/api/v1/blueprints(?:\?[^#]*)?$"),
    re.compile(r"^/api/v1/blueprints/[^/?#]+$"),
    re.compile(r"^/api/v1/devices(?:\?[^#]*)?$"),
    re.compile(r"^/api/v1/devices/[^/?#]+$"),
    re.compile(r"^/api/v1/deployments/find\?[^#]+$"),
    re.compile(r"^/api/v1/deployments/[^/?#]+/status$"),
    re.compile(r"^/api/v1/signals/[^/?#]+$"),
    re.compile(r"^/api/v1/signals/[^/?#]+/[^/?#]+$"),
)


class NativeConfigError(ValueError):
    """Local config or gateway addon is incomplete or unsafe."""


class CarSkyApiError(RuntimeError):
    """A CarSky request failed without exposing response credentials."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MutationRefused(RuntimeError):
    """A guarded mutation gate rejected its exact target or result."""


class PatchRefused(MutationRefused):
    """The guarded patch gate rejected a candidate or mutation result."""


class CloneRefused(MutationRefused):
    """The guarded clone gate rejected the request or returned clone."""


class DeployRefused(MutationRefused):
    """The guarded deployment gate rejected the request or result."""


@dataclass(frozen=True)
class KuksaConfig:
    contract: str
    reference_room_id: str
    reference_node_key: str


@dataclass(frozen=True)
class GatewayConfig:
    node_label: str
    addon_contract: str
    kuksa: KuksaConfig


@dataclass(frozen=True)
class HmiConfig:
    transport: str
    switch_label: str
    android_node_label: str
    target_host: str
    port: int


@dataclass(frozen=True)
class NativeConfig:
    schema_version: str
    base_blueprint_id: str
    candidate_blueprint_id: str | None
    target_device_id: str
    blueprint_name: str
    deployment_name: str
    gateway: GatewayConfig
    hmi: HmiConfig


@dataclass(frozen=True)
class GatewayAddon:
    path: Path
    content: str
    sha256: str
    size_bytes: int
    source_mode: str


def _compact_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _reject_secret_material(value: object, *, field: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _compact_key(key) in _SENSITIVE_KEYS:
                raise NativeConfigError(f"{field}.{key} không được chứa secret")
            _reject_secret_material(item, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_material(item, field=f"{field}[{index}]")
    elif isinstance(value, str):
        if _CREDENTIAL_VALUE.search(value) or _URI_CREDENTIAL.search(value):
            raise NativeConfigError(f"{field} chứa credential trong value")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeConfigError(f"{field} phải là object")
    return value


def _exact_keys(value: Mapping[str, Any], *, field: str, expected: set[str]) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise NativeConfigError(f"{field} keys sai; thiếu={missing}, thừa={extra}")


def _text(value: object, *, field: str, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeConfigError(f"{field} phải là chuỗi không rỗng")
    result = value.strip()
    if len(result) > maximum:
        raise NativeConfigError(f"{field} dài quá {maximum} ký tự")
    return result


def parse_config(payload: object) -> NativeConfig:
    root = _mapping(payload, field="config")
    _reject_secret_material(root)
    _exact_keys(
        root,
        field="config",
        expected={
            "schema_version",
            "base_blueprint_id",
            "candidate_blueprint_id",
            "target_device_id",
            "blueprint_name",
            "deployment_name",
            "gateway",
            "hmi",
        },
    )
    schema = _text(root["schema_version"], field="schema_version")
    if schema != SCHEMA_VERSION:
        raise NativeConfigError(f"schema_version phải là {SCHEMA_VERSION}")
    base_id = _text(root["base_blueprint_id"], field="base_blueprint_id")
    if base_id != DEFAULT_BASE_BLUEPRINT_ID:
        raise NativeConfigError(
            f"base_blueprint_id bị khóa ở {DEFAULT_BASE_BLUEPRINT_ID}"
        )
    target_id = _text(root["target_device_id"], field="target_device_id")
    if target_id != DEFAULT_TARGET_DEVICE_ID:
        raise NativeConfigError(
            f"target_device_id bị khóa ở {DEFAULT_TARGET_DEVICE_ID}"
        )
    raw_candidate = root["candidate_blueprint_id"]
    candidate_id = (
        None
        if raw_candidate is None
        else _text(raw_candidate, field="candidate_blueprint_id")
    )
    if candidate_id == base_id:
        raise NativeConfigError("candidate_blueprint_id không được trỏ vào base")

    gateway_raw = _mapping(root["gateway"], field="gateway")
    _exact_keys(
        gateway_raw,
        field="gateway",
        expected={"node_label", "addon_contract", "kuksa"},
    )
    addon_contract = _text(
        gateway_raw["addon_contract"], field="gateway.addon_contract"
    )
    if addon_contract != ADDON_CONTRACT:
        raise NativeConfigError(f"gateway.addon_contract phải là {ADDON_CONTRACT}")
    kuksa_raw = _mapping(gateway_raw["kuksa"], field="gateway.kuksa")
    _exact_keys(
        kuksa_raw,
        field="gateway.kuksa",
        expected={"contract", "reference_room_id", "reference_node_key"},
    )
    kuksa_contract = _text(kuksa_raw["contract"], field="gateway.kuksa.contract")
    if kuksa_contract != KUKSA_CONTRACT:
        raise NativeConfigError(f"gateway.kuksa.contract phải là {KUKSA_CONTRACT}")

    hmi_raw = _mapping(root["hmi"], field="hmi")
    _exact_keys(
        hmi_raw,
        field="hmi",
        expected={
            "transport",
            "switch_label",
            "android_node_label",
            "target_host",
            "port",
        },
    )
    transport = _text(hmi_raw["transport"], field="hmi.transport")
    if transport != HMI_TRANSPORT:
        raise NativeConfigError(
            f"hmi.transport phải là {HMI_TRANSPORT}; Screen Widget không phải data transport"
        )
    try:
        target_host = ipaddress.ip_address(
            _text(hmi_raw["target_host"], field="hmi.target_host")
        )
    except ValueError as exc:
        raise NativeConfigError("hmi.target_host phải là IPv4 hợp lệ") from exc
    if target_host.version != 4:
        raise NativeConfigError("hmi.target_host hiện chỉ hỗ trợ IPv4")
    port = hmi_raw["port"]
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        raise NativeConfigError("hmi.port phải là integer trong [1024, 65535]")

    return NativeConfig(
        schema_version=schema,
        base_blueprint_id=base_id,
        candidate_blueprint_id=candidate_id,
        target_device_id=target_id,
        blueprint_name=_text(root["blueprint_name"], field="blueprint_name"),
        deployment_name=_text(root["deployment_name"], field="deployment_name"),
        gateway=GatewayConfig(
            node_label=_text(
                gateway_raw["node_label"], field="gateway.node_label", maximum=100
            ),
            addon_contract=addon_contract,
            kuksa=KuksaConfig(
                contract=kuksa_contract,
                reference_room_id=_text(
                    kuksa_raw["reference_room_id"],
                    field="gateway.kuksa.reference_room_id",
                ),
                reference_node_key=_text(
                    kuksa_raw["reference_node_key"],
                    field="gateway.kuksa.reference_node_key",
                ),
            ),
        ),
        hmi=HmiConfig(
            transport=transport,
            switch_label=_text(hmi_raw["switch_label"], field="hmi.switch_label"),
            android_node_label=_text(
                hmi_raw["android_node_label"], field="hmi.android_node_label"
            ),
            target_host=str(target_host),
            port=port,
        ),
    )


def load_config(path: Path) -> NativeConfig:
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


def load_gateway_addon(path: Path, config: NativeConfig) -> GatewayAddon:
    if not path.is_file():
        raise NativeConfigError(f"Gateway addon không tồn tại: {path}")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_ADDON_BYTES:
        raise NativeConfigError(
            f"Gateway addon phải trong khoảng 1..{MAX_ADDON_BYTES} bytes"
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NativeConfigError("Gateway addon phải là UTF-8") from exc
    _reject_secret_material(content, field="gateway_script")
    generated_replay = (
        content.count(GENERATED_REPLAY_BEGIN) == 1
        and content.count(GENERATED_REPLAY_END) == 1
        and content.lstrip().startswith(GENERATED_REPLAY_BEGIN)
        and content.rstrip().endswith(GENERATED_REPLAY_END)
    )
    if ADDON_MARKER not in content and not generated_replay:
        raise NativeConfigError(
            f"Gateway addon thiếu marker {ADDON_MARKER} hoặc generated replay v1"
        )
    modes = [name for name, marker in SOURCE_MODE_MARKERS.items() if marker in content]
    if generated_replay and not modes:
        modes = ["MODEL_REPLAY"]
    if len(modes) != 1:
        raise NativeConfigError(
            "Gateway addon cần đúng một SAFELOOP_SOURCE_MODE: REAL_MODEL hoặc MODEL_REPLAY"
        )
    lower = content.lower()
    forbidden = [token for token in _FORBIDDEN_ADDON_TEXT if token in lower]
    if forbidden:
        raise NativeConfigError(f"Gateway addon chứa field/mode cấm: {forbidden}")
    requirements = {
        "KUKSA": "kuksa" in lower,
        "UDP": "udp" in lower,
        "Android host": config.hmi.target_host in content,
        "Android port": str(config.hmi.port) in content,
    }
    missing = [name for name, present in requirements.items() if not present]
    if missing:
        raise NativeConfigError(f"Gateway addon thiếu contract runtime: {missing}")
    return GatewayAddon(
        path=path.resolve(),
        content=content.rstrip() + "\n",
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        source_mode=modes[0],
    )


def combined_gateway_script(base_script: str, addon: GatewayAddon) -> str:
    if not isinstance(base_script, str) or not base_script.strip():
        raise NativeConfigError("IVI Gateway base thiếu scriptContent")
    if BOUNDARY_START in base_script or BOUNDARY_END in base_script:
        raise NativeConfigError("Base IVI Gateway không sạch: đã có SafeLoop addon")
    return (
        base_script
        + "\n"
        + f"{BOUNDARY_START} sha256={addon.sha256}\n"
        + addon.content
        + f"{BOUNDARY_END}\n"
    )


def gateway_addon_state(script: str, addon: GatewayAddon) -> dict[str, Any]:
    """Describe whether *script* is clean or has the exact verified addon suffix."""

    if not isinstance(script, str) or not script.strip():
        return {"state": "missing", "base_script": None, "expected_script": None}
    suffix = (
        "\n"
        + f"{BOUNDARY_START} sha256={addon.sha256}\n"
        + addon.content
        + f"{BOUNDARY_END}\n"
    )
    marker_present = any(
        marker in script
        for marker in (
            ADDON_MARKER,
            GENERATED_REPLAY_BEGIN,
            GENERATED_REPLAY_END,
            BOUNDARY_START,
            BOUNDARY_END,
        )
    )
    if script.endswith(suffix) and script.count(BOUNDARY_START) == 1 and script.count(BOUNDARY_END) == 1:
        base_script = script[: -len(suffix)]
        return {
            "state": "applied",
            "base_script": base_script,
            "expected_script": script,
        }
    if marker_present:
        return {"state": "conflict", "base_script": None, "expected_script": None}
    return {
        "state": "clean",
        "base_script": script,
        "expected_script": combined_gateway_script(script, addon),
    }


class CarSkyApi:
    """CarSky client exposing only the operations used by guarded workflows."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 30.0,
        urlopen: Callable[..., Any] = request.urlopen,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise NativeConfigError("Thiếu A8_API_KEY trong environment")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise NativeConfigError("timeout phải > 0")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = float(timeout_s)
        self._urlopen = urlopen

    def _request(self, method: str, path: str, payload: object | None = None) -> Any:
        headers = {"Accept": "application/json", "X-API-Key": self._api_key}
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(
            self._base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._urlopen(req, timeout=self._timeout_s) as response:
                raw = response.read()
        except error.HTTPError as exc:
            public_error = ""
            try:
                error_body = exc.read(64 * 1024)
                decoded = json.loads(error_body) if error_body else {}
                if isinstance(decoded, Mapping):
                    parts: list[str] = []
                    for key in ("error", "message"):
                        value = decoded.get(key)
                        if isinstance(value, str) and value.strip():
                            cleaned = " ".join(value.split())[:500]
                            cleaned = cleaned.replace(self._api_key, "<redacted>")
                            cleaned = _CREDENTIAL_VALUE.sub("<redacted>", cleaned)
                            cleaned = _URI_CREDENTIAL.sub("<redacted>", cleaned)
                            parts.append(f"{key}={cleaned}")
                    details = decoded.get("details")
                    if isinstance(details, Mapping):
                        public_details = {
                            key: details[key]
                            for key in ("formErrors", "fieldErrors")
                            if key in details
                        }
                        if public_details:
                            cleaned = json.dumps(
                                public_details,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )[:2000]
                            cleaned = cleaned.replace(self._api_key, "<redacted>")
                            cleaned = _CREDENTIAL_VALUE.sub("<redacted>", cleaned)
                            cleaned = _URI_CREDENTIAL.sub("<redacted>", cleaned)
                            parts.append(f"details={cleaned}")
                    if parts:
                        public_error = "; " + "; ".join(parts)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                public_error = ""
            raise CarSkyApiError(
                f"CarSky {method} {path} trả HTTP {exc.code}{public_error}",
                status_code=exc.code,
            ) from exc
        except (error.URLError, OSError) as exc:
            raise CarSkyApiError(f"Không gọi được CarSky {method} {path}") from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise CarSkyApiError(f"CarSky {method} {path} trả JSON không hợp lệ") from exc

    def get(self, path: str) -> Any:
        if not any(pattern.fullmatch(path) for pattern in _SAFE_GET_PATHS):
            raise CarSkyApiError(f"GET path ngoài read-only allow-list: {path}")
        return self._request("GET", path)

    def patch_gateway_config(self, node_id: str, config: Mapping[str, Any]) -> Any:
        """PATCH only the config object; caller owns all candidate safeguards."""

        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            raise PatchRefused("IVI Gateway node id không an toàn")
        return self._request(
            "PATCH",
            f"/api/v1/nodes/{_quote(node_id)}",
            {"config": dict(config)},
        )

    def validate_blueprint(self, blueprint_id: str) -> Any:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", blueprint_id):
            raise PatchRefused("Candidate blueprint id không an toàn")
        return self._request(
            "POST", f"/api/v1/blueprints/{_quote(blueprint_id)}/validate"
        )

    def clone_blueprint(self, base_blueprint_id: str, name: str) -> Any:
        if base_blueprint_id != DEFAULT_BASE_BLUEPRINT_ID:
            raise CloneRefused("Từ chối clone ngoài base blueprint đã khóa")
        return self._request(
            "POST",
            f"/api/v1/blueprints/{_quote(base_blueprint_id)}/clone",
            {"name": name, "isSnapshot": False},
        )

    def create_deployment(
        self, *, candidate_blueprint_id: str, target_device_id: str, name: str
    ) -> Any:
        for field, value in (
            ("candidate blueprint", candidate_blueprint_id),
            ("target device", target_device_id),
        ):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise DeployRefused(f"{field} id không an toàn")
        return self._request(
            "POST",
            "/api/v1/deployments",
            {
                "blueprintId": candidate_blueprint_id,
                "roomId": target_device_id,
                "name": name,
            },
        )


def _quote(value: str) -> str:
    return parse.quote(value, safe="")


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    for key in ("data", "blueprint", "device", "deployment"):
        if isinstance(value.get(key), dict):
            return value[key]
    return value


def _items(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "items", "nodes", "signals", "deployments"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    return []


def _nodes(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in blueprint.get("nodes") or [] if isinstance(item, dict)]


def _pins(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in node.get("pins") or [] if isinstance(item, dict)]


def _edges(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in blueprint.get("edges") or [] if isinstance(item, dict)]


def _by_label(blueprint: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    return [node for node in _nodes(blueprint) if node.get("label") == label]


def _edge_exists(
    blueprint: Mapping[str, Any], source_pin_id: object, target_pin_id: object
) -> bool:
    return any(
        edge.get("sourcePinId") == source_pin_id
        and edge.get("targetPinId") == target_pin_id
        for edge in _edges(blueprint)
    )


def _pin_address(pin: Mapping[str, Any]) -> str | None:
    properties = pin.get("properties")
    if isinstance(properties, Mapping) and isinstance(properties.get("address"), str):
        return properties["address"]
    return None


def _broker(blueprint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    brokers = [
        node for node in _nodes(blueprint) if node.get("nodeType") == "kuksa-databroker"
    ]
    if len(brokers) != 1:
        return None
    inputs = [
        pin
        for pin in _pins(brokers[0])
        if pin.get("pinType") == "KUKSA" and pin.get("direction") == "INPUT"
    ]
    return (brokers[0], inputs[0]) if len(inputs) == 1 else None


def _broker_artifact(blueprint: Mapping[str, Any]) -> tuple[str | None, str | None]:
    found = _broker(blueprint)
    config = found[0].get("config") if found is not None else None
    kuksa = config.get("kuksa") if isinstance(config, Mapping) else None
    vss = kuksa.get("vss") if isinstance(kuksa, Mapping) else None
    if not isinstance(vss, Mapping):
        return None, None
    artifact = vss.get("artifactId")
    version = vss.get("versionId")
    return (
        artifact if isinstance(artifact, str) else None,
        version if isinstance(version, str) else None,
    )


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _topology_fingerprint(blueprint: Mapping[str, Any]) -> str:
    """Hash topology and node configuration without clone-specific IDs.

    CarSky rewrites blueprint/node/pin/edge IDs while cloning.  This canonical
    form resolves edge endpoints to their semantic node/pin descriptors, while
    retaining node configs, positions, pin properties and edge multiplicity.
    """

    pin_by_id: dict[str, object] = {}
    canonical_nodes: list[object] = []
    for node in _nodes(blueprint):
        node_descriptor = {
            "label": node.get("label"),
            "nodeType": node.get("nodeType"),
            "config": node.get("config"),
            "positionX": node.get("positionX"),
            "positionY": node.get("positionY"),
        }
        canonical_pins: list[object] = []
        for pin in _pins(node):
            pin_descriptor = {
                "node": {
                    "label": node.get("label"),
                    "nodeType": node.get("nodeType"),
                },
                "pin": {
                    "name": pin.get("name"),
                    "pinType": pin.get("pinType"),
                    "direction": pin.get("direction"),
                    "side": pin.get("side"),
                    "port": pin.get("port"),
                    "properties": pin.get("properties"),
                },
            }
            pin_id = pin.get("id")
            if not isinstance(pin_id, str) or pin_id in pin_by_id:
                raise CloneRefused("Blueprint có pin id thiếu hoặc trùng")
            pin_by_id[pin_id] = pin_descriptor
            canonical_pins.append(pin_descriptor["pin"])
        node_descriptor["pins"] = sorted(
            canonical_pins,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
        canonical_nodes.append(node_descriptor)

    canonical_edges: list[object] = []
    for edge in _edges(blueprint):
        source = pin_by_id.get(str(edge.get("sourcePinId")))
        target = pin_by_id.get(str(edge.get("targetPinId")))
        if source is None or target is None:
            raise CloneRefused("Blueprint có edge trỏ tới pin không tồn tại")
        canonical_edges.append({"source": source, "target": target})

    payload = {
        "nodes": sorted(
            canonical_nodes,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        ),
        "edges": sorted(
            canonical_edges,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        ),
    }
    return _json_hash(payload)


def _working_config_path(path: Path, working_root: Path) -> Path:
    if path.is_symlink():
        raise CloneRefused("Working config không được là symlink")
    resolved_root = working_root.resolve()
    resolved_path = path.resolve()
    if resolved_root.name != ".carsky-build":
        raise CloneRefused("Working config root phải là thư mục .carsky-build")
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise CloneRefused(
            "clone-candidate chỉ được cập nhật config trong .carsky-build"
        ) from exc
    if not resolved_path.is_file():
        raise CloneRefused("Working config không tồn tại")
    return resolved_path


def _atomic_record_candidate_id(
    config_path: Path,
    candidate_id: str,
    *,
    expected_config: NativeConfig,
    working_root: Path = DEFAULT_WORKING_CONFIG_ROOT,
) -> Path:
    resolved = _working_config_path(config_path, working_root)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    current = parse_config(raw)
    if current != expected_config:
        raise CloneRefused(
            "Working config đã thay đổi trong lúc clone; không ghi đè nội dung mới"
        )
    if current.candidate_blueprint_id is not None:
        raise CloneRefused("Working config đã có candidate_blueprint_id")
    raw["candidate_blueprint_id"] = candidate_id
    parse_config(raw)
    rendered = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    original_mode = resolved.stat().st_mode & 0o777
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, original_mode)
        os.replace(temporary_name, resolved)
        temporary_name = None
        directory_fd = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    if load_config(resolved).candidate_blueprint_id != candidate_id:
        raise CloneRefused("Không xác minh được candidate id sau atomic config update")
    return resolved


def _gateway_topology(
    config: NativeConfig, blueprint: Mapping[str, Any]
) -> dict[str, Any]:
    gateway_matches = _by_label(blueprint, config.gateway.node_label)
    switch_matches = _by_label(blueprint, config.hmi.switch_label)
    android_matches = _by_label(blueprint, config.hmi.android_node_label)
    gateway = gateway_matches[0] if len(gateway_matches) == 1 else None
    switch = switch_matches[0] if len(switch_matches) == 1 else None
    android = android_matches[0] if len(android_matches) == 1 else None
    broker = _broker(blueprint)
    kuksa_edge = False
    gateway_eth_edge = False
    android_eth_edge = False
    if gateway is not None and broker is not None:
        outputs = [
            pin
            for pin in _pins(gateway)
            if pin.get("pinType") == "KUKSA"
            and pin.get("direction") == "OUTPUT"
            and pin.get("name") == "kuksa"
        ]
        kuksa_edge = len(outputs) == 1 and _edge_exists(
            blueprint, outputs[0].get("id"), broker[1].get("id")
        )
    switch_inputs = (
        [pin for pin in _pins(switch) if pin.get("pinType") == "ETHERNET"]
        if switch is not None
        else []
    )
    if gateway is not None and len(switch_inputs) == 1:
        gateway_eth = [
            pin
            for pin in _pins(gateway)
            if pin.get("pinType") == "ETHERNET"
            and pin.get("direction") == "OUTPUT"
            and pin.get("name") == "eth"
        ]
        gateway_eth_edge = len(gateway_eth) == 1 and _edge_exists(
            blueprint, gateway_eth[0].get("id"), switch_inputs[0].get("id")
        )
    if android is not None and len(switch_inputs) == 1:
        android_eth = [
            pin
            for pin in _pins(android)
            if pin.get("pinType") == "ETHERNET"
            and pin.get("direction") == "OUTPUT"
            and _pin_address(pin) == config.hmi.target_host
        ]
        android_eth_edge = len(android_eth) == 1 and _edge_exists(
            blueprint, android_eth[0].get("id"), switch_inputs[0].get("id")
        )
    gateway_config = gateway.get("config") if gateway is not None else None
    script = (
        gateway_config.get("scriptContent")
        if isinstance(gateway_config, Mapping)
        else None
    )
    return {
        "gateway": gateway,
        "gateway_config": dict(gateway_config) if isinstance(gateway_config, Mapping) else None,
        "gateway_script": script if isinstance(script, str) else None,
        "gateway_unique": gateway is not None,
        "gateway_is_script_node": gateway is not None and gateway.get("nodeType") == "script-node",
        "gateway_inline_script": isinstance(gateway_config, Mapping)
        and gateway_config.get("script") == "inline",
        "broker_ready": broker is not None,
        "kuksa_edge_ready": kuksa_edge,
        "gateway_ivi_ethernet_edge_ready": gateway_eth_edge,
        "android_ivi_ethernet_edge_ready": android_eth_edge,
    }


def _pin_type_enum(openapi_path: Path) -> set[str]:
    payload = json.loads(openapi_path.read_text(encoding="utf-8"))
    endpoint = payload.get("paths", {}).get("/api/v1/blueprints/{id}/batch", {})
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            props = value.get("properties")
            if isinstance(props, Mapping):
                pin_type = props.get("pinType")
                if isinstance(pin_type, Mapping) and isinstance(pin_type.get("enum"), list):
                    found.update(str(item) for item in pin_type["enum"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(endpoint)
    return found


def _normalize_type(value: object) -> str:
    text = str(value or "").upper()
    return {"BOOL": "BOOLEAN"}.get(text, text)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of accepting the last occurrence."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeConfigError(
                f"KUKSA bootstrap attestation có JSON key trùng: {key}"
            )
        result[key] = value
    return result


def _load_pinned_kuksa_schema_attestation(
    config: NativeConfig,
    base_artifact: tuple[str | None, str | None],
    path: Path,
) -> dict[str, Any]:
    """Verify the immutable local broker evidence used for a cold bootstrap.

    This is intentionally narrower than live discovery.  It is accepted only
    for the exact audited room/node and exact stock artifact version embedded
    in the candidate base.  The byte hash is checked before parsing, then the
    response envelope and every required VSS path/type are checked again.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeConfigError(
            f"Không đọc được KUKSA bootstrap attestation: {path}"
        ) from exc
    if not raw or len(raw) > MAX_KUKSA_ATTESTATION_BYTES:
        raise NativeConfigError(
            "KUKSA bootstrap attestation phải trong khoảng "
            f"1..{MAX_KUKSA_ATTESTATION_BYTES} bytes"
        )
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != PINNED_KUKSA_ATTESTATION_SHA256:
        raise NativeConfigError(
            "KUKSA bootstrap attestation SHA-256 không khớp giá trị đã khóa"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except UnicodeDecodeError as exc:
        raise NativeConfigError("KUKSA bootstrap attestation không phải UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise NativeConfigError("KUKSA bootstrap attestation không phải JSON hợp lệ") from exc

    root = _mapping(payload, field="kuksa_attestation")
    _exact_keys(
        root,
        field="kuksa_attestation",
        expected={"body", "content_type", "method", "observed_at", "path", "status"},
    )
    expected_api_path = (
        f"/api/v1/signals/{PINNED_KUKSA_REFERENCE_ROOM_ID}/"
        f"{PINNED_KUKSA_REFERENCE_NODE_KEY}"
    )
    if (
        root.get("method") != "GET"
        or root.get("status") != 200
        or root.get("path") != expected_api_path
        or not isinstance(root.get("content_type"), str)
        or not isinstance(root.get("observed_at"), str)
    ):
        raise NativeConfigError("KUKSA bootstrap attestation response envelope sai")
    if config.gateway.kuksa.reference_room_id != PINNED_KUKSA_REFERENCE_ROOM_ID:
        raise NativeConfigError(
            "KUKSA bootstrap attestation không dành cho reference_room_id trong config"
        )
    if config.gateway.kuksa.reference_node_key != PINNED_KUKSA_REFERENCE_NODE_KEY:
        raise NativeConfigError(
            "KUKSA bootstrap attestation không dành cho reference_node_key trong config"
        )
    pinned_artifact = (PINNED_KUKSA_ARTIFACT_ID, PINNED_KUKSA_VERSION_ID)
    if base_artifact != pinned_artifact:
        raise NativeConfigError(
            "KUKSA bootstrap attestation không khớp artifact/version của base"
        )

    body = _mapping(root["body"], field="kuksa_attestation.body")
    _exact_keys(
        body,
        field="kuksa_attestation.body",
        expected={"nodeKey", "signals"},
    )
    if body.get("nodeKey") != PINNED_KUKSA_REFERENCE_NODE_KEY:
        raise NativeConfigError("KUKSA bootstrap attestation nodeKey sai")
    signals = body.get("signals")
    if not isinstance(signals, list) or len(signals) != 1272:
        raise NativeConfigError("KUKSA bootstrap attestation signal list sai format")

    required_matches: dict[str, list[str]] = {
        path_name: [] for path_name in STANDARD_SIGNAL_TYPES
    }
    for item in signals:
        if not isinstance(item, Mapping):
            raise NativeConfigError(
                "KUKSA bootstrap attestation chứa signal không phải object"
            )
        path_name = item.get("path")
        if path_name in required_matches:
            data_type = item.get("dataType")
            if not isinstance(data_type, str):
                raise NativeConfigError(
                    f"KUKSA bootstrap attestation thiếu dataType cho {path_name}"
                )
            required_matches[path_name].append(_normalize_type(data_type))

    actual_types: dict[str, str] = {}
    for path_name, expected_type in STANDARD_SIGNAL_TYPES.items():
        matches = required_matches[path_name]
        if len(matches) != 1:
            raise NativeConfigError(
                "KUKSA bootstrap attestation cần đúng một signal cho "
                f"{path_name}; thực tế={len(matches)}"
            )
        actual_types[path_name] = matches[0]
        if matches[0] != expected_type:
            raise NativeConfigError(
                "KUKSA bootstrap attestation sai type cho "
                f"{path_name}: expected={expected_type}, actual={matches[0]}"
            )

    return {
        "source": "pinned_local_attestation",
        "path": str(path),
        "sha256": actual_sha256,
        "observed_at": root["observed_at"],
        "artifact_id": pinned_artifact[0],
        "version_id": pinned_artifact[1],
        "node_key": body["nodeKey"],
        "signal_count": len(signals),
        "required_signal_types": actual_types,
    }


def preflight(
    config: NativeConfig,
    addon: GatewayAddon,
    api: Any,
    *,
    openapi_path: Path = DEFAULT_OPENAPI,
    kuksa_attestation_path: Path = DEFAULT_KUKSA_SCHEMA_ATTESTATION,
) -> dict[str, Any]:
    """GET-only audit with separate candidate-patch and deploy gates."""

    common_blockers: list[str] = []
    warnings = [
        "Limits are advertised platform values; account-specific overrides are not proven."
    ]
    health = _object(api.get("/api/v1/healthz"))
    limits = _object(api.get("/api/v1/config/limits"))
    cpu = _object(api.get("/api/v1/config/cpu-status"))
    if health.get("status") != "ok":
        common_blockers.append("PLATFORM_HEALTH_NOT_OK")
    if cpu.get("over") is True:
        common_blockers.append("PLATFORM_CPU_OVER_THRESHOLD")

    blueprint_entries = _items(api.get("/api/v1/blueprints?limit=100"))
    device_entries = _items(api.get("/api/v1/devices?limit=100"))
    base = _object(api.get(f"/api/v1/blueprints/{_quote(config.base_blueprint_id)}"))
    target = _object(api.get(f"/api/v1/devices/{_quote(config.target_device_id)}"))
    base_topology = _gateway_topology(config, base)
    base_script = base_topology["gateway_script"]
    base_sha = (
        hashlib.sha256(base_script.encode("utf-8")).hexdigest()
        if isinstance(base_script, str)
        else None
    )
    base_addon_state = (
        gateway_addon_state(base_script, addon)
        if isinstance(base_script, str)
        else {"state": "missing"}
    )

    if base.get("id") != config.base_blueprint_id:
        common_blockers.append("BASE_BLUEPRINT_ID_MISMATCH")
    if base.get("isSnapshot") is True or base.get("locked") is True:
        common_blockers.append("BASE_BLUEPRINT_NOT_EDITABLE")
    if not all(
        base_topology[key]
        for key in (
            "gateway_unique",
            "gateway_is_script_node",
            "gateway_inline_script",
            "broker_ready",
            "kuksa_edge_ready",
            "gateway_ivi_ethernet_edge_ready",
            "android_ivi_ethernet_edge_ready",
        )
    ):
        common_blockers.append("BASE_GATEWAY_KUKSA_HMI_TOPOLOGY_MISSING")
    labels = [str(node.get("label", "")).lower() for node in _nodes(base)]
    if any("mock" in label or "replay probe" in label for label in labels):
        common_blockers.append("BASE_NOT_CLEAN")
    if base_addon_state["state"] != "clean":
        common_blockers.append("BASE_GATEWAY_NOT_CLEAN")
    base_artifact = _broker_artifact(base)

    if target.get("id") != config.target_device_id:
        common_blockers.append("TARGET_DEVICE_ID_MISMATCH")
    if target.get("status") != "PUBLISHED" or target.get("locked") is True:
        common_blockers.append("TARGET_DEVICE_NOT_DEPLOYABLE")

    deployments: dict[str, dict[str, Any]] = {}
    for device in device_entries:
        device_id = device.get("id")
        if not isinstance(device_id, str):
            continue
        response = api.get(
            "/api/v1/deployments/find?" + parse.urlencode({"device": device_id})
        )
        for deployment in _items(response):
            deployment_id = deployment.get("id")
            if isinstance(deployment_id, str):
                deployments[deployment_id] = deployment
    active_blueprint_ids = sorted(
        {
            str(item["blueprintId"])
            for item in deployments.values()
            if isinstance(item.get("blueprintId"), str)
        }
    )
    target_deployments = [
        item for item in deployments.values() if item.get("roomId") == config.target_device_id
    ]
    deployment_name_duplicates = [
        item for item in deployments.values() if item.get("name") == config.deployment_name
    ]
    if target_deployments:
        common_blockers.append("TARGET_DEVICE_ALREADY_HAS_ACTIVE_DEPLOYMENT")
    if deployment_name_duplicates:
        common_blockers.append("DEPLOYMENT_NAME_ALREADY_ACTIVE")
    max_blueprints = int(limits.get("MAX_BLUEPRINTS_PER_ACCOUNT", 0) or 0)
    max_deployments = int(
        limits.get("MAX_CONCURRENT_DEPLOYMENTS_PER_ACCOUNT", 0) or 0
    )
    if config.candidate_blueprint_id is None and (
        max_blueprints <= 0 or len(blueprint_entries) + 1 > max_blueprints
    ):
        common_blockers.append("BLUEPRINT_LIMIT_EXHAUSTED")
    if max_deployments <= 0 or len(deployments) + 1 > max_deployments:
        common_blockers.append("DEPLOYMENT_LIMIT_EXHAUSTED")

    reference = next(
        (
            item
            for item in deployments.values()
            if item.get("roomId") == config.gateway.kuksa.reference_room_id
        ),
        None,
    )
    reference_artifact = (None, None)
    kuksa_attestation: dict[str, Any] | None = None
    schema_source = "live_reference"
    source_ok = False
    actual_types: dict[str, str] = {}
    if reference is None:
        schema_source = "pinned_local_attestation"
        try:
            kuksa_attestation = _load_pinned_kuksa_schema_attestation(
                config, base_artifact, kuksa_attestation_path
            )
        except NativeConfigError as exc:
            common_blockers.extend(
                [
                    "KUKSA_REFERENCE_DEPLOYMENT_MISSING",
                    "KUKSA_BOOTSTRAP_ATTESTATION_INVALID",
                ]
            )
            warnings.append(f"KUKSA bootstrap attestation rejected: {exc}")
        else:
            reference_artifact = (
                kuksa_attestation["artifact_id"],
                kuksa_attestation["version_id"],
            )
            actual_types = dict(kuksa_attestation["required_signal_types"])
            source_ok = True
            warnings.append(
                "No live KUKSA reference deployment exists. Schema was verified "
                "against the pinned, hash-checked local broker attestation for the "
                "same stock artifact/version; live metadata must be rechecked after "
                "the new deployment starts."
            )
    else:
        reference_blueprint_id = reference.get("blueprintId")
        if reference_blueprint_id == config.base_blueprint_id:
            reference_blueprint = base
        elif isinstance(reference_blueprint_id, str):
            reference_blueprint = _object(
                api.get(f"/api/v1/blueprints/{_quote(reference_blueprint_id)}")
            )
        else:
            reference_blueprint = {}
        reference_artifact = _broker_artifact(reference_blueprint)
        if reference_artifact != base_artifact or None in reference_artifact:
            common_blockers.append("KUKSA_REFERENCE_ARTIFACT_MISMATCH")
        discovery = _object(
            api.get(
                f"/api/v1/signals/{_quote(config.gateway.kuksa.reference_room_id)}"
            )
        )
        source_ok = any(
            item.get("key") == config.gateway.kuksa.reference_node_key
            and item.get("kind") == "kuksa"
            for item in _items(discovery)
        )
        if not source_ok:
            common_blockers.append("KUKSA_REFERENCE_NODE_MISSING")
        metadata = _object(
            api.get(
                f"/api/v1/signals/{_quote(config.gateway.kuksa.reference_room_id)}/"
                f"{_quote(config.gateway.kuksa.reference_node_key)}"
            )
        )
        actual_types = {
            str(item.get("path")): _normalize_type(item.get("dataType"))
            for item in _items(metadata)
            if isinstance(item.get("path"), str)
        }
    missing_paths = sorted(set(STANDARD_SIGNAL_TYPES) - set(actual_types))
    wrong_types = {
        path: {"expected": expected, "actual": actual_types.get(path)}
        for path, expected in STANDARD_SIGNAL_TYPES.items()
        if path in actual_types and actual_types[path] != expected
    }
    if missing_paths or wrong_types:
        common_blockers.append("KUKSA_SIGNAL_CONTRACT_NOT_READY")

    documented_pin_types = _pin_type_enum(openapi_path)
    ethernet_api_supported = "ETHERNET" in documented_pin_types
    if not ethernet_api_supported:
        warnings.append(
            "OpenAPI addPin/import omits ETHERNET. Phase 1 avoids this blocker by "
            "reusing IVI Gateway's existing KUKSA and IVI-Switch Ethernet pins."
        )

    candidate = None
    candidate_topology = None
    candidate_script_sha = None
    expected_sha = None
    candidate_patch_blockers: list[str] = []
    candidate_artifact = (None, None)
    addon_applied = False
    if config.candidate_blueprint_id is None:
        candidate_patch_blockers.append("CANDIDATE_BLUEPRINT_REQUIRED")
    else:
        candidate = _object(
            api.get(f"/api/v1/blueprints/{_quote(config.candidate_blueprint_id)}")
        )
        if candidate.get("id") != config.candidate_blueprint_id:
            candidate_patch_blockers.append("CANDIDATE_BLUEPRINT_ID_MISMATCH")
        if candidate.get("name") != config.blueprint_name:
            candidate_patch_blockers.append("CANDIDATE_NAME_MISMATCH")
        if config.candidate_blueprint_id == config.base_blueprint_id:
            candidate_patch_blockers.append("CANDIDATE_IS_BASE")
        if config.candidate_blueprint_id in active_blueprint_ids:
            candidate_patch_blockers.append("CANDIDATE_IS_LIVE_DEPLOYMENT_BLUEPRINT")
        if candidate.get("parentBlueprintId") != config.base_blueprint_id:
            candidate_patch_blockers.append("CANDIDATE_NOT_DIRECT_CLONE_OF_BASE")
        if candidate.get("isSnapshot") is True or candidate.get("locked") is True:
            candidate_patch_blockers.append("CANDIDATE_NOT_EDITABLE")
        candidate_topology = _gateway_topology(config, candidate)
        candidate_artifact = _broker_artifact(candidate)
        if candidate_artifact != base_artifact or None in candidate_artifact:
            candidate_patch_blockers.append("CANDIDATE_KUKSA_ARTIFACT_MISMATCH")
        for ready_key, blocker in (
            ("gateway_unique", "CANDIDATE_IVI_GATEWAY_MISSING"),
            ("gateway_is_script_node", "CANDIDATE_IVI_GATEWAY_NOT_SCRIPT_NODE"),
            ("gateway_inline_script", "CANDIDATE_IVI_GATEWAY_NOT_INLINE"),
            ("broker_ready", "CANDIDATE_KUKSA_BROKER_MISSING"),
            ("kuksa_edge_ready", "CANDIDATE_GATEWAY_KUKSA_EDGE_MISSING"),
            (
                "gateway_ivi_ethernet_edge_ready",
                "CANDIDATE_GATEWAY_IVI_ETHERNET_EDGE_MISSING",
            ),
            (
                "android_ivi_ethernet_edge_ready",
                "CANDIDATE_ANDROID_IVI_ETHERNET_EDGE_MISSING",
            ),
        ):
            if not candidate_topology[ready_key]:
                candidate_patch_blockers.append(blocker)
        candidate_script = candidate_topology["gateway_script"]
        if isinstance(candidate_script, str):
            candidate_script_sha = hashlib.sha256(
                candidate_script.encode("utf-8")
            ).hexdigest()
            addon_state = gateway_addon_state(candidate_script, addon)
            addon_applied = addon_state["state"] == "applied"
            candidate_base_script = addon_state.get("base_script")
            if isinstance(candidate_base_script, str) and base_sha is not None:
                candidate_base_sha = hashlib.sha256(
                    candidate_base_script.encode("utf-8")
                ).hexdigest()
                if candidate_base_sha != base_sha:
                    candidate_patch_blockers.append(
                        "CANDIDATE_GATEWAY_BASE_HASH_MISMATCH"
                    )
            if addon_state["state"] == "conflict":
                candidate_patch_blockers.append(
                    "CANDIDATE_GATEWAY_ADDON_MARKER_CONFLICT"
                )
            expected_script = addon_state.get("expected_script")
            if isinstance(expected_script, str):
                expected_sha = hashlib.sha256(
                    expected_script.encode("utf-8")
                ).hexdigest()
        else:
            candidate_patch_blockers.append("CANDIDATE_GATEWAY_SCRIPT_MISSING")
        candidate_labels = [
            str(node.get("label", "")).lower() for node in _nodes(candidate)
        ]
        if any("mock" in label or "replay probe" in label for label in candidate_labels):
            candidate_patch_blockers.append("CANDIDATE_HAS_MOCK_OR_PROBE_NODE")

    common_blockers = list(dict.fromkeys(common_blockers))
    blueprint_name_duplicates = [
        item for item in blueprint_entries if item.get("name") == config.blueprint_name
    ]
    clone_blockers = list(common_blockers)
    if config.candidate_blueprint_id is not None:
        clone_blockers.append("CANDIDATE_ALREADY_CONFIGURED")
    if config.base_blueprint_id in active_blueprint_ids:
        clone_blockers.append("BASE_IS_LIVE_DEPLOYMENT_BLUEPRINT")
    if blueprint_name_duplicates:
        clone_blockers.append("BLUEPRINT_NAME_ALREADY_EXISTS")
    clone_blockers = list(dict.fromkeys(clone_blockers))
    patch_blockers = list(dict.fromkeys(common_blockers + candidate_patch_blockers))
    deploy_blockers = list(patch_blockers)
    if not addon_applied:
        deploy_blockers.append("GATEWAY_ADDON_NOT_APPLIED")
    deploy_blockers = list(dict.fromkeys(deploy_blockers))
    candidate_public = None
    gateway_node_id = None
    if candidate is not None and candidate_topology is not None:
        gateway = candidate_topology["gateway"]
        gateway_node_id = gateway.get("id") if isinstance(gateway, Mapping) else None
        candidate_public = {
            "id": candidate.get("id"),
            "name": candidate.get("name"),
            "parent_blueprint_id": candidate.get("parentBlueprintId"),
            "editable": candidate.get("isSnapshot") is not True
            and candidate.get("locked") is not True,
            "gateway_node_id": gateway_node_id,
            "gateway_script_sha256": candidate_script_sha,
            "expected_script_sha256": expected_sha,
            "addon_applied": addon_applied,
            "broker_artifact_id": candidate_artifact[0],
            "broker_version_id": candidate_artifact[1],
            "kuksa_edge_ready": candidate_topology["kuksa_edge_ready"],
            "gateway_ivi_ethernet_edge_ready": candidate_topology[
                "gateway_ivi_ethernet_edge_ready"
            ],
            "android_ivi_ethernet_edge_ready": candidate_topology[
                "android_ivi_ethernet_edge_ready"
            ],
        }

    return {
        "schema_version": "safeloop.carsky-native-preflight.v1",
        "read_only": True,
        "http_methods_used": ["GET"],
        "mutation_calls": 0,
        "config": asdict(config),
        "gateway_addon": {
            "path": str(addon.path),
            "sha256": addon.sha256,
            "bytes": addon.size_bytes,
            "source_mode": addon.source_mode,
            "expected_combined_script_sha256": expected_sha,
        },
        "platform": {
            "health": health,
            "cpu": cpu,
            "advertised_limits": limits,
            "blueprints_visible": len(blueprint_entries),
            "devices_visible": len(device_entries),
            "active_deployments": len(deployments),
            "active_blueprint_ids": active_blueprint_ids,
        },
        "base": {
            "id": base.get("id"),
            "name": base.get("name"),
            "nodes": len(_nodes(base)),
            "edges": len(_edges(base)),
            "base_gateway_script_sha256": base_sha,
            "topology_sha256": _topology_fingerprint(base),
            "broker_artifact_id": base_artifact[0],
            "broker_version_id": base_artifact[1],
            "gateway_kuksa_hmi_topology_ready": not (
                "BASE_GATEWAY_KUKSA_HMI_TOPOLOGY_MISSING" in common_blockers
            ),
        },
        "target": {
            "id": target.get("id"),
            "name": target.get("name"),
            "status": target.get("status"),
            "locked": target.get("locked"),
            "active_deployment_ids": [item.get("id") for item in target_deployments],
        },
        "kuksa": {
            "contract": config.gateway.kuksa.contract,
            "schema_source": schema_source,
            "reference_artifact_id": reference_artifact[0],
            "reference_version_id": reference_artifact[1],
            "schema_ready": not missing_paths and not wrong_types and source_ok,
            "missing_paths": missing_paths,
            "wrong_types": wrong_types,
            "bootstrap_attestation": kuksa_attestation,
        },
        "hmi": {
            "transport": config.hmi.transport,
            "destination": f"{config.hmi.target_host}:{config.hmi.port}",
            "documented_add_pin_types": sorted(documented_pin_types),
            "ethernet_pin_api_supported": ethernet_api_supported,
            "api_blocker_avoided_by_gateway_reuse": not ethernet_api_supported,
        },
        "candidate": candidate_public,
        "warnings": warnings,
        "clone_blockers": clone_blockers,
        "patch_blockers": patch_blockers,
        "deploy_blockers": deploy_blockers,
        "clone_ready": not clone_blockers,
        "patch_ready": not patch_blockers,
        "deploy_ready": not deploy_blockers,
        "deploy_execution_supported": True,
    }


def build_plan(
    config: NativeConfig, addon: GatewayAddon, result: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = result.get("candidate")
    node_id = (
        candidate.get("gateway_node_id")
        if isinstance(candidate, Mapping)
        else "{resolve IVI Gateway node id on clone}"
    )
    return {
        "schema_version": "safeloop.carsky-native-plan.v1",
        "read_only": True,
        "executed": False,
        "deploy_execution_supported": True,
        "clone_ready": bool(result.get("clone_ready")),
        "patch_ready": bool(result.get("patch_ready")),
        "deploy_ready": bool(result.get("deploy_ready")),
        "clone_blockers": list(result.get("clone_blockers") or []),
        "patch_blockers": list(result.get("patch_blockers") or []),
        "deploy_blockers": list(result.get("deploy_blockers") or []),
        "phase_1_architecture": {
            "new_container": False,
            "registry_required": False,
            "container_image": None,
            "container_image_policy": (
                "not applicable in gateway-addon phase; any later container phase "
                "must use an explicit immutable image digest"
            ),
            "reused_node": config.gateway.node_label,
            "reused_pins": ["KUKSA output to Central Broker", "ETHERNET output to IVI Switch"],
            "hmi": f"UDP {config.hmi.target_host}:{config.hmi.port}",
            "addon_source_mode": addon.source_mode,
        },
        "ethernet_api_constraint": {
            "documented_add_pin_support": False,
            "impact": "A new runtime node cannot be wired to IVI Ethernet through documented batch/import APIs.",
            "phase_1_resolution": "Reuse the existing IVI Gateway Ethernet pin and edge; do not substitute GENERIC.",
        },
        "future_reviewed_steps": [
            {
                "step": 1,
                "purpose": "clone the clean team blueprint",
                "method": "POST",
                "path": f"/api/v1/blueprints/{config.base_blueprint_id}/clone",
                "body": {"name": config.blueprint_name, "isSnapshot": False},
                "guarded_command": (
                    "python3 tools/carsky_native_ctl.py clone-candidate <config.json> "
                    f"--gateway-addon {addon.path} --confirm-base-id "
                    f"{config.base_blueprint_id}"
                ),
                "executed": False,
            },
            {
                "step": 2,
                "purpose": "set candidate_blueprint_id to the returned direct clone and rerun GET-only preflight",
                "executed": False,
            },
            {
                "step": 3,
                "purpose": "backup and patch only IVI Gateway config.scriptContent",
                "method": "PATCH",
                "path": f"/api/v1/nodes/{node_id}",
                "payload_shape": {
                    "config": {
                        "<all existing config fields>": "unchanged",
                        "scriptContent": "<current clone script + verified addon>",
                    }
                },
                "combined_script_sha256": result.get("gateway_addon", {}).get(
                    "expected_combined_script_sha256"
                ),
                "guarded_command": (
                    "python3 tools/carsky_native_ctl.py patch-gateway <config.json> "
                    f"--gateway-addon {addon.path} --confirm-candidate-id "
                    "<candidateBlueprintId>"
                ),
                "on_http_404": "Stop. Use CarSky UI/manual approved update; never patch base or a live deployment snapshot.",
                "executed": False,
            },
            {
                "step": 4,
                "purpose": "patch-gateway reads back the exact hash and POST-validates the candidate clone",
                "executed": False,
            },
            {
                "step": 5,
                "purpose": "deploy only after a new GET-only preflight says deploy_ready=true",
                "requests": [
                    {
                        "method": "POST",
                        "path": "/api/v1/deployments",
                        "body": {
                            "blueprintId": "{candidateBlueprintId}",
                            "roomId": config.target_device_id,
                            "name": config.deployment_name,
                        },
                    }
                ],
                "guarded_command": (
                    "python3 tools/carsky_native_ctl.py deploy <config.json> "
                    f"--gateway-addon {addon.path} --confirm-candidate-id "
                    "<candidateBlueprintId> --confirm-device-id "
                    f"{config.target_device_id}"
                ),
                "executed": False,
            },
        ],
    }


def _safe_resource_id(value: object, *, field: str, exc_type: type[MutationRefused]) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise exc_type(f"{field} không hợp lệ")
    return value


def clone_candidate(
    config_path: Path,
    config: NativeConfig,
    addon: GatewayAddon,
    api: Any,
    *,
    confirm_base_id: str,
    working_root: Path = DEFAULT_WORKING_CONFIG_ROOT,
    openapi_path: Path = DEFAULT_OPENAPI,
) -> dict[str, Any]:
    """Clone only the locked base and atomically record a verified clone ID."""

    if confirm_base_id != DEFAULT_BASE_BLUEPRINT_ID:
        raise CloneRefused("--confirm-base-id không khớp base blueprint đã khóa")
    if config.base_blueprint_id != DEFAULT_BASE_BLUEPRINT_ID:
        raise CloneRefused("Config không trỏ tới base blueprint đã khóa")
    if config.candidate_blueprint_id is not None:
        raise CloneRefused("Config đã có candidate_blueprint_id; từ chối clone lần nữa")
    resolved_config = _working_config_path(config_path, working_root)

    before = preflight(config, addon, api, openapi_path=openapi_path)
    if before["clone_blockers"]:
        raise CloneRefused(
            "Candidate clone gate fail: " + ", ".join(before["clone_blockers"])
        )
    base = _object(api.get(f"/api/v1/blueprints/{_quote(config.base_blueprint_id)}"))
    if base.get("id") != DEFAULT_BASE_BLUEPRINT_ID:
        raise CloneRefused("Base readback id không khớp ngay trước clone")
    base_topology_sha = _topology_fingerprint(base)

    response = api.clone_blueprint(config.base_blueprint_id, config.blueprint_name)
    returned = _object(response)
    candidate_id = returned.get("id")
    if not isinstance(candidate_id, str):
        entries = _items(api.get("/api/v1/blueprints?limit=100"))
        matches = [item for item in entries if item.get("name") == config.blueprint_name]
        if len(matches) != 1:
            raise CloneRefused(
                "Clone đã được gửi nhưng không xác định duy nhất candidate id; "
                "không tự động thử clone lại"
            )
        candidate_id = matches[0].get("id")
    candidate_id = _safe_resource_id(
        candidate_id, field="Candidate blueprint id", exc_type=CloneRefused
    )
    if candidate_id == config.base_blueprint_id:
        raise CloneRefused("Clone endpoint trả về chính base blueprint")

    candidate = _object(api.get(f"/api/v1/blueprints/{_quote(candidate_id)}"))
    candidate_topology_sha = _topology_fingerprint(candidate)
    verification_blockers: list[str] = []
    if candidate.get("id") != candidate_id:
        verification_blockers.append("CLONE_ID_MISMATCH")
    if candidate.get("name") != config.blueprint_name:
        verification_blockers.append("CLONE_NAME_MISMATCH")
    if candidate.get("parentBlueprintId") != config.base_blueprint_id:
        verification_blockers.append("CLONE_NOT_DIRECT_CHILD_OF_BASE")
    if candidate.get("isSnapshot") is True or candidate.get("locked") is True:
        verification_blockers.append("CLONE_NOT_EDITABLE")
    if candidate_topology_sha != base_topology_sha:
        verification_blockers.append("CLONE_TOPOLOGY_HASH_MISMATCH")
    if verification_blockers:
        raise CloneRefused(
            "Clone verification fail: " + ", ".join(verification_blockers)
        )

    candidate_config = replace(config, candidate_blueprint_id=candidate_id)
    after = preflight(candidate_config, addon, api, openapi_path=openapi_path)
    if after["patch_blockers"]:
        raise CloneRefused(
            "Clone fresh preflight fail: " + ", ".join(after["patch_blockers"])
        )
    recorded_path = _atomic_record_candidate_id(
        resolved_config,
        candidate_id,
        expected_config=config,
        working_root=working_root,
    )
    return {
        "cloned": True,
        "candidate_blueprint_id": candidate_id,
        "base_blueprint_id": config.base_blueprint_id,
        "candidate_name": config.blueprint_name,
        "editable": True,
        "direct_clone_verified": True,
        "topology_sha256": candidate_topology_sha,
        "base_topology_sha256": base_topology_sha,
        "working_config": str(recorded_path),
        "config_candidate_recorded": True,
        "patch_ready": True,
        "mutation_calls": 1,
        "mutation_scope": ["POST locked base clone"],
    }


def _write_backup(
    backup_dir: Path,
    *,
    config: NativeConfig,
    gateway_node_id: str,
    current_config: Mapping[str, Any],
    expected_sha256: str,
) -> Path:
    _reject_secret_material(current_config, field="candidate.gateway.config")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate_id = config.candidate_blueprint_id
    if candidate_id is None:
        raise PatchRefused("Candidate id thiếu trước backup")
    filename = f"{stamp}-{candidate_id}-{gateway_node_id}.json"
    if not re.fullmatch(r"[A-Za-z0-9TZ_-]+\.json", filename):
        raise PatchRefused("Backup filename không an toàn")
    script = current_config.get("scriptContent")
    current_sha = (
        hashlib.sha256(script.encode("utf-8")).hexdigest()
        if isinstance(script, str)
        else None
    )
    payload = {
        "schema_version": "safeloop.carsky-gateway-backup.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_blueprint_id": candidate_id,
        "gateway_node_id": gateway_node_id,
        "current_script_sha256": current_sha,
        "expected_script_sha256": expected_sha256,
        "current_config": dict(current_config),
    }
    path = backup_dir / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_manual_gateway_script(
    backup_path: Path,
    *,
    expected_script: str,
    expected_sha256: str,
) -> Path:
    """Atomically create a private, exact script for the CarSky UI fallback.

    The destination deliberately shares the backup's unique stem.  A hard-link
    publish makes the completed temporary file visible in one operation and
    refuses to replace an artifact if the name somehow already exists.
    """

    if not isinstance(expected_script, str) or not expected_script:
        raise PatchRefused("Manual IVI Gateway script không được rỗng")
    encoded = expected_script.encode("utf-8")
    rendered_sha = hashlib.sha256(encoded).hexdigest()
    if rendered_sha != expected_sha256:
        raise PatchRefused(
            "Manual IVI Gateway script hash sai trước khi ghi; "
            f"expected={expected_sha256}; actual={rendered_sha}"
        )
    if backup_path.suffix != ".json" or not backup_path.is_file():
        raise PatchRefused("Backup path không hợp lệ để tạo manual script")
    destination = backup_path.with_suffix(".manual.lua")
    if destination.exists() or destination.is_symlink():
        raise PatchRefused(
            f"Từ chối ghi đè manual IVI Gateway script đã tồn tại: {destination}"
        )

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise PatchRefused(
                f"Từ chối ghi đè manual IVI Gateway script đã tồn tại: {destination}"
            ) from exc
        Path(temporary_name).unlink()
        temporary_name = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    actual = destination.read_bytes()
    actual_sha = hashlib.sha256(actual).hexdigest()
    if actual_sha != expected_sha256:
        raise PatchRefused(
            "Manual IVI Gateway script hash sai sau atomic write; "
            f"path={destination}; expected={expected_sha256}; actual={actual_sha}"
        )
    if destination.stat().st_mode & 0o077:
        raise PatchRefused(
            f"Manual IVI Gateway script không có quyền private 0600: {destination}"
        )
    return destination.resolve()


def patch_gateway(
    config: NativeConfig,
    addon: GatewayAddon,
    api: Any,
    *,
    confirm_candidate_id: str,
    backup_dir: Path,
    openapi_path: Path = DEFAULT_OPENAPI,
) -> dict[str, Any]:
    """Guarded mutation path; never targets base or any active blueprint."""

    candidate_id = config.candidate_blueprint_id
    if candidate_id is None:
        raise PatchRefused("Config phải có candidate_blueprint_id trước PATCH")
    if confirm_candidate_id != candidate_id:
        raise PatchRefused("--confirm-candidate-id không khớp candidate trong config")
    if candidate_id == config.base_blueprint_id:
        raise PatchRefused("Từ chối PATCH base blueprint")
    result = preflight(config, addon, api, openapi_path=openapi_path)
    if result["patch_blockers"]:
        raise PatchRefused(
            "Candidate patch gate fail: " + ", ".join(result["patch_blockers"])
        )
    candidate = _object(api.get(f"/api/v1/blueprints/{_quote(candidate_id)}"))
    topology = _gateway_topology(config, candidate)
    gateway = topology["gateway"]
    current_config = topology["gateway_config"]
    if not isinstance(gateway, Mapping) or not isinstance(current_config, Mapping):
        raise PatchRefused("Không đọc được IVI Gateway config trên candidate")
    gateway_node_id = gateway.get("id")
    if not isinstance(gateway_node_id, str):
        raise PatchRefused("IVI Gateway candidate thiếu node id")
    current_script = current_config.get("scriptContent")
    if not isinstance(current_script, str):
        raise PatchRefused("Clone IVI Gateway thiếu scriptContent")
    addon_state = gateway_addon_state(current_script, addon)
    if addon_state["state"] == "conflict":
        raise PatchRefused("Clone IVI Gateway đã có marker addon khác hoặc hỏng")
    expected_script = addon_state.get("expected_script")
    if not isinstance(expected_script, str):
        raise PatchRefused("Không tạo được combined IVI Gateway script")
    expected_sha = hashlib.sha256(expected_script.encode("utf-8")).hexdigest()
    if addon_state["state"] == "applied":
        return {
            "patched": False,
            "already_applied": True,
            "candidate_blueprint_id": candidate_id,
            "gateway_node_id": gateway_node_id,
            "script_sha256": expected_sha,
            "mutation_calls": 0,
        }
    backup = _write_backup(
        backup_dir,
        config=config,
        gateway_node_id=gateway_node_id,
        current_config=current_config,
        expected_sha256=expected_sha,
    )
    updated_config = dict(current_config)
    updated_config["scriptContent"] = expected_script
    try:
        api.patch_gateway_config(gateway_node_id, updated_config)
    except CarSkyApiError as exc:
        if exc.status_code == 404:
            manual_script = _write_manual_gateway_script(
                backup,
                expected_script=expected_script,
                expected_sha256=expected_sha,
            )
            raise PatchRefused(
                "PATCH node route trả HTTP 404; đã dừng, không thử route khác. "
                f"backup={backup.resolve()}; manual_script={manual_script}; "
                f"candidate_blueprint_id={candidate_id}; gateway_node_id={gateway_node_id}; "
                f"expected_script_sha256={expected_sha}. Trên CarSky UI, chỉ mở "
                f"candidate blueprint {candidate_id}, chọn node IVI Gateway {gateway_node_id}, "
                f"rồi thay toàn bộ Script Content bằng đúng nội dung file {manual_script}. "
                "Không sửa base hoặc blueprint/snapshot đang chạy; sau khi Save, chạy lại "
                "preflight để xác minh hash trước khi deploy."
            ) from exc
        raise
    after = _object(api.get(f"/api/v1/blueprints/{_quote(candidate_id)}"))
    after_script = _gateway_topology(config, after)["gateway_script"]
    after_sha = (
        hashlib.sha256(after_script.encode("utf-8")).hexdigest()
        if isinstance(after_script, str)
        else None
    )
    if after_sha != expected_sha:
        raise PatchRefused(
            f"PATCH readback hash mismatch; backup={backup}; expected={expected_sha}; actual={after_sha}"
        )
    validation = api.validate_blueprint(candidate_id)
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise PatchRefused(
            f"Candidate validation fail sau PATCH; backup={backup}; validation={validation}"
        )
    return {
        "patched": True,
        "candidate_blueprint_id": candidate_id,
        "gateway_node_id": gateway_node_id,
        "backup": str(backup),
        "script_sha256": expected_sha,
        "readback_hash_match": True,
        "validation": dict(validation),
        "mutation_calls": 2,
        "mutation_scope": ["PATCH candidate IVI Gateway config", "POST candidate validate"],
    }


def _public_deployment(value: object) -> dict[str, Any]:
    deployment = _object(value)
    allowed = (
        "id",
        "blueprintId",
        "roomId",
        "name",
        "status",
        "namespace",
        "createdAt",
    )
    return {key: deployment.get(key) for key in allowed if key in deployment}


def _public_status(value: object) -> dict[str, Any]:
    status = _object(value)
    return {
        key: status.get(key)
        for key in ("status", "namespace")
        if isinstance(status.get(key), str) or status.get(key) is None
    }


def _exact_target_deployments(api: Any, config: NativeConfig) -> list[dict[str, Any]]:
    path = "/api/v1/deployments/find?" + parse.urlencode(
        {"device": config.target_device_id}
    )
    return [
        item
        for item in _items(api.get(path))
        if item.get("roomId") == config.target_device_id
    ]


def deploy_candidate(
    config: NativeConfig,
    addon: GatewayAddon,
    api: Any,
    *,
    confirm_candidate_id: str,
    confirm_device_id: str,
    poll_attempts: int = 30,
    poll_interval_s: float = 2.0,
    sleeper: Callable[[float], None] = time.sleep,
    openapi_path: Path = DEFAULT_OPENAPI,
) -> dict[str, Any]:
    """Create one guarded deployment; never stops or replaces another one."""

    candidate_id = config.candidate_blueprint_id
    if candidate_id is None:
        raise DeployRefused("Config chưa có candidate_blueprint_id")
    if confirm_candidate_id != candidate_id:
        raise DeployRefused("--confirm-candidate-id không khớp candidate trong config")
    if confirm_device_id != config.target_device_id:
        raise DeployRefused("--confirm-device-id không khớp device trong config")
    if candidate_id == config.base_blueprint_id:
        raise DeployRefused("Từ chối deploy base blueprint")
    if not isinstance(poll_attempts, int) or isinstance(poll_attempts, bool):
        raise DeployRefused("poll-attempts phải là integer")
    if not 1 <= poll_attempts <= 120:
        raise DeployRefused("poll-attempts phải trong [1, 120]")
    if not isinstance(poll_interval_s, (int, float)) or not 0 <= poll_interval_s <= 60:
        raise DeployRefused("poll-interval phải trong [0, 60] giây")

    fresh = preflight(config, addon, api, openapi_path=openapi_path)
    if fresh["deploy_ready"] is not True or fresh["deploy_blockers"]:
        raise DeployRefused(
            "Fresh deploy gate fail: " + ", ".join(fresh["deploy_blockers"])
        )
    # This second target read narrows the race window between inventory audit
    # and deployment creation.  Any active row is a hard stop, including an
    # exact prior attempt; the command never creates a duplicate or replaces it.
    if _exact_target_deployments(api, config):
        raise DeployRefused("Target device became busy; no deployment was created")

    validation = api.validate_blueprint(candidate_id)
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise DeployRefused("Candidate validation fail ngay trước deploy")
    created_raw = api.create_deployment(
        candidate_blueprint_id=candidate_id,
        target_device_id=config.target_device_id,
        name=config.deployment_name,
    )
    created = _public_deployment(created_raw)

    for key, expected in (
        ("blueprintId", candidate_id),
        ("roomId", config.target_device_id),
        ("name", config.deployment_name),
    ):
        if key in created and created[key] != expected:
            raise DeployRefused(f"Deployment response {key} không khớp request đã khóa")

    final_status: dict[str, Any] = {}
    terminal = False
    terminal_states = {"RUNNING", "FAILED", "ERROR", "STOPPED", "TERMINATED"}
    status_path = (
        f"/api/v1/deployments/{_quote(config.target_device_id)}/status"
    )
    polls = 0
    for attempt in range(poll_attempts):
        if attempt and poll_interval_s:
            sleeper(float(poll_interval_s))
        polls += 1
        final_status = _public_status(api.get(status_path))
        state = final_status.get("status")
        if state in terminal_states:
            terminal = True
            break

    deployment_id = created.get("id")
    if not isinstance(deployment_id, str):
        matches = [
            item
            for item in _exact_target_deployments(api, config)
            if item.get("blueprintId") == candidate_id
            and item.get("name") == config.deployment_name
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("id"), str):
            raise DeployRefused(
                "Deployment đã được gửi nhưng không xác định duy nhất deployment id; "
                "không tự động gửi lại"
            )
        deployment_id = matches[0]["id"]
        created = _public_deployment(matches[0])
    deployment_id = _safe_resource_id(
        deployment_id, field="Deployment id", exc_type=DeployRefused
    )

    return {
        "deployed": True,
        "deployment_id": deployment_id,
        "candidate_blueprint_id": candidate_id,
        "target_device_id": config.target_device_id,
        "deployment_name": config.deployment_name,
        "created": created,
        "status": final_status,
        "polls": polls,
        "poll_timed_out": not terminal,
        "running": final_status.get("status") == "RUNNING",
        "mutation_calls": 1,
        "validation_calls": 1,
        "mutation_scope": ["POST deployment create"],
        "active_deployment_untouched": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight và guarded native deployment cho SafeLoop CarSky"
    )
    parser.add_argument("--url", default=os.getenv("A8_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--openapi", type=Path, default=DEFAULT_OPENAPI)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "offline config/addon validation"),
        ("preflight", "GET-only live audit and gates"),
        ("plan", "GET-only audit plus future request plan"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument(
            "--gateway-addon",
            "--gateway-script",
            dest="gateway_addon",
            type=Path,
            required=True,
            help="SafeLoop addon only; the tool reads and preserves the clone's base script",
        )
    clone_parser = sub.add_parser(
        "clone-candidate", help="guarded clone of the locked clean base"
    )
    clone_parser.add_argument("config", type=Path)
    clone_parser.add_argument(
        "--gateway-addon",
        "--gateway-script",
        dest="gateway_addon",
        type=Path,
        required=True,
        help="SafeLoop addon used by the clone preflight gates",
    )
    clone_parser.add_argument("--confirm-base-id", required=True)
    patch_parser = sub.add_parser(
        "patch-gateway", help="guarded candidate-only IVI Gateway script PATCH"
    )
    patch_parser.add_argument("config", type=Path)
    patch_parser.add_argument(
        "--gateway-addon",
        "--gateway-script",
        dest="gateway_addon",
        type=Path,
        required=True,
        help="SafeLoop addon only; never pass or copy the 30KB base script",
    )
    patch_parser.add_argument("--confirm-candidate-id", required=True)
    patch_parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    deploy_parser = sub.add_parser(
        "deploy", help="guarded candidate deployment to the locked inactive device"
    )
    deploy_parser.add_argument("config", type=Path)
    deploy_parser.add_argument(
        "--gateway-addon",
        "--gateway-script",
        dest="gateway_addon",
        type=Path,
        required=True,
        help="SafeLoop addon whose exact hash must already be applied",
    )
    deploy_parser.add_argument("--confirm-candidate-id", required=True)
    deploy_parser.add_argument("--confirm-device-id", required=True)
    deploy_parser.add_argument("--poll-attempts", type=int, default=30)
    deploy_parser.add_argument("--poll-interval", type=float, default=2.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[..., Any] = CarSkyApi,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        addon = load_gateway_addon(args.gateway_addon, config)
        addon_summary = {
            "path": str(addon.path),
            "sha256": addon.sha256,
            "bytes": addon.size_bytes,
            "source_mode": addon.source_mode,
        }
        if args.command == "validate-config":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "read_only": True,
                        "config": asdict(config),
                        "gateway_addon": addon_summary,
                        "note": "Local config/addon valid; live gates have not run.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        api = api_factory(
            args.url,
            os.getenv("A8_API_KEY", ""),
            timeout_s=args.timeout,
        )
        if args.command == "clone-candidate":
            cloned = clone_candidate(
                args.config,
                config,
                addon,
                api,
                confirm_base_id=args.confirm_base_id,
                openapi_path=args.openapi,
            )
            print(json.dumps(cloned, ensure_ascii=False, indent=2))
            return 0
        if args.command == "patch-gateway":
            patched = patch_gateway(
                config,
                addon,
                api,
                confirm_candidate_id=args.confirm_candidate_id,
                backup_dir=args.backup_dir,
                openapi_path=args.openapi,
            )
            print(json.dumps(patched, ensure_ascii=False, indent=2))
            return 0
        if args.command == "deploy":
            deployed = deploy_candidate(
                config,
                addon,
                api,
                confirm_candidate_id=args.confirm_candidate_id,
                confirm_device_id=args.confirm_device_id,
                poll_attempts=args.poll_attempts,
                poll_interval_s=args.poll_interval,
                openapi_path=args.openapi,
            )
            print(json.dumps(deployed, ensure_ascii=False, indent=2))
            return 0 if deployed["running"] else 6
        result = preflight(config, addon, api, openapi_path=args.openapi)
        if args.command == "preflight":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["deploy_ready"] else 4
        if args.command == "plan":
            print(json.dumps(build_plan(config, addon, result), ensure_ascii=False, indent=2))
            return 0 if result["deploy_ready"] else 4
        raise NativeConfigError(f"Command không được hỗ trợ: {args.command}")
    except (
        NativeConfigError,
        CarSkyApiError,
        MutationRefused,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
