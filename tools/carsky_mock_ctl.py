#!/usr/bin/env python3
"""Install/deploy the SafeLoop mock into the retained CarSky lab device.

Credentials are read only from ``A8_API_KEY`` and are never printed or stored.
The tenant currently advertises node PATCH in OpenAPI but does not expose the
route at runtime.  Its portable import also rejects legacy pin types present
in its own export.  This command therefore uses the verified runtime path:
clone the clean base blueprint, then atomically add one mock node/pin/edge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, parse, request

DEFAULT_URL = "https://hackathon-1.carsky.io"
DEFAULT_ROOM_ID = "h5sjc3jtzl8vzl9wyqokv"
DEFAULT_BASE_BLUEPRINT_ID = "wVMseHi7XtUhuMIn_1D5e"
DEFAULT_BLUEPRINT_NAME = "SafeLoop Mock Integration Lab"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "carsky" / "scripts" / "safeloop_mock_pipeline.lua"
MOCK_PUBLISHER_LABEL = "SafeLoop Mock C1 C2 Fusion"
VERIFY_PATHS = (
    "Vehicle.Speed",
    "Vehicle.Acceleration.Longitudinal",
    "Vehicle.Acceleration.Lateral",
    "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap",
    "Vehicle.ADAS.ObstacleDetection.Front.Center.Distance",
    "Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning",
    "Vehicle.Driver.AttentiveProbability",
    "Vehicle.Driver.DistractionLevel",
    "Vehicle.Driver.FatigueLevel",
    "Vehicle.Driver.IsEyesOnRoad",
    "Vehicle.ADAS.DMS.IsWarning",
)


class ApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Api:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 30.0):
        if not api_key:
            raise ValueError("Thiếu A8_API_KEY trong environment")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def call(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None
        headers = {"Accept": "application/json", "X-API-Key": self.api_key}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(
            self.base_url + path, data=body, headers=headers, method=method
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise ApiError(
                f"CarSky HTTP {exc.code} tại {path}: {detail}",
                status_code=exc.code,
            ) from exc
        except (error.URLError, OSError) as exc:
            raise ApiError(f"Không kết nối được CarSky: {exc}") from exc
        if not raw:
            return {}
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise ApiError(f"CarSky trả response không phải object tại {path}")
        return decoded


def _quoted(value: str) -> str:
    return parse.quote(value, safe="")


def _find_blueprint(api: Api, name: str) -> dict | None:
    response = api.call(
        "GET", "/api/v1/blueprints?" + parse.urlencode({"limit": 100, "name": name})
    )
    items = response.get("data")
    if not isinstance(items, list):
        raise ApiError("Response blueprint list thiếu array data")
    matches = [item for item in items if isinstance(item, dict) and item.get("name") == name]
    if len(matches) > 1:
        raise ApiError(f"Có nhiều blueprint trùng tên {name!r}; cần dọn trên UI")
    return matches[0] if matches else None


def _blueprint_id(item: dict) -> str:
    value = item.get("id") or item.get("blueprintId")
    if not isinstance(value, str) or not value:
        raise ApiError(f"Không tìm thấy blueprint id trong response: {item}")
    return value


def _blueprint_body(response: dict) -> dict:
    """Accept the direct and wrapped response shapes observed across A8 APIs."""

    candidates = (response, response.get("data"), response.get("blueprint"))
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("nodes"), list):
            return candidate
    raise ApiError("Response get blueprint thiếu array nodes")


def _broker_input_pin_id(blueprint: dict) -> str:
    candidates = []
    for node in blueprint["nodes"]:
        if not isinstance(node, dict) or node.get("nodeType") != "kuksa-databroker":
            continue
        for pin in node.get("pins") or []:
            if (
                isinstance(pin, dict)
                and pin.get("pinType") == "KUKSA"
                and pin.get("direction") == "INPUT"
                and isinstance(pin.get("id"), str)
            ):
                candidates.append(pin["id"])
    if len(candidates) != 1:
        raise ApiError(f"Cần đúng một KUKSA broker input pin, gặp {len(candidates)}")
    return candidates[0]


def _has_mock_node(blueprint: dict) -> bool:
    return any(
        isinstance(node, dict) and node.get("label") == MOCK_PUBLISHER_LABEL
        for node in blueprint["nodes"]
    )


def _mock_node(blueprint: dict) -> dict | None:
    matches = [
        node
        for node in blueprint["nodes"]
        if isinstance(node, dict) and node.get("label") == MOCK_PUBLISHER_LABEL
    ]
    if len(matches) > 1:
        raise ApiError(f"Có {len(matches)} mock node trùng label; cần dọn trên UI")
    return matches[0] if matches else None


def _mock_output_pin(node: dict) -> dict | None:
    matches = [
        pin
        for pin in node.get("pins") or []
        if isinstance(pin, dict)
        and pin.get("name") == "kuksa"
        and pin.get("pinType") == "KUKSA"
        and pin.get("direction") == "OUTPUT"
    ]
    if len(matches) > 1:
        raise ApiError(f"Mock node có {len(matches)} KUKSA output pin trùng nhau")
    return matches[0] if matches else None


def _has_mock_edge(blueprint: dict, source_pin_id: str, target_pin_id: str) -> bool:
    return any(
        isinstance(edge, dict)
        and edge.get("sourcePinId") == source_pin_id
        and edge.get("targetPinId") == target_pin_id
        for edge in blueprint.get("edges") or []
    )


def _add_mock_node(api: Api, blueprint_id: str, blueprint: dict) -> dict:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    return api.call(
        "POST",
        f"/api/v1/blueprints/{_quoted(blueprint_id)}/batch",
        {
            "operations": [
                {
                    "op": "addNode",
                    "data": {
                        "label": MOCK_PUBLISHER_LABEL,
                        "nodeType": "script-node",
                        "config": {"script": "inline", "scriptContent": source},
                        "positionX": 550,
                        "positionY": -420,
                    },
                },
                {
                    "op": "addPin",
                    "nodeRef": 0,
                    "data": {
                        "name": "kuksa",
                        "pinType": "KUKSA",
                        "direction": "OUTPUT",
                        "side": "RIGHT",
                    },
                },
                {
                    "op": "addEdge",
                    "data": {
                        # Refs are zero-based within each operation type, not
                        # indexes in the combined operations array.  This is
                        # the first (and only) addPin operation in the batch.
                        "sourcePinRef": 0,
                        "targetPinId": _broker_input_pin_id(blueprint),
                    },
                },
            ]
        },
    )


def _ensure_mock_connection(api: Api, blueprint_id: str, blueprint: dict) -> tuple[str, dict | None]:
    node = _mock_node(blueprint)
    target_pin_id = _broker_input_pin_id(blueprint)
    if node is None:
        return "mock-added", _add_mock_node(api, blueprint_id, blueprint)

    node_id = node.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise ApiError("Mock node hiện có thiếu id")
    output_pin = _mock_output_pin(node)
    if output_pin is None:
        return "pin-edge-repaired", api.call(
            "POST",
            f"/api/v1/blueprints/{_quoted(blueprint_id)}/batch",
            {
                "operations": [
                    {
                        "op": "addPin",
                        "nodeId": node_id,
                        "data": {
                            "name": "kuksa",
                            "pinType": "KUKSA",
                            "direction": "OUTPUT",
                            "side": "RIGHT",
                        },
                    },
                    {
                        "op": "addEdge",
                        "data": {
                            "sourcePinRef": 0,
                            "targetPinId": target_pin_id,
                        },
                    },
                ]
            },
        )

    source_pin_id = output_pin.get("id")
    if not isinstance(source_pin_id, str) or not source_pin_id:
        raise ApiError("Mock KUKSA output pin hiện có thiếu id")
    if not _has_mock_edge(blueprint, source_pin_id, target_pin_id):
        return "edge-repaired", api.call(
            "POST",
            f"/api/v1/blueprints/{_quoted(blueprint_id)}/batch",
            {
                "operations": [
                    {
                        "op": "addEdge",
                        "data": {
                            "sourcePinId": source_pin_id,
                            "targetPinId": target_pin_id,
                        },
                    }
                ]
            },
        )
    return "connection-existing", None


def _connection_ready(blueprint: dict) -> bool:
    node = _mock_node(blueprint)
    if node is None:
        return False
    output_pin = _mock_output_pin(node)
    if output_pin is None or not isinstance(output_pin.get("id"), str):
        return False
    return _has_mock_edge(
        blueprint, output_pin["id"], _broker_input_pin_id(blueprint)
    )


def _signal_node_key(response: dict) -> str:
    nodes = response.get("nodes")
    if not isinstance(nodes, list):
        raise ApiError("Signal discovery thiếu array nodes")
    kuksa = [
        node.get("key")
        for node in nodes
        if isinstance(node, dict)
        and node.get("kind") == "kuksa"
        and isinstance(node.get("key"), str)
    ]
    if "central-broker-vss" in kuksa:
        return "central-broker-vss"
    if len(kuksa) != 1:
        raise ApiError(f"Không xác định duy nhất KUKSA signal node: {kuksa}")
    return kuksa[0]


def _value_map(response: dict) -> dict:
    values = response.get("values")
    if not isinstance(values, list):
        raise ApiError("Signal values response thiếu array values")
    return {
        item["path"]: item.get("value")
        for item in values
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }


def _verify_runtime(api: Api, room_id: str, samples: int, interval_s: float) -> dict:
    status = api.call("GET", f"/api/v1/deployments/{_quoted(room_id)}/status")
    discovery = api.call("GET", f"/api/v1/signals/{_quoted(room_id)}")
    node_key = _signal_node_key(discovery)
    value_samples = []
    for index in range(samples):
        response = api.call(
            "POST",
            f"/api/v1/signals/{_quoted(room_id)}/{_quoted(node_key)}/values",
            {"paths": list(VERIFY_PATHS)},
        )
        value_samples.append(_value_map(response))
        if index + 1 < samples:
            time.sleep(interval_s)

    changed = sorted(
        path
        for path in VERIFY_PATHS
        if len({sample.get(path) for sample in value_samples}) > 1
    )
    missing = sorted(
        path for path in VERIFY_PATHS if any(path not in sample for sample in value_samples)
    )
    return {
        "status": status,
        "signal_node_key": node_key,
        "samples": value_samples,
        "changed_paths": changed,
        "missing_paths": missing,
        "mock_live": (
            status.get("status") == "RUNNING"
            and not missing
            and "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap" in changed
            and "Vehicle.Speed" in changed
        ),
    }


def _android_probe(api: Api, room_id: str) -> dict:
    tunnel = api.call(
        "GET", f"/api/v1/deployments/{_quoted(room_id)}/adb-tunnel"
    )
    nodes = tunnel.get("nodes")
    if not isinstance(nodes, list):
        raise ApiError("ADB tunnel response thiếu array nodes")
    running = [
        node
        for node in nodes
        if isinstance(node, dict)
        and node.get("nodeType") == "skycraft"
        and node.get("phase") == "Running"
        and isinstance(node.get("nodeKey"), str)
    ]
    if len(running) != 1:
        raise ApiError(f"Cần đúng một Skycraft Running, gặp {len(running)}")
    node = running[0]
    node_key = node["nodeKey"]
    commands = {
        "boot_completed": "getprop sys.boot_completed",
        "display_size": "wm size",
        "browser_activity": (
            "cmd package resolve-activity --brief -a android.intent.action.VIEW "
            "-d https://example.com"
        ),
        "candidate_packages": (
            "pm list packages | grep -Ei 'chrome|browser|webview|launcher|car' | head -80"
        ),
        "base64_tool": "command -v base64 || toybox which base64",
        "temp_dir": "ls -ld /data/local/tmp",
    }
    results = {}
    for label, command in commands.items():
        try:
            response = api.call(
                "POST",
                f"/api/v1/deployments/{_quoted(room_id)}/adb-exec/{_quoted(node_key)}",
                {"command": command, "binary": False},
            )
            transport = "deployment-adb-exec"
        except ApiError as exc:
            if exc.status_code != 502:
                raise
            response = api.call(
                "POST",
                f"/api/v1/vms/{_quoted(room_id)}/{_quoted(node_key)}/shell",
                {"command": command},
            )
            transport = "vm-shell-fallback"
        results[label] = {"transport": transport, "response": response}
    return {
        "room_id": room_id,
        "namespace": tunnel.get("namespace"),
        "skycraft": {
            "node_key": node_key,
            "display_name": node.get("displayName"),
            "phase": node.get("phase"),
        },
        "checks": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Điều khiển SafeLoop CarSky mock lab")
    parser.add_argument("--url", default=os.getenv("A8_URL", DEFAULT_URL))
    parser.add_argument("--room-id", default=DEFAULT_ROOM_ID)
    parser.add_argument("--timeout", type=float, default=30.0)
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("install", help="clone base, thêm mock node và validate")
    install.add_argument("--deploy", action="store_true")
    install.add_argument("--base-blueprint-id", default=DEFAULT_BASE_BLUEPRINT_ID)
    install.add_argument("--blueprint-name", default=DEFAULT_BLUEPRINT_NAME)
    install.add_argument("--deployment-name", default="SafeLoop-Mock-C1-C2-deploy")
    sub.add_parser("status", help="đọc trạng thái deployment của lab")
    verify = sub.add_parser("verify", help="sample live C1/C2 signal để xác nhận mock")
    verify.add_argument("--samples", type=int, default=3)
    verify.add_argument("--interval", type=float, default=0.6)
    sub.add_parser("android-probe", help="kiểm tra browser/tool có sẵn trong AAOS")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        api = Api(args.url, os.getenv("A8_API_KEY", ""), args.timeout)
        if args.command == "status":
            result = api.call(
                "GET", f"/api/v1/deployments/{_quoted(args.room_id)}/status"
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "verify":
            if args.samples < 2 or args.interval <= 0:
                raise ValueError("verify cần --samples >= 2 và --interval > 0")
            result = _verify_runtime(api, args.room_id, args.samples, args.interval)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["mock_live"] else 4
        if args.command == "android-probe":
            result = _android_probe(api, args.room_id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        existing = _find_blueprint(api, args.blueprint_name)
        if existing is None:
            api.call(
                "POST",
                f"/api/v1/blueprints/{_quoted(args.base_blueprint_id)}/clone",
                {"name": args.blueprint_name, "isSnapshot": False},
            )
            existing = _find_blueprint(api, args.blueprint_name)
            if existing is None:
                raise ApiError("Clone trả thành công nhưng không tìm thấy mock blueprint")
            install_mode = "cloned"
        else:
            install_mode = "existing"

        blueprint_id = _blueprint_id(existing)
        blueprint = _blueprint_body(
            api.call("GET", f"/api/v1/blueprints/{_quoted(blueprint_id)}")
        )
        connection_action, batch = _ensure_mock_connection(api, blueprint_id, blueprint)
        install_mode += "+" + connection_action
        blueprint = _blueprint_body(
            api.call("GET", f"/api/v1/blueprints/{_quoted(blueprint_id)}")
        )
        if not _connection_ready(blueprint):
            raise ApiError("Mock node chưa nối hoàn chỉnh tới Central Broker sau batch")

        validation = api.call(
            "POST", f"/api/v1/blueprints/{_quoted(blueprint_id)}/validate"
        )
        if validation.get("valid") is not True:
            raise ApiError(f"Blueprint validation fail: {validation}")
        result = {
            "install_mode": install_mode,
            "blueprint_id": blueprint_id,
            "blueprint_name": args.blueprint_name,
            "mock_node": MOCK_PUBLISHER_LABEL,
            "connection_ready": True,
            "batch": batch,
            "validation": validation,
        }

        if args.deploy:
            result["deployment"] = api.call(
                "POST",
                "/api/v1/deployments",
                {
                    "blueprintId": blueprint_id,
                    "roomId": args.room_id,
                    "name": args.deployment_name,
                },
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ApiError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
