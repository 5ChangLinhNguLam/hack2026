"""Offline checks for the CarSky mock blueprint mutation payload."""

from __future__ import annotations

import json
from pathlib import Path

from tools.carsky_mock_ctl import (
    MOCK_PUBLISHER_LABEL,
    _add_mock_node,
    _android_probe,
    _blueprint_body,
    _broker_input_pin_id,
    _connection_ready,
    _ensure_mock_connection,
    _verify_runtime,
)


def test_mock_batch_uses_per_operation_type_refs():
    root = Path(__file__).resolve().parents[1]
    exported = json.loads(
        (root / "carsky/blueprints/5ChangLinhNguLam-base-20260803.json").read_text()
    )
    blueprint = _blueprint_body(exported)
    calls = []

    class Client:
        def call(self, method, path, payload):
            calls.append((method, path, payload))
            return {"ok": True}

    _add_mock_node(Client(), "test-blueprint", blueprint)

    method, path, payload = calls[0]
    operations = payload["operations"]
    assert (method, path) == (
        "POST",
        "/api/v1/blueprints/test-blueprint/batch",
    )
    assert [operation["op"] for operation in operations] == [
        "addNode",
        "addPin",
        "addEdge",
    ]
    assert operations[0]["data"]["label"] == MOCK_PUBLISHER_LABEL
    assert operations[1]["nodeRef"] == 0
    assert operations[2]["data"]["sourcePinRef"] == 0
    assert operations[2]["data"]["targetPinId"] == _broker_input_pin_id(blueprint)


def test_partial_mock_node_with_pin_repairs_only_missing_edge():
    blueprint = {
        "nodes": [
            {
                "id": "broker",
                "nodeType": "kuksa-databroker",
                "pins": [
                    {
                        "id": "broker-input",
                        "name": "kuksa",
                        "pinType": "KUKSA",
                        "direction": "INPUT",
                    }
                ],
            },
            {
                "id": "mock",
                "label": MOCK_PUBLISHER_LABEL,
                "nodeType": "script-node",
                "pins": [
                    {
                        "id": "mock-output",
                        "name": "kuksa",
                        "pinType": "KUKSA",
                        "direction": "OUTPUT",
                    }
                ],
            },
        ],
        "edges": [],
    }
    calls = []

    class Client:
        def call(self, method, path, payload):
            calls.append((method, path, payload))
            return {"ok": True}

    action, _response = _ensure_mock_connection(Client(), "bp", blueprint)
    operation = calls[0][2]["operations"][0]

    assert action == "edge-repaired"
    assert operation == {
        "op": "addEdge",
        "data": {"sourcePinId": "mock-output", "targetPinId": "broker-input"},
    }
    assert _connection_ready(blueprint) is False

    blueprint["edges"].append(operation["data"])
    assert _connection_ready(blueprint) is True


def test_verify_runtime_requires_live_speed_and_ttc_changes():
    class Client:
        sample = 0

        def call(self, method, path, payload=None):
            if path.endswith("/status"):
                return {"status": "RUNNING"}
            if path.endswith("/signals/room"):
                return {"nodes": [{"key": "central-broker-vss", "kind": "kuksa"}]}
            self.sample += 1
            return {
                "values": [
                    {"path": signal_path, "value": self.sample if signal_path in {
                        "Vehicle.Speed",
                        "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap",
                    } else 0}
                    for signal_path in payload["paths"]
                ]
            }

    result = _verify_runtime(Client(), "room", samples=2, interval_s=0.0001)

    assert result["mock_live"] is True
    assert result["missing_paths"] == []
    assert result["changed_paths"] == [
        "Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap",
        "Vehicle.Speed",
    ]


def test_android_probe_selects_running_skycraft_and_uses_adb_exec():
    calls = []

    class Client:
        def call(self, method, path, payload=None):
            calls.append((method, path, payload))
            if path.endswith("/adb-tunnel"):
                return {
                    "namespace": "room-test",
                    "nodes": [
                        {
                            "nodeKey": "ivi-android",
                            "nodeType": "skycraft",
                            "displayName": "IVI - Android",
                            "phase": "Running",
                            "tunnelCmd": "hidden from result",
                        }
                    ],
                }
            return {"exitCode": 0, "stdout": "ok"}

    result = _android_probe(Client(), "device-room")

    assert result["skycraft"] == {
        "node_key": "ivi-android",
        "display_name": "IVI - Android",
        "phase": "Running",
    }
    assert len(result["checks"]) == 6
    assert all("/adb-exec/ivi-android" in call[1] for call in calls[1:])
    assert "tunnelCmd" not in str(result)


def test_android_probe_falls_back_to_vm_shell_when_conduit_is_unavailable():
    class Client:
        def call(self, method, path, payload=None):
            if path.endswith("/adb-tunnel"):
                return {
                    "namespace": "room-test",
                    "nodes": [{
                        "nodeKey": "android",
                        "nodeType": "skycraft",
                        "displayName": "IVI",
                        "phase": "Running",
                        "tunnelCmd": "unused",
                    }],
                }
            if "/adb-exec/" in path:
                from tools.carsky_mock_ctl import ApiError

                raise ApiError("Conduit service not configured", status_code=502)
            assert "/api/v1/vms/device-room/android/shell" == path
            return {"ok": True, "exitCode": 0, "stdout": "fallback-ok"}

    result = _android_probe(Client(), "device-room")

    assert all(
        check["transport"] == "vm-shell-fallback"
        and check["response"]["stdout"] == "fallback-ok"
        for check in result["checks"].values()
    )
