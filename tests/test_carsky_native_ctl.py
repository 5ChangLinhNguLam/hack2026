from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from urllib import error

import pytest

from tools import carsky_native_ctl as native


BASE_ID = native.DEFAULT_BASE_BLUEPRINT_ID
TARGET_ID = native.DEFAULT_TARGET_DEVICE_ID
CANDIDATE_ID = "candidate-clean-clone"
REFERENCE_ROOM_ID = "reference-room-audit"
REFERENCE_BLUEPRINT_ID = "kdXIOlc-rVzCpJwTZ7LT4"


def config_payload(candidate_id: str | None = CANDIDATE_ID) -> dict:
    return {
        "schema_version": native.SCHEMA_VERSION,
        "base_blueprint_id": BASE_ID,
        "candidate_blueprint_id": candidate_id,
        "target_device_id": TARGET_ID,
        "blueprint_name": "SafeLoop Native Candidate",
        "deployment_name": "SafeLoop-Native-T01",
        "gateway": {
            "node_label": "IVI Gateway",
            "addon_contract": native.ADDON_CONTRACT,
            "kuksa": {
                "contract": native.KUKSA_CONTRACT,
                "reference_room_id": REFERENCE_ROOM_ID,
                "reference_node_key": "central-broker-vss",
            },
        },
        "hmi": {
            "transport": native.HMI_TRANSPORT,
            "switch_label": "IVI Switch",
            "android_node_label": "IVI - Android",
            "target_host": "10.99.0.14",
            "port": 48100,
        },
    }


def write_addon(path: Path, *, source_mode: str = "MODEL_REPLAY") -> Path:
    path.write_text(
        "\n".join(
            [
                native.ADDON_MARKER,
                f"-- SAFELOOP_SOURCE_MODE: {source_mode}",
                "-- Read model/replay envelopes and publish standard VSS through KUKSA.",
                "local safeloop_kuksa = pins.kuksa",
                "-- Send the HMI envelope over UDP through the existing Ethernet pin.",
                'local safeloop_udp_host = "10.99.0.14"',
                "local safeloop_udp_port = 48100",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def blueprint(
    blueprint_id: str,
    *,
    parent_id: str | None = None,
    snapshot: bool = False,
    script: str = "-- stock IVI Gateway\nlocal vehicle_kuksa = pins.kuksa\n",
) -> dict:
    suffix = blueprint_id[-4:]
    gateway_id = f"gateway-{suffix}"
    gateway_kuksa = f"gateway-kuksa-{suffix}"
    gateway_eth = f"gateway-eth-{suffix}"
    broker_input = f"broker-in-{suffix}"
    switch_input = f"switch-in-{suffix}"
    android_eth = f"android-eth-{suffix}"
    return {
        "id": blueprint_id,
        "name": f"Blueprint {blueprint_id}",
        "parentBlueprintId": parent_id,
        "isSnapshot": snapshot,
        "locked": False,
        "nodes": [
            {
                "id": f"broker-{suffix}",
                "label": "Central Broker",
                "nodeType": "kuksa-databroker",
                "config": {
                    "kuksa": {
                        "vss": {
                            "artifactId": "zsJexrIgGwIeyk3ODyYoh",
                            "versionId": "Cpi8WkMz_bH35uIQFviim",
                        }
                    }
                },
                "pins": [
                    {
                        "id": broker_input,
                        "name": "kuksa",
                        "pinType": "KUKSA",
                        "direction": "INPUT",
                    }
                ],
            },
            {
                "id": gateway_id,
                "label": "IVI Gateway",
                "nodeType": "script-node",
                "config": {"script": "inline", "scriptContent": script, "periodMs": 50},
                "pins": [
                    {
                        "id": gateway_kuksa,
                        "name": "kuksa",
                        "pinType": "KUKSA",
                        "direction": "OUTPUT",
                    },
                    {
                        "id": gateway_eth,
                        "name": "eth",
                        "pinType": "ETHERNET",
                        "direction": "OUTPUT",
                    },
                ],
            },
            {
                "id": f"switch-{suffix}",
                "label": "IVI Switch",
                "nodeType": "eth-bridge",
                "pins": [
                    {
                        "id": switch_input,
                        "name": "eth",
                        "pinType": "ETHERNET",
                        "direction": "INPUT",
                    }
                ],
            },
            {
                "id": f"android-{suffix}",
                "label": "IVI - Android",
                "nodeType": "skycraft",
                "pins": [
                    {
                        "id": android_eth,
                        "name": "eth",
                        "pinType": "ETHERNET",
                        "direction": "OUTPUT",
                        "properties": {"address": "10.99.0.14"},
                    }
                ],
            },
        ],
        "edges": [
            {"sourcePinId": gateway_kuksa, "targetPinId": broker_input},
            {"sourcePinId": gateway_eth, "targetPinId": switch_input},
            {"sourcePinId": android_eth, "targetPinId": switch_input},
        ],
    }


class FakeApi:
    def __init__(
        self,
        *,
        candidate_id: str | None = CANDIDATE_ID,
        candidate_script: str | None = None,
        patch_status: int | None = None,
        candidate_is_active: bool = False,
        base_is_active: bool = False,
        target_busy: bool = False,
        target_busy_after_first_find: bool = False,
        duplicate_blueprint_name: bool = False,
        duplicate_deployment_name: bool = False,
        blueprint_limit: int = 20,
        deployment_limit: int = 2,
        clone_parent_id: str = BASE_ID,
        clone_snapshot: bool = False,
        clone_topology_mismatch: bool = False,
        clone_pin_name_mismatch: bool = False,
        clone_response_has_id: bool = True,
        status_sequence: list[str] | None = None,
    ) -> None:
        base = blueprint(BASE_ID)
        base["name"] = "5ChangLinhNguLam"
        self.base_script = base["nodes"][1]["config"]["scriptContent"]
        self.candidate = (
            blueprint(
                candidate_id,
                parent_id=BASE_ID,
                script=candidate_script if candidate_script is not None else self.base_script,
            )
            if candidate_id is not None
            else None
        )
        if self.candidate is not None:
            self.candidate["name"] = "SafeLoop Native Candidate"
        reference = blueprint(REFERENCE_BLUEPRINT_ID, snapshot=True)
        reference["name"] = "Active reference snapshot"
        active_blueprint_id = (
            candidate_id
            if candidate_is_active
            else BASE_ID
            if base_is_active
            else REFERENCE_BLUEPRINT_ID
        )
        self.deployments = [
            {
                "id": "deployment-reference",
                "name": (
                    "SafeLoop-Native-T01"
                    if duplicate_deployment_name
                    else "Reference deployment"
                ),
                "blueprintId": active_blueprint_id,
                "roomId": REFERENCE_ROOM_ID,
                "status": "RUNNING",
            }
        ]
        self.target_deployments: list[dict] = []
        self.target_busy_after_first_find = target_busy_after_first_find
        self.target_find_count = 0
        if target_busy:
            self.target_deployments.append(
                {
                    "id": "deployment-target-busy",
                    "name": "Existing target deployment",
                    "blueprintId": REFERENCE_BLUEPRINT_ID,
                    "roomId": TARGET_ID,
                    "status": "RUNNING",
                }
            )
        self.patch_status = patch_status
        self.clone_parent_id = clone_parent_id
        self.clone_snapshot = clone_snapshot
        self.clone_topology_mismatch = clone_topology_mismatch
        self.clone_pin_name_mismatch = clone_pin_name_mismatch
        self.clone_response_has_id = clone_response_has_id
        self.status_sequence = list(status_sequence or ["RUNNING"])
        self.status_index = 0
        self.calls: list[tuple[str, str, object | None]] = []
        blueprint_entries = [
            {"id": BASE_ID, "name": base["name"]},
            {"id": REFERENCE_BLUEPRINT_ID, "name": reference["name"]},
        ] + (
            [{"id": candidate_id, "name": self.candidate["name"]}]
            if candidate_id and self.candidate is not None
            else []
        )
        if duplicate_blueprint_name:
            blueprint_entries.append(
                {"id": "duplicate-name", "name": "SafeLoop Native Candidate"}
            )
        self.mapping: dict[str, object] = {
            "/api/v1/healthz": {"status": "ok"},
            "/api/v1/config/limits": {
                "MAX_BLUEPRINTS_PER_ACCOUNT": blueprint_limit,
                "MAX_CONCURRENT_DEPLOYMENTS_PER_ACCOUNT": deployment_limit,
            },
            "/api/v1/config/cpu-status": {"over": False},
            "/api/v1/blueprints?limit=100": blueprint_entries,
            "/api/v1/devices?limit=100": [
                {"id": TARGET_ID},
                {"id": REFERENCE_ROOM_ID},
            ],
            f"/api/v1/blueprints/{BASE_ID}": base,
            f"/api/v1/blueprints/{REFERENCE_BLUEPRINT_ID}": reference,
            f"/api/v1/devices/{TARGET_ID}": {
                "id": TARGET_ID,
                "name": "SafeLoop target",
                "status": "PUBLISHED",
                "locked": False,
            },
            f"/api/v1/deployments/find?device={TARGET_ID}": self.target_deployments,
            f"/api/v1/deployments/find?device={REFERENCE_ROOM_ID}": self.deployments,
            f"/api/v1/signals/{REFERENCE_ROOM_ID}": {
                "nodes": [{"key": "central-broker-vss", "kind": "kuksa"}]
            },
            f"/api/v1/signals/{REFERENCE_ROOM_ID}/central-broker-vss": {
                "signals": [
                    {"path": path, "dataType": data_type}
                    for path, data_type in native.STANDARD_SIGNAL_TYPES.items()
                ]
            },
        }
        if candidate_id and self.candidate is not None:
            self.mapping[f"/api/v1/blueprints/{candidate_id}"] = self.candidate

    def get(self, path: str):
        self.calls.append(("GET", path, None))
        if path == f"/api/v1/deployments/find?device={TARGET_ID}":
            self.target_find_count += 1
            if self.target_busy_after_first_find and self.target_find_count >= 2:
                return [
                    {
                        "id": "deployment-raced",
                        "name": "Raced deployment",
                        "blueprintId": REFERENCE_BLUEPRINT_ID,
                        "roomId": TARGET_ID,
                        "status": "RUNNING",
                    }
                ]
        if path == f"/api/v1/deployments/{TARGET_ID}/status":
            index = min(self.status_index, len(self.status_sequence) - 1)
            self.status_index += 1
            return {"status": self.status_sequence[index], "namespace": "room-safe"}
        return copy.deepcopy(self.mapping[path])

    def clone_blueprint(self, base_blueprint_id: str, name: str):
        self.calls.append(
            (
                "POST_CLONE",
                base_blueprint_id,
                {"name": name, "isSnapshot": False},
            )
        )
        assert self.candidate is None
        self.candidate = blueprint(
            CANDIDATE_ID,
            parent_id=self.clone_parent_id,
            snapshot=self.clone_snapshot,
            script=self.base_script,
        )
        self.candidate["name"] = name
        if self.clone_topology_mismatch:
            self.candidate["edges"] = self.candidate["edges"][:-1]
        if self.clone_pin_name_mismatch:
            gateway = next(
                node for node in self.candidate["nodes"] if node["label"] == "IVI Gateway"
            )
            next(pin for pin in gateway["pins"] if pin["name"] == "kuksa")[
                "name"
            ] = "signals"
        self.mapping[f"/api/v1/blueprints/{CANDIDATE_ID}"] = self.candidate
        self.mapping["/api/v1/blueprints?limit=100"].append(
            {"id": CANDIDATE_ID, "name": name}
        )
        return {"id": CANDIDATE_ID} if self.clone_response_has_id else {"ok": True}

    def patch_gateway_config(self, node_id: str, config: dict):
        self.calls.append(("PATCH", node_id, copy.deepcopy(config)))
        if self.patch_status is not None:
            raise native.CarSkyApiError(
                f"fake HTTP {self.patch_status}", status_code=self.patch_status
            )
        assert self.candidate is not None
        gateway = next(node for node in self.candidate["nodes"] if node["id"] == node_id)
        gateway["config"] = copy.deepcopy(config)
        return gateway

    def validate_blueprint(self, blueprint_id: str):
        self.calls.append(("POST_VALIDATE", blueprint_id, None))
        return {"valid": True, "errors": []}

    def create_deployment(
        self, *, candidate_blueprint_id: str, target_device_id: str, name: str
    ):
        payload = {
            "id": "deployment-native",
            "blueprintId": candidate_blueprint_id,
            "roomId": target_device_id,
            "name": name,
            "status": "PENDING",
        }
        self.calls.append(("POST_DEPLOY", "/api/v1/deployments", copy.deepcopy(payload)))
        self.target_deployments.append(payload)
        return copy.deepcopy(payload)


def parsed_and_addon(tmp_path: Path, candidate_id: str | None = CANDIDATE_ID):
    config = native.parse_config(config_payload(candidate_id))
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)
    return config, addon


def test_config_and_addon_are_fail_closed(tmp_path: Path) -> None:
    payload = config_payload()
    payload["api_key"] = "must-not-be-here"
    with pytest.raises(native.NativeConfigError, match="keys sai|secret"):
        native.parse_config(payload)

    payload = config_payload()
    payload["hmi"]["transport"] = "screen-widget"
    with pytest.raises(native.NativeConfigError, match="udp-ethernet"):
        native.parse_config(payload)

    config = native.parse_config(config_payload())
    bad = tmp_path / "bad.lua"
    bad.write_text(
        f"{native.ADDON_MARKER}\n-- SAFELOOP_SOURCE_MODE: MODEL_REPLAY\n"
        "local kuksa = true\nlocal udp = '10.99.0.14:48100'\n"
        "local ground_truth = true\n",
        encoding="utf-8",
    )
    with pytest.raises(native.NativeConfigError, match="field/mode cấm"):
        native.load_gateway_addon(bad, config)

    generated = tmp_path / "generated.lua"
    generated.write_text(
        f"{native.GENERATED_REPLAY_BEGIN}\n"
        "local kuksa = pins.kuksa\n"
        "local udp = '10.99.0.14:48100'\n"
        f"{native.GENERATED_REPLAY_END}\n",
        encoding="utf-8",
    )
    assert native.load_gateway_addon(generated, config).source_mode == "MODEL_REPLAY"


def test_preflight_is_get_only_and_exposes_ethernet_api_blocker(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    api = FakeApi()

    result = native.preflight(config, addon, api)

    assert result["patch_ready"] is True
    assert result["deploy_ready"] is False
    assert result["deploy_blockers"] == ["GATEWAY_ADDON_NOT_APPLIED"]
    assert result["kuksa"]["schema_ready"] is True
    assert result["hmi"]["ethernet_pin_api_supported"] is False
    assert result["hmi"]["api_blocker_avoided_by_gateway_reuse"] is True
    assert {method for method, _, _ in api.calls} == {"GET"}

    plan = native.build_plan(config, addon, result)
    assert plan["phase_1_architecture"]["registry_required"] is False
    assert plan["phase_1_architecture"]["new_container"] is False
    assert plan["phase_1_architecture"]["container_image"] is None
    assert "immutable image digest" in plan["phase_1_architecture"]["container_image_policy"]
    assert "--gateway-addon" in plan["future_reviewed_steps"][2]["guarded_command"]


def bootstrap_config_and_addon(tmp_path: Path):
    payload = config_payload()
    payload["gateway"]["kuksa"][
        "reference_room_id"
    ] = native.PINNED_KUKSA_REFERENCE_ROOM_ID
    payload["gateway"]["kuksa"][
        "reference_node_key"
    ] = native.PINNED_KUKSA_REFERENCE_NODE_KEY
    config = native.parse_config(payload)
    addon = native.load_gateway_addon(write_addon(tmp_path / "bootstrap.lua"), config)
    return config, addon


def test_no_live_reference_uses_only_exact_pinned_kuksa_attestation(
    tmp_path: Path,
) -> None:
    config, addon = bootstrap_config_and_addon(tmp_path)
    api = FakeApi()

    result = native.preflight(config, addon, api)

    assert result["patch_ready"] is True
    assert "KUKSA_REFERENCE_DEPLOYMENT_MISSING" not in result["patch_blockers"]
    assert result["kuksa"]["schema_source"] == "pinned_local_attestation"
    assert result["kuksa"]["schema_ready"] is True
    attestation = result["kuksa"]["bootstrap_attestation"]
    assert attestation["sha256"] == native.PINNED_KUKSA_ATTESTATION_SHA256
    assert attestation["artifact_id"] == native.PINNED_KUKSA_ARTIFACT_ID
    assert attestation["version_id"] == native.PINNED_KUKSA_VERSION_ID
    assert attestation["required_signal_types"] == native.STANDARD_SIGNAL_TYPES
    assert not any("/api/v1/signals/" in path for _, path, _ in api.calls)


def test_bootstrap_attestation_tamper_or_artifact_mismatch_blocks_deploy(
    tmp_path: Path,
) -> None:
    config, addon = bootstrap_config_and_addon(tmp_path)
    tampered = tmp_path / "tampered-broker-metadata.json"
    tampered.write_bytes(native.DEFAULT_KUKSA_SCHEMA_ATTESTATION.read_bytes() + b"\n")
    api = FakeApi()

    result = native.preflight(
        config, addon, api, kuksa_attestation_path=tampered
    )

    assert result["kuksa"]["schema_ready"] is False
    assert result["kuksa"]["bootstrap_attestation"] is None
    assert "KUKSA_REFERENCE_DEPLOYMENT_MISSING" in result["deploy_blockers"]
    assert "KUKSA_BOOTSTRAP_ATTESTATION_INVALID" in result["deploy_blockers"]
    assert "KUKSA_SIGNAL_CONTRACT_NOT_READY" in result["deploy_blockers"]
    assert any("SHA-256" in warning for warning in result["warnings"])
    assert not any("/api/v1/signals/" in path for _, path, _ in api.calls)

    api = FakeApi()
    base = api.mapping[f"/api/v1/blueprints/{BASE_ID}"]
    base["nodes"][0]["config"]["kuksa"]["vss"]["versionId"] = "wrong-version"
    artifact_result = native.preflight(config, addon, api)
    assert "KUKSA_BOOTSTRAP_ATTESTATION_INVALID" in artifact_result["deploy_blockers"]
    assert any("artifact/version" in warning for warning in artifact_result["warnings"])


def test_bootstrap_attestation_hash_is_not_enough_without_strict_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, addon = bootstrap_config_and_addon(tmp_path)
    malformed = json.loads(
        native.DEFAULT_KUKSA_SCHEMA_ATTESTATION.read_text(encoding="utf-8")
    )
    malformed["status"] = 201
    path = tmp_path / "wrong-envelope.json"
    path.write_text(json.dumps(malformed), encoding="utf-8")
    monkeypatch.setattr(
        native,
        "PINNED_KUKSA_ATTESTATION_SHA256",
        native.hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    result = native.preflight(
        config, addon, FakeApi(), kuksa_attestation_path=path
    )

    assert result["kuksa"]["schema_ready"] is False
    assert "KUKSA_BOOTSTRAP_ATTESTATION_INVALID" in result["deploy_blockers"]
    assert any("response envelope" in warning for warning in result["warnings"])


def test_exact_addon_suffix_is_idempotent_and_deploy_ready(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    base_script = FakeApi().base_script
    combined = native.combined_gateway_script(base_script, addon)
    api = FakeApi(candidate_script=combined)

    result = native.preflight(config, addon, api)

    assert result["patch_ready"] is True
    assert result["deploy_ready"] is True
    assert result["candidate"]["addon_applied"] is True

    patched = native.patch_gateway(
        config,
        addon,
        api,
        confirm_candidate_id=CANDIDATE_ID,
        backup_dir=tmp_path / "backup",
    )
    assert patched["already_applied"] is True
    assert patched["mutation_calls"] == 0
    assert all(method == "GET" for method, _, _ in api.calls)


def test_existing_marker_or_non_base_clone_is_refused(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    conflict = FakeApi().base_script + "\n" + native.ADDON_MARKER + "\n"
    conflict_result = native.preflight(
        config, addon, FakeApi(candidate_script=conflict)
    )
    assert "CANDIDATE_GATEWAY_ADDON_MARKER_CONFLICT" in conflict_result["patch_blockers"]

    changed = FakeApi().base_script + "-- local unauthorized edit\n"
    changed_result = native.preflight(config, addon, FakeApi(candidate_script=changed))
    assert "CANDIDATE_GATEWAY_BASE_HASH_MISMATCH" in changed_result["patch_blockers"]


def test_missing_kuksa_or_hmi_edge_blocks_patch_and_deploy(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    api = FakeApi()
    assert api.candidate is not None
    api.candidate["edges"] = api.candidate["edges"][1:]
    result = native.preflight(config, addon, api)
    assert "CANDIDATE_GATEWAY_KUKSA_EDGE_MISSING" in result["patch_blockers"]
    assert result["patch_ready"] is False
    assert result["deploy_ready"] is False

    api = FakeApi()
    signal_path = f"/api/v1/signals/{REFERENCE_ROOM_ID}/central-broker-vss"
    api.mapping[signal_path]["signals"] = api.mapping[signal_path]["signals"][:-1]
    result = native.preflight(config, addon, api)
    assert "KUKSA_SIGNAL_CONTRACT_NOT_READY" in result["patch_blockers"]
    assert result["deploy_ready"] is False


def test_candidate_broker_must_keep_exact_base_artifact_version(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    api = FakeApi()
    assert api.candidate is not None
    api.candidate["nodes"][0]["config"]["kuksa"]["vss"][
        "versionId"
    ] = "tampered-version"

    result = native.preflight(config, addon, api)

    assert "CANDIDATE_KUKSA_ARTIFACT_MISMATCH" in result["patch_blockers"]
    assert result["patch_ready"] is False
    assert result["deploy_ready"] is False
    assert result["candidate"]["broker_artifact_id"] == native.PINNED_KUKSA_ARTIFACT_ID
    assert result["candidate"]["broker_version_id"] == "tampered-version"


def test_guarded_patch_backs_up_exact_config_and_only_patches_candidate_config(
    tmp_path: Path,
) -> None:
    config, addon = parsed_and_addon(tmp_path)
    api = FakeApi()
    before = copy.deepcopy(api.candidate["nodes"][1]["config"])

    result = native.patch_gateway(
        config,
        addon,
        api,
        confirm_candidate_id=CANDIDATE_ID,
        backup_dir=tmp_path / "backup",
    )

    assert result["patched"] is True
    assert result["mutation_calls"] == 2
    backup = json.loads(Path(result["backup"]).read_text(encoding="utf-8"))
    assert backup["candidate_blueprint_id"] == CANDIDATE_ID
    assert backup["current_config"] == before
    patch_calls = [call for call in api.calls if call[0] == "PATCH"]
    assert len(patch_calls) == 1
    _, node_id, patched_config = patch_calls[0]
    assert node_id.startswith("gateway-")
    assert set(patched_config) == set(before)
    assert patched_config["periodMs"] == 50
    assert patched_config["script"] == before["script"]
    assert patched_config["scriptContent"].startswith(before["scriptContent"])
    assert patched_config["scriptContent"].count(native.BOUNDARY_START) == 1
    assert any(call[0] == "POST_VALIDATE" for call in api.calls)


def test_patch_requires_exact_confirmation_and_inactive_candidate(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    api = FakeApi()
    with pytest.raises(native.PatchRefused, match="không khớp"):
        native.patch_gateway(
            config,
            addon,
            api,
            confirm_candidate_id="wrong-id",
            backup_dir=tmp_path / "backup",
        )
    assert api.calls == []

    active_api = FakeApi(candidate_is_active=True)
    with pytest.raises(native.PatchRefused, match="CANDIDATE_IS_LIVE"):
        native.patch_gateway(
            config,
            addon,
            active_api,
            confirm_candidate_id=CANDIDATE_ID,
            backup_dir=tmp_path / "backup-active",
        )
    assert not any(call[0] == "PATCH" for call in active_api.calls)


def test_patch_404_stops_with_manual_ui_instruction_after_backup(tmp_path: Path) -> None:
    config, addon = parsed_and_addon(tmp_path)
    api = FakeApi(patch_status=404)
    backup_dir = tmp_path / "backup"
    expected_script = native.combined_gateway_script(api.base_script, addon)
    expected_sha = native.hashlib.sha256(expected_script.encode("utf-8")).hexdigest()

    with pytest.raises(native.PatchRefused, match="CarSky UI") as caught:
        native.patch_gateway(
            config,
            addon,
            api,
            confirm_candidate_id=CANDIDATE_ID,
            backup_dir=backup_dir,
        )

    backups = list(backup_dir.glob("*.json"))
    manual_scripts = list(backup_dir.glob("*.manual.lua"))
    assert len(backups) == 1
    assert len(manual_scripts) == 1
    backup = json.loads(backups[0].read_text(encoding="utf-8"))
    manual_script = manual_scripts[0]
    assert manual_script.stem.removesuffix(".manual") == backups[0].stem
    assert manual_script.read_text(encoding="utf-8") == expected_script
    assert native.hashlib.sha256(manual_script.read_bytes()).hexdigest() == expected_sha
    assert manual_script.stat().st_mode & 0o077 == 0
    assert backup["expected_script_sha256"] == expected_sha
    message = str(caught.value)
    assert f"manual_script={manual_script.resolve()}" in message
    assert f"candidate_blueprint_id={CANDIDATE_ID}" in message
    assert "gateway_node_id=gateway-lone" in message
    assert f"expected_script_sha256={expected_sha}" in message
    assert "thay toàn bộ Script Content" in message
    assert "chỉ mở candidate blueprint" in message
    assert not list(backup_dir.glob("*.tmp"))
    assert not any(call[0] == "POST_VALIDATE" for call in api.calls)


def test_manual_gateway_script_never_overwrites_existing_artifact(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "attempt.json"
    backup.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "attempt.manual.lua"
    destination.write_text("keep this final artifact\n", encoding="utf-8")
    before = destination.read_bytes()
    script = "-- complete candidate script\n"
    expected_sha = native.hashlib.sha256(script.encode("utf-8")).hexdigest()

    with pytest.raises(native.PatchRefused, match="Từ chối ghi đè"):
        native._write_manual_gateway_script(
            backup,
            expected_script=script,
            expected_sha256=expected_sha,
        )

    assert destination.read_bytes() == before


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


def test_http_client_has_get_allowlist_and_sanitized_http_errors() -> None:
    requests = []

    def fake_urlopen(req, *, timeout):
        requests.append((req, timeout))
        return FakeResponse({"status": "ok"})

    api = native.CarSkyApi(
        "https://carsky.invalid", "test-key-redacted", urlopen=fake_urlopen
    )
    assert api.get("/api/v1/healthz") == {"status": "ok"}
    api.clone_blueprint(BASE_ID, "SafeLoop Native Candidate")
    api.create_deployment(
        candidate_blueprint_id=CANDIDATE_ID,
        target_device_id=TARGET_ID,
        name="SafeLoop-Native-T01",
    )
    assert api.get(f"/api/v1/deployments/{TARGET_ID}/status") == {"status": "ok"}
    assert requests[0][0].get_method() == "GET"
    assert requests[0][0].get_header("X-api-key") == "test-key-redacted"
    assert requests[1][0].get_method() == "POST"
    assert json.loads(requests[1][0].data) == {
        "name": "SafeLoop Native Candidate",
        "isSnapshot": False,
    }
    assert requests[2][0].get_method() == "POST"
    assert json.loads(requests[2][0].data) == {
        "blueprintId": CANDIDATE_ID,
        "roomId": TARGET_ID,
        "name": "SafeLoop-Native-T01",
    }
    assert requests[3][0].get_method() == "GET"
    with pytest.raises(native.CarSkyApiError, match="allow-list"):
        api.get("/api/v1/account-limits/save")

    def http_401(_req, *, timeout):
        raise error.HTTPError(
            "https://carsky.invalid",
            401,
            "secret body",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "error": "UNAUTHORIZED",
                        "message": "bad credential test-key-redacted a8k_secret_value",
                        "details": {
                            "fieldErrors": {
                                "apiKey": ["bad a8k_secret_value"]
                            },
                            "must_not_be_exposed": "private diagnostic",
                        },
                    }
                ).encode("utf-8")
            ),
        )

    failing = native.CarSkyApi(
        "https://carsky.invalid", "test-key-redacted", urlopen=http_401
    )
    with pytest.raises(native.CarSkyApiError) as caught:
        failing.get("/api/v1/healthz")
    assert "test-key-redacted" not in str(caught.value)
    assert "a8k_secret_value" not in str(caught.value)
    assert "UNAUTHORIZED" in str(caught.value)
    assert "bad credential <redacted> <redacted>" in str(caught.value)
    assert '"fieldErrors":{"apiKey":["bad <redacted>"]}' in str(caught.value)
    assert "private diagnostic" not in str(caught.value)


def working_config(
    tmp_path: Path, *, candidate_id: str | None = None
) -> tuple[Path, Path, native.NativeConfig]:
    root = tmp_path / ".carsky-build"
    root.mkdir(parents=True)
    path = root / "native-deployment.json"
    path.write_text(json.dumps(config_payload(candidate_id)), encoding="utf-8")
    return root, path, native.load_config(path)


def test_clone_candidate_verifies_topology_and_atomically_records_id(
    tmp_path: Path,
) -> None:
    root, config_path, config = working_config(tmp_path)
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)
    api = FakeApi(candidate_id=None)

    result = native.clone_candidate(
        config_path,
        config,
        addon,
        api,
        confirm_base_id=BASE_ID,
        working_root=root,
    )

    assert result["cloned"] is True
    assert result["direct_clone_verified"] is True
    assert result["base_topology_sha256"] == result["topology_sha256"]
    assert native.load_config(config_path).candidate_blueprint_id == CANDIDATE_ID
    clone_calls = [call for call in api.calls if call[0] == "POST_CLONE"]
    assert clone_calls == [
        (
            "POST_CLONE",
            BASE_ID,
            {"name": "SafeLoop Native Candidate", "isSnapshot": False},
        )
    ]
    assert not any(call[0] == "POST_DEPLOY" for call in api.calls)


def test_clone_can_recover_unique_id_when_response_omits_it(tmp_path: Path) -> None:
    root, config_path, config = working_config(tmp_path)
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)
    api = FakeApi(candidate_id=None, clone_response_has_id=False)

    result = native.clone_candidate(
        config_path,
        config,
        addon,
        api,
        confirm_base_id=BASE_ID,
        working_root=root,
    )

    assert result["candidate_blueprint_id"] == CANDIDATE_ID
    assert native.load_config(config_path).candidate_blueprint_id == CANDIDATE_ID
    assert len([call for call in api.calls if call[0] == "POST_CLONE"]) == 1


@pytest.mark.parametrize(
    ("api_kwargs", "blocker"),
    [
        ({"base_is_active": True}, "BASE_IS_LIVE_DEPLOYMENT_BLUEPRINT"),
        ({"target_busy": True}, "TARGET_DEVICE_ALREADY_HAS_ACTIVE_DEPLOYMENT"),
        ({"duplicate_blueprint_name": True}, "BLUEPRINT_NAME_ALREADY_EXISTS"),
        ({"blueprint_limit": 2}, "BLUEPRINT_LIMIT_EXHAUSTED"),
        ({"deployment_limit": 1}, "DEPLOYMENT_LIMIT_EXHAUSTED"),
    ],
)
def test_clone_refuses_live_name_quota_and_busy_gates_before_post(
    tmp_path: Path, api_kwargs: dict, blocker: str
) -> None:
    root, config_path, config = working_config(tmp_path)
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)
    api = FakeApi(candidate_id=None, **api_kwargs)

    with pytest.raises(native.CloneRefused, match=blocker):
        native.clone_candidate(
            config_path,
            config,
            addon,
            api,
            confirm_base_id=BASE_ID,
            working_root=root,
        )

    assert not any(call[0] == "POST_CLONE" for call in api.calls)
    assert native.load_config(config_path).candidate_blueprint_id is None


def test_clone_confirmation_existing_candidate_and_path_gates_are_fail_closed(
    tmp_path: Path,
) -> None:
    root, config_path, config = working_config(tmp_path)
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)

    wrong_api = FakeApi(candidate_id=None)
    with pytest.raises(native.CloneRefused, match="confirm-base-id"):
        native.clone_candidate(
            config_path,
            config,
            addon,
            wrong_api,
            confirm_base_id="wrong",
            working_root=root,
        )
    assert wrong_api.calls == []

    _, configured_path, configured = working_config(
        tmp_path / "configured", candidate_id=CANDIDATE_ID
    )
    configured_addon = native.load_gateway_addon(
        write_addon(tmp_path / "configured-addon.lua"), configured
    )
    configured_api = FakeApi()
    with pytest.raises(native.CloneRefused, match="đã có candidate"):
        native.clone_candidate(
            configured_path,
            configured,
            configured_addon,
            configured_api,
            confirm_base_id=BASE_ID,
            working_root=configured_path.parent,
        )
    assert configured_api.calls == []

    example = Path("carsky/native-deployment.example.json")
    before = example.read_bytes()
    outside_api = FakeApi(candidate_id=None)
    with pytest.raises(native.CloneRefused, match=r"\.carsky-build"):
        native.clone_candidate(
            example,
            config,
            addon,
            outside_api,
            confirm_base_id=BASE_ID,
            working_root=root,
        )
    assert outside_api.calls == []
    assert example.read_bytes() == before


@pytest.mark.parametrize(
    ("api_kwargs", "message"),
    [
        ({"clone_parent_id": "wrong-parent"}, "CLONE_NOT_DIRECT_CHILD_OF_BASE"),
        ({"clone_snapshot": True}, "CLONE_NOT_EDITABLE"),
        ({"clone_topology_mismatch": True}, "CLONE_TOPOLOGY_HASH_MISMATCH"),
        ({"clone_pin_name_mismatch": True}, "CLONE_TOPOLOGY_HASH_MISMATCH"),
    ],
)
def test_clone_does_not_record_unverified_returned_candidate(
    tmp_path: Path, api_kwargs: dict, message: str
) -> None:
    root, config_path, config = working_config(tmp_path)
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)
    api = FakeApi(candidate_id=None, **api_kwargs)

    with pytest.raises(native.CloneRefused, match=message):
        native.clone_candidate(
            config_path,
            config,
            addon,
            api,
            confirm_base_id=BASE_ID,
            working_root=root,
        )

    assert len([call for call in api.calls if call[0] == "POST_CLONE"]) == 1
    assert native.load_config(config_path).candidate_blueprint_id is None


def deployable_api_and_addon(tmp_path: Path, **api_kwargs):
    config = native.parse_config(config_payload())
    addon = native.load_gateway_addon(write_addon(tmp_path / "addon.lua"), config)
    base_script = FakeApi().base_script
    api = FakeApi(
        candidate_script=native.combined_gateway_script(base_script, addon),
        **api_kwargs,
    )
    return config, addon, api


def test_deploy_uses_fresh_gate_exact_confirmations_and_bounded_poll(
    tmp_path: Path,
) -> None:
    config, addon, api = deployable_api_and_addon(
        tmp_path, status_sequence=["PENDING", "DEPLOYING", "RUNNING"]
    )

    result = native.deploy_candidate(
        config,
        addon,
        api,
        confirm_candidate_id=CANDIDATE_ID,
        confirm_device_id=TARGET_ID,
        poll_attempts=5,
        poll_interval_s=0,
    )

    assert result["deployment_id"] == "deployment-native"
    assert result["candidate_blueprint_id"] == CANDIDATE_ID
    assert result["target_device_id"] == TARGET_ID
    assert result["status"] == {"status": "RUNNING", "namespace": "room-safe"}
    assert result["polls"] == 3
    assert result["poll_timed_out"] is False
    assert result["active_deployment_untouched"] is True
    deploy_calls = [call for call in api.calls if call[0] == "POST_DEPLOY"]
    assert len(deploy_calls) == 1
    assert deploy_calls[0][2]["blueprintId"] == CANDIDATE_ID
    assert deploy_calls[0][2]["roomId"] == TARGET_ID
    assert deploy_calls[0][2]["name"] == "SafeLoop-Native-T01"


def test_deploy_retry_is_idempotently_blocked_after_first_create(tmp_path: Path) -> None:
    config, addon, api = deployable_api_and_addon(tmp_path)
    native.deploy_candidate(
        config,
        addon,
        api,
        confirm_candidate_id=CANDIDATE_ID,
        confirm_device_id=TARGET_ID,
        poll_attempts=1,
        poll_interval_s=0,
    )

    with pytest.raises(native.DeployRefused, match="TARGET_DEVICE_ALREADY|DEPLOYMENT_NAME"):
        native.deploy_candidate(
            config,
            addon,
            api,
            confirm_candidate_id=CANDIDATE_ID,
            confirm_device_id=TARGET_ID,
            poll_attempts=1,
            poll_interval_s=0,
        )
    assert len([call for call in api.calls if call[0] == "POST_DEPLOY"]) == 1


@pytest.mark.parametrize(
    ("api_kwargs", "blocker"),
    [
        ({"target_busy": True}, "TARGET_DEVICE_ALREADY_HAS_ACTIVE_DEPLOYMENT"),
        ({"candidate_is_active": True}, "CANDIDATE_IS_LIVE_DEPLOYMENT_BLUEPRINT"),
        ({"duplicate_deployment_name": True}, "DEPLOYMENT_NAME_ALREADY_ACTIVE"),
        ({"deployment_limit": 1}, "DEPLOYMENT_LIMIT_EXHAUSTED"),
    ],
)
def test_deploy_refuses_busy_live_duplicate_and_quota_before_create(
    tmp_path: Path, api_kwargs: dict, blocker: str
) -> None:
    config, addon, api = deployable_api_and_addon(tmp_path, **api_kwargs)

    with pytest.raises(native.DeployRefused, match=blocker):
        native.deploy_candidate(
            config,
            addon,
            api,
            confirm_candidate_id=CANDIDATE_ID,
            confirm_device_id=TARGET_ID,
            poll_attempts=1,
            poll_interval_s=0,
        )

    assert not any(call[0] == "POST_DEPLOY" for call in api.calls)


def test_deploy_wrong_confirmation_has_zero_api_calls(tmp_path: Path) -> None:
    config, addon, api = deployable_api_and_addon(tmp_path)
    with pytest.raises(native.DeployRefused, match="confirm-candidate-id"):
        native.deploy_candidate(
            config,
            addon,
            api,
            confirm_candidate_id="wrong",
            confirm_device_id=TARGET_ID,
        )
    assert api.calls == []


def test_deploy_rechecks_target_and_closes_preflight_race(tmp_path: Path) -> None:
    config, addon, api = deployable_api_and_addon(
        tmp_path, target_busy_after_first_find=True
    )

    with pytest.raises(native.DeployRefused, match="became busy"):
        native.deploy_candidate(
            config,
            addon,
            api,
            confirm_candidate_id=CANDIDATE_ID,
            confirm_device_id=TARGET_ID,
            poll_attempts=1,
            poll_interval_s=0,
        )

    assert api.target_find_count == 2
    assert not any(call[0] == "POST_VALIDATE" for call in api.calls)
    assert not any(call[0] == "POST_DEPLOY" for call in api.calls)


def test_deploy_poll_timeout_is_bounded_and_never_reposts(tmp_path: Path) -> None:
    config, addon, api = deployable_api_and_addon(
        tmp_path, status_sequence=["PENDING"]
    )
    result = native.deploy_candidate(
        config,
        addon,
        api,
        confirm_candidate_id=CANDIDATE_ID,
        confirm_device_id=TARGET_ID,
        poll_attempts=3,
        poll_interval_s=0,
    )
    assert result["polls"] == 3
    assert result["poll_timed_out"] is True
    assert result["running"] is False
    assert len([call for call in api.calls if call[0] == "POST_DEPLOY"]) == 1


def test_deploy_command_executes_when_all_readiness_gates_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload()), encoding="utf-8")
    config = native.parse_config(config_payload())
    addon_path = write_addon(tmp_path / "addon.lua")
    addon = native.load_gateway_addon(addon_path, config)
    api = FakeApi(candidate_script=native.combined_gateway_script(FakeApi().base_script, addon))
    monkeypatch.setenv("A8_API_KEY", "test-key-not-printed")

    status = native.main(
        [
            "deploy",
            str(config_path),
            "--gateway-addon",
            str(addon_path),
            "--confirm-candidate-id",
            CANDIDATE_ID,
            "--confirm-device-id",
            TARGET_ID,
            "--poll-attempts",
            "1",
            "--poll-interval",
            "0",
        ],
        api_factory=lambda *_args, **_kwargs: api,
    )

    captured = capsys.readouterr()
    assert status == 0
    assert '"deployed": true' in captured.out
    assert "test-key-not-printed" not in captured.out + captured.err
    assert len([call for call in api.calls if call[0] == "POST_DEPLOY"]) == 1
