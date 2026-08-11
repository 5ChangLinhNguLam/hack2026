#!/usr/bin/env python3
"""Render a truth-free, real-model CarSky replay as an IVI Gateway addon.

The input JSONL is produced by ``safeloop-carsky-edge --record-envelopes``.
Every row is a strict ``safeloop.decision.v1`` packet from actual C1/C2/C3
inference.  The resulting Lua addon has two synchronized, read-only branches:

* standard scalar mirrors are published into the in-room KUKSA broker; and
* the exact atomic decision packet is sent to the native AAOS app over UDP.

This is explicitly a model-output replay, not inference inside a Script Node.
It never reads labels, targets, depth, events, or any ground-truth field.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

# A direct ``python3 tools/generate_carsky_native_replay.py --help`` invocation
# puts only ``tools/`` on sys.path.  Bootstrap the repository root so the CLI
# remains usable before an editable/package install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA = "safeloop.carsky.native-replay-addon.v1"
ADDON_BEGIN = "-- SAFELOOP_NATIVE_REPLAY_ADDON_V1_BEGIN"
ADDON_END = "-- SAFELOOP_NATIVE_REPLAY_ADDON_V1_END"
NO_FINITE_TTC_MS = 0xFFFFFFFF
DEFAULT_ANDROID_HOST = "10.99.0.14"
DEFAULT_ANDROID_PORT = 48_100
DEFAULT_LOCAL_PORT = 48_101
BOOT_TOKEN_CHARS = 24
# Kept dependency-free at import time so ``--help`` works in a fresh checkout
# before OpenCV/model packages are installed. ``_load_envelopes`` verifies it
# against the runtime transport constant before generating an addon.
MAX_SAFE_UDP_PAYLOAD_BYTES = 1_472
# Lua numbers are IEEE-754 doubles in the target runtime. Keeping the cycle at
# or below 2^53-1 preserves integer formatting and gives the MTU preflight a
# finite worst-case session identifier.
MAX_SAFE_LUA_CYCLE = 9_007_199_254_740_991


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} phải là số")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} phải là số") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} phải hữu hạn")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} phải >= {minimum}")
    return number


def _load_truth_free_frames(trip_dir: Path) -> tuple[str, float, list[Mapping[str, Any]]]:
    trip_dir = trip_dir.resolve()
    manifest_path = trip_dir / "BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Thiếu truth-free manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "safeloop.carsky.demo-bundle.v1":
        raise ValueError("Bundle manifest schema không được hỗ trợ")
    if manifest.get("truth_free") is not True:
        raise ValueError("Chỉ nhận bundle đã xác nhận truth_free=true")
    trip_id = str(manifest.get("trip_id") or "")
    if not trip_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in trip_id):
        raise ValueError("trip_id không hợp lệ cho CarSky replay")
    json_path = trip_dir / f"{trip_id}.json"
    entry = (manifest.get("files") or {}).get(json_path.name)
    if not isinstance(entry, Mapping) or entry.get("sha256") != _sha256(json_path):
        raise ValueError("Trip JSON không khớp truth-free manifest")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Truth-free bundle không có frame")
    if len(frames) != int(manifest.get("frames", -1)):
        raise ValueError("Số frame không khớp truth-free manifest")
    metadata = payload.get("metadata") or {}
    fps = _finite(metadata.get("fps"), field="metadata.fps", minimum=0.001)
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or int(frame.get("frame_id", -1)) != index:
            raise ValueError(f"Truth-free frame {index} không liên tiếp")
        if set(frame) != {"frame_id", "timestamp", "ego"}:
            raise ValueError(f"Truth-free frame {index} chứa field ngoài allowlist")
        ego = frame.get("ego")
        if not isinstance(ego, Mapping) or set(ego) != {
            "speed_kmh",
            "longitudinal_accel",
            "lateral_accel",
        }:
            raise ValueError(f"Truth-free frame {index} có ego ngoài allowlist")
        _finite(ego["speed_kmh"], field=f"frames[{index}].ego.speed_kmh", minimum=0.0)
        _finite(ego["longitudinal_accel"], field=f"frames[{index}].ego.longitudinal_accel")
        _finite(ego["lateral_accel"], field=f"frames[{index}].ego.lateral_accel")
    return trip_id, fps, frames


def _load_envelopes(path: Path, *, expected_frames: int) -> list[Any]:
    # Imported lazily so --help and bundle validation remain lightweight.
    from safeloop.carsky_decision import DecisionEnvelope
    from safeloop.carsky_hmi import (
        MAX_SAFE_UDP_PAYLOAD_BYTES as TRANSPORT_MAX_SAFE_UDP_PAYLOAD_BYTES,
    )

    if TRANSPORT_MAX_SAFE_UDP_PAYLOAD_BYTES != MAX_SAFE_UDP_PAYLOAD_BYTES:
        raise RuntimeError("Giới hạn UDP của generator lệch runtime transport")

    envelopes = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            payload = raw.strip()
            if not payload:
                raise ValueError(f"Envelope JSONL có dòng rỗng tại {line_number}")
            if len(payload) > MAX_SAFE_UDP_PAYLOAD_BYTES:
                raise ValueError(
                    f"Envelope JSONL dòng {line_number} dài {len(payload)} bytes, "
                    f"vượt MTU-safe UDP {MAX_SAFE_UDP_PAYLOAD_BYTES} bytes"
                )
            envelope = DecisionEnvelope.from_json_bytes(payload)
            index = len(envelopes)
            if envelope.source_mode != "replay":
                raise ValueError("Native Script Node chỉ nhận source_mode=replay")
            if envelope.frame_id != index or envelope.sequence != index:
                raise ValueError(
                    f"Envelope phải liên tiếp từ 0: frame={envelope.frame_id}, seq={envelope.sequence}, expected={index}"
                )
            envelopes.append(envelope)
    if len(envelopes) != expected_frames:
        raise ValueError(
            f"Số envelope {len(envelopes)} không khớp {expected_frames} frame"
        )
    session_ids = {item.session_id for item in envelopes}
    if len(session_ids) != 1:
        raise ValueError("Envelope JSONL phải chứa đúng một session")
    return envelopes


def _generated_packet_bytes(envelope: Any, *, session_id: str) -> bytes:
    """Return a conservative byte-equivalent of the Lua-built UDP packet."""

    payload = envelope.to_dict()
    payload["session_id"] = session_id
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _lua_bool(value: bool) -> str:
    return "true" if value else "false"


def _lua_number(value: float) -> str:
    return format(value, ".12g")


def _lua_string(value: str) -> str:
    """Encode arbitrary Unicode as a Lua 5.1-compatible byte string.

    JSON ``\\uXXXX`` escapes are not Lua escapes, and long-bracket literals can
    be terminated by hostile ``]=]`` input.  Three-digit decimal byte escapes
    are unambiguous in every string position and preserve the UTF-8 payload.
    """

    encoded = value.encode("utf-8")
    chunks = ['"']
    for byte in encoded:
        if 0x20 <= byte <= 0x7E and byte not in (ord('"'), ord("\\")):
            chunks.append(chr(byte))
        else:
            chunks.append(f"\\{byte:03d}")
    chunks.append('"')
    return "".join(chunks)


def _json_fragment(value: object, *, sort_keys: bool = False) -> str:
    """Serialize one JSON value for later byte-exact Lua concatenation."""

    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=sort_keys,
        allow_nan=False,
    )


def _canonical_ipv4(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Android host phải là địa chỉ IPv4 chuẩn")
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise ValueError("Android host phải là địa chỉ IPv4 chuẩn") from exc
    canonical = str(address)
    if canonical != value:
        raise ValueError("Android host phải dùng dạng IPv4 chuẩn")
    return canonical


def render_addon(
    trip_dir: Path,
    envelope_jsonl: Path,
    *,
    android_host: str = DEFAULT_ANDROID_HOST,
    android_port: int = DEFAULT_ANDROID_PORT,
    local_port: int = DEFAULT_LOCAL_PORT,
    gap_seconds: float = 3.0,
) -> tuple[str, dict[str, Any]]:
    android_host = _canonical_ipv4(android_host)
    for field, port in (("android_port", android_port), ("local_port", local_port)):
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
            raise ValueError(f"{field} phải thuộc [1, 65535]")
    if android_port == local_port:
        raise ValueError("local_port phải khác Android target port")
    gap_seconds = _finite(gap_seconds, field="gap_seconds", minimum=0.0)

    trip_id, fps, frames = _load_truth_free_frames(trip_dir)
    period_ms_float = 1_000.0 / fps
    period_ms = round(period_ms_float)
    if not math.isclose(period_ms_float, period_ms, abs_tol=1e-9):
        raise ValueError("CarSky Script Node cần FPS chia hết 1000 ms")
    envelopes = _load_envelopes(envelope_jsonl, expected_frames=len(frames))
    gap_ticks = max(1, round(gap_seconds * fps))

    first = envelopes[0]
    ttl_ms = first.ttl_ms
    # The initial quiet dwell lets a receiver observe STALE before accepting a
    # reused sequence-0 session if a platform restart happens to reuse its boot
    # identity. The packet is sent on the timer tick after this countdown.
    restart_quiet_ticks = max(1, math.ceil(ttl_ms / period_ms))
    restart_quiet_ms = (restart_quiet_ticks + 1) * period_ms
    c3_formula = str(first.c3["formula_version"])
    quality_formula = str(first.drive_quality["formula_version"])
    model_versions = dict(first.health["model_versions"])
    c3_formula_json = _json_fragment(c3_formula)
    quality_formula_json = _json_fragment(quality_formula)
    model_versions_json = _json_fragment(model_versions, sort_keys=True)

    generated_packet_sizes: list[int] = []

    rows: list[str] = []
    for index, (frame, envelope) in enumerate(zip(frames, envelopes, strict=True)):
        if round(_finite(frame["timestamp"], field=f"frames[{index}].timestamp") * 1_000) != envelope.source_timestamp_ms:
            raise ValueError(f"Timestamp envelope lệch truth-free frame {index}")
        ego = frame["ego"]
        c1 = envelope.c1
        c2 = envelope.c2
        c3 = envelope.c3
        quality = envelope.drive_quality
        risk = envelope.contextual_risk
        health = envelope.health
        if not all(envelope.validity.values()):
            raise ValueError(f"Replay deploy yêu cầu mọi component hợp lệ tại frame {index}")
        if (
            envelope.ttl_ms != ttl_ms
            or c3["formula_version"] != c3_formula
            or quality["formula_version"] != quality_formula
            or health["model_versions"] != model_versions
        ):
            raise ValueError("Replay constants thay đổi giữa các frame")
        if (
            health["mode"] != "NOMINAL"
            or health["decision_valid"] is not True
            or health["stale_or_invalid_components"]
            or risk["actuation_authorized"] is not False
            or c1["collision_probability_pct"] is None
            or any(
                c2[name] is None
                for name in (
                    "confidence_pct",
                    "attentive_probability_pct",
                    "distraction_level_pct",
                    "fatigue_level_pct",
                    "eyes_on_road",
                )
            )
            or c3["safe_score_estimate_pct"] is None
            or quality["score_available"] is not True
            or quality["score_pct"] is None
            or risk["score_pct"] is None
        ):
            raise ValueError(f"Envelope frame {index} không phải nominal replay snapshot")
        generated_size = len(
            _generated_packet_bytes(
                envelope,
                session_id=(
                    f"r:{trip_id}:{'b' * BOOT_TOKEN_CHARS}:"
                    f"c{MAX_SAFE_LUA_CYCLE}"
                ),
            )
        )
        if generated_size > MAX_SAFE_UDP_PAYLOAD_BYTES:
            raise ValueError(
                f"UDP decision frame {index} dài {generated_size} bytes, vượt "
                f"MTU-safe limit {MAX_SAFE_UDP_PAYLOAD_BYTES} bytes"
            )
        generated_packet_sizes.append(generated_size)
        ttc_ms = int(c1["ttc_ms"]) if c1["ttc_valid"] else -1
        reasons_json = _json_fragment(risk["reasons"])
        rows.append(
            "  {"
            + ",".join(
                (
                    str(index),
                    str(envelope.source_timestamp_ms),
                    _lua_number(float(ego["speed_kmh"])),
                    _lua_number(float(ego["longitudinal_accel"])),
                    _lua_number(float(ego["lateral_accel"])),
                    str(ttc_ms),
                    _lua_number(float(c1["collision_probability_pct"])),
                    _lua_bool(bool(c1["warning"])),
                    _lua_bool(bool(c1["model_updated"])),
                    str(int(c1["model_frame_id"])),
                    str(int(c1["age_ms"])),
                    _lua_string(_json_fragment(str(c2["state"]))),
                    _lua_number(float(c2["confidence_pct"])),
                    _lua_number(float(c2["attentive_probability_pct"])),
                    _lua_number(float(c2["distraction_level_pct"])),
                    _lua_number(float(c2["fatigue_level_pct"])),
                    _lua_bool(bool(c2["eyes_on_road"])),
                    _lua_bool(bool(c2["warning"])),
                    _lua_number(float(c3["safe_score_estimate_pct"])),
                    _lua_string(_json_fragment(str(c3["grade"]))),
                    _lua_string(_json_fragment(str(c3["scope"]))),
                    _lua_number(float(quality["score_pct"])),
                    _lua_string(_json_fragment(str(quality["grade"]))),
                    _lua_string(_json_fragment(str(quality["scope"]))),
                    _lua_bool(bool(quality["window_ready"])),
                    _lua_number(float(risk["score_pct"])),
                    _lua_string(_json_fragment(str(risk["level"]))),
                    _lua_string(_json_fragment(str(risk["action"]))),
                    _lua_number(float(risk["brake_request_pct"])),
                    _lua_string(reasons_json),
                    str(envelope.decision_timestamp_ms),
                )
            )
            + "},"
        )

    source_hash = _sha256(envelope_jsonl)
    lua = f'''{ADDON_BEGIN}
-- SAFELOOP_NATIVE_GATEWAY_ADDON_V1
-- SAFELOOP_SOURCE_MODE: MODEL_REPLAY
-- Generated from actual SafeLoop model decisions. Source mode is REPLAY.
-- This addon was generated only from the deployable truth-free input bundle.
-- envelope_jsonl_sha256={source_hash}
-- One timer iteration publishes the same decision to KUKSA mirrors and AAOS.

local safeloop_kuksa = pins.kuksa
assert(safeloop_kuksa and safeloop_kuksa.vss, "SafeLoop KUKSA pin unavailable")
-- CarSky's ethernet pin provisions a kernel NIC but intentionally has no
-- Script Node ``pins.<name>`` backend.  UDP therefore uses nydus.net directly;
-- the guarded deployment preflight separately verifies the Ethernet edge.
local safeloop_udp = nydus.net.udp("0.0.0.0", {local_port})
local safeloop_target_host = {_lua_string(android_host)}
local safeloop_target_port = {android_port}
local safeloop_period_ms = {period_ms}
local safeloop_max_udp_payload_bytes = {MAX_SAFE_UDP_PAYLOAD_BYTES}
local safeloop_gap_ticks = {gap_ticks}
local safeloop_restart_quiet_ticks = {restart_quiet_ticks}
local safeloop_max_cycle = {MAX_SAFE_LUA_CYCLE}
local safeloop_trip_id = {_lua_string(trip_id)}
local safeloop_ttl_ms = {ttl_ms}
local safeloop_no_finite_ttc_ms = {NO_FINITE_TTC_MS}
local safeloop_c3_formula_json = {_lua_string(c3_formula_json)}
local safeloop_quality_formula_json = {_lua_string(quality_formula_json)}
local safeloop_model_versions_json = {_lua_string(model_versions_json)}

local safeloop_vehicle = safeloop_kuksa.vss.Vehicle
local safeloop_obstacle = safeloop_vehicle.ADAS.ObstacleDetection.Front.Center
local safeloop_driver = safeloop_vehicle.Driver
local safeloop_dms = safeloop_vehicle.ADAS.DMS

local function safeloop_neutralize()
    safeloop_vehicle.Speed:publish(0.0)
    safeloop_vehicle.Acceleration.Longitudinal:publish(0.0)
    safeloop_vehicle.Acceleration.Lateral:publish(0.0)
    safeloop_obstacle.TimeGap:publish(safeloop_no_finite_ttc_ms)
    safeloop_obstacle.IsWarning:publish(false)
    safeloop_driver.AttentiveProbability:publish(100.0)
    safeloop_driver.DistractionLevel:publish(0.0)
    safeloop_driver.FatigueLevel:publish(0.0)
    safeloop_driver.IsEyesOnRoad:publish(true)
    safeloop_dms.IsWarning:publish(false)
end

-- Clear values left by a previous Script Node before materializing the replay
-- table or entering the restart quiet dwell.
safeloop_neutralize()

-- Row: frame,sourceMs,speed,aLong,aLat,ttcMs(-1=none),collisionProb,
-- warning,modelUpdated,modelFrame,age,state,confidence,attention,
-- distraction,fatigue,eyesOnRoad,dmsWarning,c3Score,c3Grade,c3Scope,
-- qualityScore,qualityGrade,qualityScope,windowReady,riskScore,riskLevel,
-- action,brakeRequest,reasonsJson,decisionMs.
local safeloop_rows = {{
{chr(10).join(rows)}
}}

local safeloop_index = 1
-- Each Script Node process gets a best-effort boot identity without relying on
-- an undocumented CarSky API. tostring(table) normally includes a VM-local
-- identity; guarded os.time adds entropy when that standard library exists.
local function safeloop_make_boot_token()
    local identity = tostring({{}}):gsub("[^%w]", "")
    local epoch = ""
    if type(os) == "table" and type(os.time) == "function" then
        local ok, value = pcall(os.time)
        if ok and type(value) == "number" then
            epoch = tostring(math.floor(value))
        end
    end
    local token = (epoch .. identity):gsub("[^%w]", "")
    if #token == 0 then
        token = "boot"
    end
    return token:sub(1, {BOOT_TOKEN_CHARS})
end

local safeloop_boot_token = safeloop_make_boot_token()
-- Start at cycle zero and wait longer than the decision TTL. If a rare boot
-- token collision occurs, the receiver can then safely accept cycle-1 seq 0
-- as an explicit restart instead of confusing it with a live replay attack.
local safeloop_cycle = 0
local safeloop_gap_remaining = safeloop_restart_quiet_ticks

local function safeloop_session_id()
    return string.format(
        "r:%s:%s:c%d",
        safeloop_trip_id,
        safeloop_boot_token,
        safeloop_cycle
    )
end

local function safeloop_bool(value)
    return value and "true" or "false"
end

local function safeloop_build_packet(row)
    local ttc_valid = row[6] >= 0
    local ttc_json = ttc_valid and tostring(row[6]) or "null"
    local expires_at_ms = row[31] + safeloop_ttl_ms
    return '{{'
        .. '"schema_version":"safeloop.decision.v1",'
        .. '"source_mode":"replay",'
        .. '"session_id":"' .. safeloop_session_id() .. '",'
        .. '"sequence":' .. row[1] .. ','
        .. '"frame_id":' .. row[1] .. ','
        .. '"source_timestamp_ms":' .. row[2] .. ','
        .. '"decision_timestamp_ms":' .. row[31] .. ','
        .. '"ttl_ms":' .. safeloop_ttl_ms .. ','
        .. '"expires_at_ms":' .. expires_at_ms .. ','
        .. '"validity":{{"ego":true,"front_camera":true,"driver_camera":true,"c1":true,"c2":true,"c3":true,"drive_quality":true,"contextual_risk":true}},'
        .. '"c1":{{"ttc_ms":' .. ttc_json
        .. ',"ttc_valid":' .. safeloop_bool(ttc_valid)
        .. ',"collision_probability_pct":' .. row[7]
        .. ',"warning":' .. safeloop_bool(row[8])
        .. ',"model_updated":' .. safeloop_bool(row[9])
        .. ',"model_frame_id":' .. row[10]
        .. ',"age_ms":' .. row[11] .. '}},'
        .. '"c2":{{"state":' .. row[12]
        .. ',"confidence_pct":' .. row[13]
        .. ',"attentive_probability_pct":' .. row[14]
        .. ',"distraction_level_pct":' .. row[15]
        .. ',"fatigue_level_pct":' .. row[16]
        .. ',"eyes_on_road":' .. safeloop_bool(row[17])
        .. ',"warning":' .. safeloop_bool(row[18]) .. '}},'
        .. '"c3":{{"safe_score_estimate_pct":' .. row[19]
        .. ',"grade":' .. row[20]
        .. ',"scope":' .. row[21]
        .. ',"formula_version":' .. safeloop_c3_formula_json
        .. ',"tailgating_penalty_omitted":true}},'
        .. '"drive_quality":{{"score_available":true,"score_pct":' .. row[22]
        .. ',"grade":' .. row[23]
        .. ',"scope":' .. row[24]
        .. ',"window_ready":' .. safeloop_bool(row[25])
        .. ',"formula_version":' .. safeloop_quality_formula_json .. '}},'
        .. '"contextual_risk":{{"score_pct":' .. row[26]
        .. ',"level":' .. row[27]
        .. ',"action":' .. row[28]
        .. ',"brake_request_pct":' .. row[29]
        .. ',"reasons":' .. row[30]
        .. ',"actuation_authorized":false}},'
        .. '"health":{{"mode":"NOMINAL","decision_valid":true,'
        .. '"stale_or_invalid_components":[],"component_age_ms":{{"c1":'
        .. row[11] .. ',"c2":0,"ego":0}},"ttl_ms":' .. safeloop_ttl_ms
        .. ',"model_versions":' .. safeloop_model_versions_json .. '}}'
        .. '}}'
end

timer.periodic(safeloop_period_ms, function()
    if safeloop_gap_remaining > 0 then
        safeloop_gap_remaining = safeloop_gap_remaining - 1
        if safeloop_gap_remaining == 0 then
            assert(safeloop_cycle < safeloop_max_cycle, "SafeLoop replay cycle exhausted")
            safeloop_cycle = safeloop_cycle + 1
            safeloop_index = 1
            log(string.format("[SafeLoop REAL MODEL / REPLAY] start session=%s", safeloop_session_id()))
        end
        return
    end
    if safeloop_index > #safeloop_rows then
        safeloop_neutralize()
        safeloop_gap_remaining = safeloop_gap_ticks
        log(string.format("[SafeLoop REAL MODEL / REPLAY] complete cycle=%d frames=%d; fail-safe gap", safeloop_cycle, #safeloop_rows))
        return
    end

    local row = safeloop_rows[safeloop_index]
    safeloop_vehicle.Speed:publish(row[3])
    safeloop_vehicle.Acceleration.Longitudinal:publish(row[4])
    safeloop_vehicle.Acceleration.Lateral:publish(row[5])
    safeloop_obstacle.TimeGap:publish(
        row[6] >= 0 and row[6] or safeloop_no_finite_ttc_ms
    )
    safeloop_obstacle.IsWarning:publish(row[8])
    safeloop_driver.AttentiveProbability:publish(row[14])
    safeloop_driver.DistractionLevel:publish(row[15])
    safeloop_driver.FatigueLevel:publish(row[16])
    safeloop_driver.IsEyesOnRoad:publish(row[17])
    safeloop_dms.IsWarning:publish(row[18])

    local packet = safeloop_build_packet(row)
    assert(
        #packet <= safeloop_max_udp_payload_bytes,
        string.format(
            "SafeLoop UDP packet exceeds MTU-safe limit: %d > %d",
            #packet, safeloop_max_udp_payload_bytes
        )
    )
    safeloop_udp:send_to(packet, safeloop_target_host, safeloop_target_port)

    if safeloop_index == 1 or safeloop_index % 100 == 0 then
        log(string.format(
            "[SafeLoop REAL MODEL / REPLAY] cycle=%d frame=%d/%d ttc_ms=%s warning=%s",
            safeloop_cycle, row[1], #safeloop_rows, tostring(row[6]), tostring(row[8])
        ))
    end
    safeloop_index = safeloop_index + 1
end)

log(string.format(
    "[SafeLoop REAL MODEL / REPLAY] ready; trip=%s frames=%d hz={fps:g} udp=%s:%d actuator=false",
    safeloop_trip_id, #safeloop_rows, safeloop_target_host, safeloop_target_port
))
{ADDON_END}
'''
    report = {
        "schema": SCHEMA,
        "trip_id": trip_id,
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps,
        "source_mode": "replay",
        "truth_free": True,
        "model_output_replay": True,
        "inference_inside_script_node": False,
        "wire_packet_built_on_node": True,
        "android_udp": {"host": android_host, "port": android_port},
        "udp_payload_limit_bytes": MAX_SAFE_UDP_PAYLOAD_BYTES,
        "udp_payload_min_bytes": min(generated_packet_sizes),
        "udp_payload_max_bytes": max(generated_packet_sizes),
        "local_udp_port": local_port,
        "gap_seconds": gap_ticks / fps,
        "session_boot_token_max_chars": BOOT_TOKEN_CHARS,
        "session_cycle_max": MAX_SAFE_LUA_CYCLE,
        "session_id_format": "r:<trip>:<boot-token>:c<cycle>",
        "restart_quiet_seconds": restart_quiet_ms / 1_000.0,
        "restart_recovery": (
            "best-effort per-process boot token plus stale-only sequence-0 fallback"
        ),
        "kuksa_standard_mirrors": 10,
        "publishes_distance": False,
        "authorizes_actuation": False,
        "envelope_jsonl_sha256": source_hash,
    }
    return lua, report


def write_addon(
    output: Path,
    content: str,
    report: Mapping[str, Any],
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace_existing:
        raise FileExistsError(f"Output đã tồn tại: {output}; dùng --force để thay")
    archived: Path | None = None
    if output.exists():
        archive = output.parent / "archive" / str(time.time_ns())
        archive.mkdir(parents=True, exist_ok=True)
        archived = archive / output.name
        output.replace(archived)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if archived is not None and not output.exists():
            archived.replace(output)
        raise
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_payload = dict(report)
    report_payload.update(
        {
            "output": str(output),
            "output_bytes": output.stat().st_size,
            "output_sha256": _sha256(output),
            "archived_previous": str(archived) if archived else None,
        }
    )
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sinh IVI Gateway addon từ actual-model decision JSONL"
    )
    parser.add_argument("--trip-dir", type=Path, required=True)
    parser.add_argument("--envelopes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--android-host", default=DEFAULT_ANDROID_HOST)
    parser.add_argument("--android-port", type=int, default=DEFAULT_ANDROID_PORT)
    parser.add_argument("--local-port", type=int, default=DEFAULT_LOCAL_PORT)
    parser.add_argument("--gap-seconds", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        content, report = render_addon(
            args.trip_dir,
            args.envelopes,
            android_host=args.android_host,
            android_port=args.android_port,
            local_port=args.local_port,
            gap_seconds=args.gap_seconds,
        )
        result = write_addon(
            args.output,
            content,
            report,
            replace_existing=args.force,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
