from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from safeloop.carsky_decision import DecisionEnvelopeBuilder, DecisionValidity
from safeloop.carsky_hmi import MAX_SAFE_UDP_PAYLOAD_BYTES
from tools.generate_carsky_native_replay import (
    ADDON_BEGIN,
    ADDON_END,
    NO_FINITE_TTC_MS,
    _json_fragment,
    _lua_string,
    render_addon,
    write_addon,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(
    frame_id: int,
    *,
    ttc: float,
    c3_formula: str = "hackathon-evaluator-v1-no-tailgating",
    quality_formula: str = "safeloop-drive-quality-v1",
    reasons: tuple[str, ...] = ("NORMAL",),
) -> SimpleNamespace:
    signals = SimpleNamespace(
        attentive_probability=91.0,
        distraction_level=5.0,
        fatigue_level=7.0,
        is_eyes_on_road=True,
        is_warning=False,
    )
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp=frame_id * 0.05,
        c1=SimpleNamespace(
            predicted_ttc_s=ttc,
            collision_probability=0.25,
            is_warning=False,
            model_updated=frame_id % 2 == 0,
            model_frame_id=frame_id - frame_id % 2,
        ),
        c2=SimpleNamespace(
            state="alert",
            confidence=0.91,
            vss_signals=lambda: signals,
        ),
        c3=SimpleNamespace(
            safe_score_estimate=95.0,
            grade="A",
            trip_complete=frame_id == 1,
            formula_version=c3_formula,
            tailgating_penalty_omitted=True,
        ),
        drive_quality=SimpleNamespace(
            score_available=True,
            score_pct=97.0,
            grade="A",
            scope="FULL_TRIP" if frame_id == 1 else "PREFIX",
            window_ready=False,
            formula_version=quality_formula,
        ),
        contextual_risk=SimpleNamespace(
            score_pct=10.0,
            level="SAFE",
            action="MONITOR",
            brake_request_pct=0.0,
            reasons=reasons,
        ),
    )


def _inputs(
    tmp_path: Path,
    *,
    c3_formula: str = "hackathon-evaluator-v1-no-tailgating",
    quality_formula: str = "safeloop-drive-quality-v1",
    reasons: tuple[str, ...] = ("NORMAL",),
    model_versions: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    trip = tmp_path / "T01-Sample"
    trip.mkdir(parents=True)
    payload = {
        "trip_id": "T01-Sample",
        "metadata": {"trip_id": "T01-Sample", "fps": 20.0, "speed_limit_kmh": 60.0},
        "frames": [
            {
                "frame_id": index,
                "timestamp": index * 0.05,
                "ego": {
                    "speed_kmh": 40.0 + index,
                    "longitudinal_accel": -0.2,
                    "lateral_accel": 0.1,
                },
            }
            for index in range(2)
        ],
    }
    trip_json = trip / "T01-Sample.json"
    trip_json.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "schema": "safeloop.carsky.demo-bundle.v1",
        "trip_id": "T01-Sample",
        "frames": 2,
        "truth_free": True,
        "files": {trip_json.name: {"sha256": _sha256(trip_json)}},
    }
    (trip / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    builder = DecisionEnvelopeBuilder(
        session_id="replay:test:once",
        source_mode="replay",
        source_fps=20.0,
        model_versions=model_versions,
        clock_ms=iter((10_000, 10_050)).__next__,
    )
    validity = DecisionValidity(True, True, True, True, True, True, True, True)
    envelopes = [
        builder.build(
            _frame(
                0,
                ttc=float("inf"),
                c3_formula=c3_formula,
                quality_formula=quality_formula,
                reasons=reasons,
            ),
            validity=validity,
        ),
        builder.build(
            _frame(
                1,
                ttc=2.5,
                c3_formula=c3_formula,
                quality_formula=quality_formula,
                reasons=reasons,
            ),
            validity=validity,
        ),
    ]
    jsonl = tmp_path / "envelopes.jsonl"
    jsonl.write_bytes(b"".join(item.to_json_bytes() + b"\n" for item in envelopes))
    return trip, jsonl


def test_render_addon_has_exact_packets_kuksa_and_no_actuation(tmp_path: Path) -> None:
    trip, jsonl = _inputs(tmp_path)
    lua, report = render_addon(trip, jsonl)

    assert lua.startswith(ADDON_BEGIN)
    assert lua.rstrip().endswith(ADDON_END)
    assert "nydus.net.udp(\"0.0.0.0\", 48101)" in lua
    assert "pins.eth" not in lua
    assert "kernel NIC" in lua
    assert 'safeloop_target_port = 48100' in lua
    assert "ObstacleDetection.Front.Center.Distance" not in lua
    assert "actuate(" not in lua
    assert str(NO_FINITE_TTC_MS) in lua
    assert "REAL MODEL / REPLAY" in lua
    assert report["frames"] == 2
    assert report["truth_free"] is True
    assert report["inference_inside_script_node"] is False
    assert report["udp_payload_limit_bytes"] == MAX_SAFE_UDP_PAYLOAD_BYTES
    assert report["udp_payload_max_bytes"] <= MAX_SAFE_UDP_PAYLOAD_BYTES
    assert report["udp_payload_min_bytes"] <= report["udp_payload_max_bytes"]

    assert "local function safeloop_build_packet(row)" in lua
    assert "local safeloop_max_udp_payload_bytes = 1472" in lua
    assert "#packet <= safeloop_max_udp_payload_bytes" in lua
    assert "SafeLoop UDP packet exceeds MTU-safe limit" in lua
    assert '"r:%s:%s:c%d"' in lua
    assert lua.count('"schema_version":"safeloop.decision.v1"') == 1
    assert "replay:T01-Sample:cycle-" not in lua
    assert report["session_id_format"] == "r:<trip>:<boot-token>:c<cycle>"
    assert report["session_boot_token_max_chars"] == 24
    assert report["restart_quiet_seconds"] > 0.2
    assert "local safeloop_cycle = 0" in lua
    assert "local safeloop_gap_remaining = safeloop_restart_quiet_ticks" in lua
    startup_neutralize = lua.index("\nsafeloop_neutralize()\n")
    assert startup_neutralize < lua.index("local safeloop_rows")
    assert startup_neutralize < lua.index("timer.periodic")
    assert "tostring({}):gsub" in lua
    assert "type(os.time)" in lua
    assert len(lua.encode()) < 15_000


def test_lua_string_and_json_fragments_survive_hostile_content(tmp_path: Path) -> None:
    hostile = '\\"]=] newline\ncontrol\x01 unicode-đ'
    quality = 'quality\\"]=]'
    reason = 'reason\\"]=]\n\x02'
    versions = {'c1\\"]=]': 'version\n\x03-đ'}
    trip, jsonl = _inputs(
        tmp_path,
        c3_formula=hostile,
        quality_formula=quality,
        reasons=(reason,),
        model_versions=versions,
    )

    lua, _report = render_addon(trip, jsonl)

    assert _lua_string(hostile) == '"\\092\\034]=] newline\\010control\\001 unicode-\\196\\145"'
    assert (
        f"local safeloop_c3_formula_json = "
        f"{_lua_string(_json_fragment(hostile))}"
    ) in lua
    assert (
        f"local safeloop_quality_formula_json = "
        f"{_lua_string(_json_fragment(quality))}"
    ) in lua
    assert _lua_string(_json_fragment([reason])) in lua
    assert _lua_string(_json_fragment(versions, sort_keys=True)) in lua
    assert "[=[" not in lua
    assert hostile not in lua
    assert reason not in lua


@pytest.mark.parametrize(
    "host",
    ("localhost", "::1", "127.1", "010.0.0.1", "127.0.0.1\n", " 127.0.0.1"),
)
def test_render_rejects_noncanonical_or_non_ipv4_host(
    tmp_path: Path, host: str
) -> None:
    trip, jsonl = _inputs(tmp_path)
    with pytest.raises(ValueError, match="IPv4"):
        render_addon(trip, jsonl, android_host=host)


def test_direct_help_works_without_site_packages(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "tools" / "generate_carsky_native_replay.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--android-host" in result.stdout


def test_render_rejects_non_truth_free_or_frame_mismatch(tmp_path: Path) -> None:
    trip, jsonl = _inputs(tmp_path)
    manifest_path = trip / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["truth_free"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="truth_free"):
        render_addon(trip, jsonl)

    trip, jsonl = _inputs(tmp_path / "second")
    lines = jsonl.read_text().splitlines()
    jsonl.write_text(lines[0] + "\n")
    with pytest.raises(ValueError, match="không khớp"):
        render_addon(trip, jsonl)


def test_render_rejects_envelope_above_safe_udp_payload(tmp_path: Path) -> None:
    trip, jsonl = _inputs(tmp_path)
    packets = [json.loads(line) for line in jsonl.read_bytes().splitlines()]
    packets[0]["health"]["model_versions"] = {
        "c1": "x" * MAX_SAFE_UDP_PAYLOAD_BYTES
    }
    jsonl.write_bytes(
        b"\n".join(
            json.dumps(
                packet,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            for packet in packets
        )
        + b"\n"
    )

    with pytest.raises(ValueError, match=r"vượt MTU-safe UDP 1472 bytes"):
        render_addon(trip, jsonl)


def test_write_is_atomic_and_refuses_silent_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "addon.lua"
    result = write_addon(destination, "safe", {"frames": 1})
    assert destination.read_text() == "safe"
    assert result["output_sha256"] == _sha256(destination)
    with pytest.raises(FileExistsError):
        write_addon(destination, "changed", {"frames": 1})
    assert destination.read_text() == "safe"
