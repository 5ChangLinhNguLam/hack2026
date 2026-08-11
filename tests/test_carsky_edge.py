from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from safeloop.carsky_decision import DecisionEnvelope, DecisionEnvelopeBuilder
from safeloop.carsky_edge import (
    AtomicEnvelopeJsonlRecorder,
    CarSkyEdgeSink,
    DEFAULT_ANDROID_HOST,
    DEFAULT_ANDROID_PORT,
    MODEL_VERSION_SHA256_PREFIX_HEX,
    build_model_provenance,
    build_parser,
    create_edge_sink,
    main,
)
from safeloop.carsky_hmi import HmiTransportError, MAX_SAFE_UDP_PAYLOAD_BYTES
from safeloop.carsky_kuksa import KuksaPublishError
from safeloop.replay_models import SUBMISSION_FIELDS
from safeloop import replay_models


class FakeUdp:
    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.envelopes = []
        self.closed = False

    def publish(self, envelope):
        self.order.append(f"udp:{envelope.sequence}")
        if self.fail:
            raise HmiTransportError("udp unavailable")
        self.envelopes.append(envelope)

    def close(self):
        self.closed = True


class FakeKuksa:
    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.calls = []
        self.closed = False

    def publish(self, frame, envelope):
        self.order.append(f"kuksa:{envelope.sequence}")
        if self.fail:
            raise KuksaPublishError("broker unavailable")
        self.calls.append((frame, envelope))

    def close(self):
        self.closed = True


def frame(*, frame_id: int = 0, model_frame_id: int | None = None):
    signals = SimpleNamespace(
        attentive_probability=90.0,
        distraction_level=5.0,
        fatigue_level=7.0,
        is_eyes_on_road=True,
        is_warning=False,
    )
    if model_frame_id is None:
        model_frame_id = frame_id
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp=frame_id / 20.0,
        source_bundle=SimpleNamespace(
            ego={
                "speed_kmh": 40.0,
                "longitudinal_accel": -0.2,
                "lateral_accel": 0.1,
            }
        ),
        c1=SimpleNamespace(
            predicted_ttc_s=2.5,
            collision_probability=0.5,
            is_warning=False,
            model_updated=model_frame_id == frame_id,
            model_frame_id=model_frame_id,
        ),
        c2=SimpleNamespace(
            state="alert",
            confidence=0.9,
            vss_signals=lambda: signals,
        ),
        c3=SimpleNamespace(
            safe_score_estimate=80.0,
            grade="B",
            trip_complete=False,
            formula_version="hackathon-evaluator-v1-no-tailgating",
            tailgating_penalty_omitted=True,
        ),
        drive_quality=SimpleNamespace(
            score_available=True,
            score_pct=90.0,
            grade="A",
            scope="PREFIX",
            window_ready=False,
            formula_version="safeloop-drive-quality-v1",
        ),
        contextual_risk=SimpleNamespace(
            score_pct=20.0,
            level="SAFE",
            action="MONITOR",
            brake_request_pct=0.0,
            reasons=("NORMAL",),
        ),
    )


def builder(*, ttl_ms: int = 200) -> DecisionEnvelopeBuilder:
    return DecisionEnvelopeBuilder(
        session_id="replay:T01:test",
        source_mode="replay",
        source_fps=20.0,
        ttl_ms=ttl_ms,
        clock_ms=lambda: 10_000,
    )


def test_model_provenance_uses_compact_wire_ids_and_full_audit_hashes(
    tmp_path,
) -> None:
    c1 = tmp_path / "student_ttc.pth"
    c1.write_bytes(b"c1-checkpoint")
    c2 = tmp_path / "driver-state-bundle"
    c2.mkdir()
    (c2 / "manifest.json").write_text('{"version":1}', encoding="utf-8")
    (c2 / "weights.bin").write_bytes(b"c2-weights")

    wire, report = build_model_provenance(c1, c2)

    assert set(wire) == {"c1", "c2", "c3", "dq", "risk"}
    assert MODEL_VERSION_SHA256_PREFIX_HEX == 12
    assert all(
        len(identifier) == MODEL_VERSION_SHA256_PREFIX_HEX
        and set(identifier) <= set("0123456789abcdef")
        for identifier in wire.values()
    )
    assert report["wire_contract"] == "health.model_versions"
    for name, identifier in wire.items():
        artifact = report["artifacts"][name]
        assert artifact["wire_id"] == identifier
        assert len(artifact["sha256"]) == 64
        assert artifact["sha256"].startswith(identifier)

    previous_c2 = wire["c2"]
    (c2 / "weights.bin").write_bytes(b"changed-c2-weights")
    changed_wire, _ = build_model_provenance(c1, c2)
    assert changed_wire["c2"] != previous_c2


def test_compact_runtime_identity_keeps_strict_packet_mtu_safe() -> None:
    model_versions = {
        name: character * MODEL_VERSION_SHA256_PREFIX_HEX
        for name, character in zip(
            ("c1", "c2", "c3", "dq", "risk"),
            "12345",
            strict=True,
        )
    }
    sink = create_edge_sink(
        trip_id="T01-Sample",
        source_fps=20.0,
        model_versions=model_versions,
        enable_udp=False,
        enable_kuksa=False,
    )

    sink.publish(frame())
    envelope = sink.latest

    assert envelope is not None
    assert envelope.session_id.startswith("r:T01-Sample:")
    assert len(envelope.session_id) == len("r:T01-Sample:") + 16
    assert envelope.health["model_versions"] == model_versions
    assert len(envelope.to_json_bytes()) <= MAX_SAFE_UDP_PAYLOAD_BYTES


def test_edge_sink_fans_same_ordered_envelope_to_udp_then_kuksa() -> None:
    order: list[str] = []
    udp = FakeUdp(order)
    kuksa = FakeKuksa(order)
    sink = CarSkyEdgeSink(
        builder(), udp_publisher=udp, kuksa_publisher=kuksa
    )

    first = sink.publish(frame())
    second = sink.publish(frame(frame_id=1, model_frame_id=0))
    sink.close()

    assert order == ["udp:0", "kuksa:0", "udp:1", "kuksa:1"]
    assert first.udp_sent and first.kuksa_sent
    assert second.sequence == 1
    assert udp.envelopes[0] is kuksa.calls[0][1]
    assert sink.summary()["frames_enveloped"] == 2
    assert sink.summary()["udp_sent"] == 2
    assert sink.summary()["kuksa_sent"] == 2
    assert udp.closed and kuksa.closed


def test_transport_failures_are_isolated_and_observable() -> None:
    order: list[str] = []
    warnings: list[str] = []
    sink = CarSkyEdgeSink(
        builder(),
        udp_publisher=FakeUdp(order, fail=True),
        kuksa_publisher=FakeKuksa(order),
        transport_error_handler=warnings.append,
    )

    receipt = sink.publish(frame())

    assert receipt.udp_sent is False
    assert receipt.kuksa_sent is True
    assert order == ["udp:0", "kuksa:0"]
    assert receipt.transport_errors == (
        "udp: HmiTransportError: udp unavailable",
    )
    assert warnings == list(receipt.transport_errors)
    assert sink.summary()["udp_errors"] == 1
    assert sink.summary()["kuksa_sent"] == 1

    order.clear()
    sink = CarSkyEdgeSink(
        builder(),
        udp_publisher=FakeUdp(order),
        kuksa_publisher=FakeKuksa(order, fail=True),
    )
    receipt = sink.publish(frame())
    assert receipt.udp_sent is True
    assert receipt.kuksa_sent is False
    assert order == ["udp:0", "kuksa:0"]
    assert sink.summary()["live_transport_pass"] is False

    empty = CarSkyEdgeSink(builder(), udp_publisher=FakeUdp([]))
    assert empty.summary()["live_transport_enabled"] is True
    assert empty.summary()["live_transport_pass"] is False


def test_stale_forward_fill_suppresses_c1_and_contextual_risk() -> None:
    sink = CarSkyEdgeSink(builder(ttl_ms=40))

    receipt = sink.publish(frame(frame_id=1, model_frame_id=0))
    envelope = sink.latest

    assert receipt.sequence == 0
    assert envelope is not None
    assert envelope.validity["c1"] is False
    assert envelope.validity["contextual_risk"] is False
    assert envelope.c1["ttc_ms"] is None
    assert envelope.contextual_risk["action"] == "MONITOR"
    assert envelope.health["mode"] == "DEGRADED"


def test_disabled_transports_still_validate_every_frame() -> None:
    sink = create_edge_sink(
        trip_id="T01-Sample",
        source_fps=20.0,
        enable_udp=False,
        enable_kuksa=False,
    )

    sink(frame())

    summary = sink.summary()
    assert summary["frames_enveloped"] == 1
    assert summary["udp_enabled"] is False
    assert summary["kuksa_enabled"] is False


def test_atomic_jsonl_recorder_writes_one_strict_packet_per_frame(
    tmp_path,
) -> None:
    destination = tmp_path / "decision.jsonl"
    destination.write_text("previous-complete-file\n", encoding="utf-8")
    recorder = AtomicEnvelopeJsonlRecorder(destination)
    sink = CarSkyEdgeSink(builder(), envelope_recorder=recorder)

    sink(frame())
    sink(frame(frame_id=1, model_frame_id=0))
    assert destination.read_text(encoding="utf-8") == "previous-complete-file\n"
    sink.close()

    lines = destination.read_bytes().splitlines()
    packets = [DecisionEnvelope.from_json_bytes(line) for line in lines]
    assert [packet.sequence for packet in packets] == [0, 1]
    assert [packet.frame_id for packet in packets] == [0, 1]
    assert recorder.records == 2
    assert not recorder.temporary.exists()


def test_aborted_jsonl_recorder_preserves_previous_complete_file(
    tmp_path,
) -> None:
    destination = tmp_path / "decision.jsonl"
    destination.write_text("previous\n", encoding="utf-8")
    recorder = AtomicEnvelopeJsonlRecorder(destination)
    source = CarSkyEdgeSink(builder())
    source(frame())
    envelope = source.latest
    assert envelope is not None
    recorder.publish(envelope)

    recorder.abort()

    assert destination.read_text(encoding="utf-8") == "previous\n"
    assert not recorder.temporary.exists()


def test_edge_context_aborts_recording_on_exception_and_closes_transports(
    tmp_path,
) -> None:
    destination = tmp_path / "decision.jsonl"
    destination.write_text("previous-complete\n", encoding="utf-8")
    recorder = AtomicEnvelopeJsonlRecorder(destination)
    udp = FakeUdp([])
    kuksa = FakeKuksa([])

    with pytest.raises(RuntimeError, match="inference failed"):
        with CarSkyEdgeSink(
            builder(),
            udp_publisher=udp,
            kuksa_publisher=kuksa,
            envelope_recorder=recorder,
        ) as sink:
            sink.publish(frame())
            raise RuntimeError("inference failed")

    assert destination.read_text(encoding="utf-8") == "previous-complete\n"
    assert not recorder.temporary.exists()
    assert udp.closed and kuksa.closed


def test_edge_parser_defaults_do_not_change_original_replay_cli() -> None:
    edge = build_parser().parse_args(["--dataset", "data", "--trip", "T01"])
    original = replay_models.build_parser().parse_args(
        ["--dataset", "data", "--trip", "T01"]
    )

    assert edge.mode == "realtime"
    assert edge.udp_host == DEFAULT_ANDROID_HOST
    assert edge.udp_port == DEFAULT_ANDROID_PORT
    assert edge.no_udp is False
    assert edge.no_kuksa is False
    assert original.mode == "fast"
    assert not hasattr(original, "no_udp")
    assert SUBMISSION_FIELDS == (
        "frame_id",
        "timestamp",
        "predicted_ttc",
        "predicted_driver_state",
        "predicted_risk_score",
    )
    assert inspect.signature(replay_models.run_trip).parameters[
        "frame_sink"
    ].default is None
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'carsky-runtime = [' in pyproject
    assert '"kuksa-client>=0.5.1,<0.6"' in pyproject
    assert 'safeloop-carsky-edge = "safeloop.carsky_edge:main"' in pyproject


def test_local_smoke_cli_uses_frame_sink_without_network(
    monkeypatch, tmp_path, capsys
) -> None:
    class FakeLoader:
        def __init__(self, _path):
            self.trip_id = "T01-Sample"
            self.fps = 20.0

    captured = {}

    def fake_run_trip(_path, **kwargs):
        captured.update(kwargs)
        kwargs["frame_sink"](frame())
        return {
            "trip_id": "T01-Sample",
            "frames": 1,
            "trip_complete": False,
            "stopped_by_user": False,
        }

    monkeypatch.setattr("safeloop.carsky_edge.TripLoader", FakeLoader)
    monkeypatch.setattr("safeloop.carsky_edge.replay_models.run_trip", fake_run_trip)
    output = tmp_path / "output"

    result = main(
        [
            "--dataset",
            str(tmp_path),
            "--trip",
            "T01-Sample",
            "--output-dir",
            str(output),
            "--no-udp",
            "--no-kuksa",
            "--record-envelopes",
            str(output / "decision.jsonl"),
        ]
    )

    assert result == 0
    assert isinstance(captured["frame_sink"], CarSkyEdgeSink)
    report = json.loads(
        (output / "carsky_edge_report.json").read_text(encoding="utf-8")
    )
    assert report["completed_trips"] == 1
    assert report["summaries"][0]["carsky_transport"][
        "frames_enveloped"
    ] == 1
    assert report["android_udp"]["enabled"] is False
    assert report["kuksa"]["enabled"] is False
    assert report["record_envelopes"] == str(output / "decision.jsonl")
    recorded = (output / "decision.jsonl").read_bytes().splitlines()
    assert len(recorded) == 1
    assert DecisionEnvelope.from_json_bytes(recorded[0]).frame_id == 0
    assert "REPORT=" in capsys.readouterr().out


def test_invalid_local_limit_matches_existing_cli_contract(tmp_path) -> None:
    assert main(["--dataset", str(tmp_path), "--limit", "-1"]) == 2


def test_recording_path_rejects_implicit_multi_trip_overwrite(
    tmp_path,
) -> None:
    assert main(
        [
            "--dataset",
            str(tmp_path),
            "--record-envelopes",
            str(tmp_path / "decision.jsonl"),
        ]
    ) == 2


def test_live_transport_error_runs_frame_but_fails_cli_gate(
    monkeypatch, tmp_path
) -> None:
    class FakeLoader:
        def __init__(self, _path):
            self.trip_id = "T01-Sample"
            self.fps = 20.0

    failing_sink = CarSkyEdgeSink(
        builder(), udp_publisher=FakeUdp([], fail=True)
    )

    def fake_run_trip(_path, **kwargs):
        kwargs["frame_sink"](frame())
        return {
            "trip_id": "T01-Sample",
            "frames": 1,
            "trip_complete": False,
            "stopped_by_user": False,
        }

    monkeypatch.setattr("safeloop.carsky_edge.TripLoader", FakeLoader)
    monkeypatch.setattr(
        "safeloop.carsky_edge.create_edge_sink",
        lambda **_kwargs: failing_sink,
    )
    monkeypatch.setattr(
        "safeloop.carsky_edge.replay_models.run_trip", fake_run_trip
    )
    output = tmp_path / "output"

    result = main(
        [
            "--dataset",
            str(tmp_path),
            "--trip",
            "T01-Sample",
            "--output-dir",
            str(output),
            "--no-kuksa",
        ]
    )

    report = json.loads(
        (output / "carsky_edge_report.json").read_text(encoding="utf-8")
    )
    assert result == 1
    assert report["processed_frames"] == 1
    assert report["failures"][0]["error"].startswith(
        "live transport incomplete"
    )
    assert report["summaries"][0]["carsky_transport"][
        "live_transport_pass"
    ] is False


def test_user_stop_aborts_official_envelope_recording(
    monkeypatch, tmp_path
) -> None:
    class FakeLoader:
        def __init__(self, _path):
            self.trip_id = "T01-Sample"
            self.fps = 20.0

    def fake_run_trip(_path, **kwargs):
        kwargs["frame_sink"](frame())
        return {
            "trip_id": "T01-Sample",
            "frames": 1,
            "trip_complete": False,
            "stopped_by_user": True,
        }

    monkeypatch.setattr("safeloop.carsky_edge.TripLoader", FakeLoader)
    monkeypatch.setattr(
        "safeloop.carsky_edge.replay_models.run_trip", fake_run_trip
    )
    destination = tmp_path / "decision.jsonl"
    output = tmp_path / "output"

    result = main(
        [
            "--dataset",
            str(tmp_path),
            "--trip",
            "T01-Sample",
            "--output-dir",
            str(output),
            "--no-udp",
            "--no-kuksa",
            "--record-envelopes",
            str(destination),
        ]
    )

    report = json.loads(
        (output / "carsky_edge_report.json").read_text(encoding="utf-8")
    )
    assert result == 1
    assert not destination.exists()
    assert report["summaries"][0]["carsky_transport"][
        "recording_committed"
    ] is False
