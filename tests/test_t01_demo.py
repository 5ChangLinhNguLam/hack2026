"""Checks for the T01-Sample GT demo artifacts."""

from __future__ import annotations

from pathlib import Path

from tools.build_t01_demo import build_dashboard, build_lua, build_trace
from tripkit import TripLoader


def test_t01_trace_is_gt_based_and_reaches_official_final_score():
    root = Path(__file__).resolve().parents[1]
    loader = TripLoader(root / "data/T01-Sample")
    trace = build_trace(loader, step=4)

    assert len(trace) == 151
    assert trace[0][8] == "distracted"
    assert next(row for row in trace if row[1] == 300)[8] == "alert"
    assert min(row[5] for row in trace if row[5] is not None) < 1.1
    assert trace[-1][13:15] == [0.0, "E"]
    assert trace[-1][13] == loader.trip_aggregate["safe_driving_score"]


def test_dashboard_is_self_contained_and_never_serializes_infinity():
    root = Path(__file__).resolve().parents[1]
    loader = TripLoader(root / "data/T01-Sample")
    template = (root / "carsky/screen/safeloop_dashboard.html").read_text()

    html = build_dashboard(template, loader)

    assert "T01-SAMPLE · GT REPLAY" in html
    assert "GROUND TRUTH" in html
    assert "PEDESTRIAN JAYWALK" in html
    assert "Infinity" not in html
    assert html.count("<script>") == 1


def test_lua_keeps_all_native_20hz_frames_and_gt_marker():
    root = Path(__file__).resolve().parents[1]
    loader = TripLoader(root / "data/T01-Sample")

    lua = build_lua(loader)

    assert lua.count("-- frame=") == 600
    assert "timer.periodic(50" in lua
    assert "model_inference=false" in lua
    assert "[SafeLoop T01 GT]" in lua
