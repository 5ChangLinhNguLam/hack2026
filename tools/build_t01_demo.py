#!/usr/bin/env python3
"""Build an honest T01-Sample GT replay for the CarSky Screen and broker.

The generated artifacts are demo/reference material, not model predictions.
The dashboard is downsampled to 5 Hz for a compact ADB payload; the CarSky
Script Node keeps all source frames and publishes at the native 20 Hz.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tripkit import TripLoader
from tripkit.types import parse_kitti_label_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIP = ROOT / "data" / "T01-Sample"
DEFAULT_TEMPLATE = ROOT / "carsky" / "screen" / "safeloop_dashboard.html"
DEFAULT_HTML = ROOT / "carsky" / "screen" / "safeloop_t01_dashboard.html"
DEFAULT_LUA = ROOT / "carsky" / "scripts" / "safeloop_t01_gt_replay.lua"


def _number(value: object, digits: int = 3) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "E"


def _pedestrian_distance(loader: TripLoader, frame_id: int) -> float | None:
    distances = [
        label.z
        for label in parse_kitti_label_file(loader.label_path(frame_id))
        if label.type.lower() == "pedestrian" and label.z > 0
    ]
    return round(min(distances), 2) if distances else None


def build_trace(loader: TripLoader, step: int = 4) -> list[list]:
    """Build compact trace rows while accumulating the official C3 formula."""

    if step < 1:
        raise ValueError("step must be >= 1")
    if not loader.has_gt():
        raise ValueError("T01 demo requires a practice trip with ground truth")

    counts = {"brake": 0, "accel": 0, "corner": 0, "near": 0, "speed": 0, "tail": 0}
    rows: list[list] = []
    for frame_id in range(loader.n_frames):
        raw = loader.raw_frame(frame_id)
        flags = raw.get("behavior_flags") or {}
        counts["brake"] += int(bool(flags.get("harsh_brake")))
        counts["accel"] += int(bool(flags.get("harsh_accel")))
        counts["corner"] += int(bool(flags.get("harsh_corner")))
        counts["speed"] += int(bool(flags.get("speeding")))
        counts["tail"] += int(bool(flags.get("tailgating")))
        ttc = _number(raw.get("min_ttc"))
        counts["near"] += int(ttc is not None and ttc < 1.5)

        if frame_id % step and frame_id != loader.n_frames - 1:
            continue

        elapsed_frames = frame_id + 1
        speeding_pct = 100.0 * counts["speed"] / elapsed_frames
        tailgating_pct = 100.0 * counts["tail"] / elapsed_frames
        maneuver_penalty = 3 * counts["brake"] + 2 * counts["accel"] + 2 * counts["corner"]
        collision_penalty = 5 * counts["near"]
        compliance_penalty = 0.15 * speeding_pct + 0.10 * tailgating_pct
        score = max(0.0, 100.0 - maneuver_penalty - collision_penalty - compliance_penalty)

        ego = raw.get("ego") or {}
        driver = raw.get("driver") or {}
        state = str(driver.get("state") or "unknown")
        attention = round(100 * float(driver.get("alertness_score") or 0), 1)
        distraction = round(max(0.0, 100.0 - attention), 1)
        fatigue = round(float((loader.driver_summary or {}).get("fatigue_score") or 0), 1)
        eyes_on_road = driver.get("eye_state") == "open" and driver.get("head_pose") == "normal"
        risk = _number((raw.get("risk") or {}).get("final_risk_score"), 1) or 0.0
        event_active = any(
            event.get("event_type") == "pedestrian_jaywalk"
            for event in (raw.get("events_active") or [])
        )
        rows.append(
            [
                round(float(raw.get("timestamp", frame_id / loader.fps)) * 1000),
                frame_id,
                _number(ego.get("speed_kmh"), 2) or 0.0,
                _number(ego.get("longitudinal_accel")) or 0.0,
                _number(ego.get("lateral_accel")) or 0.0,
                ttc,
                _pedestrian_distance(loader, frame_id),
                risk,
                state,
                attention,
                distraction,
                fatigue,
                eyes_on_road,
                round(score, 1),
                _grade(score),
                round(max(0.0, 100.0 - collision_penalty), 1),
                round(max(0.0, 100.0 - maneuver_penalty), 1),
                round(max(0.0, 100.0 - compliance_penalty), 1),
                event_active,
            ]
        )
    return rows


def _dashboard_script(trace: list[list], duration_ms: int) -> str:
    payload = json.dumps(trace, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f"""
  const trace={payload},duration={duration_ms};
  const $=id=>document.getElementById(id),clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const obstacle=document.querySelector('.obstacle');let started=performance.now();
  function render(now){{
    const elapsed=(now-started)%duration;
    let lo=0,hi=trace.length-1;
    while(lo<hi){{const mid=Math.ceil((lo+hi)/2);if(trace[mid][0]<=elapsed)lo=mid;else hi=mid-1}}
    const f=trace[lo];
    const [time,frame,speed,lon,lat,ttc,distance,risk,state,attention,distraction,fatigue,eyes,c3,grade,collision,maneuver,compliance,eventActive]=f;
    let level='SAFE',action='MONITOR',reason='NORMAL',brake=0,color='var(--cyan)';
    if(ttc!==null&&ttc<=1.2){{level='CRITICAL';action='EMERGENCY BRAKE REQUEST';reason='PEDESTRIAN + LOW TTC';brake=70;color='var(--red)'}}
    else if((ttc!==null&&ttc<1.5)||risk>=50){{level='HIGH';action='VISUAL · AUDIO · HAPTIC WARNING';reason='PEDESTRIAN NEAR-MISS';color='var(--red)'}}
    else if(state!=='alert'){{level='CAUTION';action='DRIVER ATTENTION WARNING';reason='DRIVER DISTRACTED';color='var(--amber)'}}
    else if(eventActive&&time<19000){{level='CAUTION';action='PEDESTRIAN TRACKING';reason='PEDESTRIAN JAYWALK';color='var(--amber)'}}
    document.documentElement.style.setProperty('--risk',color);
    $('ttc').textContent=ttc===null?'--':ttc.toFixed(2);$('distance').textContent=distance===null?'--':distance.toFixed(1);$('roadDistance').textContent=distance===null?'NO TARGET':distance.toFixed(1)+' m';
    obstacle.style.opacity=distance===null?'0':'1';if(distance!==null)obstacle.style.top=clamp(42-distance,15,36)+'%';
    $('speed').textContent=Math.round(speed);$('c1trend').textContent=ttc===null?'NO THREAT':ttc<1.5?'COLLISION WARNING':'TRACKING';$('c1trend').className='trend'+(ttc!==null&&ttc<1.5?' warn':'');
    $('riskScore').textContent=String(Math.round(risk)).padStart(2,'0');$('riskLevel').textContent=level;$('roadRisk').textContent=level;$('reason').textContent=reason;$('brake').textContent=brake+'%';
    $('action').textContent=action;$('footerReason').textContent='Frame '+frame+' · '+reason.replaceAll('_',' ');
    $('driverState').textContent=state;$('eyes').textContent=eyes?'◉':'—';
    [['attention',attention],['distraction',distraction],['fatigue',fatigue]].forEach(([name,value])=>{{$(`${{name}}Text`).textContent=Math.round(value)+'%';$(`${{name}}Bar`).style.width=value+'%';$(`${{name}}Bar`).style.background=value>50&&name!=='attention'?'var(--red)':'var(--cyan)'}});
    $('c3Score').textContent=Math.round(c3);$('grade').textContent=grade;$('scoreRing').style.setProperty('--score',c3);$('collisionPart').textContent=Math.round(collision);$('driverPart').textContent=Math.round(maneuver);$('comfortPart').textContent=Math.round(compliance);
    $('clock').textContent='T+'+(time/1000).toFixed(1)+' / 30s';requestAnimationFrame(render)
  }}
  requestAnimationFrame(render);
""".strip()


def build_dashboard(template: str, loader: TripLoader, step: int = 4) -> str:
    trace = build_trace(loader, step=step)
    duration_ms = round(loader.n_frames / loader.fps * 1000)
    html = template
    replacements = {
        "INTELLIGENT SAFETY CO-PILOT": "T01-SAMPLE · TOWN10HD · 30 SEC @ 20 FPS",
        "MOCK PRODUCT SLICE": "T01-SAMPLE · GT REPLAY",
        "<span id=\"distance\">100.0</span> m": "<span id=\"distance\">--</span> m",
        "<span>Confidence</span><b>91%</b>": "<span>Data source</span><b>GROUND TRUTH</b>",
        "<span>Collision</span>": "<span>Near-miss</span>",
        "<span>Driver</span>": "<span>Maneuver</span>",
        "<span>Comfort</span>": "<span>Compliance</span>",
        '<div class="obstacle">': '<div class="obstacle" id="obstacle">',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    html, count = re.subn(
        r"<script>.*?</script>",
        "<script>\n" + _dashboard_script(trace, duration_ms) + "\n</script>",
        html,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("dashboard template must contain exactly one script block")
    return html


def build_lua(loader: TripLoader) -> str:
    fatigue = round(float((loader.driver_summary or {}).get("fatigue_score") or 0), 1)
    rows = []
    for frame_id in range(loader.n_frames):
        raw = loader.raw_frame(frame_id)
        ego = raw.get("ego") or {}
        driver = raw.get("driver") or {}
        state = str(driver.get("state") or "unknown")
        attention = round(100 * float(driver.get("alertness_score") or 0), 1)
        distraction = round(max(0.0, 100.0 - attention), 1)
        eyes = driver.get("eye_state") == "open" and driver.get("head_pose") == "normal"
        ttc = _number(raw.get("min_ttc"))
        distance = _pedestrian_distance(loader, frame_id)
        row = (
            f"  {{{_number(ego.get('speed_kmh'), 3) or 0:.3f},"
            f"{_number(ego.get('longitudinal_accel'), 3) or 0:.3f},"
            f"{_number(ego.get('lateral_accel'), 3) or 0:.3f},"
            f"{round(ttc * 1000) if ttc is not None else 60000},"
            f"{distance or 0:.2f},{str(ttc is not None and ttc < 1.5).lower()},"
            f"{attention:.1f},{distraction:.1f},{fatigue:.1f},{str(eyes).lower()},"
            f"{str(state != 'alert').lower()}}}, -- frame={frame_id}"
        )
        rows.append(row)
    frames = "\n".join(rows)
    return f"""-- Generated from T01-Sample ground truth. DEMO/REFERENCE ONLY.
-- Native replay: 600 frames, 30 seconds, 20 Hz. This is not model inference.
local kuksa = pins.kuksa
assert(kuksa and kuksa.vss, "pins.kuksa.vss missing")
local Vehicle = kuksa.vss.Vehicle
local frames = {{
{frames}
}}
local index = 1
timer.periodic(50, function()
  local f = frames[index]
  Vehicle.Speed:publish(f[1])
  Vehicle.Acceleration.Longitudinal:publish(f[2])
  Vehicle.Acceleration.Lateral:publish(f[3])
  Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap:publish(f[4])
  Vehicle.ADAS.ObstacleDetection.Front.Center.Distance:publish(f[5])
  Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning:publish(f[6])
  Vehicle.Driver.AttentiveProbability:publish(f[7])
  Vehicle.Driver.DistractionLevel:publish(f[8])
  Vehicle.Driver.FatigueLevel:publish(f[9])
  Vehicle.Driver.IsEyesOnRoad:publish(f[10])
  Vehicle.ADAS.DMS.IsWarning:publish(f[11])
  if index % 20 == 0 then log(string.format("[SafeLoop T01 GT] frame=%d/599", index-1)) end
  index = index + 1
  if index > #frames then index = 1; log("[SafeLoop T01 GT] replay loop") end
end)
log("[SafeLoop T01 GT] ready; frames=600; fps=20; model_inference=false")
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build T01 GT Screen + CarSky Lua replay")
    parser.add_argument("--trip-dir", type=Path, default=DEFAULT_TRIP)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--html-output", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--lua-output", type=Path, default=DEFAULT_LUA)
    parser.add_argument("--screen-step", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = TripLoader(args.trip_dir)
    dashboard = build_dashboard(args.template.read_text(encoding="utf-8"), loader, args.screen_step)
    lua = build_lua(loader)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.lua_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(dashboard, encoding="utf-8")
    args.lua_output.write_text(lua, encoding="utf-8")
    print(json.dumps({
        "trip_id": loader.trip_id,
        "source_frames": loader.n_frames,
        "screen_frames": len(build_trace(loader, args.screen_step)),
        "html": str(args.html_output),
        "lua": str(args.lua_output),
        "gt_replay": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
