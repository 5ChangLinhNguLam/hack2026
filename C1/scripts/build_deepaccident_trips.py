"""Convert DeepAccident into the same trip + TTC format as datasets/carcrash.

One trip = one agent's front-camera recording of one scenario. DeepAccident records
5 synchronised V2X agents per scenario (4 vehicles + 1 roadside unit), each with 6
cameras; we render Camera_Front at the native 10 Hz.

Where the collision time comes from
-----------------------------------
The scenario `meta/<scenario>.txt` first line is:

    <weather+time> <id1> <type1> <id2> <type2> <impulse> <direction> <impact> <raw_end>

`<raw_end>` is NOT the collision frame -- it is CARLA's end frame including 10
unsaved warm-up frames (verified: raw_end - saved_frames == 10 for every scenario).
`id1 == -1` means the scenario ends without a collision.

So the collision instant is recovered geometrically: the frame where the two
colliding objects' box centres are closest. Their separation falls monotonically
and bottoms out one or two frames before the recording ends, then ticks up as the
vehicles rebound -- that minimum is the impact.

TTC convention (identical to datasets/carcrash)
    ttc = (collision_frame - i) / 10   before impact
    ttc = 0.0                          from impact onward
    ttc = -1.0                         scenarios that never collide (sentinel)
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2

FPS = 10.0
NO_COLLISION = -1.0
AGENTS = ["ego_vehicle", "ego_vehicle_behind", "other_vehicle",
          "other_vehicle_behind", "infrastructure"]

GREEN = (80, 220, 90)
AMBER = (40, 190, 255)
RED = (60, 60, 240)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# --------------------------------------------------------------------------- labels

def parse_label(path: Path):
    """-> (ego_speed_xy, {vehicle_id: (x, y, z, speed)}).

    Field layout, per tools/data_converter/carla_converter.py:
      class x y z l w h yaw vx vy vehicle_id num_lidar_pts camera_visibility
    The recording agent itself is written with vehicle_id -100.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return (0.0, 0.0), {}
    head = lines[0].split(" ")
    ego = (float(head[0]), float(head[1]))
    objs = {}
    for ln in lines[1:]:
        f = ln.split(" ")
        if len(f) <= 1:
            continue
        objs[int(f[-3])] = (float(f[1]), float(f[2]), float(f[3]),
                            math.hypot(float(f[8]), float(f[9])))
    return ego, objs


def parse_meta(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    t = lines[0].split(" ")
    meta = {
        "weather_time": t[0],
        "obj1_id": int(t[1]), "obj1_type": t[2],
        "obj2_id": int(t[3]), "obj2_type": t[4],
        "impulse": t[5], "relative_direction": t[6], "impact_location": t[7],
        "raw_end_frame": int(t[8]),
        "agent_ids": [],
        "road_type": "", "ego_direction": "", "other_direction": "",
    }
    for ln in lines[1:]:
        if "agents id:" in ln:
            meta["agent_ids"] = [int(x) for x in ln.split(": ")[1].split(" ")]
        elif "road_type:" in ln:
            meta["road_type"] = ln.split(": ", 1)[1]
        elif "ego_vehicle_direction:" in ln:
            meta["ego_direction"] = ln.split(": ", 1)[1]
        elif "other_vehicle_direction:" in ln:
            meta["other_direction"] = ln.split(": ", 1)[1]
    return meta


def find_collision_frame(label_dir: Path, meta):
    """0-based frame of closest approach between the two colliding objects.

    Returns (frame, distances, method). Distances are centre-to-centre in metres,
    measured in the ego frame -- a relative distance, so it is the same physical
    quantity whichever agent recorded it.
    """
    files = sorted(label_dir.glob("*.txt"))
    if not files:
        return None, [], "no_labels"

    ego_id = meta["agent_ids"][0] if meta["agent_ids"] else None
    a = -100 if meta["obj1_id"] == ego_id else meta["obj1_id"]
    b = -100 if meta["obj2_id"] == ego_id else meta["obj2_id"]

    dists = []
    for p in files:
        _, objs = parse_label(p)
        if a in objs and b in objs:
            dists.append(math.dist(objs[a][:2], objs[b][:2]))
        else:
            dists.append(None)

    seen = [(i, d) for i, d in enumerate(dists) if d is not None]
    if not seen:
        # e.g. a collision with static map geometry, whose id never appears as a box
        return len(files) - 1, dists, "fallback_last_frame"
    return min(seen, key=lambda t: t[1])[0], dists, "closest_approach"


def load_splits(*dirs: Path):
    """(scenario_type, scenario) -> train | val | test. First directory wins."""
    out = {}
    for name in ("train", "val", "test"):
        p = next((d / f"{name}.txt" for d in dirs if (d / f"{name}.txt").exists()), None)
        if p is None:
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) == 2:
                out[(parts[0], parts[1])] = name
    return out


# --------------------------------------------------------------------------- render

def ttc_for_frame(i: int, collision_frame):
    if collision_frame is None:
        return NO_COLLISION
    if i >= collision_frame:
        return 0.0
    return round((collision_frame - i) / FPS, 3)


def ttc_color(ttc):
    if ttc == NO_COLLISION:
        return GREY
    if ttc < 0.5:
        return RED
    if ttc < 1.5:
        return AMBER
    return GREEN


def ttc_text(ttc):
    if ttc == NO_COLLISION:
        return "NEGATIVE - no collision"
    if ttc <= 0.0:
        return "COLLISION"
    return f"TTC: {ttc:.1f}s"


def draw_overlay(frame, trip_label, agent, i, n, ttc, dist):
    h, w = frame.shape[:2]
    s = w / 1600.0
    pad = int(20 * s)
    bar = int(96 * s)
    strip = frame[:bar].copy()
    cv2.rectangle(frame, (0, 0), (w, bar), BLACK, -1)
    cv2.addWeighted(strip, 0.35, frame[:bar], 0.65, 0, frame[:bar])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{trip_label}  [{agent}]  frame {i + 1}/{n}",
                (pad, int(32 * s)), font, 0.62 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(frame, ttc_text(ttc), (pad, int(72 * s)), font,
                1.0 * s, ttc_color(ttc), max(1, int(2 * s)), cv2.LINE_AA)
    if dist is not None:
        cv2.putText(frame, f"gap {dist:.1f} m", (int(w * 0.55), int(72 * s)), font,
                    0.75 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    if ttc != NO_COLLISION:
        bw, bh = int(340 * s), int(14 * s)
        bx, by = w - bw - pad, int(40 * s)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        frac = max(0.0, min(1.0, ttc / 3.0))
        cv2.rectangle(frame, (bx, by), (bx + int(bw * (1 - frac)), by + bh),
                      ttc_color(ttc), -1)
    return frame


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/datasets/deepaccident")
    ap.add_argument("--overlay", dest="overlay", action="store_true", default=True)
    ap.add_argument("--no-overlay", dest="overlay", action="store_false")
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios")
    args = ap.parse_args()

    out: Path = args.out or args.root / "datasets" / "deepaccident"
    raw = out / "raw"
    mini = raw / "mini"
    if not mini.is_dir():
        raise SystemExit(f"not found: {mini} -- extract DeepAccident_mini.zip there first")

    # Mirror datasets/carcrash: the authors' own annotation files live in annotations/,
    # lifted out of raw/ so they are readable without digging through the archive.
    ann = out / "annotations"
    (ann / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        src = raw / f"{name}.txt"
        if src.exists() and not (ann / f"{name}.txt").exists():
            shutil.move(str(src), str(ann / f"{name}.txt"))
    for m in sorted(mini.glob("*/meta/*.txt")):
        shutil.copy2(m, ann / "meta" / f"{m.parts[-3]}__{m.name}")

    splits = load_splits(ann, raw)
    scenarios = sorted(mini.glob("*/meta/*.txt"))
    if args.limit:
        scenarios = scenarios[:args.limit]
    print(f"{len(scenarios)} scenarios, split entries: {len(splits)}")

    for cls in ("positive", "negative"):
        (out / "trips" / cls).mkdir(parents=True, exist_ok=True)
        if args.overlay:
            (out / "overlay" / cls).mkdir(parents=True, exist_ok=True)

    mf = (out / "trip_manifest.csv").open("w", newline="", encoding="utf-8")
    ff = (out / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8")
    mw, fw = csv.writer(mf), csv.writer(ff)
    mw.writerow([
        "trip_id", "class", "label", "split", "scenario_type", "scenario", "agent",
        "clean_path", "overlay_path", "num_frames", "fps", "width", "height",
        "collision_frame", "ttc_at_trip_start_s", "collision_method",
        "obj1_id", "obj1_type", "obj2_id", "obj2_type", "impact_location",
        "relative_direction", "impulse", "weather_time", "road_type",
        "ego_direction", "other_direction",
    ])
    fw.writerow(["trip_id", "class", "split", "frame_idx", "time_s",
                 "binlabel", "ttc_s", "pair_distance_m", "ego_speed_mps"])

    # One row per scenario -- the DeepAccident counterpart of carcrash/Crash_Table.csv.
    sf = (ann / "scenario_table.csv").open("w", newline="", encoding="utf-8")
    sw = csv.writer(sf)
    sw.writerow([
        "scenario_type", "scenario", "split", "has_collision", "num_frames", "fps",
        "collision_frame", "ttc_at_scenario_start_s", "collision_method",
        "raw_end_frame", "raw_end_minus_frames", "min_pair_distance_m",
        "obj1_id", "obj1_type", "obj2_id", "obj2_type", "impact_location",
        "relative_direction", "impulse", "weather_time", "road_type",
        "ego_direction", "other_direction", "agent_ids",
    ])

    n_trips = n_frames = 0
    problems = []

    for n, meta_path in enumerate(scenarios, 1):
        stype = meta_path.parts[-3]
        scen = meta_path.stem
        meta = parse_meta(meta_path)
        split = splits.get((stype, scen), "unassigned")

        has_collision = meta["obj1_id"] != -1
        cls = "positive" if has_collision else "negative"
        label_dir = mini / stype / "ego_vehicle" / "label" / scen

        if has_collision:
            cframe, dists, method = find_collision_frame(label_dir, meta)
        else:
            cframe, dists, method = None, [], "no_collision"

        for agent in AGENTS:
            cam = mini / stype / agent / "Camera_Front" / scen
            imgs = sorted(cam.glob("*.jpg"))
            if not imgs:
                problems.append(f"{stype}/{scen}/{agent}: no Camera_Front frames")
                continue

            trip_id = f"{stype}__{scen}__{agent}"
            clean = out / "trips" / cls / f"{trip_id}.mp4"
            ov = out / "overlay" / cls / f"{trip_id}.mp4" if args.overlay else None

            first = cv2.imread(str(imgs[0]))
            if first is None:
                problems.append(f"{trip_id}: unreadable first frame")
                continue
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            wc = cv2.VideoWriter(str(clean), fourcc, FPS, (w, h))
            wo = cv2.VideoWriter(str(ov), fourcc, FPS, (w, h)) if ov else None

            ego_speeds = []
            for i, ip in enumerate(imgs):
                img = cv2.imread(str(ip))
                if img is None:
                    problems.append(f"{trip_id}: unreadable frame {i}")
                    continue
                ttc = ttc_for_frame(i, cframe)
                binlabel = 0 if cframe is None else int(i >= cframe)
                dist = dists[i] if i < len(dists) else None

                lp = mini / stype / agent / "label" / scen / f"{scen}_{i + 1:03d}.txt"
                espeed = ""
                if lp.exists():
                    ego, _ = parse_label(lp)
                    espeed = round(math.hypot(*ego), 3)
                    ego_speeds.append(espeed)

                wc.write(img)
                if wo is not None:
                    wo.write(draw_overlay(img.copy(), scen, agent, i, len(imgs), ttc, dist))
                fw.writerow([trip_id, cls, split, i, round(i / FPS, 3), binlabel, ttc,
                             "" if dist is None else round(dist, 3), espeed])
                n_frames += 1

            wc.release()
            if wo is not None:
                wo.release()

            mw.writerow([
                trip_id, cls, 1 if has_collision else 0, split, stype, scen, agent,
                clean.relative_to(out).as_posix(),
                ov.relative_to(out).as_posix() if ov else "",
                len(imgs), FPS, w, h,
                "" if cframe is None else cframe,
                ttc_for_frame(0, cframe), method,
                meta["obj1_id"], meta["obj1_type"], meta["obj2_id"], meta["obj2_type"],
                meta["impact_location"], meta["relative_direction"], meta["impulse"],
                meta["weather_time"], meta["road_type"],
                meta["ego_direction"], meta["other_direction"],
            ])
            n_trips += 1

        n_scen_frames = len(sorted(label_dir.glob("*.txt")))
        finite = [d for d in dists if d is not None]
        sw.writerow([
            stype, scen, split, int(has_collision), n_scen_frames, FPS,
            "" if cframe is None else cframe,
            ttc_for_frame(0, cframe), method,
            meta["raw_end_frame"], meta["raw_end_frame"] - n_scen_frames,
            "" if not finite else round(min(finite), 3),
            meta["obj1_id"], meta["obj1_type"], meta["obj2_id"], meta["obj2_type"],
            meta["impact_location"], meta["relative_direction"], meta["impulse"],
            meta["weather_time"], meta["road_type"],
            meta["ego_direction"], meta["other_direction"],
            " ".join(str(x) for x in meta["agent_ids"]),
        ])

        print(f"  [{n}/{len(scenarios)}] {stype}/{scen} -> {cls}"
              f"{'' if cframe is None else f', collision @ frame {cframe} ({method})'}")

    mf.close()
    ff.close()
    sf.close()
    print(f"annotations -> {ann}")
    print(f"\n{n_trips} trips, {n_frames} frames -> {out}")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("no problems")


if __name__ == "__main__":
    main()
