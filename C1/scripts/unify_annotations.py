"""Emit one shared annotation schema across every dataset under datasets/.

Each dataset keeps its own manifest, because each carries fields the other simply
does not have (CCD has `egoinvolve` and a YouTube id; DeepAccident has 3D collider
ids, impact geometry and impulse). This script projects both onto a common core so
they can be loaded, filtered and trained on as a single corpus:

  datasets/trips.csv    one row per trip, identical columns for every dataset
  datasets/frames.csv   one row per (trip, frame), identical columns
  datasets/SCHEMA.md    the column contract and the per-dataset mapping

Nothing is invented. A field a dataset genuinely lacks is left empty rather than
filled with a plausible-looking default.
"""

import argparse
import csv
from pathlib import Path

TRIP_COLUMNS = [
    "trip_id",            # globally unique: "<dataset>/<local id>"
    "dataset",            # carcrash | deepaccident
    "source",             # real | simulated
    "class",              # positive | negative
    "label",              # 1 | 0  (trip-level classification target)
    "split",              # train | val | test | unassigned
    "clean_path",         # relative to datasets/
    "overlay_path",
    "num_frames",
    "fps",
    "width",
    "height",
    "collision_frame",    # 0-based frame of impact; empty for negatives
    "ttc_at_trip_start_s",
    "collision_method",   # how collision_frame was established
    "ego_involved",       # 1 if the recording agent is itself in the collision
    "timing",             # Day | Night
    "weather",            # Clear | Rainy | Snowy | Cloudy | Wet
    "agent",              # which sensor rig recorded this trip
]

FRAME_COLUMNS = [
    "trip_id", "dataset", "class", "split",
    "frame_idx", "time_s", "binlabel", "ttc_s",
    "pair_distance_m",    # DeepAccident only; empty for CCD
    "ego_speed_mps",      # DeepAccident only; empty for CCD
]


def read_csv(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------- normalisation

def norm_weather_carcrash(w: str) -> str:
    # CCD calls a clear day "Normal"
    return {"Normal": "Clear", "Rainy": "Rainy", "Snowy": "Snowy"}.get(w, w)


def norm_weather_time_deepaccident(wt: str):
    """'HardRainNight' -> ('Night', 'Rainy'). CARLA presets concatenate the two."""
    timing = "Night" if "Night" in wt else "Day"          # Noon and Sunset are daylight
    if "Rain" in wt:
        weather = "Rainy"
    elif "Cloud" in wt:
        weather = "Cloudy"
    elif "Wet" in wt:
        weather = "Wet"                                    # wet road, no active rain
    else:
        weather = "Clear"
    return timing, weather


# --------------------------------------------------------------------- loaders

def load_carcrash(root: Path):
    d = root / "carcrash"
    trips, frames = [], []
    for r in read_csv(d / "trip_manifest.csv"):
        tid = f"carcrash/{r['trip_id']}"
        trips.append({
            "trip_id": tid,
            "dataset": "carcrash",
            "source": "real",
            "class": r["class"],
            "label": r["label"],
            "split": r["split"],
            "clean_path": f"carcrash/{r['clean_path']}",
            "overlay_path": f"carcrash/{r['overlay_path']}" if r["overlay_path"] else "",
            "num_frames": r["num_frames"],
            "fps": r["fps"],
            "width": r["width"],
            "height": r["height"],
            # CCD's accident_start_frame is exactly "the frame impact begins"
            "collision_frame": r["accident_start_frame"],
            "ttc_at_trip_start_s": r["ttc_at_trip_start_s"],
            "collision_method": ("binary_frame_annotation" if r["class"] == "positive"
                                 else "no_collision"),
            # A negative trip contains no collision at all, so the recording vehicle
            # is definitionally not in one -- 0, not "unknown". CCD only annotates
            # egoinvolve on its crash clips.
            "ego_involved": (0 if r["class"] == "negative"
                             else {"Yes": 1, "No": 0}.get(r["egoinvolve"], "")),
            "timing": r["timing"],
            "weather": norm_weather_carcrash(r["weather"]),
            "agent": "dashcam",
        })
    for r in read_csv(d / "ttc_per_frame.csv"):
        frames.append({
            "trip_id": f"carcrash/{r['trip_id']}",
            "dataset": "carcrash",
            "class": r["class"],
            "split": r["split"],
            "frame_idx": r["frame_idx"],
            "time_s": r["time_s"],
            "binlabel": r["binlabel"],
            "ttc_s": r["ttc_s"],
            "pair_distance_m": "",
            "ego_speed_mps": "",
        })
    return trips, frames


def load_deepaccident(root: Path):
    d = root / "deepaccident"
    if not (d / "trip_manifest.csv").exists():
        return [], []

    # agent -> which entry of the scenario's "agents id" line is that agent's own id
    agent_slot = {"ego_vehicle": 0, "ego_vehicle_behind": 1,
                  "other_vehicle": 2, "other_vehicle_behind": 3}
    scen_agents = {}
    st = d / "annotations" / "scenario_table.csv"
    if st.exists():
        for r in read_csv(st):
            scen_agents[(r["scenario_type"], r["scenario"])] = r["agent_ids"].split()

    trips, frames = [], []
    for r in read_csv(d / "trip_manifest.csv"):
        tid = f"deepaccident/{r['trip_id']}"
        timing, weather = norm_weather_time_deepaccident(r["weather_time"])

        ego_involved = ""
        ids = scen_agents.get((r["scenario_type"], r["scenario"]))
        if r["class"] == "negative":
            ego_involved = 0
        elif r["agent"] == "infrastructure":
            ego_involved = 0                      # a roadside camera never collides
        elif ids and r["agent"] in agent_slot:
            own = ids[agent_slot[r["agent"]]]
            ego_involved = int(own in (r["obj1_id"], r["obj2_id"]))

        trips.append({
            "trip_id": tid,
            "dataset": "deepaccident",
            "source": "simulated",
            "class": r["class"],
            "label": r["label"],
            "split": r["split"],
            "clean_path": f"deepaccident/{r['clean_path']}",
            "overlay_path": f"deepaccident/{r['overlay_path']}" if r["overlay_path"] else "",
            "num_frames": r["num_frames"],
            "fps": r["fps"],
            "width": r["width"],
            "height": r["height"],
            "collision_frame": r["collision_frame"],
            "ttc_at_trip_start_s": r["ttc_at_trip_start_s"],
            "collision_method": r["collision_method"],
            "ego_involved": ego_involved,
            "timing": timing,
            "weather": weather,
            "agent": r["agent"],
        })
    for r in read_csv(d / "ttc_per_frame.csv"):
        frames.append({
            "trip_id": f"deepaccident/{r['trip_id']}",
            "dataset": "deepaccident",
            "class": r["class"],
            "split": r["split"],
            "frame_idx": r["frame_idx"],
            "time_s": r["time_s"],
            "binlabel": r["binlabel"],
            "ttc_s": r["ttc_s"],
            "pair_distance_m": r["pair_distance_m"],
            "ego_speed_mps": r["ego_speed_mps"],
        })
    return trips, frames


# ------------------------------------------------------------------------ main

def write(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "datasets")
    args = ap.parse_args()
    root: Path = args.root

    trips, frames = [], []
    for loader in (load_carcrash, load_deepaccident):
        t, f = loader(root)
        trips += t
        frames += f
        if t:
            print(f"{t[0]['dataset']}: {len(t)} trips, {len(f)} frames")

    # every trip must resolve to a video that actually exists
    missing = [t["trip_id"] for t in trips if not (root / t["clean_path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} trips point at a missing video, "
                         f"e.g. {missing[:3]}")

    ids = [t["trip_id"] for t in trips]
    if len(set(ids)) != len(ids):
        raise SystemExit("trip_id collision across datasets")

    known = {t["trip_id"] for t in trips}
    orphan = {f["trip_id"] for f in frames} - known
    if orphan:
        raise SystemExit(f"{len(orphan)} frame rows reference unknown trips")

    write(root / "trips.csv", TRIP_COLUMNS, trips)
    write(root / "frames.csv", FRAME_COLUMNS, frames)

    pos = sum(1 for t in trips if t["class"] == "positive")
    print(f"\ntotal: {len(trips)} trips ({pos} positive / {len(trips) - pos} negative), "
          f"{len(frames)} frames")


if __name__ == "__main__":
    main()
