"""Build datasets/combined -- every trip from every dataset in one place.

Input  : one mp4 per trip (native pixels, nothing re-encoded)
Label  : time-to-collision per frame

Videos are hardlinked from the source datasets, so the combined set costs no extra
disk and its files are byte-identical to the originals.

Layout
  combined/trips/positive/<trip>.mp4
  combined/trips/negative/<trip>.mp4
  combined/annotations/trips.csv          one row per trip
  combined/annotations/ttc_per_frame.csv  one row per (trip, frame)
  combined/annotations/{train,val,test}.txt   "<trip_id> <label>" per line

Run scripts/unify_annotations.py first -- this reads datasets/trips.csv and
datasets/frames.csv.
"""

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def safe_name(trip_id: str) -> str:
    """'carcrash/positive_000001' -> 'carcrash__positive_000001' (a valid filename)."""
    return trip_id.replace("/", "__")


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def read_csv(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "datasets")
    ap.add_argument("--with-overlay", action="store_true",
                    help="also link the label-burned-in videos (inspection only)")
    ap.add_argument("--ego-only", action="store_true",
                    help="keep only trips where the recording vehicle is itself in the "
                         "collision; drops crashes merely filmed from a passing car")
    ap.add_argument("--name", default=None, help="output folder name under datasets/")
    args = ap.parse_args()

    root: Path = args.root
    out = root / (args.name or ("combined_ego" if args.ego_only else "combined"))
    trips_csv, frames_csv = root / "trips.csv", root / "frames.csv"
    for p in (trips_csv, frames_csv):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run scripts/unify_annotations.py first")

    trips = read_csv(trips_csv)
    frames = read_csv(frames_csv)
    print(f"{len(trips)} trips, {len(frames)} frames")

    if args.ego_only:
        # positives must have the recorder in the crash; negatives have no crash at
        # all, so they stay -- they are still valid "nothing happened to me" examples
        before = len(trips)
        keep = {t["trip_id"] for t in trips
                if t["class"] == "negative" or t["ego_involved"] == "1"}
        trips = [t for t in trips if t["trip_id"] in keep]
        frames = [f for f in frames if f["trip_id"] in keep]
        dropped = before - len(trips)
        print(f"--ego-only: bỏ {dropped} trip có va chạm nhưng xe quay không tham gia; "
              f"còn {len(trips)} trip, {len(frames)} frame")

    for cls in ("positive", "negative"):
        (out / "trips" / cls).mkdir(parents=True, exist_ok=True)
        if args.with_overlay:
            (out / "overlay" / cls).mkdir(parents=True, exist_ok=True)
    ann = out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    rows = []
    for t in trips:
        name = safe_name(t["trip_id"])
        src = root / t["clean_path"]
        if not src.exists():
            raise SystemExit(f"{t['trip_id']}: source video missing at {src}")
        dst = out / "trips" / t["class"] / f"{name}.mp4"
        counts[link_or_copy(src, dst)] += 1

        overlay_rel = ""
        if args.with_overlay and t["overlay_path"]:
            osrc = root / t["overlay_path"]
            if osrc.exists():
                odst = out / "overlay" / t["class"] / f"{name}.mp4"
                link_or_copy(osrc, odst)
                overlay_rel = odst.relative_to(out).as_posix()

        r = dict(t)
        r["trip_id"] = name                       # filename and id now agree
        r["video"] = dst.relative_to(out).as_posix()
        r["overlay"] = overlay_rel
        r["source_dataset_path"] = t["clean_path"]
        del r["clean_path"], r["overlay_path"]
        rows.append(r)

    trip_cols = (["trip_id", "video", "dataset", "source", "class", "label", "split",
                  "num_frames", "fps", "width", "height", "collision_frame",
                  "ttc_at_trip_start_s", "collision_method", "ego_involved",
                  "timing", "weather", "agent", "overlay", "source_dataset_path"])
    with (ann / "trips.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=trip_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {ann / 'trips.csv'} ({len(rows)} rows)")

    frame_cols = ["trip_id", "dataset", "class", "split", "frame_idx", "time_s",
                  "binlabel", "ttc_s", "pair_distance_m", "ego_speed_mps"]
    n = 0
    with (ann / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=frame_cols)
        w.writeheader()
        for f in frames:
            f = dict(f)
            f["trip_id"] = safe_name(f["trip_id"])
            w.writerow(f)
            n += 1
    print(f"wrote {ann / 'ttc_per_frame.csv'} ({n} rows)")

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    for split, items in sorted(by_split.items()):
        p = ann / f"{split}.txt"
        p.write_text("".join(f"{r['trip_id']} {r['label']}\n" for r in sorted(
            items, key=lambda x: x["trip_id"])), encoding="utf-8")
        print(f"  {split}: {len(items)} trips -> {p.name}")

    # every declared frame count must match the label table
    declared = sum(int(r["num_frames"]) for r in rows)
    if declared != n:
        raise SystemExit(f"manifest declares {declared} frames but label table has {n}")

    print(f"\nlinked: {dict(counts)}")
    print(f"class: {dict(Counter(r['class'] for r in rows))}")
    print(f"dataset: {dict(Counter(r['dataset'] for r in rows))}")
    print(f"combined dataset at {out}")


if __name__ == "__main__":
    main()
