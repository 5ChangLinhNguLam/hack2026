"""Build the CCD trip dataset: one trip per source clip, positive and negative.

Each CCD clip is a single continuous 5 s dashcam segment (50 frames @ 10 fps), so
a clip *is* a trip -- nothing is concatenated and no frame is dropped.

Layout produced under --out (default: datasets/carcrash/):
  trips/positive/000001.mp4      1500 crash trips   (hardlinked, pixels untouched)
  trips/negative/000001.mp4      3000 normal trips  (hardlinked)
  overlay/positive/000001.mp4    same trips with the TTC label burned in
  overlay/negative/000001.mp4
  trip_manifest.csv              one row per trip
  ttc_per_frame.csv              one row per (trip, frame) -- 225,000 rows

TTC convention
  positive: ttc = (accident_start_frame - frame_idx) / 10 s, clipped to 0 once the
            collision has begun
  negative: ttc = -1.0 (sentinel: no collision in this trip)

Every clip is decoded and its real frame count checked against the annotation, so
a truncated or unreadable file shows up as an error instead of silently shrinking
the dataset.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2

FPS = 10.0
N_FRAMES = 50
NO_COLLISION = -1.0

GREEN = (80, 220, 90)
AMBER = (40, 190, 255)
RED = (60, 60, 240)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def ttc_for_frame(frame_idx: int, accident_start):
    if accident_start is None:
        return NO_COLLISION
    if frame_idx >= accident_start:
        return 0.0
    return round((accident_start - frame_idx) / FPS, 3)


def ttc_color(ttc: float):
    if ttc == NO_COLLISION:
        return GREY
    if ttc < 0.5:
        return RED
    if ttc < 1.5:
        return AMBER
    return GREEN


def ttc_text(ttc: float) -> str:
    if ttc == NO_COLLISION:
        return "NEGATIVE - no collision"
    if ttc <= 0.0:
        return "COLLISION"
    return f"TTC: {ttc:.1f}s"


def draw_overlay(frame, vidname, cls, local_idx, ttc):
    h, w = frame.shape[:2]
    s = w / 1280.0
    pad = int(16 * s)
    bar_h = int(78 * s)

    strip = frame[:bar_h].copy()
    cv2.rectangle(frame, (0, 0), (w, bar_h), BLACK, -1)
    cv2.addWeighted(strip, 0.35, frame[:bar_h], 0.65, 0, frame[:bar_h])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{cls}/{vidname}  frame {local_idx + 1}/{N_FRAMES}",
                (pad, int(28 * s)), font, 0.62 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(frame, ttc_text(ttc), (pad, int(62 * s)), font,
                0.95 * s, ttc_color(ttc), max(1, int(2 * s)), cv2.LINE_AA)

    if ttc != NO_COLLISION:
        bw, bh = int(300 * s), int(12 * s)
        bx, by = w - bw - pad, int(34 * s)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        frac = max(0.0, min(1.0, ttc / 2.5))
        cv2.rectangle(frame, (bx, by), (bx + int(bw * (1.0 - frac)), by + bh),
                      ttc_color(ttc), -1)
    return frame


def load_crash_table(path: Path):
    """vidname -> annotation row."""
    with path.open(encoding="utf-8") as fh:
        return {r["vidname"]: r for r in csv.DictReader(fh)}


def load_official_split(root: Path):
    """(cls, vidname) -> 'train' | 'test', from the CCD feature split lists."""
    out = {}
    for name in ("train", "test"):
        p = root / "CarCrash" / f"{name}.txt"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0]                     # e.g. negative/001355.npz
            folder, stem = rel.split("/")
            cls = "positive" if folder == "positive" else "negative"
            out[(cls, Path(stem).stem)] = name
    return out


def link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink when possible (same volume, no extra bytes), else copy."""
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/datasets/carcrash")
    ap.add_argument("--overlay", dest="overlay", action="store_true", default=True)
    ap.add_argument("--no-overlay", dest="overlay", action="store_false")
    ap.add_argument("--classes", choices=["positive", "negative", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="first N trips per class")
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out or root / "datasets" / "carcrash"
    vroot = root / "CarCrash" / "videos"

    crash = load_crash_table(out / "annotations" / "Crash_Table.csv")
    splits = load_official_split(root)
    print(f"annotations: {len(crash)} crash videos; official split entries: {len(splits)}")

    trips = []
    if args.classes in ("positive", "both"):
        ps = sorted((vroot / "Crash-1500").glob("*.mp4"))
        trips += [("positive", p) for p in (ps[:args.limit] if args.limit else ps)]
    if args.classes in ("negative", "both"):
        ns = sorted((vroot / "Normal").glob("*.mp4"))
        trips += [("negative", p) for p in (ns[:args.limit] if args.limit else ns)]
    if not trips:
        raise SystemExit("no clips found -- is the dataset extracted?")

    for cls in ("positive", "negative"):
        (out / "trips" / cls).mkdir(parents=True, exist_ok=True)
        if args.overlay:
            (out / "overlay" / cls).mkdir(parents=True, exist_ok=True)

    manifest_path = out / "trip_manifest.csv"
    frames_path = out / "ttc_per_frame.csv"
    mf = manifest_path.open("w", newline="", encoding="utf-8")
    ff = frames_path.open("w", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    fw = csv.writer(ff)
    mw.writerow([
        "trip_id", "class", "label", "split", "clean_path", "overlay_path",
        "num_frames", "fps", "width", "height", "accident_start_frame",
        "accident_end_frame", "ttc_at_trip_start_s", "timing", "weather",
        "egoinvolve", "youtubeID", "source_startframe",
    ])
    fw.writerow(["trip_id", "class", "split", "frame_idx", "time_s", "binlabel", "ttc_s"])

    total_frames = 0
    problems = []
    n_link = n_copy = 0

    for n, (cls, src) in enumerate(trips, 1):
        vid = src.stem
        trip_id = f"{cls}_{vid}"
        ann = crash.get(vid) if cls == "positive" else None
        acc_start = int(ann["accident_start_frame"]) if ann else None
        acc_end = int(ann["accident_end_frame"]) if ann else None
        if cls == "positive" and ann is None:
            problems.append(f"{trip_id}: no annotation row")
            continue

        clean_dst = out / "trips" / cls / f"{vid}.mp4"
        r = link_or_copy(src, clean_dst)
        n_link += r == "link"
        n_copy += r == "copy"

        overlay_dst = out / "overlay" / cls / f"{vid}.mp4" if args.overlay else None
        writer = None

        cap = cv2.VideoCapture(str(src))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        local = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ttc = ttc_for_frame(local, acc_start)
            binlabel = 0 if acc_start is None else int(local >= acc_start)
            fw.writerow([trip_id, cls, splits.get((cls, vid), "unassigned"),
                         local, round(local / FPS, 3), binlabel, ttc])
            if overlay_dst is not None:
                if writer is None:
                    writer = cv2.VideoWriter(str(overlay_dst),
                                             cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise SystemExit(f"cannot open writer for {overlay_dst}")
                writer.write(draw_overlay(frame, vid, cls, local, ttc))
            local += 1
        cap.release()
        if writer is not None:
            writer.release()

        if local != N_FRAMES:
            problems.append(f"{trip_id}: decoded {local} frames, expected {N_FRAMES}")
        total_frames += local

        mw.writerow([
            trip_id, cls, 1 if cls == "positive" else 0,
            splits.get((cls, vid), "unassigned"),
            clean_dst.relative_to(out).as_posix(),
            overlay_dst.relative_to(out).as_posix() if overlay_dst else "",
            local, FPS, w, h,
            acc_start if acc_start is not None else "",
            acc_end if acc_end is not None else "",
            ttc_for_frame(0, acc_start),
            ann["timing"] if ann else "", ann["weather"] if ann else "",
            ann["egoinvolve"] if ann else "", ann["youtubeID"] if ann else "",
            ann["startframe"] if ann else "",
        ])

        if n % 250 == 0 or n == len(trips):
            print(f"  {n}/{len(trips)} trips, {total_frames} frames")

    mf.close()
    ff.close()

    print(f"\ntrips written : {len(trips)}  (hardlink {n_link}, copy {n_copy})")
    print(f"frames labelled: {total_frames} (expected {len(trips) * N_FRAMES})")
    print(f"manifest      : {manifest_path}")
    print(f"per-frame     : {frames_path}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("\nall trips decoded with the expected frame count")


if __name__ == "__main__":
    main()
