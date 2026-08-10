"""Concatenate CCD clips into one continuous video and emit a per-frame TTC index.

Every clip is 50 frames @ 10 fps. Frames are copied in clip order into a single
mp4; alongside it we write a CSV that maps each frame of the output video back to
its source clip and its time-to-collision label.

Examples
  python scripts/concat_videos.py --split crash --limit 50 --scale 0.5
  python scripts/concat_videos.py --split both --out out/ccd_all.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2

FPS = 10.0
NO_COLLISION = -1.0

# BGR
GREEN = (80, 220, 90)
AMBER = (40, 190, 255)
RED = (60, 60, 240)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def ttc_color(ttc: float):
    if ttc == NO_COLLISION:
        return GREY
    if ttc <= 0.0:
        return RED
    if ttc < 0.5:
        return RED
    if ttc < 1.5:
        return AMBER
    return GREEN


def ttc_text(ttc: float) -> str:
    if ttc == NO_COLLISION:
        return "TTC: n/a (no collision)"
    if ttc <= 0.0:
        return "COLLISION"
    return f"TTC: {ttc:.1f}s"


def draw_overlay(frame, vidname, split, local_idx, ttc, n_frames):
    h, w = frame.shape[:2]
    scale = w / 1280.0
    pad = int(16 * scale)
    bar_h = int(78 * scale)

    strip = frame[:bar_h].copy()
    cv2.rectangle(frame, (0, 0), (w, bar_h), BLACK, -1)
    cv2.addWeighted(strip, 0.35, frame[:bar_h], 0.65, 0, frame[:bar_h])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{split}/{vidname}  frame {local_idx + 1}/{n_frames}",
                (pad, int(28 * scale)), font, 0.62 * scale, WHITE, max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, ttc_text(ttc), (pad, int(62 * scale)), font,
                0.95 * scale, ttc_color(ttc), max(1, int(2 * scale)), cv2.LINE_AA)

    # countdown bar: full at 2.5s out, empty at impact
    if ttc != NO_COLLISION:
        bw, bh = int(300 * scale), int(12 * scale)
        bx, by = w - bw - pad, int(34 * scale)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        frac = max(0.0, min(1.0, ttc / 2.5))
        cv2.rectangle(frame, (bx, by), (bx + int(bw * (1.0 - frac)), by + bh),
                      ttc_color(ttc), -1)
    return frame


def load_crash_labels(labels_csv: Path):
    """vidname -> accident_start_frame."""
    out = {}
    with labels_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["vidname"]] = int(row["accident_start_frame"])
    return out


def ttc_for_frame(frame_idx: int, accident_start):
    if accident_start is None:
        return NO_COLLISION
    if frame_idx >= accident_start:
        return 0.0
    return round((accident_start - frame_idx) / FPS, 3)


def collect_clips(root: Path, split: str, limit: int | None, crash_labels):
    """[(path, vidname, split_name, accident_start_or_None)] in clip order."""
    vroot = root / "CarCrash" / "videos"
    clips = []
    if split in ("crash", "both"):
        for p in sorted((vroot / "Crash-1500").glob("*.mp4")):
            clips.append((p, p.stem, "Crash-1500", crash_labels.get(p.stem)))
    if split in ("normal", "both"):
        ndir = vroot / "Normal"
        if ndir.is_dir():
            for p in sorted(ndir.glob("*.mp4")):
                clips.append((p, p.stem, "Normal", None))
    if limit:
        clips = clips[:limit]
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--split", choices=["crash", "normal", "both"], default="crash")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N clips")
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor for output")
    ap.add_argument("--no-overlay", action="store_true", help="write clean frames")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root: Path = args.root
    out_video = args.out or root / "out" / f"ccd_{args.split}.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    index_csv = out_video.with_name(out_video.stem + "_frame_index.csv")

    crash_labels = load_crash_labels(
        root / "datasets" / "carcrash" / "annotations" / "Crash_Table.csv")
    clips = collect_clips(root, args.split, args.limit, crash_labels)
    if not clips:
        raise SystemExit(f"no clips found for split={args.split}")
    print(f"{len(clips)} clips -> {out_video}")

    probe = cv2.VideoCapture(str(clips[0][0]))
    ok, frame = probe.read()
    probe.release()
    if not ok:
        raise SystemExit(f"cannot read {clips[0][0]}")
    h, w = frame.shape[:2]
    w, h = int(w * args.scale), int(h * args.scale)

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        raise SystemExit("could not open VideoWriter")

    index_fh = index_csv.open("w", newline="", encoding="utf-8")
    idx = csv.writer(index_fh)
    idx.writerow(["global_frame", "global_time_s", "vidname", "split",
                  "local_frame", "binlabel", "ttc_s"])

    g = 0
    skipped = []
    for n, (path, vidname, split_name, acc_start) in enumerate(clips, 1):
        cap = cv2.VideoCapture(str(path))
        local = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            ttc = ttc_for_frame(local, acc_start)
            binlabel = 0 if acc_start is None else int(local >= acc_start)
            if not args.no_overlay:
                frame = draw_overlay(frame, vidname, split_name, local, ttc, 50)
            writer.write(frame)
            idx.writerow([g, round(g / FPS, 3), vidname, split_name, local, binlabel, ttc])
            g += 1
            local += 1
        cap.release()
        if local == 0:
            skipped.append(str(path))
        if n % 100 == 0 or n == len(clips):
            print(f"  {n}/{len(clips)} clips, {g} frames ({g / FPS / 60:.1f} min)")

    writer.release()
    index_fh.close()

    if skipped:
        print(f"WARNING: {len(skipped)} unreadable clips skipped, e.g. {skipped[:3]}")
    size_mb = out_video.stat().st_size / 1e6
    print(f"done: {g} frames, {g / FPS / 60:.1f} min, {size_mb:.0f} MB")
    print(f"index: {index_csv}")


if __name__ == "__main__":
    main()
