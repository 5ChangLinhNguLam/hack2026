"""Parse CCD annotations (Crash-1500.txt) into per-video and per-frame TTC labels.

TTC (time-to-collision) convention used here:
  - accident_start = index of the first frame whose binary label is 1
  - for frame i < accident_start:  ttc = (accident_start - i) / FPS   (seconds, > 0)
  - for frame i >= accident_start: ttc = 0.0                          (collision in progress)
  - normal (non-crash) videos:     ttc = -1.0                         (no collision, sentinel)

Outputs (under datasets/carcrash/annotations/)
  Crash_Table.csv      one row per crash video (mirrors the Kaggle Crash_Table schema)
  ttc_per_frame.csv    one row per (video, frame)

These are the intermediate tables; build_trips.py consumes them to produce the
packaged dataset in datasets/carcrash/.
"""

import argparse
import csv
import re
from pathlib import Path

FPS = 10.0
N_FRAMES = 50
NO_COLLISION = -1.0

LINE_RE = re.compile(
    r"^(?P<vid>\d+),"
    r"\[(?P<bin>[^\]]*)\],"
    r"(?P<startframe>\d+),"
    r"(?P<youtube>[^,]*),"
    r"(?P<timing>[^,]*),"
    r"(?P<weather>[^,]*),"
    r"(?P<ego>[^,\s]*)\s*$"
)


def parse_crash_file(path: Path):
    """Yield one dict per crash video."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                raise ValueError(f"{path.name}:{lineno}: unparseable line: {line[:80]!r}")

            binlabels = [int(x) for x in m.group("bin").split(",")]
            if len(binlabels) != N_FRAMES:
                raise ValueError(
                    f"{path.name}:{lineno}: expected {N_FRAMES} labels, got {len(binlabels)}"
                )

            ones = [i for i, v in enumerate(binlabels) if v == 1]
            if not ones:
                raise ValueError(f"{path.name}:{lineno}: no accident frame flagged")

            yield {
                "vidname": m.group("vid"),
                "split": "Crash-1500",
                "binlabels": binlabels,
                "accident_start_frame": ones[0],
                "accident_end_frame": ones[-1],
                "startframe": m.group("startframe"),
                "youtubeID": m.group("youtube"),
                "timing": m.group("timing"),
                "weather": m.group("weather"),
                "egoinvolve": m.group("ego"),
            }


def ttc_for_frame(frame_idx: int, accident_start: int | None) -> float:
    """Seconds until the collision starts; 0 once it has, -1 if the clip has none."""
    if accident_start is None:
        return NO_COLLISION
    if frame_idx >= accident_start:
        return 0.0
    return round((accident_start - frame_idx) / FPS, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument(
        "--normal-dir",
        type=Path,
        default=None,
        help="Folder of Normal/*.mp4; if present they are emitted with ttc = -1.",
    )
    args = ap.parse_args()

    root: Path = args.root
    ann = root / "CarCrash" / "videos" / "Crash-1500.txt"
    out_dir = root / "datasets" / "carcrash" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(parse_crash_file(ann))
    print(f"parsed {len(videos)} crash videos from {ann}")

    table_path = out_dir / "Crash_Table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "vidname", "split", "startframe", "youtubeID", "timing", "weather",
                "egoinvolve", "accident_start_frame", "accident_end_frame",
                "ttc_at_clip_start_s", "num_frames", "fps",
            ]
        )
        for v in videos:
            w.writerow(
                [
                    v["vidname"], v["split"], v["startframe"], v["youtubeID"],
                    v["timing"], v["weather"], v["egoinvolve"],
                    v["accident_start_frame"], v["accident_end_frame"],
                    ttc_for_frame(0, v["accident_start_frame"]), N_FRAMES, FPS,
                ]
            )
    print(f"wrote {table_path}")

    normal_ids = []
    if args.normal_dir and args.normal_dir.is_dir():
        normal_ids = sorted(p.stem for p in args.normal_dir.glob("*.mp4"))
        print(f"found {len(normal_ids)} normal videos")

    frames_path = out_dir / "ttc_per_frame.csv"
    rows = 0
    with frames_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["vidname", "split", "frame_idx", "time_s", "binlabel", "ttc_s"])
        for v in videos:
            start = v["accident_start_frame"]
            for i in range(N_FRAMES):
                w.writerow(
                    [v["vidname"], "Crash-1500", i, round(i / FPS, 3),
                     v["binlabels"][i], ttc_for_frame(i, start)]
                )
                rows += 1
        for vid in normal_ids:
            for i in range(N_FRAMES):
                w.writerow([vid, "Normal", i, round(i / FPS, 3), 0, NO_COLLISION])
                rows += 1
    print(f"wrote {frames_path} ({rows} rows)")


if __name__ == "__main__":
    main()
