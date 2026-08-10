"""Move everything into the final datasets/<name>/ layout.

Before:
  dataset/            packaged CCD trips (built by build_trips.py)
  labels/             intermediate annotation tables
  CarCrash/           raw CCD download -- the mp4s here are hardlinks, the same
                      inodes already reachable from dataset/trips/
  out/                preview render from the (abandoned) concat approach

After:
  datasets/carcrash/  trips/ overlay/ annotations/ preview/ + CSVs + README
  datasets/deepaccident/

CarCrash/ is only removed once every clip is confirmed present under trips/.
Because the mp4s are hardlinks, unlinking the CarCrash/ names frees no space and
loses no data -- the inodes stay alive through their trips/ names.
"""

import os
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_DATASET = ROOT / "dataset"
LABELS = ROOT / "labels"
RAW = ROOT / "CarCrash"
REPO = ROOT / "CarCrashDataset"      # cloned repo, README only -- data lives on Drive
OUT = ROOT / "out"
OLD_DA = ROOT / "DeepAccident"       # first download location, before the datasets/ layout
DEST = ROOT / "datasets" / "carcrash"
DA_DEST = ROOT / "datasets" / "deepaccident"

EXPECTED = {"positive": 1500, "negative": 3000}


def fail(msg: str):
    print(f"ABORT: {msg}")
    sys.exit(1)


def verify_built() -> None:
    """Refuse to touch the raw download unless the package is complete."""
    for cls, n in EXPECTED.items():
        for kind in ("trips", "overlay"):
            got = len(list((OLD_DATASET / kind / cls).glob("*.mp4")))
            if got != n:
                fail(f"{kind}/{cls}: {got} clips, expected {n} -- build not finished")
    import csv
    with (OLD_DATASET / "trip_manifest.csv").open(encoding="utf-8") as fh:
        man = list(csv.DictReader(fh))
    n_frm = sum(1 for _ in (OLD_DATASET / "ttc_per_frame.csv").open(encoding="utf-8")) - 1
    if len(man) != 4500:
        fail(f"trip_manifest.csv has {len(man)} rows, expected 4500")

    # Two CCD Normal clips genuinely hold 49 frames, so don't hardcode 4500*50 --
    # require the frame table and the manifest to agree instead.
    declared = sum(int(r["num_frames"]) for r in man)
    if n_frm != declared:
        fail(f"ttc_per_frame.csv has {n_frm} rows but manifest declares {declared} frames")
    short = [(r["trip_id"], r["num_frames"]) for r in man if r["num_frames"] != "50"]
    print(f"verified: 4500 trips, {n_frm} labelled frames, overlay complete")
    if short:
        print(f"  note: {len(short)} trip(s) shorter than 50 frames (source files): {short}")


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  skip (absent): {src.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  skip (exists): {dst.relative_to(ROOT)}")
        return
    shutil.move(str(src), str(dst))
    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> None:
    if not OLD_DATASET.exists() and DEST.exists():
        print("already reorganized")
        return
    verify_built()

    print("\nmoving packaged dataset:")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    move(OLD_DATASET, DEST)

    print("\ncollecting annotations:")
    ann = DEST / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for name in ("Crash_Table.csv", "ttc_per_frame.csv"):
        move(LABELS / name, ann / name)
    for src in (RAW / "train.txt", RAW / "test.txt",
                RAW / "videos" / "Crash-1500.txt", RAW / "videos" / "ytb_list.txt"):
        move(src, ann / src.name)

    print("\nmoving preview render:")
    if OUT.exists():
        for p in OUT.iterdir():
            move(p, DEST / "preview" / p.name)

    print("\nremoving redundant raw copies (hardlinks -- no data lost):")
    for cls, folder in (("positive", "Crash-1500"), ("negative", "Normal")):
        raw_dir = RAW / "videos" / folder
        if not raw_dir.exists():
            continue
        missing = [p.name for p in raw_dir.glob("*.mp4")
                   if not (DEST / "trips" / cls / p.name).exists()]
        if missing:
            fail(f"{len(missing)} clip(s) in {folder} have no trips/ counterpart, "
                 f"e.g. {missing[:3]} -- refusing to delete")
        print(f"  {folder}: all {len(list(raw_dir.glob('*.mp4')))} clips present in trips/{cls}")

    print("\nmoving DeepAccident split lists:")
    if OLD_DA.exists():
        for p in OLD_DA.iterdir():
            move(p, DA_DEST / "raw" / p.name)

    for leftover in (RAW, LABELS, OUT, REPO, OLD_DA):
        if leftover.exists():
            # git pack files are read-only on Windows; clear the bit and retry
            shutil.rmtree(leftover, onexc=lambda f, p, _: (os.chmod(p, stat.S_IWRITE), f(p)))
            print(f"  removed {leftover.relative_to(ROOT)}/")

    print("\nfinal layout:")
    for p in sorted((ROOT / "datasets").rglob("*")):
        if p.is_dir() and len(p.relative_to(ROOT).parts) <= 3:
            n = len(list(p.glob("*")))
            print(f"  {p.relative_to(ROOT).as_posix()}/  ({n} entries)")


if __name__ == "__main__":
    main()
