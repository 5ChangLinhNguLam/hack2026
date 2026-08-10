"""Download DeepAccident from Google Drive, retrying past the shared-quota block.

Drive rejects heavily-shared files with "Too many users have viewed or downloaded
this file recently"; the block lifts on its own, so this retries with backoff
instead of failing. Already-complete files are skipped, so it is safe to re-run.
"""

import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path

import gdown

FILES = {
    "split_txt_files.zip": "1n_EG6mS18s4EfXhBPf3XM1aEr6LRjgi7",
    "DeepAccident_mini.zip": "1NXC_-zTWFdHj-30g3zSUfNFMp4Mk_7Hh",
}

QUOTA_MARKERS = ("Too many users", "Quota exceeded", "quota")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def already_done(path: Path, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def try_once(fid: str, out: Path) -> bool:
    try:
        gdown.download(id=fid, output=str(out), quiet=False, resume=True)
        return out.exists() and out.stat().st_size > 0
    except Exception as e:
        text = str(e)
        kind = "quota" if any(m in text for m in QUOTA_MARKERS) else "other"
        log(f"  attempt failed ({kind}): {text.splitlines()[0][:120]}")
        if kind == "other":
            traceback.print_exc()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "datasets" / "deepaccident" / "raw")
    ap.add_argument("--interval", type=int, default=900, help="seconds between retries")
    ap.add_argument("--max-hours", type=float, default=26.0)
    ap.add_argument("--min-mb", type=int, default=100,
                    help="a mini/train archive smaller than this is treated as incomplete")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    pending = dict(FILES)
    attempt = 0

    while pending and time.time() < deadline:
        attempt += 1
        log(f"--- attempt {attempt}, {len(pending)} file(s) pending ---")
        for name, fid in list(pending.items()):
            dst = args.out / name
            floor = 1024 if name.startswith("split") else args.min_mb * 1024 * 1024
            if already_done(dst, floor):
                log(f"{name}: already complete ({dst.stat().st_size / 1e6:.1f} MB)")
                pending.pop(name)
                continue
            log(f"{name}: downloading...")
            if try_once(fid, dst) and already_done(dst, floor):
                log(f"{name}: OK ({dst.stat().st_size / 1e6:.1f} MB)")
                pending.pop(name)
            elif dst.exists() and dst.stat().st_size < floor:
                dst.unlink(missing_ok=True)   # drop the HTML error page Drive served

        if pending:
            left = (deadline - time.time()) / 3600
            log(f"still pending: {sorted(pending)}; sleeping {args.interval}s ({left:.1f}h budget left)")
            time.sleep(args.interval)

    if pending:
        log(f"GAVE UP after {attempt} attempts, still blocked: {sorted(pending)}")
        raise SystemExit(2)
    log("ALL DOWNLOADS COMPLETE")


if __name__ == "__main__":
    main()
