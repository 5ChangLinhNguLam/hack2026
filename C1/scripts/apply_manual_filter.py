"""Áp danh sách trip bị loại tay -> dataset mới đã lọc + file zip.

Nhận quyết định của bạn từ HAI nguồn, gộp lại:
  1. file .mp4 đã bị XOÁ khỏi trips_preview/  (so với review.txt)
  2. tên trip liệt kê trong trips_preview/drop_list.txt

Chỉ những trip ĐÃ ĐƯỢC RENDER mới tính. Trip không nằm trong preview (ví dụ
render mặc định thì 919 trip `da_che` không có mặt) được GIỮ NGUYÊN — nếu không,
render thiếu một nhóm là im lặng xoá sạch nhóm đó khỏi dataset.

Video hardlink sang bộ mới nên không tốn thêm đĩa. Nhãn chép nguyên, không tính
lại: cái đã verify rồi thì đừng đụng vào.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=root / "datasets" / "hackathon_ttc_ego")
    ap.add_argument("--preview", type=Path, default=None,
                    help="mặc định: <data>/../trips_preview_<tên bộ>")
    ap.add_argument("--out", type=Path, default=None,
                    help="mặc định: <data>_filtered")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--assume-all-rendered", action="store_true",
                    help="dùng khi review.txt bị mất VÀ preview đã render --all")
    args = ap.parse_args()

    prev = args.preview or args.data.parent / f"trips_preview_{args.data.name}"
    out = args.out or args.data.with_name(args.data.name + "_filtered")
    review = prev / "review.txt"
    trips_all = list(csv.DictReader(
        (args.data / "annotations" / "trips.csv").open(encoding="utf-8")))

    rendered = {}
    if review.exists():
        for ln in review.read_text(encoding="utf-8").splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            f = ln.split("\t")
            rendered[f[0]] = f[1]
    elif args.assume_all_rendered:
        # Không có review.txt thì không biết trip nào ĐÃ từng được render. Chỉ
        # suy ra được khi biết chắc preview phủ CẢ bộ (render bằng --all) —
        # đoán bừa là xoá nhầm cả nhóm chưa bao giờ được render.
        for r in trips_all:
            b = ("co_nhan" if int(r["n_finite_ttc"] or 0) > 0
                 else "khong_thay" if r["ego_involved"] == "1" else None)
            for cand in ([b] if b else ["an_toan", "da_che"]):
                rendered[r["trip_id"]] = cand
                if (prev / cand / f"{r['trip_id']}.mp4").exists():
                    break
        print(f"không có review.txt — dựng lại danh sách từ dataset "
              f"({len(rendered)} trip, giả định preview render bằng --all)")
    else:
        raise SystemExit(
            f"chưa thấy {review}.\n"
            f"  - preview render bằng --all  -> thêm cờ --assume-all-rendered\n"
            f"  - ngược lại -> chạy: python scripts/make_preview.py --index-only")

    gone = {t for t, b in rendered.items() if not (prev / b / f"{t}.mp4").exists()}
    listed = set()
    dl = prev / "drop_list.txt"
    if dl.exists():
        for ln in dl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                listed.add(ln.removesuffix(".mp4"))

    unknown = listed - rendered.keys()
    if unknown:
        print(f"CẢNH BÁO: {len(unknown)} tên trong drop_list.txt không khớp trip nào:")
        for u in sorted(unknown)[:5]:
            print("   ", u)
    drop = (gone | listed) & rendered.keys()
    print(f"{len(rendered)} trip đã render · bỏ {len(drop)} "
          f"(xoá file {len(gone)} · drop_list {len(listed & rendered.keys())})")

    ann_in = args.data / "annotations"
    trips = trips_all
    keep = [r for r in trips if r["trip_id"] not in drop]
    keep_ids = {r["trip_id"] for r in keep}
    if not keep:
        raise SystemExit("bỏ hết trip rồi — không có gì để đóng gói")

    (out / "trips").mkdir(parents=True, exist_ok=True)
    ann = out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for r in keep:
        link_or_copy(args.data / r["video"], out / r["video"])

    with (ann / "trips.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trips[0].keys()))
        w.writeheader()
        w.writerows(keep)

    n = 0
    with (ann_in / "ttc_per_frame.csv").open(encoding="utf-8") as fi, \
         (ann / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            if r["trip_id"] in keep_ids:
                w.writerow(r)
                n += 1

    print(f"\n{len(keep)} trip · {n} frame -> {out}")
    print("  " + " · ".join(f"{k} {v}" for k, v in
                            Counter(r["dataset"] for r in keep).items()))
    print("  " + " · ".join(f"{k} {v}" for k, v in
                            Counter(r["class"] for r in keep).items()))
    print(f"  group còn lại: {len({r.get('group') or r['trip_id'] for r in keep})}")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for script, extra in [("verify_hackathon_dataset.py", ["--data", str(out)])] + \
                         ([] if args.no_zip else [("zip_dataset.py", ["--data", str(out)])]):
        print(f"\n$ {script}")
        r = subprocess.run([sys.executable, str(root / "scripts" / script), *extra],
                           env=env)
        if r.returncode != 0:
            raise SystemExit(f"{script} thất bại (mã {r.returncode})")


if __name__ == "__main__":
    main()
