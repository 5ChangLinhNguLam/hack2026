"""Cắt ngưỡng TTC: trên ngưỡng thì coi như VÔ CỰC, không phải một con số to.

Vì sao cần: `ttc_simple = gap / closing_speed` không có chặn trên. Target nào
tiến lại rất chậm thì mẫu số ~0 và TTC nổ ra vô nghĩa — đo được trên
hackathon_ttc_all: giá trị lớn nhất là 1,05e14 giây, và 31% frame có TTC > 10s.

Ba lý do bỏ chúng đi thay vì giữ:

1. NHÃN MÂU THUẪN VỚI CHÍNH ĐƯỜNG SUY LUẬN. Notebook chặn ở TTC_CEIL = 10s:
   `to_ttc()` trả inf khi 1/TTC < 0.1, và `evaluate()` chỉ tính MAE trên frame
   "hữu hạn" theo đúng ngưỡng đó. Nhãn 60s dạy model xuất 0.017 — một giá trị
   mà lúc chấm luôn bị quy về inf. Học xong không dùng được.

2. Đích hồi quy dồn cục. 1/TTC của vùng > 10s nằm trong [0, 0.1], lẫn hoàn toàn
   vào đích 0 của frame "không có nguy cơ". Model không phân biệt được, chỉ tốn
   capacity để khớp nhiễu.

3. Hai đầu ra tự mâu thuẫn: phân loại nói "an toàn" trong khi hồi quy vẫn đưa ra
   một con số thời gian. Hệ cảnh báo mà báo "an toàn, còn 47 giây" là vô nghĩa.

Sau khi cắt: `ttc_s = -1` (sentinel vô cực) cho mọi frame trên ngưỡng, rồi tính
lại class/label/n_finite_ttc/n_ttc_lt2/min_ttc_overall cho khớp. Video hardlink
sang bộ mới nên không tốn thêm đĩa.

CHỌN NGƯỠNG (đo trên hackathon_ttc_all, 91.128 frame):
    2.0  = đúng ngưỡng nguy hiểm POS_TTC. Hai đầu ra khớp tuyệt đối: có TTC
           <=> được cảnh báo. Nhưng xoá luôn vùng đệm (2,4) mà notebook cố ý
           loại khỏi loss phân loại, nên frame 2.1s thành negative cứng.
    4.0  = NEG_TTC. Giữ vùng đệm làm dải chuyển tiếp cho hồi quy, vẫn bỏ hết
           giá trị vô lý. Ít gãy nhất ở chỗ ranh giới quyết định.
    10.0 = TTC_CEIL. Chỉ sửa đúng phần mâu thuẫn với đường suy luận, giữ nguyên
           mọi thứ khác.
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

POS_TTC = 2.0


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
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", type=float, default=2.0,
                    help="TTC lớn hơn giá trị này -> -1 (vô cực). Mặc định 2.0")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    ann_in = args.data / "annotations"
    trips = list(csv.DictReader((ann_in / "trips.csv").open(encoding="utf-8")))
    cols = list(trips[0].keys())

    (args.out / "trips").mkdir(parents=True, exist_ok=True)
    ann = args.out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    agg = {r["trip_id"]: {"fin": 0, "lt2": 0, "best": float("inf")} for r in trips}
    n_cut = n_frame = 0
    with (ann_in / "ttc_per_frame.csv").open(encoding="utf-8") as fi, \
         (ann / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            v = float(r["ttc_s"])
            if v >= 0 and v > args.cutoff:
                r["ttc_s"], r["binlabel"] = -1.0, 0
                v = -1.0
                n_cut += 1
            a = agg[r["trip_id"]]
            if v >= 0:
                a["fin"] += 1
                a["best"] = min(a["best"], v)
                if v <= POS_TTC:
                    a["lt2"] += 1
                    r["binlabel"] = 1
            w.writerow(r)
            n_frame += 1

    for r in trips:
        a = agg[r["trip_id"]]
        r["n_finite_ttc"] = a["fin"]
        r["n_ttc_lt2"] = a["lt2"]
        r["min_ttc_overall"] = ("inf" if not math.isfinite(a["best"])
                                else round(a["best"], 3))
        r["class"] = "positive" if a["lt2"] else "negative"
        r["label"] = 1 if a["lt2"] else 0
        # cột class trong ttc_per_frame.csv chỉ để tra cứu, notebook không đọc
        link_or_copy(args.data / r["video"], args.out / r["video"])

    with (ann / "trips.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(trips)

    pos = sum(1 for r in trips if r["class"] == "positive")
    print(f"cắt ở {args.cutoff}s: {n_cut}/{n_frame} frame ({n_cut / n_frame:.1%}) "
          f"-> vô cực")
    print(f"{len(trips)} trip · positive {pos} · -> {args.out}")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    steps = [("verify_hackathon_dataset.py", ["--data", str(args.out)])]
    if not args.no_zip:
        steps.append(("zip_dataset.py", ["--data", str(args.out)]))
    for script, extra in steps:
        print(f"\n$ {script}")
        if subprocess.run([sys.executable, str(root / "scripts" / script), *extra],
                          env=env).returncode != 0:
            raise SystemExit(f"{script} thất bại")


if __name__ == "__main__":
    main()
