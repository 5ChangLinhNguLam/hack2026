"""Kiểm tra một bộ đã đóng gói trước khi zip lên Colab.

Kiểm đúng những thứ mà notebook GIẢ ĐỊNH ngầm — sai cái nào cũng chỉ vỡ ra lúc
đang train, khi đã mất một tiếng upload:

  1. `assert len(imgs) == len(y) == r.num_frames` (cell 3) — nên phải GIẢI MÃ
     thật từng video mà đếm, không tin cột num_frames lẫn siêu dữ liệu container.
  2. frame_idx liên tục 0..n-1, không trùng, không hổng.
  3. ttc_s chỉ được là -1 (vô hạn) hoặc số dương; class/label khớp n_ttc_lt2.
  4. group không nằm cả ở train lẫn test (rò rỉ: 4 agent cùng quay 1 vụ).
  5. Feature vô hướng CÓ MẶT Ở CẢ HAI DOMAIN, và không phải hằng số ở domain nào.

Kiểm tra 5 đáng bằng cả bốn cái trên gộp lại. Một cột feature chỉ khác 0 ở đúng
một domain KHÔNG phải feature — nó là nhãn domain, và model sẽ học nó rồi lấy
điểm cao trên deepaccident (99% dữ liệu) trong khi sập trên practice. Không có
gì trong loss train báo chuyện đó; chỉ bảng band tách domain phát hiện được, và
lúc ấy đã mất một lượt train đầy đủ.

Rồi in phân bố để biết bộ dữ liệu thực sự có gì: tỉ lệ frame nguy hiểm là con số
quyết định trọng số lớp, mà nó luôn khác kỳ vọng.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

POS_TTC = 2.0
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float("inf"))]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=root / "datasets" / "hackathon_ttc")
    ap.add_argument("--no-decode", action="store_true",
                    help="bỏ qua bước giải mã video (nhanh, nhưng bỏ luôn kiểm tra 1)")
    args = ap.parse_args()

    ann = args.data / "annotations"
    with (ann / "trips.csv").open(encoding="utf-8") as fh:
        trips = list(csv.DictReader(fh))
    per = defaultdict(list)
    with (ann / "ttc_per_frame.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per[r["trip_id"]].append(r)

    errs = []
    n_frame = 0
    ttc_all = []
    for k, t in enumerate(trips, 1):
        tid, n = t["trip_id"], int(t["num_frames"])
        rows = sorted(per[tid], key=lambda r: int(r["frame_idx"]))
        n_frame += len(rows)

        if len(rows) != n:
            errs.append(f"{tid}: {len(rows)} dòng nhãn vs num_frames {n}")
        if [int(r["frame_idx"]) for r in rows] != list(range(len(rows))):
            errs.append(f"{tid}: frame_idx không phải 0..n-1")

        vals = [float(r["ttc_s"]) for r in rows]
        ttc_all += vals
        if any(v != -1.0 and v < 0 for v in vals):
            errs.append(f"{tid}: có ttc_s âm khác sentinel -1")
        lt2 = sum(1 for v in vals if 0 <= v <= POS_TTC)
        if lt2 != int(t["n_ttc_lt2"]):
            errs.append(f"{tid}: n_ttc_lt2 {t['n_ttc_lt2']} vs đếm lại {lt2}")
        if (t["class"] == "positive") != (lt2 > 0) or int(t["label"]) != int(lt2 > 0):
            errs.append(f"{tid}: class/label không khớp n_ttc_lt2")

        if not args.no_decode:
            vid = args.data / t["video"]
            if not vid.exists():
                errs.append(f"{tid}: thiếu video {vid}")
            else:
                cap = cv2.VideoCapture(str(vid))
                d = 0
                while cap.grab():
                    d += 1
                cap.release()
                if d != n:
                    errs.append(f"{tid}: video giải mã {d} frame vs num_frames {n}")
            if k % 50 == 0:
                print(f"  đã kiểm {k}/{len(trips)} trip")

    by_split = defaultdict(set)
    for t in trips:
        by_split[t["split"]].add(t.get("group") or t["trip_id"])
    leak = by_split.get("train", set()) & by_split.get("test", set())
    if leak:
        errs.append(f"{len(leak)} group nằm cả train lẫn test: {sorted(leak)[:3]}")

    print(f"\n{len(trips)} trip · {n_frame} frame · "
          f"{len(set(t.get('group') or t['trip_id'] for t in trips))} group")
    print("split :", dict(Counter(t["split"] for t in trips)))
    print("dataset:", dict(Counter(t["dataset"] for t in trips)))
    print("class :", dict(Counter(t["class"] for t in trips)))
    for ds in sorted(set(t["dataset"] for t in trips)):
        sub = [t for t in trips if t["dataset"] == ds]
        pos = sum(1 for t in sub if t["class"] == "positive")
        print(f"   {ds:14s} {len(sub):4d} trip · positive {pos} ({pos / len(sub):.0%})")

    # ---- (5) feature vô hướng ------------------------------------------------
    FEAT = ["ego_speed_kmh", "ego_accel_mps2", "ego_jerk_mps3", "ego_lat_accel",
            "n_targets"]
    TRIP_FEAT = ["speed_limit_kmh", "w_cloud", "w_rain", "w_wet", "w_fog", "w_sun_alt"]
    ds_of = {t["trip_id"]: t["dataset"] for t in trips}
    vals = defaultdict(lambda: defaultdict(list))
    for tid, rows in per.items():
        for r in rows:
            for c in FEAT:
                try:
                    vals[ds_of.get(tid, "?")][c].append(float(r.get(c, "")))
                except (TypeError, ValueError):
                    pass
    domains = sorted(set(ds_of.values()))
    print("\nfeature vô hướng (phủ = tỉ lệ ô đọc được thành số):")
    for ds in domains:
        n_rows = sum(len(per[t]) for t in per if ds_of.get(t) == ds)
        parts = []
        for c in FEAT:
            v = vals[ds][c]
            cov = len(v) / max(n_rows, 1)
            spread = (max(v) - min(v)) if v else 0.0
            parts.append(f"{c.replace('ego_', '')} {cov:3.0%}"
                         + ("!" if cov < 0.99 or spread == 0 else ""))
            if cov < 0.99:
                errs.append(f"{ds}: cột {c} chỉ có số ở {cov:.0%} số dòng")
            elif spread == 0:
                errs.append(f"{ds}: cột {c} là HẰNG SỐ ({v[0] if v else '?'}) "
                            f"-> model đọc được nó như nhãn domain")
        print(f"  {ds:14s} " + " · ".join(parts))
    for c in TRIP_FEAT:
        got = {ds: sum(1 for t in trips if t["dataset"] == ds
                       and str(t.get(c, "")).strip() != "") for ds in domains}
        tot = {ds: sum(1 for t in trips if t["dataset"] == ds) for ds in domains}
        print(f"  {c:16s} " + " · ".join(f"{ds} {got[ds]}/{tot[ds]}" for ds in domains))
    # speed_limit_kmh RỖNG ở deepaccident là ĐÚNG THIẾT KẾ (nguồn không có giới
    # hạn tốc độ), và cell 3 của notebook đi kèm cờ has_limit để model phân biệt
    # "không biết" với "0 km/h". Vì vậy không tính là lỗi — chỉ in ra.

    fin = [v for v in ttc_all if v >= 0]
    print(f"\nframe: {len(ttc_all)} · TTC hữu hạn {len(fin)} "
          f"({len(fin) / max(1, len(ttc_all)):.1%}) · "
          f"nguy hiểm (<= {POS_TTC}s) {sum(1 for v in fin if v <= POS_TTC)} "
          f"({sum(1 for v in fin if v <= POS_TTC) / max(1, len(ttc_all)):.1%})")
    for lo, hi in BANDS:
        c = sum(1 for v in fin if lo <= v < hi)
        print(f"   TTC [{lo:>3.0f},{hi:>3.0f}) : {c:6d}  ({c / max(1, len(ttc_all)):5.1%})")

    if errs:
        print(f"\n{len(errs)} LỖI:")
        for e in errs[:30]:
            print("  " + e)
        raise SystemExit(1)
    print("\nOK — không có lỗi")


if __name__ == "__main__":
    main()
