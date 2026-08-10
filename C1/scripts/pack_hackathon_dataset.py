"""Đóng gói practice + deepaccident_ttc thành 1 bộ để zip lên Colab.

Cấu trúc và TÊN CỘT giống hệt datasets/combined_ego, nên notebook chạy được mà
chỉ cần đổi một dòng:  DATASET = "hackathon_ttc"

Mẹo mã hoá nhãn: `min_ttc` vô hạn được ghi thành `ttc_s = -1`. Nhờ vậy hàm
frame_targets sẵn có trong notebook (cell 6) hoạt động ĐÚNG mà không sửa gì:
  t < 0      -> cls 0, keep 1, inv 0     ("không có nguy cơ" = 1/TTC bằng 0)
  t <= 2.0   -> cls 1, keep 1, inv 1/max(t, 0.5)
  t >= 4.0   -> cls 0, keep 1
  ở giữa     -> keep 0 (vùng biên)
Đó chính là ngữ nghĩa của min_ttc, chỉ khác cách viết.

MỌI THỨ Ở 10 Hz: practice gốc 20 fps nên lấy stride 2 và mã hoá lại video ở
10 fps, để chỉ số frame trong video khớp 1-1 với dòng nhãn. TSM và TCN đo ngữ
cảnh bằng ĐƠN VỊ FRAME nên trộn hai tần số là hỏng nhãn thời gian.

Ba thứ thêm vào khi dựng lại bộ lớn từ DeepAccident_train_01/02.zip
-----------------------------------------------------------------
`--carry-from`  Nguồn thô trên máy KHÔNG dựng lại được mọi trip của bộ cũ:
    16 scenario chỉ có trong DeepAccident_mini (đã xoá), và T06-Sample giờ chỉ
    còn 298/600 ảnh. Trip nào bộ cũ có mà lần dựng này thiếu — hoặc dựng ra
    NGẮN HƠN — thì lấy nguyên video + nhãn của bộ cũ sang. Schema y hệt nên
    chỉ là hardlink + chép dòng CSV; không mất dữ liệu đã có.

`--reencode`    Mặc định TẮT với nguồn đã đúng 10 fps / 640x360 (deepaccident):
    stride 1 nghĩa là giữ đúng mọi frame, nên mã hoá lại chỉ tạo thêm một đời
    nén mp4v (mờ thêm) và tốn một lượt decode+encode toàn bộ. Hardlink thay vào
    đó: file byte-identical, không tốn đĩa. Practice 20 fps vẫn phải mã hoá lại
    vì có lấy mẫu stride 2.

cột `group`     Bốn agent của CÙNG một scenario quay CÙNG một cảnh. Chia
    train/val theo TRIP là rò rỉ: cùng vụ va chạm nằm cả hai bên. `group` gộp
    chúng lại (practice: mỗi trip một group) để chia theo nhóm cho đúng.

Feature phi-thị-giác (thêm 2026-08)
-----------------------------------
Bộ chấm điểm (Hackathon_Dataset_Redacted) KHÔNG xoá `ego.speed_kmh`,
`longitudinal_accel`, `lateral_accel`, `metadata.weather` và `speed_limit_kmh` —
đã kiểm bằng cách mở T01d.json.gz. Đó là input hợp lệ lúc chấm, và model cũ chỉ
ăn ảnh nên vứt hết. Đo trên nhãn thật, một scalar duy nhất là GIẢM TỐC đạt
AUC 0.781 khi dự đoán frame nguy hiểm (ttc<=2s) trên practice — cao hơn nhiều so
với ego_speed (0.548).

Hai quyết định về tính NHẤT QUÁN GIỮA HAI DOMAIN, quan trọng hơn bản thân feature:

1. `ego_accel_mps2` và `ego_jerk_mps3` tính LẠI Ở ĐÂY bằng sai phân lùi của tốc
   độ, cho CẢ HAI nguồn — không dùng trường IMU của practice. Lý do: DeepAccident
   không có IMU nên buộc phải sai phân; trộn hai cách là cho model một dấu hiệu
   nhận biết domain. Đã đo trên 6 trip practice: sai phân so với IMU đạt
   corr 0.91-0.98 và std khớp (trừ T03 mưa đêm, IMU có spike std 11.0 vs 6.5),
   nên bỏ IMU gần như không mất gì mà được sự đồng nhất.
2. Sai phân chạy SAU khi lấy stride, tức trên trục thời gian 10 Hz của bộ đóng
   gói. Làm trước thì practice sẽ có dt = 0.05 s còn deepaccident 0.1 s.

`ego_lat_accel` là ngoại lệ: practice lấy IMU, deepaccident suy từ hướng vector
vận tốc (xem lateral_accels() bên kia). Hai cách chỉ đồng ý về ĐỘ LỚN, nên phía
notebook chỉ dùng |a_lat| đã tanh-nén.
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

import weather_features as wx

# console Windows mặc định cp1252, không in nổi tiếng Việt trong log
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

FPS = 10.0
OUT_SIZE = (640, 360)

TRIP_COLS = ["trip_id", "dataset", "source", "ttc_label", "class", "label",
             "split", "video", "num_frames", "fps", "width", "height",
             "n_finite_ttc", "n_ttc_lt2", "min_ttc_overall", "notes", "group",
             "ego_involved", "speed_limit_kmh", *wx.COLS]
FRAME_COLS = ["trip_id", "dataset", "class", "split", "frame_idx", "time_s",
              "binlabel", "ttc_s", "ego_speed_kmh", "ego_accel_mps2",
              "ego_jerk_mps3", "ego_lat_accel", "n_targets"]


def to_float(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def add_kinematics(frows):
    """Điền ego_accel_mps2 / ego_jerk_mps3 bằng sai phân LÙI của tốc độ.

    Lùi chứ không phải trung tâm: lúc chạy thật frame t+1 chưa tồn tại. Sai phân
    trung tâm ở đây sẽ làm điểm eval đẹp lên bằng một feature không có thật lúc
    stream — đúng kiểu lỗi mà cell 5 của notebook assert để chặn.

    Frame 0 nhận 0, khớp trạng thái bộ đệm rỗng lúc bắt đầu stream.
    """
    prev_v = prev_a = None
    for r in frows:
        v = to_float(r.get("ego_speed_kmh")) / 3.6
        a = 0.0 if prev_v is None else (v - prev_v) * FPS
        j = 0.0 if prev_a is None else (a - prev_a) * FPS
        r["ego_accel_mps2"] = round(a, 4)
        r["ego_jerk_mps3"] = round(j, 4)
        prev_v, prev_a = v, a
    return frows


def read_src(root: Path):
    ann = root / "annotations"
    with (ann / "trips.csv").open(encoding="utf-8") as fh:
        trips = {r["trip_id"]: r for r in csv.DictReader(fh)}
    per = defaultdict(list)
    with (ann / "ttc_per_frame.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per[r["trip_id"]].append(r)
    for v in per.values():
        v.sort(key=lambda r: int(r["frame_id"]))
    return trips, per


def group_of(trip_id: str) -> str:
    """Trip nào quay cùng một cảnh thì cùng group -> chia tập không rò rỉ.

    deepaccident__<type>__<scenario>__<agent>  -> bỏ phần <agent>
    practice__<trip>                           -> chính nó
    """
    parts = trip_id.split("__")
    if parts[0] == "deepaccident" and len(parts) == 4:
        return "__".join(parts[:3])
    return trip_id


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def count_frames(path: Path) -> int:
    """Số frame GIẢI MÃ ĐƯỢC thật sự. Không tin CAP_PROP_FRAME_COUNT: đó là siêu
    dữ liệu container, lệch một frame là notebook assert chết ngay lúc train."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def transcode(src_vid: Path, dst_vid: Path, keep: set) -> int:
    """Chép ra đúng các frame được lấy mẫu, ghi lại ở 10 fps / 640x360.

    PHẢI unlink trước. Thư mục đích của lần chạy trước chứa toàn HARDLINK trỏ về
    dataset nguồn; mở VideoWriter đè lên một hardlink là ghi thẳng vào inode
    dùng chung — tức phá luôn file gốc ở dataset kia. Đã mất bản T06-Sample 300
    frame theo đúng cách này. unlink() bẻ liên kết trước, inode cũ còn nguyên
    cho những link còn lại.
    """
    dst_vid.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(src_vid))
    w = cv2.VideoWriter(str(dst_vid), cv2.VideoWriter_fourcc(*"mp4v"), FPS, OUT_SIZE)
    i = written = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in keep:
            if (im.shape[1], im.shape[0]) != OUT_SIZE:
                im = cv2.resize(im, OUT_SIZE, interpolation=cv2.INTER_AREA)
            w.write(im)
            written += 1
        i += 1
    cap.release()
    w.release()
    return written


def native_ok(src_vid: Path) -> bool:
    """Video nguồn đã đúng 10 fps và 640x360 chưa -> có được phép hardlink không."""
    cap = cv2.VideoCapture(str(src_vid))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (w, h) == OUT_SIZE and abs(fps - FPS) < 0.05


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=root / "datasets" / "hackathon_ttc")
    ap.add_argument("--val-trips", nargs="*", default=["T05-Sample", "T06-Sample"],
                    help="trip practice giữ lại làm test — đây là domain thật của bài chấm")
    ap.add_argument("--carry-from", type=Path, default=None,
                    help="bộ đã đóng gói trước đó; trip nào lần này không dựng "
                         "được (hoặc dựng ra ngắn hơn) thì lấy nguyên sang")
    ap.add_argument("--reencode", action="store_true",
                    help="mã hoá lại cả nguồn vốn đã 10 fps/640x360 (mặc định hardlink)")
    ap.add_argument("--ego-policy", choices=("keep", "no-label", "drop"), default="keep",
                    help="trip DeepAccident mà CHÍNH xe mang cam không nằm trong vụ "
                         "va chạm: keep = giữ nguyên min_ttc · no-label = ttc_s = -1 "
                         "toàn bộ (an toàn) · drop = bỏ hẳn trip")
    args = ap.parse_args()

    sources = [
        ("practice", root / "datasets" / "practice", 2, "real_gt"),
        ("deepaccident", root / "datasets" / "deepaccident_ttc", 1, "derived"),
    ]
    for _, p, _, _ in sources:
        if not (p / "annotations" / "trips.csv").exists():
            raise SystemExit(f"thiếu {p}")
    if args.carry_from and args.carry_from.resolve() == args.out.resolve():
        raise SystemExit("--carry-from trùng --out: phải ghi ra thư mục khác")

    (args.out / "trips").mkdir(parents=True, exist_ok=True)
    ann = args.out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    trip_rows: dict[str, dict] = {}
    frame_rows: dict[str, list] = {}
    stats = defaultdict(lambda: [0, 0])
    n_link = n_enc = n_masked = n_dropped = 0

    for name, src, stride, kind in sources:
        trips, per = read_src(src)
        for tid, rows in sorted(per.items()):
            rows = rows[::stride]
            meta = trips[tid]
            full_id = f"{name}__{tid}"
            split = "test" if tid in args.val_trips else "train"

            # practice là GT thật của bài chấm và không có khái niệm "va chạm của
            # ego" — nó toàn near-miss. Chính sách ego chỉ áp cho DeepAccident.
            ego_inv = meta.get("ego_involved", "")
            masked = (name == "deepaccident" and args.ego_policy != "keep"
                      and ego_inv != "1")
            if masked and args.ego_policy == "drop":
                n_dropped += 1
                continue

            src_vid = src / meta["video"]
            dst_vid = args.out / "trips" / f"{full_id}.mp4"
            if stride == 1 and not args.reencode and native_ok(src_vid):
                written = count_frames(src_vid)
                link_or_copy(src_vid, dst_vid)
                n_link += 1
            else:
                written = transcode(src_vid, dst_vid,
                                    {int(r["frame_id"]) for r in rows})
                n_enc += 1

            rows = rows[:written]          # nhãn không được nhiều hơn frame có thật
            if masked:
                n_masked += 1
            n_fin = 0 if masked else sum(1 for r in rows if r["min_ttc_s"] != "inf")
            n_lt2 = 0 if masked else sum(1 for r in rows if r["cls"] == "1")
            cls = "positive" if n_lt2 else "negative"

            best = float("inf")
            frows = []
            for k, r in enumerate(rows):
                m = (float("inf") if masked or r["min_ttc_s"] == "inf"
                     else float(r["min_ttc_s"]))
                best = min(best, m)
                frows.append({
                    "trip_id": full_id, "dataset": name, "class": cls, "split": split,
                    "frame_idx": k, "time_s": round(k / FPS, 3),
                    "binlabel": 1 if (math.isfinite(m) and m <= 2.0) else 0,
                    "ttc_s": -1.0 if not math.isfinite(m) else round(m, 4),
                    "ego_speed_kmh": r.get("ego_speed_kmh", ""),
                    "ego_lat_accel": r.get("ego_lat_accel", ""),
                    "n_targets": r.get("n_targets", ""),
                })
            frame_rows[full_id] = add_kinematics(frows)
            trip_rows[full_id] = {
                "trip_id": full_id, "dataset": name, "source": kind,
                "ttc_label": "min_ttc", "class": cls, "label": 1 if n_lt2 else 0,
                "split": split, "video": f"trips/{full_id}.mp4",
                "num_frames": written, "fps": FPS,
                "width": OUT_SIZE[0], "height": OUT_SIZE[1],
                "n_finite_ttc": n_fin, "n_ttc_lt2": n_lt2,
                "min_ttc_overall": "inf" if not math.isfinite(best) else round(best, 3),
                "notes": meta.get("events", ""), "group": group_of(full_id),
                "ego_involved": ego_inv,
                # deepaccident không có giới hạn tốc độ -> để RỖNG, không điền 0.
                # Điền 0 thì tỉ số speed/limit hoá vô cực và notebook không còn
                # phân biệt được "không biết" với "biển báo 0 km/h".
                "speed_limit_kmh": meta.get("speed_limit_kmh", ""),
                **{c: meta.get(c, "") for c in wx.COLS},
            }
            stats[name][0] += 1
            stats[name][1] += written
        print(f"  {name}: {stats[name][0]} trip · {stats[name][1]} frame")

    # ---- trip của bộ cũ mà nguồn thô hiện tại không tái tạo được -------------
    n_carry = 0
    if args.carry_from:
        old_ann = args.carry_from / "annotations"
        with (old_ann / "trips.csv").open(encoding="utf-8") as fh:
            old_trips = list(csv.DictReader(fh))
        old_frames = defaultdict(list)
        with (old_ann / "ttc_per_frame.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                old_frames[r["trip_id"]].append(r)

        for t in old_trips:
            tid = t["trip_id"]
            n_old = int(t["num_frames"])
            cur = trip_rows.get(tid)
            if cur is not None and int(cur["num_frames"]) >= n_old:
                continue
            why = "thiếu hẳn" if cur is None else \
                  f"dựng lại chỉ được {cur['num_frames']}/{n_old} frame"

            # Trip carry sang từ bộ mini KHÔNG còn file meta -> không biết được
            # xe mang cam có nằm trong vụ va chạm hay không. Với chính sách ego,
            # "không biết" phải coi như KHÔNG dính: chỉ trip chứng minh được là
            # ego bị đâm mới được mang nhãn nguy hiểm.
            c_inv = t.get("ego_involved", "")
            c_masked = (t.get("dataset") == "deepaccident"
                        and args.ego_policy != "keep" and c_inv != "1")
            if c_masked and args.ego_policy == "drop":
                n_dropped += 1
                continue

            src_vid = args.carry_from / t["video"]
            if not src_vid.exists():
                print(f"  ! bỏ qua {tid}: không thấy {src_vid}")
                continue
            link_or_copy(src_vid, args.out / "trips" / f"{tid}.mp4")

            rows = sorted(old_frames[tid], key=lambda r: int(r["frame_idx"]))[:n_old]
            # Bộ cũ chỉ có ego_speed_kmh; accel/jerk tính lại tại chỗ nên trip
            # carry sang không bị thiếu feature so với trip dựng mới.
            frows = add_kinematics([{c: r.get(c, "") for c in FRAME_COLS}
                                    for r in rows])
            row = {c: t.get(c, "") for c in TRIP_COLS}
            if c_masked:
                for fr in frows:
                    fr["ttc_s"], fr["binlabel"], fr["class"] = -1.0, 0, "negative"
                row.update({"class": "negative", "label": 0, "n_finite_ttc": 0,
                            "n_ttc_lt2": 0, "min_ttc_overall": "inf"})
                n_masked += 1
            frame_rows[tid] = frows
            row["video"] = f"trips/{tid}.mp4"
            row["group"] = group_of(tid)
            trip_rows[tid] = row
            n_carry += 1
            stats["carried"][0] += 1
            stats["carried"][1] += n_old
            print(f"  carry {tid} ({why})")

    with (ann / "trips.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIP_COLS)
        w.writeheader()
        for tid in sorted(trip_rows):
            w.writerow(trip_rows[tid])
    n_frame = 0
    with (ann / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAME_COLS)
        w.writeheader()
        for tid in sorted(frame_rows):
            w.writerows(frame_rows[tid])
            n_frame += len(frame_rows[tid])

    # mọi num_frames phải khớp số dòng nhãn, nếu không notebook assert chết
    bad = [tid for tid, r in trip_rows.items()
           if int(r["num_frames"]) != len(frame_rows[tid])]
    if bad:
        raise SystemExit(f"lệch số frame ở {len(bad)} trip: {bad[:5]}")

    # ---- độ phủ feature, tách theo domain -----------------------------------
    # In ra vì một cột RỖNG ở đúng một domain là lỗi im lặng tệ nhất của bộ này:
    # notebook sẽ thay bằng 0 + cờ mask, model vẫn train, và cái nó thật sự học
    # được là "cột này rỗng nghĩa là deepaccident".
    print("\nđộ phủ feature (tỉ lệ ô có số, theo domain):")
    cov = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for tid, rows in frame_rows.items():
        ds = trip_rows[tid].get("dataset", "?")
        for c in ("ego_speed_kmh", "ego_accel_mps2", "ego_lat_accel", "n_targets"):
            for r in rows:
                cov[ds][c][0] += str(r.get(c, "")).strip() not in ("", "nan")
                cov[ds][c][1] += 1
    for ds in sorted(cov):
        tl = [t for t in trip_rows.values() if t.get("dataset") == ds]
        has_lim = sum(1 for t in tl if str(t.get("speed_limit_kmh", "")).strip())
        has_wx = sum(1 for t in tl if str(t.get("w_cloud", "")).strip())
        print(f"  {ds:13s} " + " ".join(f"{c} {n / max(d, 1):.0%}"
                                        for c, (n, d) in cov[ds].items())
              + f" | speed_limit {has_lim}/{len(tl)} trip · weather {has_wx}/{len(tl)} trip")

    n_ego = sum(1 for r in trip_rows.values() if r.get("ego_involved") == "1")
    print(f"\nchính sách ego = {args.ego_policy} · {n_ego} trip có xe mang cam bị đâm"
          f" · {n_masked} trip bị xoá nhãn TTC · {n_dropped} trip bị bỏ")
    print(f"video: hardlink {n_link} · mã hoá lại {n_enc} · carry {n_carry}")
    print(f"{len(trip_rows)} trip · {n_frame} frame · {len(set(r['group'] for r in trip_rows.values()))} group -> {args.out}")


if __name__ == "__main__":
    main()
