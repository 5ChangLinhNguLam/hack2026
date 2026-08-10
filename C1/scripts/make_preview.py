"""Vẽ nhãn TTC lên video để soi tay, kèm danh sách để đánh dấu trip cần bỏ.

Ra hai thứ:

  <data>/../trips_preview/<nhóm>/<trip_id>.mp4   video có burn nhãn
  <data>/../trips_preview/review.txt             1 dòng / trip, kèm số liệu
  <data>/../trips_preview/drop_list.txt          file rỗng để bạn điền tên trip bỏ

Chia thư mục theo NHÓM vì ba nhóm này cần nhìn bằng ba câu hỏi khác nhau:

  co_nhan/     trip có nhãn TTC hữu hạn — toàn bộ tín hiệu dương nằm ở đây.
               Hỏi: cú va chạm có THẬT SỰ nhìn thấy được từ camera trước không?
  khong_thay/  ego bị đâm nhưng KHÔNG frame nào có target trong cone trước
               (bị húc đuôi, đâm ngang). Hỏi: có nên giữ làm negative không?
  da_che/      trip bị chính sách no-label xoá nhãn (chỉ khi --all).

Tên file GIỮ NGUYÊN trip_id để ánh xạ ngược không nhập nhằng. Bốn agent của cùng
một scenario nằm sát nhau khi sort, vì trip_id có sẵn tiền tố group — xem liền 4
góc của cùng một vụ nhanh hơn nhiều so với xem rời rạc.

Sau khi soi xong, có HAI cách báo trip cần bỏ, dùng cách nào cũng được:
  1. Xoá thẳng file .mp4 trong trips_preview/  (tiện nhất, vừa xem vừa xoá)
  2. Ghi tên trip vào drop_list.txt, mỗi dòng một tên
rồi chạy scripts/apply_manual_filter.py — nó gộp cả hai nguồn.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

GREEN, AMBER, RED = (80, 220, 90), (40, 190, 255), (60, 60, 240)
GREY, WHITE, BLACK = (170, 170, 170), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
NO_LABEL = -1.0


def ttc_style(v: float, masked: bool):
    """cv2 chỉ vẽ được ASCII với font Hershey -> bỏ dấu, không dùng tiếng Việt."""
    if v < 0:
        return (GREY, "KHONG CO NHAN") if masked else (GREEN, "AN TOAN (inf)")
    if v <= 0.0:
        return RED, "VA CHAM"
    if v < 0.5:
        return RED, f"TTC {v:.2f}s"
    if v < 1.5:
        return AMBER, f"TTC {v:.2f}s"
    if v <= 2.0:
        return AMBER, f"TTC {v:.2f}s"
    return GREEN, f"TTC {v:.2f}s"


def draw(frame, trip_id, agent, i, n, v, masked, tag):
    h, w = frame.shape[:2]
    s = w / 640.0
    pad = int(10 * s)

    band = int(58 * s)
    strip = frame[:band].copy()
    cv2.rectangle(frame, (0, 0), (w, band), BLACK, -1)
    cv2.addWeighted(strip, 0.25, frame[:band], 0.75, 0, frame[:band])

    color, text = ttc_style(v, masked)
    cv2.putText(frame, text, (pad, int(26 * s)), FONT, 0.72 * s, color,
                max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(frame, f"{agent}  |  {tag}", (pad, int(48 * s)), FONT, 0.42 * s,
                WHITE, max(1, int(1 * s)), cv2.LINE_AA)

    # thanh TTC: đầy dần khi TTC tiến về 0, trần 3s
    if v >= 0:
        bw, bh = int(150 * s), int(9 * s)
        bx, by = w - bw - pad, int(16 * s)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        frac = max(0.0, min(1.0, 1.0 - v / 3.0))
        if frac > 0:
            cv2.rectangle(frame, (bx, by), (bx + int(bw * frac), by + bh), color, -1)

    foot = f"frame {i + 1}/{n}   t={i / 10:.1f}s"
    cv2.putText(frame, foot, (pad, h - pad), FONT, 0.42 * s, WHITE,
                max(1, int(1 * s)), cv2.LINE_AA)
    cv2.putText(frame, trip_id[:78], (pad, h - pad - int(16 * s)), FONT, 0.34 * s,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=root / "datasets" / "hackathon_ttc_ego")
    ap.add_argument("--out", type=Path, default=None,
                    help="mặc định: <data>/../trips_preview_<tên bộ>")
    ap.add_argument("--all", action="store_true",
                    help="render CẢ trip không có nhãn nào (chậm hơn)")
    ap.add_argument("--index-only", action="store_true",
                    help="chỉ ghi lại review.txt + drop_list.txt, KHÔNG render. "
                         "Dùng khi lỡ xoá mất review.txt — video đã xoá vẫn đứng "
                         "trong danh sách nên vẫn được tính là bỏ")
    ap.add_argument("--missing-only", action="store_true",
                    help="chỉ render clip CHƯA có file. ⚠️ Nó DỰNG LẠI cả những "
                         "clip bạn đã cố ý xoá -> chỉ dùng để sửa lỗi render dở, "
                         "đừng dùng sau khi đã bắt đầu lọc tay")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ann = args.data / "annotations"
    # Thư mục preview gắn TÊN BỘ. Hai bộ dùng chung trip_id nên nếu đổ chung một
    # chỗ thì clip của bộ này đè clip bộ kia, và apply_manual_filter đọc phải
    # review.txt của bộ khác -> lọc nhầm dataset mà không có dấu hiệu gì.
    out = args.out or args.data.parent / f"trips_preview_{args.data.name}"
    trips = list(csv.DictReader((ann / "trips.csv").open(encoding="utf-8")))
    per = defaultdict(list)
    with (ann / "ttc_per_frame.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per[r["trip_id"]].append(r)
    for v in per.values():
        v.sort(key=lambda r: int(r["frame_idx"]))

    # Bộ no-label: MỌI trip deepaccident không phải ego bị đâm đều bị xoá nhãn.
    # Bộ keep: ttc = -1 nghĩa là thật sự không có target nào trong cone, tức AN
    # TOÀN — hiển thị "khong co nhan" ở đó là nói sai hẳn nghĩa của nhãn.
    da = [r for r in trips if r["dataset"] == "deepaccident"]
    no_label_ds = all(int(r["n_finite_ttc"] or 0) == 0
                      for r in da if r["ego_involved"] != "1")
    third = "da_che" if no_label_ds else "an_toan"
    print(f"chính sách nhãn của bộ: {'no-label' if no_label_ds else 'keep'}")

    def bucket(r):
        if int(r["n_finite_ttc"] or 0) > 0:
            return "co_nhan"
        if r["ego_involved"] == "1":
            return "khong_thay"
        return third

    sel = [r for r in trips if args.all or bucket(r) != third]
    sel.sort(key=lambda r: r["trip_id"])
    if args.limit:
        sel = sel[:args.limit]
    print(f"{len(sel)}/{len(trips)} trip sẽ render -> {out}")

    for b in ("co_nhan", "khong_thay", third):
        (out / b).mkdir(parents=True, exist_ok=True)

    lines = ["# Danh sách trip để soi tay. Cột: trip_id | nhóm | frame | min_ttc |",
             "# số frame TTC<=2s | ego_involved | ghi chú scenario",
             "# Bỏ trip nào thì XOÁ FILE .mp4 tương ứng trong trips_preview/,",
             "# hoặc chép tên trip đó vào drop_list.txt. Rồi chạy:",
             "#     python scripts/apply_manual_filter.py", "#"]
    n_done = n_skip = 0
    for r in sel:
        tid = r["trip_id"]
        b = bucket(r)
        agent = tid.split("__")[-1] if tid.startswith("deepaccident") else "practice"
        masked = no_label_ds and b == third
        vals = [float(x["ttc_s"]) for x in per[tid]]

        # review.txt LUÔN liệt kê đủ mọi trip đã chọn, kể cả lần này không render:
        # apply_manual_filter suy ra "bị bỏ" từ chỗ THIẾU FILE so với danh sách,
        # nên danh sách mà khuyết trip là trip đó âm thầm được giữ lại.
        if args.index_only or (args.missing_only and (out / b / f"{tid}.mp4").exists()):
            lines.append("\t".join([tid, b, r["num_frames"], r["min_ttc_overall"],
                                    r["n_ttc_lt2"], r["ego_involved"] or "-",
                                    r["notes"][:44]]))
            n_skip += 1
            continue

        # Đổi ngưỡng TTC làm trip NHẢY NHÓM (an_toan -> co_nhan...). Bản cũ ở
        # nhóm cũ mà không dọn thì cùng một trip tồn tại hai chỗ: xoá bản này
        # tưởng đã loại trip, nhưng bản kia vẫn còn nên nó vẫn nằm trong dataset.
        for other in ("co_nhan", "khong_thay", third, "da_che", "an_toan"):
            if other != b:
                (out / other / f"{tid}.mp4").unlink(missing_ok=True)

        cap = cv2.VideoCapture(str(args.data / r["video"]))
        dst = out / b / f"{tid}.mp4"
        wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                             (int(r["width"]), int(r["height"])))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            v = vals[i] if i < len(vals) else NO_LABEL
            wr.write(draw(im, tid, agent, i, len(vals), v, masked, r["notes"][:40]))
            i += 1
        cap.release()
        wr.release()

        lines.append("\t".join([tid, b, r["num_frames"], r["min_ttc_overall"],
                                r["n_ttc_lt2"], r["ego_involved"] or "-",
                                r["notes"][:44]]))
        n_done += 1
        if n_done % 25 == 0 or n_done == len(sel):
            print(f"  {n_done}/{len(sel)}", flush=True)

    (out / "review.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dl = out / "drop_list.txt"
    if not dl.exists():
        dl.write_text(
            "# Mỗi dòng một trip_id cần BỎ. Dòng bắt đầu bằng # bị bỏ qua.\n"
            "# Cách khác: xoá thẳng file .mp4 trong trips_preview/ — script\n"
            "# apply_manual_filter.py hiểu cả hai và gộp lại.\n", encoding="utf-8")

    counts = defaultdict(int)
    for r in sel:
        counts[bucket(r)] += 1
    print(f"\nrender {n_done} clip · bỏ qua {n_skip} · "
          + " · ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"review.txt liệt kê {len(sel)} trip · drop_list.txt -> {out}")


if __name__ == "__main__":
    main()
