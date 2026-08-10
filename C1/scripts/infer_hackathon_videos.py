"""Chạy student_ttc.pth trên CẢ 10 trip Hackathon_Dataset_Redacted và ghi video.

Vì sao là script chứ không phải một cell nữa
--------------------------------------------
Cell 9 của notebooks/infer_checkpoint.ipynb dựng animation bằng matplotlib và giữ
ảnh trong RAM -- hợp cho MỘT cửa sổ quanh MỘT event. Cả 10 trip là 9.000 frame:
matplotlib render tốn ~50 ms/frame (7-8 phút chỉ để vẽ) và 620 MB ảnh mỗi trip.
Ở đây overlay vẽ thẳng bằng cv2 rồi đẩy ngay vào VideoWriter, không frame nào
nằm lại trong RAM.

Kiến trúc model KHÔNG chép lại ở đây. Script import implementation deployable
từ ``C1.model`` và feature contract từ ``C1.features``; production không exec
notebook.

    python scripts/infer_hackathon_videos.py
    python scripts/infer_hackathon_videos.py --trips T01d T02d --fourcc avc1

Ra:
    outputs/hackathon_redacted/T01d.mp4 ...   video có overlay dự đoán
    outputs/hackathon_redacted/predictions.csv   p và TTC từng frame, mọi trip
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Console Windows mặc định cp1252; giữ cấu hình UTF-8 để log tiếng Việt từ
# runtime/imported modules không gây UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
IMG_SIZE = (224, 224)      # (W, H) -- phải trùng lúc train
STRIDE = 2                 # nguồn 20 fps -> 10 fps như lúc train
OUT_FPS = 10.0
OUT_W = 960                # bề ngang video xuất ra
EMA = 0.9                  # hệ số EMA trên 1/TTC; 1.0 = tắt làm mượt
WARN_ON, WARN_OFF = 0.85, 0.5      # hysteresis cho cờ cảnh báo

GREY = (170, 170, 170)
GREEN = (90, 200, 120)
AMBER = (0, 165, 235)
RED = (60, 60, 240)
WHITE = (255, 255, 255)


def load_arch():
    """Return the canonical importable architecture namespace."""
    from C1.features import FEATURES_USED, build_scalars
    from C1.model import StudentTTC

    return {
        "StudentTTC": StudentTTC,
        "FEAT_USE": FEATURES_USED,
        "build_scalars": build_scalars,
    }


def ttc_style(t):
    """(màu, chữ) cho một giá trị TTC giây. inf = không có nguy cơ."""
    if not np.isfinite(t):
        return GREEN, "TTC inf"
    if t < 0.5:
        return RED, f"TTC {t:.1f}s"
    if t < 1.5:
        return AMBER, f"TTC {t:.1f}s"
    return GREEN, f"TTC {t:.1f}s"


class Smoother:
    """Làm mượt NHÂN QUẢ đầu ra, một bộ cho mỗi trip.

    HAI ĐƯỜNG RIÊNG, cố ý không dùng chung một bộ lọc:

    * Số TTC hiển thị -> EMA trên 1/TTC, KHÔNG phải trên TTC. TTC có thể bằng
      inf ("không có nguy cơ") mà lấy trung bình inf thì vô nghĩa; còn 1/TTC thì
      inf trở thành 0, một con số bình thường, và bị chặn trong [0, 2].

    * Cờ cảnh báo -> HYSTERESIS, không lọc. Mọi bộ lọc đều làm trễ CẢ HAI chiều,
      mà chiều bật mới là chiều có giá. Đo trên nhãn thật: EMA a=0.5 làm cảnh báo
      trễ 1 frame và mất 5/103 lần; a=0.2 trễ 18.7 frame ở p90 và mất 23/103.
      Ở 10 fps, 1 frame = 0.1s ~ 1.4 m quãng đường phanh tại 50 km/h.
      Hysteresis khử nhấp nháy quanh ngưỡng mà KHÔNG thêm frame trễ nào lúc bật;
      nó chỉ "trễ" lúc tắt, vốn vô hại.

    ema=1.0 nghĩa là tắt hẳn phần làm mượt, chỉ còn hysteresis.
    """

    def __init__(self, ema=0.5, warn_on=0.5, warn_off=0.35):
        self.a, self.on, self.off = ema, warn_on, warn_off
        self.reset()

    def reset(self):
        self.inv = None
        self.warn = False

    def step(self, p, ttc):
        """-> (ttc đã mượt, cờ cảnh báo). p giữ nguyên, không lọc."""
        raw = 0.0 if not np.isfinite(ttc) else 1.0 / max(ttc, 1e-6)
        self.inv = raw if self.inv is None else self.a * raw + (1 - self.a) * self.inv
        # cùng ngưỡng TTC_CEIL=10s như lúc train: dưới 1/10 thì coi như vô cực
        ttc_s = 1.0 / self.inv if self.inv > 0.1 else float("inf")
        self.warn = p >= self.on if not self.warn else p >= self.off
        return ttc_s, self.warn


def draw_overlay(img, trip, frame_id, t_now, speed, p, ttc, event, warn=False):
    """Overlay vẽ bằng cv2 nên chỉ dùng ASCII: font Hershey không có dấu tiếng
    Việt, đưa chữ có dấu vào thì ra một dãy '?' trong video.

    ttc đưa vào là bản ĐÃ MƯỢT, warn là cờ từ hysteresis — đúng thứ tài xế thấy."""
    h, w = img.shape[:2]
    s = w / 960.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = int(14 * s)

    cv2.rectangle(img, (0, 0), (w, int(64 * s)), (18, 18, 18), -1)
    col, txt = ttc_style(ttc)
    cv2.putText(img, f"PRED  {txt}   p {p:.2f}", (pad, int(42 * s)),
                font, 0.95 * s, col, max(1, int(2 * s)), cv2.LINE_AA)
    if warn:
        (tw, _), _ = cv2.getTextSize("WARN", font, 0.95 * s, max(1, int(2 * s)))
        wx = pad + int(cv2.getTextSize(f"PRED  {txt}   p {p:.2f}", font,
                                       0.95 * s, max(1, int(2 * s)))[0][0]) + int(24 * s)
        cv2.rectangle(img, (wx - int(8 * s), int(14 * s)),
                      (wx + tw + int(8 * s), int(52 * s)), RED, -1)
        cv2.putText(img, "WARN", (wx, int(42 * s)), font, 0.95 * s, WHITE,
                    max(1, int(2 * s)), cv2.LINE_AA)
    if event:
        right_text(img, event, s)

    # thanh xác suất: vạch mốc ở hai ngưỡng hysteresis
    bx, by, bw, bh = int(w * 0.30), int(72 * s), int(w * 0.40), int(10 * s)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (58, 58, 58), -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * p), by + bh),
                  RED if warn else GREY, -1)
    for thr in (WARN_OFF, WARN_ON):
        tx = bx + int(bw * thr)
        cv2.line(img, (tx, by - int(3 * s)), (tx, by + bh + int(3 * s)), WHITE, 1)

    foot = f"{trip} | frame {frame_id} | t={t_now:5.1f}s | {speed:.0f} km/h"
    cv2.rectangle(img, (0, h - int(30 * s)), (w, h), (18, 18, 18), -1)
    cv2.putText(img, foot, (pad, h - int(10 * s)), font, 0.5 * s, WHITE,
                max(1, int(1 * s)), cv2.LINE_AA)
    return img


def right_text(img, text, s, color=WHITE):
    """Căn phải một dòng ở hàng tiêu đề."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, int(2 * s))
    (tw, _), _ = cv2.getTextSize(text, font, 0.65 * s, th)
    cv2.putText(img, text, (img.shape[1] - tw - int(14 * s), int(42 * s)),
                font, 0.65 * s, color, th, cv2.LINE_AA)


def trip_scalars(ns, meta, frame_ids):
    """Feature vô hướng cho ĐÚNG các frame sẽ chạy, dựng từ JSON của bộ chấm.

    Bộ redacted xoá nhãn nhưng GIỮ NGUYÊN ego.speed_kmh, lateral_accel, danh
    sách targets, weather và speed_limit_kmh — đã mở T01d.json.gz kiểm từng
    trường. Đây là chỗ biến chúng thành đúng vector mà model đã học.

    HAI ĐIỀU PHẢI KHỚP TUYỆT ĐỐI VỚI LÚC TRAIN, nếu không model nhận một thang
    đo khác mà chẳng có gì báo lỗi:
      - accel/jerk là sai phân LÙI trên trục 10 Hz, tức SAU khi lấy stride 2.
        Tính trên 20 Hz rồi mới lấy mẫu thì biên độ gia tốc chỉ còn một nửa.
      - chuẩn hoá do chính `build_scalars` của notebook làm, không chép lại.
    """
    fr = {f["frame_id"]: f for f in meta["frames"]}
    sel = [fr[i] for i in frame_ids if i in fr]
    speed = [f["ego"].get("speed_kmh", 0.0) for f in sel]
    lat = [f["ego"].get("lateral_accel", 0.0) for f in sel]
    ntg = [len(f.get("targets") or []) for f in sel]

    v = [s / 3.6 for s in speed]
    acc = [0.0] + [(v[i] - v[i - 1]) * OUT_FPS for i in range(1, len(v))]
    jrk = [0.0] + [(acc[i] - acc[i - 1]) * OUT_FPS for i in range(1, len(acc))]

    md = meta.get("metadata") or {}
    w = md.get("weather") or {}
    return ns["build_scalars"](
        speed, acc, jrk, lat, ntg,
        speed_limit_kmh=md.get("speed_limit_kmh", ""),
        weather={"w_cloud": w.get("cloudiness", 0.0),
                 "w_rain": w.get("precipitation", 0.0),
                 "w_wet": max(w.get("wetness", 0.0),
                              w.get("precipitation_deposits", 0.0)),
                 "w_fog": w.get("fog_density", 0.0),
                 "w_sun_alt": w.get("sun_altitude_angle", 75.0)})


def run_trip(ns, model, trip_dir, out_path, fourcc, device, smoother):
    trip = trip_dir.name
    with gzip.open(trip_dir / f"{trip}.json.gz", "rt", encoding="utf-8") as fh:
        meta = json.load(fh)
    fps_src = meta["metadata"]["fps"]
    events = [(e["t"], e["type"]) for e in meta["events_log"]]
    speed = {f["frame_id"]: f["ego"]["speed_kmh"] for f in meta["frames"]}

    files = sorted((trip_dir / "kitti" / "image_2").glob("*.jpg"))[::STRIDE]
    if not files:
        raise SystemExit(f"{trip}: không có ảnh trong kitti/image_2")
    feats = (trip_scalars(ns, meta, [int(f.stem) for f in files])
             if getattr(model, "n_scalar", 0) else None)

    probe = cv2.imread(str(files[0]))
    out_h = int(round(probe.shape[0] * OUT_W / probe.shape[1]))
    out_w, out_h = OUT_W - OUT_W % 2, out_h - out_h % 2   # codec đòi cạnh chẵn
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                         OUT_FPS, (out_w, out_h))
    if not vw.isOpened():
        raise SystemExit(f"không mở được VideoWriter cho {out_path} (fourcc {fourcc})")

    scale = out_w / 960.0
    model.reset()
    smoother.reset()          # state của bộ lọc thuộc về MỘT trip, không mang sang trip sau
    rows, t0 = [], time.perf_counter()
    try:
        for k, fp in enumerate(files):
            im = cv2.imread(str(fp))
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, IMG_SIZE)
            x = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) / 255.0
            f_sc = (torch.from_numpy(feats[min(k, len(feats) - 1)])[None].to(device)
                    if feats is not None else None)
            p, ttc = model.predict(x, f_sc)
            ttc_sm, warn = smoother.step(p, ttc)

            fid = int(fp.stem)
            t_now = fid / fps_src
            # event gần nhất đã xảy ra trong 10s qua -- đủ để đối chiếu bằng mắt
            ev = next((f"{n} {t_now - t:+.1f}s" for t, n in reversed(events)
                       if 0 <= t_now - t <= 10), "")
            frame = cv2.resize(im, (out_w, out_h))
            draw_overlay(frame, trip, fp.stem, t_now,
                         speed.get(fid, float("nan")), p, ttc_sm, ev, warn)
            vw.write(frame)
            # giữ CẢ raw lẫn smoothed: chấm điểm nên dùng raw (không lệch pha),
            # còn smoothed là thứ hiển thị cho người.
            rows.append({"trip": trip, "frame_id": fid, "t_s": round(t_now, 3),
                         "p": round(p, 4), "warn": int(warn),
                         "ttc_raw_s": "" if not np.isfinite(ttc) else round(ttc, 3),
                         "ttc_s": "" if not np.isfinite(ttc_sm) else round(ttc_sm, 3),
                         "speed_kmh": round(speed.get(fid, float("nan")), 2)})
    finally:
        vw.release()          # thiếu release() thì file không có moov atom
        model.reset()

    dt = time.perf_counter() - t0
    p_arr = np.array([r["p"] for r in rows])
    t_arr = np.array([r["t_s"] for r in rows])

    def _inv(key):
        v = np.array([1.0 / r[key] if r[key] != "" else 0.0 for r in rows])
        return np.abs(np.diff(v)).mean() if len(v) > 1 else float("nan")

    # độ giật đo TRÊN 1/TTC vì đó là thang bị chặn [0,2]; trên TTC thì một frame
    # nhảy sang inf là đủ làm mọi con số vô nghĩa
    n_sw = int((np.diff([r["warn"] for r in rows]) != 0).sum()) if len(rows) > 1 else 0
    print(f"{trip}: {len(rows)} frame · {dt:.0f}s ({len(rows) / max(dt, 1e-9):.1f} fps) · "
          f"p>=0.5 ở {(p_arr >= 0.5).mean():.1%} thời lượng · "
          f"{out_path.stat().st_size / 1e6:.1f} MB")
    print(f"    giật 1/TTC: raw {_inv('ttc_raw_s'):.4f} -> mượt {_inv('ttc_s'):.4f} "
          f"· cờ cảnh báo đảo {n_sw} lần")
    for t_e, n_e in events:
        w = p_arr[(t_arr >= t_e - 2) & (t_arr <= t_e + 10)]
        print(f"    {n_e:24s} @{t_e:5.1f}s → p tối đa [-2s,+10s] = "
              f"{(w.max() if len(w) else float('nan')):.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=ROOT / "datasets" / "Hackathon_Dataset_Redacted")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "student_ttc.pth")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "hackathon_redacted")
    ap.add_argument("--trips", nargs="*", default=None, help="mặc định: tất cả")
    ap.add_argument("--fourcc", default="mp4v", help="mp4v (luôn có) hoặc avc1 nếu máy hỗ trợ")
    ap.add_argument("--ema", type=float, default=EMA,
                    help="hệ số EMA trên 1/TTC (1.0 = tắt làm mượt)")
    ap.add_argument("--warn-on", type=float, default=WARN_ON)
    ap.add_argument("--warn-off", type=float, default=WARN_OFF)
    args = ap.parse_args()
    if not 0 < args.ema <= 1:
        raise SystemExit("--ema phải nằm trong (0, 1]")
    if args.warn_off > args.warn_on:
        raise SystemExit("--warn-off phải <= --warn-on, nếu không hysteresis đảo chiều")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ns = load_arch()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # aux_depth=False: head depth là công cụ REGULARIZE lúc train, `step()` không
    # bao giờ gọi nó. Dựng nó ở đây chỉ tốn tham số, và làm mọi checkpoint cũ
    # (chưa có head đó) trượt assert bên dưới vì thiếu key.
    # Số chiều nhánh scalar lấy TỪ CHECKPOINT, không từ cấu hình notebook: file
    # trọng số là thứ duy nhất biết nó đã được train với gì. Nhờ vậy checkpoint
    # chỉ-ảnh cũ (không có khoá n_scalar -> 0) vẫn chạy được ở đây.
    model = ns["StudentTTC"](backbone=ck["backbone"], pretrained=False,
                             shift_at=ck["shift_at"], auxiliary_depth=False,
                             n_scalar=ck.get("n_scalar", 0),
                             tcn_dilations=ck.get("tcn_dil", (1, 2, 4, 8))).to(device)
    state = {k: v for k, v in ck["state"].items() if not k.startswith("head_depth.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.eval()
    # Checkpoint train VỚI nhánh scalar mà notebook hiện đang khai báo FEAT_USE
    # khác đi (hoặc USE_SCALARS = False) thì load_state_dict ở trên đã báo lỗi.
    # Chiều ngược lại mới nguy: checkpoint CŨ chỉ-ảnh + notebook mới có scalar —
    # lúc đó nhánh scalar là trọng số ngẫu nhiên chưa từng được train.
    # Số chiều khớp là chưa đủ: đổi THỨ TỰ hay THÀNH PHẦN của FEAT_USE mà vẫn
    # giữ nguyên số lượng thì model nhận đúng 11 số nhưng sai ý nghĩa từng ô —
    # không có lỗi nào nổ ra, chỉ có kết quả tệ đi mà không giải thích được.
    _nck = ck.get("n_scalar", 0)
    _fck = ck.get("feat_use")
    if _nck and _fck and list(_fck) != list(ns["FEAT_USE"]):
        raise SystemExit(
            f"FEAT_USE lúc train {list(_fck)} khác notebook hiện tại "
            f"{list(ns['FEAT_USE'])} — sửa cell 3 về đúng danh sách của checkpoint")
    print(f"checkpoint {args.ckpt.name} · backbone {ck['backbone']} · "
          f"nhãn {ck.get('label', 'collision_countdown')} · "
          f"scalar {_nck or 'tắt'} · device {device}")
    print(f"làm mượt: EMA a={args.ema:g} trên 1/TTC"
          f"{' (TẮT)' if args.ema >= 1 else ''} · "
          f"hysteresis bật {args.warn_on:g} / tắt {args.warn_off:g}\n")
    smoother = Smoother(args.ema, args.warn_on, args.warn_off)

    dirs = sorted(d for d in args.data.iterdir() if d.is_dir() and (d / "kitti").is_dir())
    if args.trips:
        want = set(args.trips)
        dirs = [d for d in dirs if d.name in want]
        missing_trips = want - {d.name for d in dirs}
        if missing_trips:
            raise SystemExit(f"không thấy trip: {sorted(missing_trips)}")

    all_rows = []
    for d in dirs:
        all_rows += run_trip(ns, model, d, args.out / f"{d.name}.mp4", args.fourcc,
                             device, smoother)

    csv_path = args.out / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n{len(dirs)} video trong {args.out}\n{csv_path} ({len(all_rows):,} dòng)")


if __name__ == "__main__":
    main()
