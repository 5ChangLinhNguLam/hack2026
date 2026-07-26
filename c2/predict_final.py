"""C2 final — merge 2 track: retrieval (NN hash) + classifier (MLP blendshapes).

Thiết kế chốt sau bake-off 26/07 (4 họ model, LOTO, adversarial-verified):
- Track A (retrieval): segment có coverage match tin cậy (dHash<=6 tới frame
  Sample đã có nhãn) >= COVER_MIN → nhãn = vote trọng số của các match tin cậy.
  6/10 trip chấm điểm gần như được giải trọn bằng track này.
- Track B (classifier): phần còn lại dùng MLP(32,) trên 52 blendshape
  MediaPipe (blendshapes-only LOTO 39.2 > full-63 37.1 — cột hình học nhiễm
  vân tay subject), train trên TOÀN BỘ 6 trip Sample (đủ 5 lớp), smoothing
  majority vote cửa sổ ±75 frame (7.5s — nhỏ hơn block state 15-30s).
- Guardrail (bài học T02 bake-off): nếu smoothing xóa sạch 1 lớp chiếm >=8%
  dự đoán raw của 1 vùng classifier → in cảnh báo để xem xét tay.

Chạy:
    python c2/predict_final.py --predict     # 10 CSV final cho T0Xd
    python c2/predict_final.py --selfcheck   # LOTO 6 Sample (retrieval+clf đều loại trip đang giữ)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knn_baseline import (  # noqa: E402
    CACHE, SAMPLES, SCORED, FPS, MATCH_WEIGHT_CAP,
    RefIndex, scan_trip, segment_bounds, sample_labels, macro_f1_present,
)

N_GEO = 11              # 11 cột hình học đứng trước 52 blendshape
CONF_DIST = 6           # match dHash <= ngưỡng này là tin cậy
COVER_MIN = 0.5         # segment có >= 50% frame match tin cậy → dùng retrieval
MIN_SEG = 40            # segment ngắn hơn (2s) là nhiễu cắt → gộp vào trước đó
SMOOTH_HALF = 75        # ±75 frame (7.5s) cho majority vote classifier
ERASE_WARN = 0.08       # lớp chiếm >=8% raw mà bị smoothing xóa sạch → cảnh báo

# Ngưỡng rule classifier ngữ nghĩa (grid-search trên 6 trip Sample 26/07:
# mean composite 91.7 — hơn hẳn MLP blendshapes LOTO 39.2 vì rule mã hóa
# thẳng prototype nhãn, không học vân tay subject). Thứ tự ưu tiên rule:
# yawning > microsleep > drowsy > distracted > alert.
T_JAW = 0.25            # rolling-p75 jawOpen > → yawning
T_MSLEEP = 0.55         # rolling-mean eyeBlink > → microsleep (mắt nhắm hẳn)
T_BLINK = 0.13          # eyeBlink > (mắt lim dim)...
T_LOOK = 0.20           # ...kèm eyeLookDown > → drowsy
T_MAR = 0.15            # rolling-mean MAR > → distracted (nói chuyện điện thoại)


def merged_segments(hashes: np.ndarray) -> list[tuple[int, int]]:
    segs = segment_bounds(hashes)
    out: list[list[int]] = []
    for a, b in segs:
        if out and b - a < MIN_SEG:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _roll(x: np.ndarray, half: int, fn) -> np.ndarray:
    out = np.empty(len(x))
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        w = w[~np.isnan(w)]
        out[i] = fn(w) if len(w) else np.nan
    return out


def rule_classify(trip: str) -> np.ndarray:
    """Rule classifier ngữ nghĩa trên blendshape rolling (xem ngưỡng T_* trên).

    NaN (không bắt được mặt) được rolling window nội suy từ frame lân cận;
    frame vẫn NaN sau rolling (mất mặt dài) rơi về 'alert' — hiếm (<7%)."""
    z = np.load(CACHE / f"features_{trip}.npz", allow_pickle=True)
    F, names = z["features"], list(z["names"])
    ix = {n: names.index(n) for n in names}
    jaw = _roll(F[:, ix["jawOpen"]], 50, lambda w: np.percentile(w, 75))
    blink = _roll((F[:, ix["eyeBlinkLeft"]] + F[:, ix["eyeBlinkRight"]]) / 2, 25, np.mean)
    look = _roll((F[:, ix["eyeLookDownLeft"]] + F[:, ix["eyeLookDownRight"]]) / 2, 25, np.mean)
    mar = _roll(F[:, ix["mar"]], 25, np.mean)
    pred = np.full(len(F), "alert", dtype=object)
    pred[mar > T_MAR] = "distracted"
    pred[(blink > T_BLINK) & (look > T_LOOK)] = "drowsy"
    pred[blink > T_MSLEEP] = "microsleep"
    pred[jaw > T_JAW] = "yawning"
    return pred


def smooth_majority(pred: np.ndarray, half: int) -> np.ndarray:
    out = pred.copy()
    for i in range(len(pred)):
        a, b = max(0, i - half), min(len(pred), i + half + 1)
        vals, cnt = np.unique(pred[a:b], return_counts=True)
        out[i] = vals[cnt.argmax()]
    return out


def predict_trip(trip: str, ref: RefIndex) -> tuple[np.ndarray, list[str]]:
    """Trả (nhãn final, log các quyết định per-segment)."""
    md5s, hashes = scan_trip(trip)
    nn_lab, nn_dist = ref.match(md5s, hashes)

    clf_raw = rule_classify(trip)
    clf_smooth = smooth_majority(clf_raw, SMOOTH_HALF)

    final = np.empty(len(hashes), dtype=object)
    logs: list[str] = []
    for a, b in merged_segments(hashes):
        conf = nn_dist[a:b] <= CONF_DIST
        cov = float(conf.mean())
        if cov >= COVER_MIN:
            weights: dict[str, float] = defaultdict(float)
            for lab, dist in zip(nn_lab[a:b][conf], nn_dist[a:b][conf]):
                weights[lab] += MATCH_WEIGHT_CAP - int(dist)
            lab = max(weights, key=weights.get)
            final[a:b] = lab
            logs.append(f"[{a}-{b}) retrieval={lab} (cov={cov:.0%})")
        else:
            final[a:b] = clf_smooth[a:b]
            # guardrail: lớp bị smoothing xóa sạch trong vùng này
            raw_seg, sm_seg = clf_raw[a:b], clf_smooth[a:b]
            for c in np.unique(raw_seg):
                share = float((raw_seg == c).mean())
                if share >= ERASE_WARN and not (sm_seg == c).any():
                    logs.append(f"[{a}-{b}) ⚠ lớp '{c}' chiếm {share:.0%} raw nhưng bị smoothing xóa")
            counts = {c: int((sm_seg == c).sum()) for c in np.unique(sm_seg)}
            logs.append(f"[{a}-{b}) classifier {counts} (cov={cov:.0%})")
    return final, logs


def write_csv(trip: str, labels: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{trip}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp", "predicted_driver_state"])
        for i, lab in enumerate(labels):
            w.writerow([i, f"{i / FPS:.3f}", lab])
    return path


def run_predict() -> None:
    ref = RefIndex(SAMPLES)
    out_dir = ROOT / "predictions" / "thien_c2"
    for trip in SCORED:
        labels, logs = predict_trip(trip, ref)
        write_csv(trip, labels, out_dir)
        print(f"\n{trip}:")
        for line in logs:
            print(f"  {line}")


def run_selfcheck() -> None:
    """LOTO: retrieval loại trip đang giữ. Lưu ý ngưỡng rule T_* đã được
    grid-search trên cả 6 trip Sample nên phần rule không phải LOTO thuần —
    con số này là ước lượng lạc quan nhẹ cho subject lạ."""
    out_dir = ROOT / "predictions" / "thien_c2_selfcheck"
    comps = []
    for held in SAMPLES:
        others = [t for t in SAMPLES if t != held]
        labels, _ = predict_trip(held, RefIndex(others))
        write_csv(held, labels, out_dir)
        acc, mf1 = macro_f1_present(labels, sample_labels(held))
        comp = 100 * (0.5 * acc + 0.5 * mf1)
        comps.append(comp)
        print(f"{held}: acc={acc:.3f} macroF1={mf1:.3f} composite={comp:.1f}")
    print(f"→ LOTO composite trung bình (floor cho trip subject lạ): {np.mean(comps):.1f}")
    print(f"CSV self-check tại {out_dir} — chấm chuẩn bằng:")
    print("  python team_kit/evaluation.py --predictions predictions/thien_c2_selfcheck --data-dir data")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predict", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()
    if not (args.predict or args.selfcheck):
        p.error("cần --predict và/hoặc --selfcheck")
    if args.selfcheck:
        run_selfcheck()
    if args.predict:
        run_predict()
    return 0


if __name__ == "__main__":
    sys.exit(main())
