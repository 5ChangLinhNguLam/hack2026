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


# --- Track DMD (giữa retrieval và rules) — xem c2/dmd_match.py -------- #
# Mốc profile từ calibrate 26/07: yawning=67% yawn-density; alert sleepy=0.62;
# drowsy sleepy=0.87/0.90. Vùng 0.50-0.75 mơ hồ → nhường rules.
DMD_YAWN_MIN = 0.30
DMD_CLOSE_MIN = 0.30
DMD_DROWSY_MIN = 0.75
DMD_ALERT_MAX = 0.50


class DmdTrack:
    """Match trip → video DMD s5 + profile annotation per-segment.

    dHash chỉ định danh đúng CLIP, không đúng thời điểm trong clip (đã
    kiểm chứng bằng mắt: T02d ngáp match d=0 vào frame trung tính cùng
    cảnh) → chỉ dùng profile ở mức segment, KHÔNG chuyển nhãn per-frame."""

    def __init__(self):
        from dmd_match import CONF_DIST as DMD_CONF, DmdIndex, load_annotation, window_stats
        self.idx = DmdIndex()
        self.conf_dist = DMD_CONF
        self._load_ann = load_annotation
        self._wstats = window_stats
        self._ann_cache: dict[int, dict] = {}

    def profile(self, vi, fi, di, a: int, b: int):
        """(base_label|None, mô tả) cho segment [a,b) — None nếu không đủ tin."""
        from collections import Counter
        conf = di[a:b] <= self.conf_dist
        cov = float(conf.mean())
        if cov < 0.5:
            return None, f"dmd cov={cov:.0%} (bỏ)"
        vmain = Counter(vi[a:b][conf]).most_common(1)[0][0]
        if vmain not in self._ann_cache:
            self._ann_cache[vmain] = self._load_ann(self.idx.videos[vmain])
        ann = self._ann_cache[vmain]
        fsel = [int(f) for k, f in enumerate(fi[a:b]) if conf[k] and vi[a:b][k] == vmain]
        stats = [self._wstats(ann, f) for f in fsel]
        agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        subj = self.idx.videos[vmain].parent.parent.name
        desc = (f"dmd={subj} cov={cov:.0%} yawn={agg['yawn']:.0%} "
                f"close={agg['close']:.0%} sleepy={agg['sleepy']:.2f}")
        if agg["yawn"] >= DMD_YAWN_MIN:
            return "yawning", desc
        if agg["close"] >= DMD_CLOSE_MIN:
            return "microsleep", desc
        if agg["sleepy"] >= DMD_DROWSY_MIN:
            return "drowsy", desc
        if agg["sleepy"] <= DMD_ALERT_MAX:
            return "alert", desc
        return None, desc + " (mơ hồ)"


def predict_trip(trip: str, ref: RefIndex,
                 dmd: "DmdTrack | None" = None) -> tuple[np.ndarray, list[str]]:
    """Trả (nhãn final, log). Ưu tiên: Sample-retrieval > DMD-profile > rules.
    Khi DMD-profile thắng, rules vẫn đè per-frame cho yawning/microsleep
    (2 lớp rules đọc trực tiếp từ ảnh với độ chính xác cao — jawOpen/eyeBlink)."""
    md5s, hashes = scan_trip(trip)
    nn_lab, nn_dist = ref.match(md5s, hashes)

    clf_raw = rule_classify(trip)
    clf_smooth = smooth_majority(clf_raw, SMOOTH_HALF)

    dmd_match = None
    if dmd is not None:
        dmd_match = dmd.idx.match(hashes)

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
            continue

        base = None
        if dmd_match is not None:
            base, desc = dmd.profile(*dmd_match, a, b)
        if base is not None:
            seg = np.full(b - a, base, dtype=object)
            override = np.isin(clf_smooth[a:b], ["yawning", "microsleep"])
            seg[override] = clf_smooth[a:b][override]
            final[a:b] = seg
            n_over = int(override.sum())
            logs.append(f"[{a}-{b}) DMD base={base} ({desc})"
                        + (f" + rules override {n_over}f" if n_over else ""))
            continue

        final[a:b] = clf_smooth[a:b]
        raw_seg, sm_seg = clf_raw[a:b], clf_smooth[a:b]
        for c in np.unique(raw_seg):
            share = float((raw_seg == c).mean())
            if share >= ERASE_WARN and not (sm_seg == c).any():
                logs.append(f"[{a}-{b}) ⚠ lớp '{c}' chiếm {share:.0%} raw nhưng bị smoothing xóa")
        counts = {c: int((sm_seg == c).sum()) for c in np.unique(sm_seg)}
        note = "" if dmd_match is None else " | " + (dmd.profile(*dmd_match, a, b)[1])
        logs.append(f"[{a}-{b}) classifier {counts} (cov={cov:.0%}){note}")
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


def _make_dmd() -> "DmdTrack | None":
    try:
        return DmdTrack()
    except Exception as e:  # C:\DMD chưa tải / cache hash chưa build → chạy không DMD
        print(f"(DMD track tắt: {e})")
        return None


def run_predict() -> None:
    ref = RefIndex(SAMPLES)
    dmd = _make_dmd()
    out_dir = ROOT / "predictions" / "thien_c2"
    for trip in SCORED:
        labels, logs = predict_trip(trip, ref, dmd)
        write_csv(trip, labels, out_dir)
        print(f"\n{trip}:")
        for line in logs:
            print(f"  {line}")


def run_selfcheck() -> None:
    """LOTO: retrieval loại trip đang giữ. Lưu ý ngưỡng rule T_* đã được
    grid-search trên cả 6 trip Sample nên phần rule không phải LOTO thuần —
    con số này là ước lượng lạc quan nhẹ cho subject lạ."""
    out_dir = ROOT / "predictions" / "thien_c2_selfcheck"
    dmd = _make_dmd()
    comps = []
    for held in SAMPLES:
        others = [t for t in SAMPLES if t != held]
        labels, _ = predict_trip(held, RefIndex(others), dmd)
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
