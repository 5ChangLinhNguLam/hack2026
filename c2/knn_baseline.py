"""C2 v1 — lan truyền nhãn driver state bằng nearest-neighbor perceptual hash.

Phát hiện nền tảng (26/07): ảnh driver của cả 16 trip đều composite từ DMD;
6/10 trip chấm điểm tái dùng gần nguyên clip đã có nhãn trong 6 trip Sample
(T05d 1798/1800 frame có near-dup dHash≤6; T03d 1702; T04d 1622; T07d 1701;
T06d 900; T10d 845). Với các trip đó, tra ngược frame → nhãn Sample gần như
giải xong C2. Trip còn lại (T01d/T02d/T08d/T09d) chứa clip/subject mới —
cần classifier thật (v2), file này chỉ đo được độ phủ và vote tạm.

Cách chạy:
    python c2/knn_baseline.py --loto              # leave-one-trip-out trên 6 Sample
    python c2/knn_baseline.py --predict           # sinh CSV cho 10 trip T0Xd
    python c2/knn_baseline.py --loto --no-smooth  # tắt smoothing để so sánh

Nhãn state đi theo block dài (15-30s) nên sau khi vote per-frame, nhãn được
làm mượt theo segment: điểm cắt = bước nhảy dHash giữa 2 frame liên tiếp
(>CUT_THRESHOLD), trong mỗi segment lấy vote có trọng số theo độ tin cậy
match (trọng số = max(0, 12 - hamming)).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from team_kit.dataset_loader import TripDataset  # noqa: E402

DATA = ROOT / "data"
CACHE = Path(__file__).resolve().parent / "cache"
SAMPLES = [f"T{i:02d}-Sample" for i in range(1, 7)]
SCORED = [f"T{i:02d}d" for i in range(1, 11)]

STATES = ["alert", "drowsy", "yawning", "distracted", "microsleep"]
# alertness là hằng số theo state trong toàn bộ 6 trip Sample (đã kiểm chứng)
ALERTNESS = {"alert": 0.95, "yawning": 0.55, "distracted": 0.45,
             "drowsy": 0.35, "microsleep": 0.05}

CUT_THRESHOLD = 16      # bước nhảy dHash liên tiếp > ngưỡng này = điểm cắt clip
MATCH_WEIGHT_CAP = 12   # trọng số vote = max(0, cap - hamming)
FPS = 20.0

_POP = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    d = np.zeros(x.shape, dtype=np.uint16)
    for shift in (0, 16, 32, 48):
        d += _POP[((x >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint16)]
    return d


def dhash64(gray: np.ndarray) -> np.uint64:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def scan_trip(trip: str) -> tuple[list[str], np.ndarray]:
    """(md5 list, dHash array) cho mọi frame driver của 1 trip — cache .npz."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"{trip}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        return list(z["md5"]), z["dhash"]
    ddir = DATA / trip / "driver"
    files = sorted(ddir.glob("frame_*.jpg"))
    md5s: list[str] = []
    hashes = np.empty(len(files), dtype=np.uint64)
    for i, f in enumerate(files):
        raw = f.read_bytes()
        md5s.append(hashlib.md5(raw).hexdigest())
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        hashes[i] = dhash64(img)
    np.savez(cache_file, md5=np.array(md5s), dhash=hashes)
    return md5s, hashes


def sample_labels(trip: str) -> list[str]:
    ds = TripDataset(DATA / trip)
    return [fr.driver_state for fr in ds.frame_records]


class RefIndex:
    """Chỉ mục tham chiếu: md5 → nhãn (khớp chính xác) + dHash stack (khớp gần)."""

    def __init__(self, trips: list[str]):
        self.md5_label: dict[str, str] = {}
        stacks, labels = [], []
        for t in trips:
            md5s, hashes = scan_trip(t)
            labs = sample_labels(t)
            for h, lab in zip(md5s, labs):
                self.md5_label.setdefault(h, lab)
            stacks.append(hashes)
            labels.extend(labs)
        self.hashes = np.concatenate(stacks)
        self.labels = np.array(labels)

    def match(self, md5s: list[str], hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(nhãn NN, khoảng cách) cho từng frame; khớp md5 → khoảng cách 0."""
        n = len(hashes)
        out_lab = np.empty(n, dtype=object)
        out_dist = np.empty(n, dtype=np.uint16)
        for s in range(0, n, 512):
            chunk = hashes[s:s + 512]
            d = _popcount(chunk[:, None] ^ self.hashes[None, :])
            idx = d.argmin(axis=1)
            out_lab[s:s + 512] = self.labels[idx]
            out_dist[s:s + 512] = d[np.arange(len(chunk)), idx]
        for i, h in enumerate(md5s):
            lab = self.md5_label.get(h)
            if lab is not None:
                out_lab[i], out_dist[i] = lab, 0
        return out_lab, out_dist


def segment_bounds(hashes: np.ndarray) -> list[tuple[int, int]]:
    """Chia trip thành các segment theo điểm cắt dHash liên tiếp."""
    d = _popcount(hashes[:-1] ^ hashes[1:])
    cuts = [0] + [int(i) + 1 for i in np.where(d > CUT_THRESHOLD)[0]] + [len(hashes)]
    cuts = sorted(set(cuts))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


def smooth_by_segment(labels: np.ndarray, dists: np.ndarray,
                      hashes: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Vote có trọng số trong từng segment; trả (nhãn mượt, thống kê segment)."""
    out = labels.copy()
    stats = []
    for a, b in segment_bounds(hashes):
        weights = defaultdict(float)
        for lab, dist in zip(labels[a:b], dists[a:b]):
            weights[lab] += max(0, MATCH_WEIGHT_CAP - int(dist))
        if weights and max(weights.values()) > 0:
            winner = max(weights, key=weights.get)
            out[a:b] = winner
        else:
            winner = None  # segment không có match tin cậy — giữ nhãn NN thô
        conf = float((dists[a:b] <= 6).mean())
        stats.append({"range": (a, b), "label": winner, "coverage<=6": round(conf, 3)})
    return out, stats


def macro_f1_present(pred: np.ndarray, gt: list[str]) -> tuple[float, float]:
    """(accuracy, macro-F1 trên lớp xuất hiện trong GT) — khớp evaluation.py."""
    gt_arr = np.array(gt)
    acc = float((pred == gt_arr).mean())
    f1s = []
    for c in set(gt):
        tp = int(((pred == c) & (gt_arr == c)).sum())
        fp = int(((pred == c) & (gt_arr != c)).sum())
        fn = int(((pred != c) & (gt_arr == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, float(np.mean(f1s))


def run_loto(smooth: bool) -> None:
    print(f"=== Leave-one-trip-out trên 6 Sample (smooth={smooth}) ===")
    print("(mỗi trip Sample là 1 subject chưa từng thấy → đây là proxy cho "
          "T01d/T02d/T08d/T09d, kỳ vọng THẤP — đo floor cho clip mới)")
    comps = []
    for held in SAMPLES:
        ref = RefIndex([t for t in SAMPLES if t != held])
        md5s, hashes = scan_trip(held)
        labels, dists = ref.match(md5s, hashes)
        if smooth:
            labels, _ = smooth_by_segment(labels, dists, hashes)
        acc, mf1 = macro_f1_present(labels, sample_labels(held))
        comp = 100 * (0.5 * acc + 0.5 * mf1)
        comps.append(comp)
        print(f"  {held}: acc={acc:.3f} macroF1={mf1:.3f} composite={comp:.1f} "
              f"(median NN dist={int(np.median(dists))})")
    print(f"  → composite trung bình: {np.mean(comps):.1f}")


def run_predict(smooth: bool) -> None:
    out_dir = ROOT / "predictions" / "thien_c2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = RefIndex(SAMPLES)
    print(f"=== Sinh predictions cho 10 trip T0Xd (smooth={smooth}) ===")
    for trip in SCORED:
        md5s, hashes = scan_trip(trip)
        labels, dists = ref.match(md5s, hashes)
        if smooth:
            labels, stats = smooth_by_segment(labels, dists, hashes)
        else:
            stats = []
        cov = float((dists <= 6).mean())
        path = out_dir / f"{trip}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "timestamp", "predicted_driver_state"])
            for i, lab in enumerate(labels):
                w.writerow([i, f"{i / FPS:.3f}", lab])
        seg_desc = ", ".join(
            f"[{a}-{b}) {s['label'] or '?'} cov={s['coverage<=6']}"
            for (a, b), s in [((s["range"][0], s["range"][1]), s) for s in stats]
        )
        print(f"  {trip}: coverage dist<=6 = {cov:.1%} → {path.name}")
        if seg_desc:
            print(f"      segments: {seg_desc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loto", action="store_true", help="đánh giá leave-one-trip-out trên 6 Sample")
    p.add_argument("--predict", action="store_true", help="sinh CSV cho 10 trip T0Xd")
    p.add_argument("--no-smooth", action="store_true", help="tắt smoothing theo segment")
    args = p.parse_args()
    if not (args.loto or args.predict):
        p.error("cần --loto và/hoặc --predict")
    if args.loto:
        run_loto(smooth=not args.no_smooth)
    if args.predict:
        run_predict(smooth=not args.no_smooth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
