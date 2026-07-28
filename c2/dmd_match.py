"""C2 — match ngược frame hackathon → video nguồn DMD + đọc nhãn OpenLABEL.

Phát hiện smoke test 26/07: ảnh driver hackathon (640×360) là camera **FACE**
của DMD (không phải body như đoán ban đầu) — subject 14 khớp p50 hamming=6.

Hai nguồn được index chung:
* Drowsiness ``s5`` KHÔNG có nhãn state trực tiếp — chỉ có
  ``eyes_state/*``, ``blinks/blinking``, ``yawning/*``.
* Distraction ``s2`` có nhãn trực tiếp ``driver_actions/phonecall_*``,
  ``driver_actions/texting_*`` và ``driver_actions/safe_drive``.

Mapping sang 5 lớp:
    - yawning/*                     → yawning (trực tiếp)
    - eyes_state/close kéo dài      → microsleep
    - phonecall_* / texting_*       → distracted (trực tiếp)
    - alert vs drowsy               → calibrate qua match với 6 trip Sample
      (đã biết GT hackathon) — xem run_calibrate().

Cách chạy (theo thứ tự):
    python c2/dmd_match.py --hash --workers 4  # hash video face s5+s2
    python c2/dmd_match.py --calibrate   # bảng: Sample GT state ↔ ngữ cảnh annotation DMD
    python c2/dmd_match.py --label       # match các trip T0Xd → nhãn đề xuất per-segment
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Windows terminals used by the team may default to cp1252.  The diagnostic
# output intentionally contains Vietnamese and arrows, so make the CLI
# deterministic instead of failing halfway through calibration.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knn_baseline import (  # noqa: E402
    CACHE, SAMPLES, SCORED, dhash64, _popcount, scan_trip, sample_labels,
)
from predict_final import merged_segments  # noqa: E402

DMD_ROOT = Path(r"C:\DMD\dmd")
MICROSLEEP_MIN_FRAMES = 45      # close liên tục >=1.5s (@29.76fps) = microsleep
# Calibrate 26/07: dist 6-8 sinh match GIẢ giữa các subject (dHash mặt cận
# cảnh va chạm) — vd T06-distracted (nguồn s2 chưa tải) match nhầm subject 9.
# Match thật (cùng clip) có dist 0-4.
CONF_DIST = 4
WIN = 30                        # cửa sổ ±30 frame DMD để đo mật độ annotation


# ---------------------------------------------------------------------- #
# Hash video DMD
# ---------------------------------------------------------------------- #
def face_videos() -> list[Path]:
    videos = list(DMD_ROOT.glob("g*/*/s5/*_rgb_face.mp4"))
    videos.extend(DMD_ROOT.glob("g*/*/s2/*_rgb_face.mp4"))
    return sorted(videos)


def video_key(v: Path) -> str:
    return v.stem.replace(";", "_")


def hash_video(v: Path) -> np.ndarray:
    dest = CACHE / f"dmd_{video_key(v)}.npz"
    if dest.exists():
        return np.load(dest)["dhash"]
    cap = cv2.VideoCapture(str(v))
    hashes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        hashes.append(dhash64(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
    cap.release()
    arr = np.array(hashes, dtype=np.uint64)
    np.savez(dest, dhash=arr)
    return arr


# ---------------------------------------------------------------------- #
# Annotation OpenLABEL
# ---------------------------------------------------------------------- #
def load_annotation(v: Path) -> dict:
    """Per-frame masks thống nhất cho annotation Drowsiness và Distraction.

    Annotation đánh trên timeline chủ (mosaic-aligned); frame video lệch
    ``frame_shift`` của stream face_camera → cộng shift khi tra."""
    session = v.parent.name
    if session == "s5":
        pattern = "*_ann_drowsiness.json"
        kind = "drowsiness"
    elif session == "s2":
        pattern = "*_ann_distraction.json"
        kind = "distraction"
    else:
        raise ValueError(f"session DMD không hỗ trợ: {v}")

    try:
        ann_path = next(v.parent.glob(pattern))
    except StopIteration as exc:
        raise FileNotFoundError(f"thiếu annotation {pattern} cạnh {v}") from exc
    with open(ann_path, encoding="utf-8") as f:
        doc = json.load(f)
    ol = doc[next(iter(doc))]
    shift = 0
    streams = ol.get("streams") or (ol.get("metadata") or {}).get("streams") or {}
    fc = streams.get("face_camera") or {}
    shift = int(((fc.get("stream_properties") or {}).get("sync") or {})
                .get("frame_shift", 0))
    actions = ol.get("actions") or {}
    n_actions = max(
        (iv["frame_end"] for action in actions.values()
         for iv in action.get("frame_intervals", [])),
        default=-1,
    ) + 1
    n_stream = int(((fc.get("stream_properties") or {}).get("total_frames") or 0))
    n = max(n_actions, n_stream)
    if n <= 0:
        raise ValueError(f"annotation không có timeline: {ann_path}")
    masks: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n, dtype=bool))
    for action in actions.values():
        t = action["type"]
        for iv in action.get("frame_intervals", []):
            masks[t][iv["frame_start"]:iv["frame_end"] + 1] = True
    yawn = masks["yawning/Yawning with hand"] | masks["yawning/Yawning without hand"]
    close = masks["eyes_state/close"]
    transition = masks["eyes_state/closing"] | masks["eyes_state/opening"]
    blink = masks["blinks/blinking"]
    distracted = (
        masks["driver_actions/phonecall_left"]
        | masks["driver_actions/phonecall_right"]
        | masks["driver_actions/texting_left"]
        | masks["driver_actions/texting_right"]
    )
    safe = masks["driver_actions/safe_drive"]
    # run-length của close để tách microsleep (close dài) khỏi chớp mắt
    close_run = np.zeros(n, dtype=np.int32)
    run = 0
    for i in range(n):
        run = run + 1 if close[i] else 0
        close_run[i] = run
    for i in range(n - 2, -1, -1):          # lan ngược: cả run mang max length
        if close[i] and close[i + 1]:
            close_run[i] = close_run[i + 1]
    return {"kind": kind, "yawn": yawn, "close": close, "close_run": close_run,
            "transition": transition, "blink": blink,
            "open": masks["eyes_state/open"], "distracted": distracted,
            "safe": safe, "n": n, "shift": shift}


def window_stats(ann: dict, fidx: int) -> dict[str, float]:
    """Mật độ annotation trong ±WIN frame quanh frame match (đã cộng shift)."""
    m = fidx + ann["shift"]
    a, b = max(0, m - WIN), min(ann["n"], m + WIN + 1)
    if b <= a:
        return {
            "yawn": 0.0, "close": 0.0, "trans": 0.0, "blink": 0.0,
            "sleepy": 0.0, "distracted": 0.0, "safe": 0.0,
        }
    out = {
        "yawn": float(ann["yawn"][a:b].mean()),
        "close": float(ann["close"][a:b].mean()),
        "trans": float(ann["transition"][a:b].mean()),
        "blink": float(ann["blink"][a:b].mean()),
        "distracted": float(ann["distracted"][a:b].mean()),
        "safe": float(ann["safe"][a:b].mean()),
    }
    # "sleepy" = mật độ hoạt động mí mắt tổng hợp — alert sạch, drowsy dày
    out["sleepy"] = out["close"] + out["trans"] + out["blink"]
    return out


def derive_state(ann: dict, fidx: int, drowsy_start: float) -> str:
    """Map frame DMD → lớp hackathon. drowsy_start = mốc phần-trăm video
    (calibrate được) mà trước đó coi là alert, sau đó là drowsy (protocol
    s5 quay tuần tự: safe → sleepy → yawn → microsleep)."""
    n = ann["n"]
    if fidx >= n:
        fidx = n - 1
    if ann["distracted"][fidx]:
        return "distracted"
    if ann["safe"][fidx]:
        return "alert"
    if ann["yawn"][fidx]:
        return "yawning"
    if ann["close"][fidx] and ann["close_run"][fidx] >= MICROSLEEP_MIN_FRAMES:
        return "microsleep"
    return "drowsy" if fidx / n >= drowsy_start else "alert"


# ---------------------------------------------------------------------- #
# Match
# ---------------------------------------------------------------------- #
class DmdIndex:
    def __init__(self):
        self.videos = face_videos()
        self.stacks = [np.load(CACHE / f"dmd_{video_key(v)}.npz")["dhash"]
                       for v in self.videos]
        self.all = np.concatenate(self.stacks)
        self.owner = np.concatenate([np.full(len(s), i, dtype=np.int32)
                                     for i, s in enumerate(self.stacks)])
        offs = np.cumsum([0] + [len(s) for s in self.stacks])
        self.local = np.concatenate([np.arange(len(s)) for s in self.stacks])
        self.offsets = offs

    def match(self, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(video_idx, frame_idx, dist) cho từng hash."""
        vi = np.empty(len(hashes), dtype=np.int32)
        fi = np.empty(len(hashes), dtype=np.int32)
        di = np.empty(len(hashes), dtype=np.uint16)
        for s in range(0, len(hashes), 256):
            chunk = hashes[s:s + 256]
            d = _popcount(chunk[:, None] ^ self.all[None, :])
            idx = d.argmin(axis=1)
            vi[s:s + 256] = self.owner[idx]
            fi[s:s + 256] = self.local[idx]
            di[s:s + 256] = d[np.arange(len(chunk)), idx]
        return vi, fi, di


def run_hash(workers: int = 1) -> None:
    videos = face_videos()

    def report(v: Path, arr: np.ndarray) -> None:
        print(
            f"{v.parent.parent.name}/{v.parent.name}/{v.name}: {len(arr)} frames",
            flush=True,
        )

    if workers <= 1:
        for v in videos:
            report(v, hash_video(v))
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(hash_video, v): v for v in videos}
        for future in as_completed(futures):
            v = futures[future]
            report(v, future.result())


def run_calibrate() -> None:
    """Với mỗi trip Sample: match từng frame → video/frame DMD, in ngữ cảnh
    annotation theo GT state — từ đây chốt drowsy_start + xác nhận mapping."""
    idx = DmdIndex()
    for trip in SAMPLES:
        _, hashes = scan_trip(trip)
        gt = sample_labels(trip)
        vi, fi, di = idx.match(hashes)
        print(f"\n{trip}:")
        by_state: dict[str, list] = defaultdict(list)
        for k in range(len(hashes)):
            by_state[gt[k]].append((vi[k], fi[k], di[k]))
        for state, rows in by_state.items():
            good = [(v, f) for v, f, d in rows if d <= CONF_DIST]
            if not good:
                print(f"  {state}: KHÔNG match dist<={CONF_DIST} (p50="
                      f"{int(np.median([d for _, _, d in rows]))}) — nguồn ngoài bộ s5?")
                continue
            vids = Counter(v for v, _ in good)
            vmain, cnt = vids.most_common(1)[0]
            fpos = sorted(f for v, f in good if v == vmain)
            ann = load_annotation(idx.videos[vmain])
            n = ann["n"]
            stats = [window_stats(ann, int(f)) for v, f in good if v == vmain]
            agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            purity = cnt / len(good)
            print(f"  {state}: match {len(good)}/{len(rows)} → "
                  f"{idx.videos[vmain].parent.parent.name}/{idx.videos[vmain].parent.name}/"
                  f"{idx.videos[vmain].stem} "
                  f"(purity={purity:.0%}, shift={ann['shift']}) "
                  f"vị trí {fpos[0] / n:.2f}→{fpos[-1] / n:.2f} | ±{WIN}f: "
                  f"yawn={agg['yawn']:.0%} close={agg['close']:.0%} "
                  f"trans={agg['trans']:.0%} blink={agg['blink']:.0%} "
                  f"sleepy={agg['sleepy']:.2f} "
                  f"distracted={agg['distracted']:.0%} safe={agg['safe']:.0%}")


def suggest_label(agg: dict[str, float]) -> str:
    """Mapping mật độ annotation → lớp hackathon (mốc từ --calibrate 26/07:
    T03-yawning yawn=67%; T01-alert sleepy=0.62 vs T02/T06-drowsy 0.87/0.90)."""
    if agg["distracted"] >= 0.30:
        return "distracted"
    if agg["yawn"] >= 0.30:
        return "yawning"
    if agg["close"] >= 0.30:
        return "microsleep"
    return "drowsy" if agg["sleepy"] >= 0.75 else "alert"


def run_label(
    drowsy_start: float,
    trips: list[str] | None = None,
) -> None:  # drowsy_start giữ cho CLI cũ, không dùng
    idx = DmdIndex()
    for trip in (trips or SCORED):
        _, hashes = scan_trip(trip)
        vi, fi, di = idx.match(hashes)
        print(f"\n{trip}:")
        for a, b in merged_segments(hashes):
            conf = di[a:b] <= CONF_DIST
            cov = float(conf.mean())
            if cov < 0.3:
                print(f"  [{a}-{b}) KHÔNG match DMD s5 (cov={cov:.0%}) — nguồn s2/removed?")
                continue
            vmain, cnt = Counter(vi[a:b][conf]).most_common(1)[0]
            purity = cnt / int(conf.sum())
            v = idx.videos[vmain]
            ann = load_annotation(v)
            fsel = [int(f) for k, f in enumerate(fi[a:b])
                    if conf[k] and vi[a:b][k] == vmain]
            stats = [window_stats(ann, f) for f in fsel]
            agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            subj = f"{v.parent.parent.parent.name}/{v.parent.parent.name}"
            print(f"  [{a}-{b}) → {subj} cov={cov:.0%} purity={purity:.0%} "
                  f"session={v.parent.name} "
                  f"pos={min(fsel) / ann['n']:.2f}→{max(fsel) / ann['n']:.2f} | "
                  f"yawn={agg['yawn']:.0%} close={agg['close']:.0%} "
                  f"sleepy={agg['sleepy']:.2f} distracted={agg['distracted']:.0%} "
                  f"safe={agg['safe']:.0%} → ĐỀ XUẤT: {suggest_label(agg)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hash", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--label", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="số video hash song song (khuyên 4 trên máy hiện tại)")
    p.add_argument("--trips", nargs="*", choices=SCORED,
                   help="chỉ label các trip đã chọn (mặc định: toàn bộ T01d..T10d)")
    p.add_argument("--drowsy-start", type=float, default=0.15,
                   help="mốc %% video: trước=alert, sau=drowsy (chốt sau --calibrate)")
    args = p.parse_args()
    if args.hash:
        run_hash(max(1, args.workers))
    if args.calibrate:
        run_calibrate()
    if args.label:
        run_label(args.drowsy_start, args.trips)
    if not (args.hash or args.calibrate or args.label):
        p.error("cần --hash / --calibrate / --label")
    return 0


if __name__ == "__main__":
    sys.exit(main())
