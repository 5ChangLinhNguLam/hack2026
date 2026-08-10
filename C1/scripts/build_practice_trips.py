"""Chuyển Practice_Dataset (6 trip có ground truth) sang schema trip dùng chung.

Đây là nhãn ĐÚNG NGHĨA cho bài Collision Risk Monitor của hackathon:
`min_ttc` là TTC động học tới target trong collision cone — khác hẳn nhãn của
datasets/combined_ego (thời gian tới vụ va chạm đã thực sự xảy ra). Một chiếc xe
đang tiến gần xe trước có min_ttc hữu hạn dù chẳng có tai nạn nào.

Đầu ra
  datasets/practice/trips/<trip>.mp4        600 frame @20fps, 640x360 (từ kitti/image_2)
  datasets/practice/annotations/trips.csv   1 dòng / trip
  datasets/practice/annotations/ttc_per_frame.csv   1 dòng / frame

Quy ước nhãn (giữ nguyên ngữ nghĩa của bộ gốc, không phát minh gì thêm)
  min_ttc_s  = min ttc_simple trên các target CÓ in_collision_cone; Infinity nếu
               không có target nào trong cone. Ghi ra CSV là chuỗi "inf".
  inv_ttc    = 1/min_ttc, và = 0 khi min_ttc vô hạn. Đây mới là đích hồi quy:
               "không có nguy cơ" là inv = 0, một con số bình thường, trong khi
               TTC = vô cùng thì không đưa vào loss được.
  cls        = 1 nếu min_ttc <= POS_TTC, 0 nếu min_ttc >= NEG_TTC hoặc vô hạn
  keep       = 0 ở khoảng giữa (POS_TTC, NEG_TTC) — vùng biên, gán nhãn bừa ở đó
               là dạy model sai chỗ đường biên nằm.
"""

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import cv2

import weather_features as wx

# console Windows mặc định cp1252, không in nổi tiếng Việt trong log
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

POS_TTC, NEG_TTC = 2.0, 4.0
TTC_FLOOR = 0.5              # 1/TTC chặn trên ở 2.0, giống lúc train student
CONE_HALF_WIDTH = 2.5        # suy ngược từ chính Practice_Dataset: khớp 100%
                             # trên 21.810 target-frame (x>0 và |y| <= 2.5 m)


def load_trip_json(trip_dir: Path):
    gz = trip_dir / f"{trip_dir.name}.json.gz"
    raw = trip_dir / f"{trip_dir.name}.json"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.load(fh)          # Python đọc được token Infinity trần
    with raw.open(encoding="utf-8") as fh:
        return json.load(fh)


def frame_targets(min_ttc: float):
    """min_ttc -> (cls, keep, inv_ttc)."""
    if not math.isfinite(min_ttc):
        return 0.0, 1.0, 0.0
    inv = 1.0 / max(min_ttc, TTC_FLOOR)
    if min_ttc <= POS_TTC:
        return 1.0, 1.0, inv
    if min_ttc >= NEG_TTC:
        return 0.0, 1.0, inv
    return 0.0, 0.0, inv                  # vùng biên: có đích hồi quy, bỏ khỏi phân loại


def recompute_min_ttc(targets):
    """Tính lại min_ttc từ targets để kiểm chứng cột có sẵn."""
    vals = [t["ttc_simple"] for t in targets
            if t["longitudinal_distance"] > 0
            and abs(t["lateral_distance"]) <= CONE_HALF_WIDTH]
    return min(vals) if vals else float("inf")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=root / "datasets" / "Practice_Dataset")
    ap.add_argument("--out", type=Path, default=root / "datasets" / "practice")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    (args.out / "trips").mkdir(parents=True, exist_ok=True)
    ann = args.out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    trips = sorted(d for d in args.src.iterdir() if d.is_dir())
    if not trips:
        raise SystemExit(f"không thấy trip nào trong {args.src}")

    tf = (ann / "trips.csv").open("w", newline="", encoding="utf-8")
    ff = (ann / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8")
    tw, fw = csv.writer(tf), csv.writer(ff)
    tw.writerow(["trip_id", "video", "num_frames", "fps", "width", "height",
                 "map", "weather_desc", "speed_limit_kmh", "events",
                 "n_finite_ttc", "n_ttc_lt2", "n_ttc_lt15", "min_ttc_overall",
                 "driver_states", "safe_driving_score", *wx.COLS])
    fw.writerow(["trip_id", "frame_id", "timestamp", "min_ttc_s", "inv_ttc",
                 "cls", "keep", "headway_s", "ego_speed_kmh",
                 "ego_long_accel", "ego_lat_accel", "n_targets", "n_in_cone",
                 "driver_state", "alertness", "risk_final", "events_active"])

    mismatch = 0
    for d in trips:
        j = load_trip_json(d)
        md, frames = j["metadata"], j["frames"]
        imgs = sorted((d / "kitti" / "image_2").glob("*.jpg"))
        if len(imgs) != len(frames):
            print(f"  CẢNH BÁO {d.name}: {len(imgs)} ảnh vs {len(frames)} frame JSON")

        vid = args.out / "trips" / f"{d.name}.mp4"
        writer = None
        if not args.no_video:
            h, w = cv2.imread(str(imgs[0])).shape[:2]
            writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"),
                                     md["fps"], (w, h))

        n_fin = n_lt2 = n_lt15 = 0
        best = float("inf")
        states = {}
        for i, fr in enumerate(frames):
            m = float(fr["min_ttc"])
            # đối chiếu cột có sẵn với công thức cone đã suy ngược
            if not math.isclose(m, recompute_min_ttc(fr["targets"]),
                                rel_tol=1e-6, abs_tol=1e-6):
                mismatch += 1
            cls, keep, inv = frame_targets(m)
            in_cone = sum(1 for t in fr["targets"] if t["in_collision_cone"])
            drv = fr.get("driver") or {}
            n_fin += math.isfinite(m)
            n_lt2 += m < POS_TTC
            n_lt15 += m < 1.5
            best = min(best, m)
            states[drv.get("state", "")] = states.get(drv.get("state", ""), 0) + 1

            fw.writerow([
                d.name, fr["frame_id"], fr["timestamp"],
                "inf" if not math.isfinite(m) else round(m, 4), round(inv, 6),
                int(cls), int(keep),
                "inf" if not math.isfinite(float(fr["headway_sec"])) else round(float(fr["headway_sec"]), 4),
                fr["ego"]["speed_kmh"], fr["ego"]["longitudinal_accel"],
                fr["ego"]["lateral_accel"], len(fr["targets"]), in_cone,
                drv.get("state", ""), drv.get("alertness_score", ""),
                (fr.get("risk") or {}).get("final_risk_score", ""),
                "|".join(e["event_type"] for e in fr.get("events_active", [])),
            ])
            if writer is not None and i < len(imgs):
                writer.write(cv2.imread(str(imgs[i])))

        if writer is not None:
            writer.release()

        tw.writerow([
            d.name, f"trips/{d.name}.mp4", len(frames), md["fps"],
            640, 360, md.get("map", ""), md.get("description", ""),
            md.get("speed_limit_kmh", ""),
            "|".join(f"{e['type']}@{e['t']:.0f}s" for e in j["events_log"]),
            n_fin, n_lt2, n_lt15,
            "inf" if not math.isfinite(best) else round(best, 3),
            "|".join(f"{k}:{v}" for k, v in states.items() if k),
            (j.get("trip_aggregate") or {}).get("safe_driving_score", ""),
            *(wx.from_carla_dict(md.get("weather"))[c] for c in wx.COLS),
        ])
        print(f"  {d.name}: {len(frames)} frame · TTC hữu hạn {n_fin} "
              f"({n_fin / len(frames):.1%}) · <2s {n_lt2} · <1.5s {n_lt15} · "
              f"min {best:.2f}s")

    tf.close()
    ff.close()
    print(f"\ncông thức cone tự suy: lệch {mismatch} frame so với cột min_ttc có sẵn")
    print(f"ghi vào {args.out}")


if __name__ == "__main__":
    main()
