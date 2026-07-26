"""Nhóm A — Toàn vẹn & alignment.

- T-A1 integrity.csv (P+S)
- F-A1 contact_sheet_T0X.png (P, 6 file)
- T-A2 redaction_shape.csv (S)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from .common import Out, finite_or_none
from .io import Trip, _is_erased

CONTACT_SHEET_SEED = 1337  # cố định để chạy lại ra cùng frame
CONTACT_SHEET_N = 5


# ---------------------------------------------------------------- T-A1
def t_a1_integrity(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    for t in trips:
        labels = t.label_paths()
        n_label_nonempty = sum(1 for p in labels if p.stat().st_size > 0)
        w = t.metadata.get("weather") or {}
        rows.append({
            "trip_id": t.trip_id,
            "has_gt": t.has_gt,
            "n_frames": t.n_frames,
            "fps": t.fps,
            "n_image_2": t.count_left(),
            "n_image_3": t.count_right(),
            "n_driver": t.count_driver(),
            "n_depth": t.count_depth(),
            "n_label": len(labels),
            "n_label_nonempty": n_label_nonempty,
            "map": t.metadata.get("map"),
            "speed_limit_kmh": t.metadata.get("speed_limit_kmh"),
            "weather_cloudiness": w.get("cloudiness"),
            "weather_precipitation": w.get("precipitation"),
            "weather_fog_density": w.get("fog_density"),
            "weather_sun_altitude": w.get("sun_altitude_angle"),
            "weather_wetness": w.get("wetness"),
            "duration_sec": t.metadata.get("duration_sec"),
            "n_events_log": len(t.events_log),
        })
    df = pd.DataFrame(rows)
    out.table(df, "integrity.csv")
    return df


# ---------------------------------------------------------------- F-A1
def _caption(fr: Dict[str, Any]) -> str:
    ego = fr.get("ego") or {}
    drv = fr.get("driver") or {}
    ttc = fr.get("min_ttc")
    ttc_s = "n/a" if ttc is None else ("inf" if finite_or_none(ttc) is None else f"{ttc:.2f}s")
    evs = [e.get("event_type", "?") for e in (fr.get("events_active") or [])]
    ev_lines = "\n      ".join(evs) if evs else "-"
    return (f"f={fr.get('frame_id')} t={fr.get('timestamp', 0):.2f}s  "
            f"state={drv.get('state', 'n/a')}\n"
            f"min_ttc={ttc_s}  speed={ego.get('speed_kmh', float('nan')):.1f}km/h\n"
            f"event={ev_lines}")


def f_a1_contact_sheets(trips: List[Trip], out: Out) -> List[Path]:
    """5 frame ngẫu nhiên/trip: ảnh road-trái trên, ảnh driver dưới, caption JSON."""
    paths = []
    for t in trips:
        rng = random.Random(CONTACT_SHEET_SEED + hash(t.trip_id) % 1000)
        ids = sorted(rng.sample(range(t.n_frames), min(CONTACT_SHEET_N, t.n_frames)))
        fig, axes = plt.subplots(2, len(ids), figsize=(3.6 * len(ids), 6.2))
        if len(ids) == 1:
            axes = axes.reshape(2, 1)
        for col, i in enumerate(ids):
            fr = t.frames[i]
            for row, getter in ((0, t.left_path), (1, t.driver_path)):
                ax = axes[row, col]
                p = getter(i)
                if p is not None:
                    img = cv2.imread(str(p))
                    if img is not None:
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        ax.text(.5, .5, "không đọc được", ha="center")
                else:
                    ax.text(.5, .5, "thiếu ảnh", ha="center")
                ax.set_xticks([]); ax.set_yticks([])
            axes[0, col].set_title(_caption(fr), fontsize=7.5, loc="left")
        axes[0, 0].set_ylabel("road (image_2)", fontsize=9)
        axes[1, 0].set_ylabel("driver (cabin)", fontsize=9)
        fig.suptitle(f"Contact sheet — {t.trip_id}  "
                     f"({t.n_frames} frame, {t.metadata.get('map', '?')}, "
                     f"limit {t.metadata.get('speed_limit_kmh', '?')} km/h)", fontsize=11)
        paths.append(out.figure(fig, f"contact_sheet_{t.trip_id}.png"))
    return paths


# ---------------------------------------------------------------- T-A2
def _status(value: Any, *, missing_key: bool) -> str:
    if missing_key:
        return "MẤT (không có key)"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)) and len(value) == 0:
        return "RỖNG"
    return "còn"


def _field_status(container: Dict[str, Any], key: str) -> str:
    return _status(container.get(key), missing_key=key not in container)


def t_a2_redaction_shape(trips: List[Trip], out: Out) -> pd.DataFrame:
    """Bằng chứng máy móc cho fact #8 + phát hiện 10 trip có đồng nhất không."""
    rows = []
    for t in trips:
        fr = t.frames[0] if t.frames else {}
        ego = fr.get("ego") or {}
        tgt0 = (fr.get("targets") or [{}])[0] if (fr.get("targets") or []) else {}
        ev0 = (fr.get("events_active") or [{}])[0] if (fr.get("events_active") or []) else {}
        # events_active có thể rỗng ở frame 0 → tìm frame đầu tiên có event
        if not ev0:
            for f in t.frames:
                if f.get("events_active"):
                    ev0 = f["events_active"][0]
                    break
        # targets có thể rỗng ở frame 0 → tìm frame đầu tiên có target
        if not tgt0:
            for f in t.frames:
                if f.get("targets"):
                    tgt0 = f["targets"][0]
                    break
        evlog0 = t.events_log[0] if t.events_log else {}

        row = {
            "trip_id": t.trip_id,
            "has_gt": t.has_gt,
            # top-level
            "trip_aggregate": _field_status(t.raw, "trip_aggregate"),
            "driver_summary": _field_status(t.raw, "driver_summary"),
            # frame-level GT
            "frames.driver": _field_status(fr, "driver"),
            "frames.min_ttc": _field_status(fr, "min_ttc"),
            "frames.headway_sec": _field_status(fr, "headway_sec"),
            "frames.behavior_flags": _field_status(fr, "behavior_flags"),
            "frames.risk": _field_status(fr, "risk"),
            # ego
            "ego.speed_kmh": _field_status(ego, "speed_kmh"),
            "ego.longitudinal_accel": _field_status(ego, "longitudinal_accel"),
            "ego.lateral_accel": _field_status(ego, "lateral_accel"),
            "ego.location": _field_status(ego, "location"),
            "ego.rotation": _field_status(ego, "rotation"),
            "ego.geolocation": _field_status(ego, "geolocation"),
            # targets[]
            "targets[].target_id": _field_status(tgt0, "target_id"),
            "targets[].target_class": _field_status(tgt0, "target_class"),
            "targets[].rel_pos": _field_status(tgt0, "rel_pos"),
            "targets[].closing_speed": _field_status(tgt0, "closing_speed"),
            "targets[].ttc_simple": _field_status(tgt0, "ttc_simple"),
            "targets[].ttc_2d": _field_status(tgt0, "ttc_2d"),
            "targets[].in_collision_cone": _field_status(tgt0, "in_collision_cone"),
            # events
            "events_active[].event_id": _field_status(ev0, "event_id"),
            "events_active[].event_type": _field_status(ev0, "event_type"),
            "events_active[].age_sec": _field_status(ev0, "age_sec"),
            "events_active[].actor_ids": _field_status(ev0, "actor_ids"),
            "events_active[].params": _field_status(ev0, "params"),
            "events_log[].t": _field_status(evlog0, "t"),
            "events_log[].type": _field_status(evlog0, "type"),
            "events_log[].params": _field_status(evlog0, "params"),
            # metadata
            "metadata.speed_limit_kmh": _field_status(t.metadata, "speed_limit_kmh"),
            "metadata.weather": _field_status(t.metadata, "weather"),
        }
        # KITTI label location (fact: xyz bị zero ở trip chấm điểm)
        row["label_2.location_xyz"], row["label_2.n_lines_checked"] = _label_xyz_status(t)
        rows.append(row)
    df = pd.DataFrame(rows)
    out.table(df, "redaction_shape.csv")
    return df


def _label_xyz_status(t: Trip) -> tuple[str, int]:
    """Kiểm x/y/z trong label_2 có bị zero hết không (chỉ nhìn file khác rỗng).

    Trả (trạng thái, số dòng đã kiểm) — tách rời để so sánh đồng nhất giữa
    các trip không bị số đếm làm nhiễu.
    """
    nonempty = [p for p in t.label_paths() if p.stat().st_size > 0]
    if not nonempty:
        return "không có label khác rỗng", 0
    n_check, n_zero = 0, 0
    for p in nonempty[:200]:
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) != 15:
                continue
            n_check += 1
            if all(abs(float(v)) < 1e-9 for v in parts[11:14]):
                n_zero += 1
    if n_check == 0:
        return "không parse được", 0
    if n_zero == n_check:
        return "ZERO hết", n_check
    if n_zero == 0:
        return "còn", n_check
    return f"lẫn lộn ({n_zero}/{n_check} zero)", n_check


def run(trips_p: List[Trip], trips_s: List[Trip], out: Out) -> Dict[str, Any]:
    print("[Nhóm A] Toàn vẹn & alignment")
    res = {}
    res["integrity"] = t_a1_integrity(trips_p + trips_s, out)
    res["contact_sheets"] = f_a1_contact_sheets(trips_p, out)
    # T-A2 dành cho S; kèm luôn dòng của P làm mốc đối chiếu "còn" trong CÙNG bảng
    res["redaction"] = t_a2_redaction_shape(trips_p + trips_s, out)
    return res
