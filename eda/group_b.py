"""Nhóm B — C1: TTC, targets, events (cho Hải).

- T-B1 ttc_summary.csv (P)          - F-B1 min_ttc_timeline_T0X.png (P, 6)
- T-B2 targets_summary.csv (P)      - F-B2 ttc_hist.png (P)
- T-B3 events_lifetime.csv (P+S)    - F-B3 ttc_simple_vs_2d.png (P)
- T-B4 labels_vs_events.csv (P)     - F-B4 relpos_scatter.png (P)
- T-B5 event_density_compare.csv    - F-B5 event_density.png (P+S)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (CRITICAL_TTC_SEC, EVENT_COLORS, EVENT_ORDER,
                     NEAR_MISS_TTC_SEC, Out, event_intervals, finite_or_none)
from .io import Trip


# ---------------------------------------------------------------- T-B1
def t_b1_ttc_summary(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    for t in trips:
        vals = [fr.get("min_ttc") for fr in t.frames]
        finite = [v for v in (finite_or_none(v) for v in vals) if v is not None]
        n = len(vals)
        n_inf = n - len(finite)
        arr = np.array(finite) if finite else np.array([])
        rows.append({
            "trip_id": t.trip_id,
            "n_frames": n,
            "n_inf": n_inf,
            "pct_inf": round(100 * n_inf / n, 2) if n else None,
            "n_finite": len(finite),
            "n_lt_3s": int((arr < CRITICAL_TTC_SEC).sum()) if finite else 0,
            "pct_lt_3s": round(100 * float((arr < CRITICAL_TTC_SEC).mean()) if finite else 0, 2)
            if finite else 0.0,
            "n_lt_1p5s": int((arr < NEAR_MISS_TTC_SEC).sum()) if finite else 0,
            "pct_lt_1p5s": round(100 * (arr < NEAR_MISS_TTC_SEC).sum() / n, 2) if n else None,
            "ttc_min": round(float(arr.min()), 3) if finite else None,
            "ttc_p25": round(float(np.percentile(arr, 25)), 3) if finite else None,
            "ttc_median": round(float(np.median(arr)), 3) if finite else None,
            "ttc_max": round(float(arr.max()), 3) if finite else None,
        })
    df = pd.DataFrame(rows)
    # pct_lt_3s tính trên TOÀN BỘ frame để so sánh giữa trip
    df["pct_lt_3s"] = (df["n_lt_3s"] / df["n_frames"] * 100).round(2)
    out.table(df, "ttc_summary.csv")
    return df


# ---------------------------------------------------------------- F-B1
def f_b1_min_ttc_timeline(trips: List[Trip], out: Out) -> None:
    for t in trips:
        ts = [fr.get("timestamp", i / t.fps) for i, fr in enumerate(t.frames)]
        vals = [finite_or_none(fr.get("min_ttc")) for fr in t.frames]
        fig, ax = plt.subplots(figsize=(13, 4))

        # nền tô khoảng event theo loại (dùng khoảng active THẬT, fact #9)
        used = {}
        for sp in event_intervals(t):
            typ = sp["event_type"]
            c = EVENT_COLORS.get(typ, "#888888")
            t0 = sp["frame_start"] / t.fps
            t1 = (sp["frame_end"] + 1) / t.fps
            ax.axvspan(t0, t1, color=c, alpha=0.16, zorder=0)
            used[typ] = c

        finite_mask = [v is not None for v in vals]
        ax.plot([x for x, m in zip(ts, finite_mask) if m],
                [v for v, m in zip(vals, finite_mask) if m],
                lw=1.2, color="#222222", label="min_ttc (hữu hạn)")
        # đánh dấu frame inf ở đáy
        inf_ts = [x for x, m in zip(ts, finite_mask) if not m]
        if inf_ts:
            ax.plot(inf_ts, [0] * len(inf_ts), "|", ms=6, color="#bbbbbb",
                    label=f"inf ({len(inf_ts)} frame)")

        ax.axhline(NEAR_MISS_TTC_SEC, color="#e74c3c", ls="--", lw=1,
                   label=f"near-miss {NEAR_MISS_TTC_SEC}s")
        ax.axhline(CRITICAL_TTC_SEC, color="#f39c12", ls="--", lw=1,
                   label=f"critical {CRITICAL_TTC_SEC}s")
        finite_vals = [v for v in vals if v is not None]
        ax.set_ylim(0, min(30, max(finite_vals) * 1.05) if finite_vals else 10)
        ax.set_xlabel("thời gian (s)"); ax.set_ylabel("min_ttc (s)")
        ax.set_title(f"{t.trip_id} — min_ttc theo thời gian "
                     f"(nền = khoảng event active)")
        handles, labels = ax.get_legend_handles_labels()
        for typ, c in used.items():
            handles.append(mpatches.Patch(color=c, alpha=0.3, label=typ))
        ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper right")
        out.figure(fig, f"min_ttc_timeline_{t.trip_id}.png")


# ---------------------------------------------------------------- F-B2
def f_b2_ttc_hist(trips: List[Trip], out: Out) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, t in zip(axes.ravel(), trips):
        finite = [v for v in (finite_or_none(fr.get("min_ttc")) for fr in t.frames)
                  if v is not None]
        n_inf = t.n_frames - len(finite)
        clipped = [min(v, 30) for v in finite]
        ax.hist(clipped, bins=40, color="#3498db", edgecolor="white")
        ax.axvline(NEAR_MISS_TTC_SEC, color="#e74c3c", ls="--", lw=1.2)
        ax.axvline(CRITICAL_TTC_SEC, color="#f39c12", ls="--", lw=1.2)
        ax.set_title(f"{t.trip_id}  (hữu hạn {len(finite)}, inf {n_inf})", fontsize=10)
        ax.set_xlabel("min_ttc (s, cắt ở 30)")
    fig.suptitle("F-B2 — Histogram min_ttc hữu hạn (đỏ 1.5s, cam 3.0s); "
                 "frame inf tách riêng không vẽ", fontsize=12)
    fig.tight_layout()
    out.figure(fig, "ttc_hist.png")


# ---------------------------------------------------------------- T-B2
def t_b2_targets_summary(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    for t in trips:
        per_frame, cls_counter, in_cone = [], Counter(), 0
        total_targets = 0
        track_frames: Dict[Any, int] = defaultdict(int)
        track_class: Dict[Any, str] = {}
        for fr in t.frames:
            tg = fr.get("targets") or []
            per_frame.append(len(tg))
            for x in tg:
                total_targets += 1
                cls_counter[x.get("target_class")] += 1
                if x.get("in_collision_cone"):
                    in_cone += 1
                tid = x.get("target_id")
                track_frames[tid] += 1
                track_class[tid] = x.get("target_class")
        life = np.array(list(track_frames.values())) if track_frames else np.array([])
        rows.append({
            "trip_id": t.trip_id,
            "n_frames": t.n_frames,
            "targets_total": total_targets,
            "targets_per_frame_mean": round(float(np.mean(per_frame)), 2),
            "targets_per_frame_max": int(np.max(per_frame)) if per_frame else 0,
            "n_frames_no_target": int(sum(1 for v in per_frame if v == 0)),
            "n_vehicle": cls_counter.get("vehicle", 0),
            "n_walker": cls_counter.get("walker", 0),
            "n_bike": cls_counter.get("bike", 0),
            "pct_in_cone": round(100 * in_cone / total_targets, 2) if total_targets else None,
            "n_tracks": len(track_frames),
            "track_life_min": int(life.min()) if life.size else None,
            "track_life_median": float(np.median(life)) if life.size else None,
            "track_life_max": int(life.max()) if life.size else None,
        })
    df = pd.DataFrame(rows)
    out.table(df, "targets_summary.csv")
    return df


# ---------------------------------------------------------------- F-B3
def f_b3_ttc_simple_vs_2d(trips: List[Trip], out: Out) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.ravel(), trips):
        xs, ys, n_both_inf = [], [], 0
        for fr in t.frames:
            for tg in fr.get("targets") or []:
                if not tg.get("in_collision_cone"):
                    continue  # chỉ target trong cone
                a = finite_or_none(tg.get("ttc_simple"))
                b = finite_or_none(tg.get("ttc_2d"))
                if a is None or b is None:
                    n_both_inf += 1
                    continue
                xs.append(a); ys.append(b)
        ax.scatter(xs, ys, s=6, alpha=0.35, color="#2980b9")
        lim = max(xs + ys + [1]) if xs else 1
        lim = min(lim, 60)
        ax.plot([0, lim], [0, lim], "--", color="#e74c3c", lw=1, label="y = x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("ttc_simple (s)"); ax.set_ylabel("ttc_2d (s)")
        ax.set_title(f"{t.trip_id}  (n={len(xs)}, có inf: {n_both_inf})", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("F-B3 — ttc_simple vs ttc_2d (chỉ target in_collision_cone=True)",
                 fontsize=12)
    fig.tight_layout()
    out.figure(fig, "ttc_simple_vs_2d.png")


# ---------------------------------------------------------------- F-B4
Y_VIEW_M = 40.0  # cắt trục ngang: target xa hàng trăm mét làm mất hình dạng cone


def f_b4_relpos_scatter(trips: List[Trip], out: Out) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, t in zip(axes.ravel(), trips):
        n_off = 0
        for cone, color, lbl in ((False, "#bdc3c7", "ngoài cone"),
                                 (True, "#e74c3c", "trong cone")):
            pts = [(tg["rel_pos"]["y"], tg["rel_pos"]["x"])
                   for fr in t.frames for tg in (fr.get("targets") or [])
                   if bool(tg.get("in_collision_cone")) is cone and tg.get("rel_pos")]
            n_off += sum(1 for y, _ in pts if abs(y) > Y_VIEW_M)
            ax.scatter([y for y, _ in pts], [x for _, x in pts],
                       s=4, alpha=0.4, color=color, label=f"{lbl} ({len(pts)})")
        ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        ax.set_xlim(-Y_VIEW_M, Y_VIEW_M)
        ax.set_xlabel("lateral y (m)"); ax.set_ylabel("longitudinal x (m)")
        ax.set_title(f"{t.trip_id}  ({n_off} điểm ngoài khung |y|>{Y_VIEW_M:.0f}m)",
                     fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"F-B4 — rel_pos của target, tô màu theo in_collision_cone "
                 f"(ego ở gốc, x = phía trước; trục y cắt ở ±{Y_VIEW_M:.0f}m)", fontsize=12)
    fig.tight_layout()
    out.figure(fig, "relpos_scatter.png")


# ---------------------------------------------------------------- T-B3
def t_b3_events_lifetime(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    for t in trips:
        # bản đồ target_id -> class để join actor_ids (fact #10)
        id2class: Dict[Any, str] = {}
        for fr in t.frames:
            for tg in fr.get("targets") or []:
                tid = tg.get("target_id")
                if tid is not None and tid not in id2class:
                    id2class[tid] = tg.get("target_class")
        for sp in event_intervals(t):
            actors = sorted(sp["actor_ids"])
            classes = sorted({id2class.get(a) for a in actors if id2class.get(a)})
            rows.append({
                "trip_id": t.trip_id,
                "has_gt": t.has_gt,
                "event_id": sp["event_id"],
                "event_type": sp["event_type"],
                "frame_start": sp["frame_start"],
                "frame_end": sp["frame_end"],
                "t_start_sec": round(sp["frame_start"] / t.fps, 2),
                "n_frames_active": sp["n_frames_active"],
                "lifetime_sec_age_max": round(sp["age_sec_max"], 2),
                "lifetime_sec_by_frames": round(sp["n_frames_active"] / t.fps, 2),
                "actor_ids": ";".join(str(a) for a in actors),
                "actor_class": ";".join(classes) if classes else "(không join được)",
                "reaches_last_frame": bool(sp["frame_end"] >= t.n_frames - 1),
            })
    df = pd.DataFrame(rows)
    out.table(df, "events_lifetime.csv")

    # thống kê theo type, đối chiếu fact #9
    if not df.empty:
        stat = (df.groupby(["event_type", "has_gt"])
                  .agg(n_instance=("event_id", "size"),
                       lifetime_min=("lifetime_sec_age_max", "min"),
                       lifetime_max=("lifetime_sec_age_max", "max"),
                       frames_min=("n_frames_active", "min"),
                       frames_max=("n_frames_active", "max"),
                       n_reach_last_frame=("reaches_last_frame", "sum"))
                  .reset_index())
        out.table(stat, "events_lifetime_by_type.csv")
    return df


# ---------------------------------------------------------------- T-B4
def t_b4_labels_vs_events(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    for t in trips:
        labels = {int(p.stem): p.stat().st_size > 0 for p in t.label_paths()}
        active = {fr.get("frame_id"): bool(fr.get("events_active"))
                  for fr in t.frames}
        n_ev = n_ev_lbl = n_noev = n_noev_lbl = 0
        for fid, has_lbl in labels.items():
            if active.get(fid):
                n_ev += 1; n_ev_lbl += has_lbl
            else:
                n_noev += 1; n_noev_lbl += has_lbl
        spans = event_intervals(t)
        first_event_frame = min((s["frame_start"] for s in spans), default=None)
        n_before = (sum(1 for fid, h in labels.items()
                        if h and first_event_frame is not None and fid < first_event_frame))
        rows.append({
            "trip_id": t.trip_id,
            "n_label_files": len(labels),
            "n_label_nonempty": sum(labels.values()),
            "first_event_frame": first_event_frame,
            "n_frames_event_active": n_ev,
            "n_label_when_event": n_ev_lbl,
            "pct_label_when_event": round(100 * n_ev_lbl / n_ev, 2) if n_ev else None,
            "n_frames_no_event": n_noev,
            "n_label_when_no_event": n_noev_lbl,
            "pct_label_when_no_event": round(100 * n_noev_lbl / n_noev, 2) if n_noev else None,
            "n_label_before_first_event": n_before,
        })
    df = pd.DataFrame(rows)
    out.table(df, "labels_vs_events.csv")
    return df


# ---------------------------------------------------------------- T-B5 / F-B5
def t_b5_event_density(trips_p: List[Trip], trips_s: List[Trip],
                       out: Out) -> pd.DataFrame:
    rows = []
    for t in trips_p + trips_s:
        n_active = sum(1 for fr in t.frames if fr.get("events_active"))
        max_concurrent = max((len(fr.get("events_active") or []) for fr in t.frames),
                             default=0)
        spans = event_intervals(t)
        by_type = Counter(s["event_type"] for s in spans)
        row = {
            "trip_id": t.trip_id,
            "set": "P (mẫu)" if t.has_gt else "S (chấm điểm)",
            "n_frames": t.n_frames,
            "n_frames_event_active": n_active,
            "pct_frames_event_active": round(100 * n_active / t.n_frames, 2)
            if t.n_frames else None,
            "n_event_instances": len(spans),
            "max_concurrent_events": max_concurrent,
        }
        for typ in EVENT_ORDER:
            row[f"n_{typ}"] = by_type.get(typ, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    out.table(df, "event_density_compare.csv")

    # F-B5 bar chart P cạnh S
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    colors = ["#2ecc71" if s.startswith("P") else "#e67e22" for s in df["set"]]
    ax1.bar(df["trip_id"], df["pct_frames_event_active"], color=colors)
    ax1.set_ylabel("% frame có ≥1 event active")
    ax1.set_title("F-B5a — Mật độ event theo trip (xanh = P mẫu, cam = S chấm điểm)")
    ax1.tick_params(axis="x", rotation=60)
    for grp, c in (("P (mẫu)", "#2ecc71"), ("S (chấm điểm)", "#e67e22")):
        m = df[df["set"] == grp]["pct_frames_event_active"].mean()
        if not np.isnan(m):
            ax1.axhline(m, color=c, ls="--", lw=1.2, label=f"{grp} TB={m:.1f}%")
    ax1.legend(fontsize=8)

    tot = df.groupby("set")[[f"n_{t}" for t in EVENT_ORDER]].sum()
    x = np.arange(len(EVENT_ORDER)); w = 0.38
    for k, (grp, vals) in enumerate(tot.iterrows()):
        ax2.bar(x + (k - .5) * w, vals.values, w, label=grp,
                color="#2ecc71" if grp.startswith("P") else "#e67e22")
    ax2.set_xticks(x); ax2.set_xticklabels(EVENT_ORDER, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("số instance event"); ax2.legend(fontsize=8)
    ax2.set_title("F-B5b — Tổng instance theo loại event")
    fig.tight_layout()
    out.figure(fig, "event_density.png")
    return df


def run(trips_p: List[Trip], trips_s: List[Trip], out: Out) -> Dict[str, Any]:
    print("[Nhóm B] C1 — TTC, targets, events")
    res = {}
    if trips_p:
        res["ttc_summary"] = t_b1_ttc_summary(trips_p, out)
        f_b1_min_ttc_timeline(trips_p, out)
        f_b2_ttc_hist(trips_p, out)
        res["targets_summary"] = t_b2_targets_summary(trips_p, out)
        f_b3_ttc_simple_vs_2d(trips_p, out)
        f_b4_relpos_scatter(trips_p, out)
        res["labels_vs_events"] = t_b4_labels_vs_events(trips_p, out)
    res["events_lifetime"] = t_b3_events_lifetime(trips_p + trips_s, out)
    res["event_density"] = t_b5_event_density(trips_p, trips_s, out)
    return res
