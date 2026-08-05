"""Nhóm D — C3: kinematics & score (cho Hải). TRỌNG TÂM v2.

Thứ tự bắt buộc: T-D1 → T-D2 → T-D3 (chốt công thức) → T-D4 → T-D5.

T-D3 là cổng chặn: nếu KHÔNG tổ hợp nào cho sai số = 0 tuyệt đối trên cả 6
trip mẫu thì DỪNG, báo bảng sai số, KHÔNG tự chọn "gần đúng nhất" và
KHÔNG chạy T-D5.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (G_MS2, HARSH_ACCEL_G, HARSH_BRAKE_G, HARSH_LATERAL_G,
                     NEAR_MISS_TTC_SEC, PENALTY_W, SPEEDING_TOLERANCE_KMH, Out,
                     finite_or_none)
from .io import Trip

FLAGS = ["harsh_brake", "harsh_accel", "harsh_corner", "speeding"]


# ================================================================= T-D1
def t_d1_trip_aggregate(trips: List[Trip], out: Out) -> pd.DataFrame:
    """trip_aggregate của JSON đặt cạnh số đếm tái tạo TỪ frames. Lệch phải = 0."""
    rows = []
    for t in trips:
        agg = t.trip_aggregate or {}
        n = t.n_frames
        cnt = {f: 0 for f in ["harsh_brake", "harsh_accel", "harsh_corner",
                              "speeding", "tailgating"]}
        near_miss = 0
        headways, risks = [], []
        for fr in t.frames:
            bf = fr.get("behavior_flags") or {}
            for f in cnt:
                if bf.get(f):
                    cnt[f] += 1
            mt = finite_or_none(fr.get("min_ttc"))
            if mt is not None and mt < NEAR_MISS_TTC_SEC:
                near_miss += 1
            hw = finite_or_none(fr.get("headway_sec"))
            if hw is not None:
                headways.append(hw)
            r = (fr.get("risk") or {}).get("final_risk_score")
            if r is not None:
                risks.append(float(r))

        recomputed = {
            "harsh_brake_count": cnt["harsh_brake"],
            "harsh_accel_count": cnt["harsh_accel"],
            "harsh_corner_count": cnt["harsh_corner"],
            "near_miss_count": near_miss,
            "speeding_pct_time": round(100 * cnt["speeding"] / n, 2) if n else 0.0,
            "tailgating_pct_time": round(100 * cnt["tailgating"] / n, 2) if n else 0.0,
            "avg_headway_sec": round(float(np.mean(headways)), 2) if headways else None,
            "max_risk_score": round(float(np.max(risks)), 2) if risks else None,
            "avg_risk_score": round(float(np.mean(risks)), 2) if risks else None,
        }
        row = {"trip_id": t.trip_id, "n_frames": n}
        for k, v in recomputed.items():
            j = agg.get(k)
            row[f"{k}__json"] = j
            row[f"{k}__recomputed"] = v
            if j is None or v is None:
                row[f"{k}__delta"] = None
            else:
                row[f"{k}__delta"] = round(float(v) - float(j), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    out.table(df, "trip_aggregate.csv")
    return df


# ================================================================= T-D2
def t_d2_safe_score_check(trips: List[Trip], out: Out) -> pd.DataFrame:
    """safe tính lại vs JSON + giá trị TRƯỚC clamp từng trip."""
    rows = []
    for t in trips:
        a = t.trip_aggregate or {}
        pen = (a.get("harsh_brake_count", 0) * PENALTY_W["harsh_brake"]
               + a.get("harsh_accel_count", 0) * PENALTY_W["harsh_accel"]
               + a.get("harsh_corner_count", 0) * PENALTY_W["harsh_corner"]
               + a.get("near_miss_count", 0) * PENALTY_W["near_miss"]
               + a.get("speeding_pct_time", 0.0) * PENALTY_W["speeding_pct"]
               + a.get("tailgating_pct_time", 0.0) * PENALTY_W["tailgating_pct"])
        pre_clamp = 100.0 - pen
        rows.append({
            "trip_id": t.trip_id,
            "safe_json": a.get("safe_driving_score"),
            "penalty_total": round(pen, 3),
            "safe_pre_clamp": round(pre_clamp, 3),
            "safe_recomputed": round(max(0.0, pre_clamp), 3),
            "delta": round(max(0.0, pre_clamp) - float(a.get("safe_driving_score", 0)), 6),
            "clamped": pre_clamp < 0,
            "margin_to_zero": round(-pre_clamp, 3),
            "pen_harsh_brake": round(a.get("harsh_brake_count", 0) * PENALTY_W["harsh_brake"], 2),
            "pen_harsh_accel": round(a.get("harsh_accel_count", 0) * PENALTY_W["harsh_accel"], 2),
            "pen_harsh_corner": round(a.get("harsh_corner_count", 0) * PENALTY_W["harsh_corner"], 2),
            "pen_near_miss": round(a.get("near_miss_count", 0) * PENALTY_W["near_miss"], 2),
            "pen_speeding": round(a.get("speeding_pct_time", 0.0) * PENALTY_W["speeding_pct"], 3),
            "pen_tailgating": round(a.get("tailgating_pct_time", 0.0) * PENALTY_W["tailgating_pct"], 3),
        })
    df = pd.DataFrame(rows)
    out.table(df, "safe_score_check.csv")
    return df


# ================================================================= T-D3
def _apply_flags(t: Trip, unit: str, boundary: str) -> Dict[str, List[bool]]:
    """Áp ngưỡng evaluation.py lên ego kinematics với 1 lựa chọn đơn vị+biên."""
    scale = 1.0 if unit == "g" else G_MS2  # ngưỡng quy về cùng đơn vị với dữ liệu
    thr_brake = -HARSH_BRAKE_G * scale
    thr_accel = HARSH_ACCEL_G * scale
    thr_lat = HARSH_LATERAL_G * scale
    limit = t.metadata.get("speed_limit_kmh")
    thr_speed = (float(limit) + SPEEDING_TOLERANCE_KMH) if limit is not None else None

    ge = boundary == ">="
    out: Dict[str, List[bool]] = {f: [] for f in FLAGS}
    for fr in t.frames:
        ego = fr.get("ego") or {}
        la = float(ego.get("longitudinal_accel", 0.0) or 0.0)
        lat = abs(float(ego.get("lateral_accel", 0.0) or 0.0))
        sp = float(ego.get("speed_kmh", 0.0) or 0.0)
        out["harsh_brake"].append(la <= thr_brake if ge else la < thr_brake)
        out["harsh_accel"].append(la >= thr_accel if ge else la > thr_accel)
        out["harsh_corner"].append(lat >= thr_lat if ge else lat > thr_lat)
        out["speeding"].append(
            False if thr_speed is None
            else (sp >= thr_speed if ge else sp > thr_speed))
    return out


def _count_mode(mask: List[bool], mode: str) -> int:
    """frame = đếm số frame True; event = đếm số đoạn True liên tiếp."""
    if mode == "frame":
        return int(sum(mask))
    n, prev = 0, False
    for v in mask:
        if v and not prev:
            n += 1
        prev = v
    return n


def t_d3_flag_recipe_verify(trips: List[Trip], out: Out) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Duyệt mọi tổ hợp {đơn vị} × {biên} × {đếm}; tổ hợp sai số 0 tuyệt đối = chốt."""
    rows = []
    for unit, boundary, mode in itertools.product(("g", "m/s2"), (">", ">="),
                                                  ("frame", "event")):
        err_flag_total = 0        # lệch so với trip_aggregate counts
        err_perframe_total = 0    # lệch so với behavior_flags từng frame
        per_trip = []
        for t in trips:
            agg = t.trip_aggregate or {}
            masks = _apply_flags(t, unit, boundary)
            n = t.n_frames
            # so với trip_aggregate
            e = 0
            for f, key in (("harsh_brake", "harsh_brake_count"),
                           ("harsh_accel", "harsh_accel_count"),
                           ("harsh_corner", "harsh_corner_count")):
                e += abs(_count_mode(masks[f], mode) - int(agg.get(key, 0)))
            pred_speed_pct = 100 * _count_mode(masks["speeding"], mode) / n if n else 0.0
            e_speed = abs(round(pred_speed_pct, 2) - float(agg.get("speeding_pct_time", 0.0)))
            e += 0 if e_speed < 1e-9 else e_speed
            # so với behavior_flags từng frame (không phụ thuộc mode đếm)
            pf = 0
            for i, fr in enumerate(t.frames):
                bf = fr.get("behavior_flags") or {}
                for f in FLAGS:
                    if bool(bf.get(f)) != masks[f][i]:
                        pf += 1
            err_flag_total += e
            err_perframe_total += pf
            per_trip.append((t.trip_id, e, pf))
        rows.append({
            "unit": unit, "boundary": boundary, "count_mode": mode,
            "err_vs_trip_aggregate": round(err_flag_total, 4),
            "err_vs_behavior_flags_perframe": err_perframe_total,
            "exact_match": err_flag_total == 0 and err_perframe_total == 0,
            **{f"err_agg_{tid}": e for tid, e, _ in per_trip},
            **{f"err_frame_{tid}": pf for tid, _, pf in per_trip},
        })
    df = pd.DataFrame(rows).sort_values(
        ["err_vs_trip_aggregate", "err_vs_behavior_flags_perframe"])
    out.table(df, "flag_recipe_verify.csv")

    winners = df[df["exact_match"]]
    if len(winners) == 0:
        return df, None
    w = winners.iloc[0]
    return df, {"unit": w["unit"], "boundary": w["boundary"],
                "count_mode": w["count_mode"], "n_winners": len(winners)}


# ================================================================= T-D4
def t_d4_error_budget(trips: List[Trip], safe_df: pd.DataFrame, out: Out) -> pd.DataFrame:
    """Đóng góp từng hạng mục vào tổng phạt + độ nhạy (Δsafe → ΔC3 = 2×Δsafe)."""
    rows = []
    for _, r in safe_df.iterrows():
        pen = r["penalty_total"]
        row = {"trip_id": r["trip_id"], "penalty_total": pen,
               "safe_pre_clamp": r["safe_pre_clamp"], "margin_to_zero": r["margin_to_zero"]}
        for k in ("harsh_brake", "harsh_accel", "harsh_corner", "near_miss",
                  "speeding", "tailgating"):
            p = r[f"pen_{k}"]
            row[f"pen_{k}"] = p
            row[f"share_{k}_pct"] = round(100 * p / pen, 2) if pen else None
        # độ nhạy: sai 1 đơn vị đếm → Δpenalty → Δsafe → ΔC3 (=2×Δsafe)
        row["d_safe_per_1_near_miss"] = PENALTY_W["near_miss"]
        row["d_c3_per_1_near_miss"] = 2 * PENALTY_W["near_miss"]
        row["d_safe_per_1_harsh_brake"] = PENALTY_W["harsh_brake"]
        row["d_c3_per_1_harsh_brake"] = 2 * PENALTY_W["harsh_brake"]
        row["d_safe_per_1pct_tailgating"] = PENALTY_W["tailgating_pct"]
        row["d_c3_per_1pct_tailgating"] = 2 * PENALTY_W["tailgating_pct"]
        row["d_safe_per_1pct_speeding"] = PENALTY_W["speeding_pct"]
        # còn phải sai bao nhiêu near_miss nữa thì safe mới thoát khỏi 0
        row["near_miss_error_to_escape_clamp"] = (
            None if not r["clamped"]
            else int(np.ceil(r["margin_to_zero"] / PENALTY_W["near_miss"])))
        row["tailgating_pct_error_to_escape_clamp"] = (
            None if not r["clamped"]
            else round(r["margin_to_zero"] / PENALTY_W["tailgating_pct"], 1))
        rows.append(row)
    df = pd.DataFrame(rows)
    out.table(df, "error_budget.csv")
    return df


# ================================================================= T-D5
def t_d5_clamp_floor(trips_s: List[Trip], recipe: Dict, out: Out) -> pd.DataFrame:
    """Sàn phạt deterministic trên trip chấm điểm, dùng công thức đã chốt ở T-D3."""
    rows = []
    for t in trips_s:
        masks = _apply_flags(t, recipe["unit"], recipe["boundary"])
        n = t.n_frames
        hb = _count_mode(masks["harsh_brake"], recipe["count_mode"])
        ha = _count_mode(masks["harsh_accel"], recipe["count_mode"])
        hc = _count_mode(masks["harsh_corner"], recipe["count_mode"])
        sp_pct = 100 * _count_mode(masks["speeding"], recipe["count_mode"]) / n if n else 0.0
        floor = (hb * PENALTY_W["harsh_brake"] + ha * PENALTY_W["harsh_accel"]
                 + hc * PENALTY_W["harsh_corner"] + sp_pct * PENALTY_W["speeding_pct"])
        certain_zero = floor >= 100.0
        rows.append({
            "trip_id": t.trip_id,
            "n_frames": n,
            "speed_limit_kmh": t.metadata.get("speed_limit_kmh"),
            "harsh_brake": hb, "harsh_accel": ha, "harsh_corner": hc,
            "speeding_pct_time": round(sp_pct, 2),
            "pen_harsh_brake": round(hb * PENALTY_W["harsh_brake"], 2),
            "pen_harsh_accel": round(ha * PENALTY_W["harsh_accel"], 2),
            "pen_harsh_corner": round(hc * PENALTY_W["harsh_corner"], 2),
            "pen_speeding": round(sp_pct * PENALTY_W["speeding_pct"], 3),
            "deterministic_floor": round(floor, 3),
            "safe_upper_bound": round(max(0.0, 100.0 - floor), 3),
            "ket_luan": ("MIỄN PHÍ (sàn ≥100 → safe chắc chắn = 0)" if certain_zero
                         else "CẦN PERCEPTION (near_miss + tailgating)"),
            "khoang_cach_toi_bien_100": round(100.0 - floor, 3),
            "near_miss_can_de_ve_0": (0 if certain_zero
                                      else int(np.ceil((100.0 - floor)
                                                       / PENALTY_W["near_miss"]))),
        })
    df = pd.DataFrame(rows)
    out.table(df, "clamp_floor_scoring.csv")
    return df


# ================================================================= F-D1 / F-D2
def f_d1_ego_kinematics(trips: List[Trip], out: Out) -> None:
    for t in trips:
        ts = [fr.get("timestamp", i / t.fps) for i, fr in enumerate(t.frames)]
        sp = [(fr.get("ego") or {}).get("speed_kmh", np.nan) for fr in t.frames]
        la = [(fr.get("ego") or {}).get("longitudinal_accel", np.nan) for fr in t.frames]
        lat = [(fr.get("ego") or {}).get("lateral_accel", np.nan) for fr in t.frames]
        bf = [fr.get("behavior_flags") or {} for fr in t.frames]
        limit = t.metadata.get("speed_limit_kmh")

        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        axes[0].plot(ts, sp, lw=1, color="#2c3e50")
        if limit is not None:
            axes[0].axhline(float(limit), color="#7f8c8d", ls=":", lw=1, label="speed_limit")
            axes[0].axhline(float(limit) + SPEEDING_TOLERANCE_KMH, color="#e74c3c",
                            ls="--", lw=1, label=f"limit + {SPEEDING_TOLERANCE_KMH:.0f}")
        m = [x for x, f in zip(ts, bf) if f.get("speeding")]
        axes[0].plot(m, [sp[ts.index(x)] for x in m], "o", ms=3, color="#e74c3c",
                     label=f"speeding ({len(m)})")
        axes[0].set_ylabel("speed (km/h)"); axes[0].legend(fontsize=8, ncol=3)

        axes[1].plot(ts, la, lw=1, color="#2c3e50")
        for g, c, lbl in ((-HARSH_BRAKE_G * G_MS2, "#e74c3c", "-0.40g"),
                          (HARSH_ACCEL_G * G_MS2, "#27ae60", "+0.35g")):
            axes[1].axhline(g, color=c, ls="--", lw=1, label=f"{lbl} = {g:.3f} m/s²")
        for key, c in (("harsh_brake", "#e74c3c"), ("harsh_accel", "#27ae60")):
            m = [x for x, f in zip(ts, bf) if f.get(key)]
            axes[1].plot(m, [la[ts.index(x)] for x in m], "o", ms=3, color=c,
                         label=f"{key} ({len(m)})")
        axes[1].set_ylabel("long. accel (m/s²)"); axes[1].legend(fontsize=8, ncol=2)

        axes[2].plot(ts, np.abs(lat), lw=1, color="#2c3e50")
        axes[2].axhline(HARSH_LATERAL_G * G_MS2, color="#8e44ad", ls="--", lw=1,
                        label=f"0.30g = {HARSH_LATERAL_G * G_MS2:.3f} m/s²")
        m = [x for x, f in zip(ts, bf) if f.get("harsh_corner")]
        axes[2].plot(m, [abs(lat[ts.index(x)]) for x in m], "o", ms=3, color="#8e44ad",
                     label=f"harsh_corner ({len(m)})")
        axes[2].set_ylabel("|lat. accel| (m/s²)"); axes[2].set_xlabel("thời gian (s)")
        axes[2].legend(fontsize=8, ncol=2)
        fig.suptitle(f"F-D1 — {t.trip_id}: ego kinematics + frame có flag bật")
        fig.tight_layout()
        out.figure(fig, f"ego_kinematics_{t.trip_id}.png")


def f_d2_risk_decomposition(trips: List[Trip], out: Out) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.ravel(), trips):
        prod, final = [], []
        for fr in t.frames:
            r = fr.get("risk") or {}
            b, d, f = r.get("base_risk"), r.get("driver_factor"), r.get("final_risk_score")
            if b is None or d is None or f is None:
                continue
            prod.append(float(b) * float(d)); final.append(float(f))
        ax.scatter(prod, final, s=8, alpha=0.4, color="#16a085")
        lim = max(prod + final + [1])
        ax.plot([0, lim], [0, lim], "--", color="#e74c3c", lw=1, label="y = x")
        ax.axhline(100, color="#7f8c8d", ls=":", lw=1, label="clamp 100")
        n_clamp = sum(1 for p, f in zip(prod, final) if p > 100 + 1e-9)
        ax.set_xlabel("base_risk × driver_factor"); ax.set_ylabel("final_risk_score")
        ax.set_title(f"{t.trip_id}  (bị clamp: {n_clamp})", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("F-D2 — final_risk_score vs base_risk × driver_factor", fontsize=12)
    fig.tight_layout()
    out.figure(fig, "risk_decomposition.png")


# ================================================================= run
def run(trips_p: List[Trip], trips_s: List[Trip], out: Out) -> Dict[str, Any]:
    print("[Nhóm D] C3 — kinematics & score")
    res: Dict[str, Any] = {}
    if not trips_p:
        print("  (không có trip GT — không verify được công thức, bỏ qua)")
        return res

    d1 = t_d1_trip_aggregate(trips_p, out); res["d1"] = d1
    delta_cols = [c for c in d1.columns if c.endswith("__delta")]
    d1_bad = d1[delta_cols].abs().max().max()
    print(f"  T-D1: |lệch| lớn nhất giữa trip_aggregate và số đếm tái tạo = {d1_bad}")

    d2 = t_d2_safe_score_check(trips_p, out); res["d2"] = d2
    print(f"  T-D2: |lệch| safe lớn nhất = {d2['delta'].abs().max()}  "
          f"(trip bị clamp: {int(d2['clamped'].sum())}/{len(d2)})")

    d3, recipe = t_d3_flag_recipe_verify(trips_p, out); res["d3"] = d3
    res["recipe"] = recipe
    if recipe is None:
        print("  T-D3: ✗ KHÔNG tổ hợp nào cho sai số 0 tuyệt đối → DỪNG, "
              "không chạy T-D5 (xem flag_recipe_verify.csv)")
    else:
        print(f"  T-D3: ✓ chốt công thức = đơn vị {recipe['unit']}, "
              f"biên '{recipe['boundary']}', đếm theo {recipe['count_mode']} "
              f"({recipe['n_winners']} tổ hợp đạt)")

    res["d4"] = t_d4_error_budget(trips_p, d2, out)

    if recipe is not None and trips_s:
        res["d5"] = t_d5_clamp_floor(trips_s, recipe, out)
    elif recipe is None:
        print("  T-D5: BỎ QUA — chờ chốt công thức ở T-D3")

    f_d1_ego_kinematics(trips_p, out)
    f_d2_risk_decomposition(trips_p, out)
    return res
