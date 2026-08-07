"""Nhóm C — C2: driver state (cho Thiện). Chỉ chạy trên P (6 trip mẫu).

- T-C1 state_distribution.csv      - F-C1 state_timeline_T0X.png (6)
- T-C2 state_segments.csv          - T-C3 state_transition_matrix.csv
- T-C4 alertness_check.csv (verify fact #4)
- T-C5 face_features_by_state.csv
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import STATE_COLORS, STATE_ORDER, Out, driver_state
from .io import Trip

# fact #4 — alertness cố định theo state
ALERTNESS_EXPECTED = {
    "alert": 0.95, "yawning": 0.55, "distracted": 0.45,
    "drowsy": 0.35, "microsleep": 0.05,
}


def _states(t: Trip) -> List[str]:
    return [driver_state(fr) for fr in t.frames]


# ---------------------------------------------------------------- T-C1
def t_c1_state_distribution(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    for t in trips:
        c = Counter(s for s in _states(t) if s)
        n = sum(c.values())
        row = {"trip_id": t.trip_id, "n_frames_with_state": n}
        for s in STATE_ORDER:
            row[f"n_{s}"] = c.get(s, 0)
            row[f"pct_{s}"] = round(100 * c.get(s, 0) / n, 2) if n else 0.0
        present = [v for v in c.values() if v > 0]
        row["n_classes_present"] = len(present)
        row["imbalance_ratio"] = round(max(present) / min(present), 2) if present else None
        rows.append(row)
    df = pd.DataFrame(rows)

    # dòng tổng hợp toàn bộ P
    tot = Counter()
    for t in trips:
        tot.update(s for s in _states(t) if s)
    n = sum(tot.values())
    agg = {"trip_id": "TỔNG (6 trip)", "n_frames_with_state": n}
    for s in STATE_ORDER:
        agg[f"n_{s}"] = tot.get(s, 0)
        agg[f"pct_{s}"] = round(100 * tot.get(s, 0) / n, 2) if n else 0.0
    present = [v for v in tot.values() if v > 0]
    agg["n_classes_present"] = len(present)
    agg["imbalance_ratio"] = round(max(present) / min(present), 2) if present else None
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    out.table(df, "state_distribution.csv")
    return df


# ---------------------------------------------------------------- F-C1
def f_c1_state_timeline(trips: List[Trip], out: Out) -> None:
    for t in trips:
        states = _states(t)
        fig, ax = plt.subplots(figsize=(13, 2.4))
        for i, s in enumerate(states):
            if s:
                ax.axvspan(i / t.fps, (i + 1) / t.fps,
                           color=STATE_COLORS.get(s, "#888"), lw=0)
        ax.set_xlim(0, len(states) / t.fps); ax.set_yticks([])
        ax.set_xlabel("thời gian (s)")
        ax.set_title(f"F-C1 — {t.trip_id}: dải trạng thái tài xế theo thời gian")
        used = [s for s in STATE_ORDER if s in set(states)]
        ax.legend(handles=[mpatches.Patch(color=STATE_COLORS[s], label=s) for s in used],
                  ncol=len(used), fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.35))
        out.figure(fig, f"state_timeline_{t.trip_id}.png")


# ---------------------------------------------------------------- T-C2
def t_c2_state_segments(trips: List[Trip], out: Out) -> pd.DataFrame:
    seg_rows = []
    for t in trips:
        states = _states(t)
        if not states:
            continue
        cur, start = states[0], 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != cur:
                if cur:
                    seg_rows.append({
                        "trip_id": t.trip_id, "state": cur,
                        "frame_start": start, "frame_end": i - 1,
                        "n_frames": i - start,
                        "duration_sec": round((i - start) / t.fps, 2),
                    })
                if i < len(states):
                    cur, start = states[i], i
    df = pd.DataFrame(seg_rows)
    out.table(df, "state_segments.csv")

    if not df.empty:
        stat = (df.groupby("state")["duration_sec"]
                  .agg(n_segments="size", min_sec="min", median_sec="median",
                       max_sec="max", total_sec="sum")
                  .reset_index()
                  .sort_values("state"))
        out.table(stat, "state_segments_by_state.csv")
    return df


# ---------------------------------------------------------------- T-C3
def t_c3_transition_matrix(trips: List[Trip], out: Out) -> pd.DataFrame:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for t in trips:
        states = [s for s in _states(t)]
        for a, b in zip(states, states[1:]):
            if a and b:
                counts[a][b] += 1
    present = [s for s in STATE_ORDER if s in counts or any(s in c for c in counts.values())]
    rows = []
    for a in present:
        tot = sum(counts[a].values())
        row = {"from_state": a, "n_transitions_out": tot}
        for b in present:
            row[f"to_{b}"] = counts[a].get(b, 0)
            row[f"p_{b}"] = round(counts[a].get(b, 0) / tot, 6) if tot else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    out.table(df, "state_transition_matrix.csv")
    return df


# ---------------------------------------------------------------- T-C4
def t_c4_alertness_check(trips: List[Trip], out: Out) -> pd.DataFrame:
    """Verify fact #4: mỗi state đúng 1 giá trị alertness, trên 6/6 trip."""
    rows = []
    for t in trips:
        by_state: Dict[str, set] = defaultdict(set)
        for fr in t.frames:
            d = fr.get("driver") or {}
            s, a = d.get("state"), d.get("alertness_score")
            if s is not None and a is not None:
                by_state[s].add(round(float(a), 6))
        for s in sorted(by_state):
            vals = sorted(by_state[s])
            exp = ALERTNESS_EXPECTED.get(s)
            rows.append({
                "trip_id": t.trip_id, "state": s,
                "n_distinct_alertness": len(vals),
                "alertness_values": ";".join(str(v) for v in vals),
                "expected_fact4": exp,
                "match_fact4": (len(vals) == 1 and exp is not None
                                and abs(vals[0] - exp) < 1e-9),
            })
    df = pd.DataFrame(rows)
    out.table(df, "alertness_check.csv")
    return df


# ---------------------------------------------------------------- T-C5
def t_c5_face_features(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    combo: Dict[str, Counter] = defaultdict(Counter)
    for t in trips:
        for fr in t.frames:
            d = fr.get("driver") or {}
            s = d.get("state")
            if not s:
                continue
            combo[s][(d.get("eye_state"), d.get("head_pose"), d.get("mouth_state"))] += 1
    for s in STATE_ORDER:
        if s not in combo:
            continue
        for (eye, head, mouth), n in sorted(combo[s].items(), key=lambda kv: -kv[1]):
            rows.append({
                "state": s, "eye_state": eye, "head_pose": head,
                "mouth_state": mouth, "n_frames": n,
                "pct_within_state": round(100 * n / sum(combo[s].values()), 2),
            })
    df = pd.DataFrame(rows)
    # tổ hợp (eye,head,mouth) có ánh xạ 1-1 sang state không?
    if not df.empty:
        triple = df.groupby(["eye_state", "head_pose", "mouth_state"])["state"].nunique()
        df["triple_maps_to_n_states"] = df.apply(
            lambda r: int(triple[(r.eye_state, r.head_pose, r.mouth_state)]), axis=1)
    out.table(df, "face_features_by_state.csv")
    return df


def run(trips_p: List[Trip], out: Out) -> Dict[str, Any]:
    print("[Nhóm C] C2 — driver state (chỉ trên 6 trip mẫu)")
    if not trips_p:
        print("  (không có trip nào có GT — bỏ qua)")
        return {}
    res = {
        "state_distribution": t_c1_state_distribution(trips_p, out),
        "state_segments": t_c2_state_segments(trips_p, out),
        "transition": t_c3_transition_matrix(trips_p, out),
        "alertness": t_c4_alertness_check(trips_p, out),
        "face_features": t_c5_face_features(trips_p, out),
    }
    f_c1_state_timeline(trips_p, out)
    return res
