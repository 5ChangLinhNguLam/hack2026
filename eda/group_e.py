"""Nhóm E — Kiểm định độc lập driver state ⊥ event CARLA (fact #6). Chỉ trên P.

- T-E1 independence_test.csv: contingency state × event-active, Cramér's V + p
- F-E1 drowsy_vs_ttc.png: boxplot min_ttc hữu hạn theo state
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .common import STATE_COLORS, STATE_ORDER, Out, driver_state, finite_or_none
from .io import Trip


def _cramers_v(table: np.ndarray):
    """Trả (V, p, chi2, dof). None nếu bảng suy biến (một chiều chỉ 1 mức)."""
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None, None
    if table.sum() == 0 or (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return None, None, None, None
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return v, float(p), float(chi2), int(dof)


def _rows_for(label: str, states: List[str], actives: List[bool]) -> Dict[str, Any]:
    present = [s for s in STATE_ORDER if s in set(states)]
    tab = np.array([[sum(1 for s, a in zip(states, actives) if s == st and a == flag)
                     for flag in (True, False)] for st in present], dtype=float)
    v, p, chi2, dof = _cramers_v(tab)
    row = {
        "scope": label,
        "n_frames": len(states),
        "n_states_present": len(present),
        "states": ";".join(present),
        "n_event_active": int(sum(actives)),
        "n_event_inactive": int(len(actives) - sum(actives)),
        "cramers_v": None if v is None else round(v, 4),
        "p_value": None if p is None else float(f"{p:.6g}"),
        "chi2": None if chi2 is None else round(chi2, 3),
        "dof": dof,
        "testable": v is not None,
        "ket_luan": None,
    }
    if v is None:
        row["ket_luan"] = "KHÔNG kiểm được (state hoặc event không biến thiên)"
    else:
        assoc = ("không đáng kể" if v < 0.1 else "yếu" if v < 0.3
                 else "trung bình" if v < 0.5 else "MẠNH")
        sig = "có ý nghĩa (p<0.05)" if p < 0.05 else "không có ý nghĩa (p≥0.05)"
        row["ket_luan"] = f"liên hệ {assoc} (V={v:.3f}), {sig}"
    return row


def t_e1_independence(trips: List[Trip], out: Out) -> pd.DataFrame:
    rows = []
    all_states, all_active = [], []
    for t in trips:
        states = [driver_state(fr) for fr in t.frames]
        actives = [bool(fr.get("events_active")) for fr in t.frames]
        keep = [(s, a) for s, a in zip(states, actives) if s]
        if not keep:
            continue
        st = [s for s, _ in keep]; ac = [a for _, a in keep]
        all_states += st; all_active += ac
        rows.append(_rows_for(t.trip_id, st, ac))
    rows.insert(0, _rows_for("GỘP 6 trip (có confound theo trip)",
                             all_states, all_active))
    df = pd.DataFrame(rows)
    out.table(df, "independence_test.csv")
    return df


def f_e1_drowsy_vs_ttc(trips: List[Trip], out: Out) -> None:
    by_state: Dict[str, List[float]] = {}
    n_inf: Dict[str, int] = {}
    for t in trips:
        for fr in t.frames:
            s = driver_state(fr)
            if not s:
                continue
            v = finite_or_none(fr.get("min_ttc"))
            if v is None:
                n_inf[s] = n_inf.get(s, 0) + 1
            else:
                by_state.setdefault(s, []).append(v)
    present = [s for s in STATE_ORDER if s in by_state]
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [np.clip(by_state[s], 0, 30) for s in present]
    bp = ax.boxplot(data, labels=present, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=3, alpha=.4))
    for patch, s in zip(bp["boxes"], present):
        patch.set_facecolor(STATE_COLORS.get(s, "#888")); patch.set_alpha(.65)
    ax.axhline(1.5, color="#e74c3c", ls="--", lw=1, label="near-miss 1.5s")
    ax.axhline(3.0, color="#f39c12", ls="--", lw=1, label="critical 3.0s")
    ax.set_ylabel("min_ttc hữu hạn (s, cắt ở 30)")
    ax.set_title("F-E1 — Phân bố min_ttc hữu hạn theo driver state\n"
                 + "  ".join(f"{s}: n={len(by_state[s])}, inf={n_inf.get(s, 0)}"
                             for s in present), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.figure(fig, "drowsy_vs_ttc.png")


def run(trips_p: List[Trip], out: Out) -> Dict[str, Any]:
    print("[Nhóm E] Kiểm định độc lập driver ⊥ event")
    if not trips_p:
        print("  (không có trip GT — bỏ qua)")
        return {}
    res = {"independence": t_e1_independence(trips_p, out)}
    f_e1_drowsy_vs_ttc(trips_p, out)
    return res
