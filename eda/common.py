"""Tiện ích dùng chung cho các nhóm phân tích EDA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # spec: backend Agg, không cần GUI

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Màu cố định cho 5 driver state (thống nhất giữa mọi figure)
STATE_COLORS = {
    "alert": "#2ecc71",
    "drowsy": "#f39c12",
    "yawning": "#3498db",
    "distracted": "#9b59b6",
    "microsleep": "#e74c3c",
}
STATE_ORDER = ["alert", "distracted", "drowsy", "yawning", "microsleep"]

EVENT_COLORS = {
    "pedestrian_jaywalk": "#e74c3c",
    "motorcycle_cut_in": "#f39c12",
    "lead_brake": "#3498db",
    "stopped_vehicle_ahead": "#9b59b6",
}
EVENT_ORDER = list(EVENT_COLORS)

# Ngưỡng chấm điểm — trích từ team_kit/evaluation.py:117-134
CRITICAL_TTC_SEC = 3.0
DANGER_TTC_SEC = 2.0
NEAR_MISS_TTC_SEC = 1.5
HARSH_BRAKE_G = 0.40
HARSH_ACCEL_G = 0.35
HARSH_LATERAL_G = 0.30
G_MS2 = 9.81
SPEEDING_TOLERANCE_KMH = 5.0

# Trọng số phạt của công thức safe_driving_score (fact #5)
PENALTY_W = {
    "harsh_brake": 3.0,
    "harsh_accel": 2.0,
    "harsh_corner": 2.0,
    "near_miss": 5.0,
    "speeding_pct": 0.15,
    "tailgating_pct": 0.10,
}


class Out:
    """Quản lý thư mục output: figures/ và tables/."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.figures = self.root / "figures"
        self.tables = self.root / "tables"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)

    def table(self, df: pd.DataFrame, name: str) -> Path:
        path = self.tables / name
        df.to_csv(path, index=False)
        print(f"  [table] {path.relative_to(self.root)}  ({len(df)} dòng)")
        return path

    def figure(self, fig, name: str, dpi: int = 110) -> Path:
        path = self.figures / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [figure] {path.relative_to(self.root)}")
        return path


def is_inf(x: Any) -> bool:
    return isinstance(x, float) and math.isinf(x)


def finite_or_none(x: Any) -> Optional[float]:
    """Trả None cho inf/None/không phải số — để tách nhóm inf trước khi vẽ (fact #1)."""
    if x is None or isinstance(x, bool):
        return None
    if not isinstance(x, (int, float)):
        return None
    return None if math.isinf(x) or math.isnan(x) else float(x)


def frame_gt(fr: Dict[str, Any]) -> Dict[str, Any]:
    """Phần GT của 1 frame (rỗng nếu đã bị che)."""
    from .io import FRAME_GT_KEYS, _is_erased

    return {k: fr[k] for k in FRAME_GT_KEYS if k in fr and not _is_erased(fr.get(k))}


def driver_state(fr: Dict[str, Any]) -> Optional[str]:
    d = fr.get("driver")
    if not isinstance(d, dict) or not d:
        return None
    return d.get("state")


def active_event_types(fr: Dict[str, Any]) -> List[str]:
    return [e.get("event_type") for e in (fr.get("events_active") or [])]


def event_intervals(trip) -> List[Dict[str, Any]]:
    """Các khoảng frame mà từng instance event đang active.

    Đo trực tiếp từ ``events_active`` (fact #9) — KHÔNG giả định event kéo
    dài tới frame cuối.
    """
    spans: Dict[Any, Dict[str, Any]] = {}
    for fr in trip.frames:
        fid = fr.get("frame_id")
        for e in fr.get("events_active") or []:
            key = (e.get("event_id"), e.get("event_type"))
            s = spans.setdefault(key, {
                "event_id": e.get("event_id"),
                "event_type": e.get("event_type"),
                "frame_start": fid,
                "frame_end": fid,
                "n_frames_active": 0,
                "age_sec_max": 0.0,
                "actor_ids": set(),
            })
            s["frame_end"] = fid
            s["n_frames_active"] += 1
            s["age_sec_max"] = max(s["age_sec_max"], float(e.get("age_sec") or 0.0))
            for a in e.get("actor_ids") or []:
                s["actor_ids"].add(a)
    return sorted(spans.values(), key=lambda s: (s["frame_start"], str(s["event_type"])))


def savefig_grid(nrows: int, ncols: int, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes
