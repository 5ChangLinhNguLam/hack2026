"""Kiểu dữ liệu dùng chung: Calib (Phase 1), KittiLabel + FrameBundle (Phase 2).

API contract đã thống nhất với downstream (C1/C2/1.4/HUD) — không đổi
tên field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calib:
    """Thông số camera stereo của 1 trip.

    Load đúng 1 lần/trip từ ``kitti/calibration_info.txt`` (fact #6) —
    600 file trong ``kitti/calib/`` giống hệt nhau, không đọc lại.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float          # 0.3
    width: int                 # 640
    height: int                # 360

    @property
    def depth_factor(self) -> float:
        """= fx × baseline_m (= 96.0 với dataset này).

        Công thức stereo: depth(m) = depth_factor / disparity(px).
        """
        return self.fx * self.baseline_m
