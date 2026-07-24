"""Kiểu dữ liệu dùng chung: Calib, KittiLabel, FrameBundle.

API contract đã thống nhất với downstream (C1/C2/1.4/HUD) — không đổi
tên field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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


@dataclass(frozen=True)
class KittiLabel:
    """1 dòng trong ``kitti/label_2/{i:06d}.txt`` — đúng 15 trường KITTI.

    Trong dataset này chỉ ``type`` + kích thước 3D (height/width/length)
    + vị trí 3D (x/y/z, hệ camera trái: x-phải, y-xuống, z-tiến, mét) có
    giá trị thật; bbox 2D / truncated / occluded / alpha / rotation_y
    luôn = 0 (fact #7) — muốn bbox 2D phải tự chiếu 3D→2D qua P2.
    Ở trip chấm điểm, x/y/z bị zero-out.
    """

    type: str                  # Pedestrian (walker) | Car (vehicle) | Cyclist (bike)
    truncated: float
    occluded: int
    alpha: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    height: float              # dims 3D (m)
    width: float
    length: float
    x: float                   # location 3D hệ cam trái (m)
    y: float
    z: float
    rotation_y: float


def parse_kitti_label_file(path: str | Path) -> List[KittiLabel]:
    """Parse 1 file label KITTI. File rỗng 0 byte / không tồn tại → ``[]`` (fact #7)."""
    path = Path(path)
    if not path.exists():
        return []
    labels: List[KittiLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 15:
            raise ValueError(
                f"Dòng KITTI label phải có 15 trường, gặp {len(parts)}: {path}: {line!r}"
            )
        labels.append(
            KittiLabel(
                type=parts[0],
                truncated=float(parts[1]),
                occluded=int(float(parts[2])),
                alpha=float(parts[3]),
                bbox_left=float(parts[4]),
                bbox_top=float(parts[5]),
                bbox_right=float(parts[6]),
                bbox_bottom=float(parts[7]),
                height=float(parts[8]),
                width=float(parts[9]),
                length=float(parts[10]),
                x=float(parts[11]),
                y=float(parts[12]),
                z=float(parts[13]),
                rotation_y=float(parts[14]),
            )
        )
    return labels


def _imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


@dataclass(eq=False)
class FrameBundle:
    """1 frame đồng bộ đủ các modality theo ``frame_id``.

    Ảnh lazy-load: chỉ đọc từ đĩa khi gọi ``left()/right()/driver()``,
    trả ``np.ndarray`` BGR (cv2.imread), cache trong bundle (tắt được
    qua ``TripLoader.frame(i, cache_images=False)``).
    """

    trip_id: str
    frame_id: int
    timestamp: float                       # giây, = frame_id / fps (fact #10)
    depth: Optional[np.ndarray]            # CHỈ khi frame_id % 5 == 0 (fact #5)
    ego: Optional[Dict[str, Any]]          # speed_kmh, longitudinal/lateral_accel, ...
    targets: List[Dict[str, Any]]          # [] nếu không có / bị xoá
    labels: List[KittiLabel]               # [] nếu file rỗng
    gt: Optional[Dict[str, Any]]           # driver/min_ttc/risk... — None ở trip chấm điểm
    events_active: List[Dict[str, Any]]

    # nội bộ — không thuộc contract
    _loader: Any = field(default=None, repr=False)
    _cache: bool = field(default=True, repr=False)
    _left: Optional[np.ndarray] = field(default=None, repr=False)
    _right: Optional[np.ndarray] = field(default=None, repr=False)
    _driver: Optional[np.ndarray] = field(default=None, repr=False)

    def left(self) -> np.ndarray:
        """Ảnh camera trái (kitti/image_2), BGR."""
        if self._left is not None:
            return self._left
        img = _imread(self._loader.left_path(self.frame_id))
        if self._cache:
            self._left = img
        return img

    def right(self) -> np.ndarray:
        """Ảnh camera phải (kitti/image_3), BGR."""
        if self._right is not None:
            return self._right
        img = _imread(self._loader.right_path(self.frame_id))
        if self._cache:
            self._right = img
        return img

    def driver(self) -> np.ndarray:
        """Ảnh cabin tài xế (driver/frame_*.jpg), BGR."""
        if self._driver is not None:
            return self._driver
        img = _imread(self._loader.driver_path(self.frame_id))
        if self._cache:
            self._driver = img
        return img
