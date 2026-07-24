"""TripLoader — đánh index 1 thư mục trip, đọc JSON telemetry, tra cứu
đường dẫn ảnh/depth/calib/label theo frame_id.

Các fact dataset mà module này tôn trọng (spec mục 2):
- #1  Không hardcode số frame — ``n_frames = len(frames)``, ``fps`` đọc từ metadata.
- #2  JSON chứa token ``Infinity`` trần: ``json.load`` của Python đọc được,
      giữ nguyên ``float('inf')`` trong bộ nhớ.
- #3  Ưu tiên ``.json`` giải nén nếu có, fallback ``.json.gz``.
- #4  Hai pattern tên ảnh khác nhau: ``driver/frame_{i:06d}.jpg`` vs
      ``kitti/image_2/{i:06d}.jpg``.
- #6  Calib load 1 lần/trip từ ``kitti/calibration_info.txt``.
- #8  Mọi field ground truth là Optional — trip chấm điểm bị xoá GT,
      loader không được crash vì thiếu key.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Calib

# Thứ tự thử extension ảnh (dataset hiện tại dùng .jpg; README nhắc PNG/JPG)
_IMAGE_EXTS = (".jpg", ".png", ".jpeg")

# Depth chỉ có ở keyframe i % 5 == 0 (fact #5)
DEPTH_KEYFRAME_STEP = 5


class TripLoader:
    """Index 1 thư mục trip: JSON telemetry + đường dẫn ảnh/depth/calib/label.

    JSON được đọc đúng 1 lần khi khởi tạo; calib đọc lazy đúng 1 lần;
    ảnh không load ở đây (lazy trong FrameBundle — Phase 2).
    """

    def __init__(self, trip_dir: str | Path):
        self.trip_dir = Path(trip_dir)
        if not self.trip_dir.is_dir():
            raise FileNotFoundError(f"Thư mục trip không tồn tại: {self.trip_dir}")

        self.json_path = self._resolve_json_path()
        self._raw: Dict[str, Any] = self._load_json(self.json_path)
        self._frames: List[Dict[str, Any]] = self._raw.get("frames") or []
        self._calib: Optional[Calib] = None
        # cache extension ảnh tìm được cho mỗi thư mục — tránh stat 600 lần
        self._ext_cache: Dict[Path, str] = {}

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #
    def _resolve_json_path(self) -> Path:
        """Ưu tiên ``<tên thư mục>.json``, fallback ``.json.gz`` (fact #3)."""
        name = self.trip_dir.name
        for cand in (self.trip_dir / f"{name}.json", self.trip_dir / f"{name}.json.gz"):
            if cand.exists():
                return cand
        # phòng khi file JSON không trùng tên thư mục
        others = sorted(self.trip_dir.glob("*.json")) + sorted(self.trip_dir.glob("*.json.gz"))
        if others:
            return others[0]
        raise FileNotFoundError(
            f"Không tìm thấy {name}.json hoặc {name}.json.gz trong {self.trip_dir}"
        )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        # json.load chấp nhận token `Infinity` trần → float('inf') (fact #2)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    # Thuộc tính cấp trip
    # ------------------------------------------------------------------ #
    @property
    def trip_id(self) -> str:
        return self._raw.get("trip_id") or self.trip_dir.name

    @property
    def n_frames(self) -> int:
        """Số frame thật của trip (600, 1800 hoặc bất kỳ) — không hardcode."""
        return len(self._frames)

    @property
    def fps(self) -> float:
        fps = (self._raw.get("metadata") or {}).get("fps")
        if fps:
            return float(fps)
        # metadata thiếu fps → suy ra từ bước timestamp (fact #10)
        if len(self._frames) >= 2:
            dt = self._frames[1].get("timestamp", 0.0) - self._frames[0].get("timestamp", 0.0)
            if dt > 0:
                return 1.0 / dt
        return 20.0

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._raw.get("metadata") or {}

    @property
    def trip_aggregate(self) -> Optional[Dict[str, Any]]:
        """GT cấp trip (C3) — None ở trip chấm điểm (fact #8)."""
        return self._raw.get("trip_aggregate")

    @property
    def driver_summary(self) -> Optional[Dict[str, Any]]:
        """GT tổng kết tài xế (C2) — None ở trip chấm điểm (fact #8)."""
        return self._raw.get("driver_summary")

    @property
    def events_log(self) -> List[Dict[str, Any]]:
        return self._raw.get("events_log") or []

    @property
    def calib(self) -> Calib:
        if self._calib is None:
            self._calib = self._load_calibration_info()
        return self._calib

    def _load_calibration_info(self) -> Calib:
        path = self.trip_dir / "kitti" / "calibration_info.txt"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file calib: {path}")
        info = json.loads(path.read_text(encoding="utf-8"))
        k = info["K_left"]
        return Calib(
            fx=float(k[0][0]),
            fy=float(k[1][1]),
            cx=float(k[0][2]),
            cy=float(k[1][2]),
            baseline_m=float(info["baseline_m"]),
            width=int(info["image_width"]),
            height=int(info["image_height"]),
        )

    def has_gt(self) -> bool:
        """Tự phát hiện trip mẫu (đủ GT) vs trip chấm điểm (GT bị xoá).

        Trip chấm điểm bị xoá: driver state, min_ttc/risk trong frame,
        trip_aggregate, driver_summary (fact #8).
        """
        if self._raw.get("trip_aggregate") or self._raw.get("driver_summary"):
            return True
        if self._frames:
            f0 = self._frames[0]
            driver = f0.get("driver") or {}
            if "min_ttc" in f0 or driver.get("state") is not None:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Truy cập frame thô
    # ------------------------------------------------------------------ #
    def raw_frame(self, i: int) -> Dict[str, Any]:
        """Bản ghi JSON thô của frame ``i`` (giữ nguyên inf, không copy)."""
        self._check_frame_id(i)
        return self._frames[i]

    def _check_frame_id(self, i: int) -> None:
        if not 0 <= i < self.n_frames:
            raise IndexError(
                f"frame_id {i} ngoài phạm vi [0, {self.n_frames}) của {self.trip_id}"
            )

    # ------------------------------------------------------------------ #
    # Tra cứu đường dẫn theo frame_id (fact #4 — 2 pattern tên khác nhau)
    # ------------------------------------------------------------------ #
    def _find_image(self, directory: Path, stem: str) -> Path:
        ext = self._ext_cache.get(directory)
        if ext is not None:
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        for e in _IMAGE_EXTS:
            p = directory / f"{stem}{e}"
            if p.exists():
                self._ext_cache[directory] = e
                return p
        raise FileNotFoundError(
            f"Không tìm thấy ảnh {stem}{{{'/'.join(_IMAGE_EXTS)}}} trong {directory}"
        )

    def left_path(self, i: int) -> Path:
        """kitti/image_2/{i:06d}.jpg — camera trái (không tiền tố)."""
        self._check_frame_id(i)
        return self._find_image(self.trip_dir / "kitti" / "image_2", f"{i:06d}")

    def right_path(self, i: int) -> Path:
        """kitti/image_3/{i:06d}.jpg — camera phải."""
        self._check_frame_id(i)
        return self._find_image(self.trip_dir / "kitti" / "image_3", f"{i:06d}")

    def driver_path(self, i: int) -> Path:
        """driver/frame_{i:06d}.jpg — ảnh cabin, CÓ tiền tố ``frame_``."""
        self._check_frame_id(i)
        return self._find_image(self.trip_dir / "driver", f"frame_{i:06d}")

    def label_path(self, i: int) -> Path:
        """kitti/label_2/{i:06d}.txt — có thể rỗng 0 byte (bình thường, fact #7)."""
        self._check_frame_id(i)
        return self.trip_dir / "kitti" / "label_2" / f"{i:06d}.txt"

    def depth_path(self, i: int) -> Optional[Path]:
        """Đường dẫn .npy depth — None nếu ``i`` không phải keyframe (fact #5)."""
        self._check_frame_id(i)
        if i % DEPTH_KEYFRAME_STEP != 0:
            return None
        return self.trip_dir / "kitti" / "depth" / f"{i:06d}.npy"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TripLoader({self.trip_id!r}, n_frames={self.n_frames}, "
            f"fps={self.fps}, has_gt={self.has_gt()})"
        )
