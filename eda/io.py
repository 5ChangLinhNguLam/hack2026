"""Adapter đọc trip cho EDA — ưu tiên tripkit, fallback raw JSON.

Xử lý đúng 3 điểm dữ liệu (spec mục 2):
- fact #1: token ``Infinity`` trần → json của Python giữ thành ``float('inf')``.
- fact #2: chỉ T01-Sample có ``.json`` giải nén; còn lại ``gzip.open(path,'rt')``.
- fact #8: trip chấm điểm che GT bằng ``driver = {}`` (dict rỗng, KHÔNG mất key)
  → ``has_gt`` phải phát hiện theo HÌNH DẠNG dữ liệu, không theo tên trip.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:  # phụ thuộc mềm vào task 1.2
    from tripkit import TripLoader
    from tripkit.loader import DEPTH_KEYFRAME_STEP
    HAS_TRIPKIT = True
except ImportError:  # pragma: no cover - fallback khi tripkit chưa merge
    TripLoader = None  # type: ignore
    DEPTH_KEYFRAME_STEP = 5
    HAS_TRIPKIT = False

_IMG_EXTS = (".jpg", ".jpeg", ".png")

# Các key GT mức frame; ở trip chấm điểm chúng bị xoá hoặc để rỗng
FRAME_GT_KEYS = ("driver", "min_ttc", "headway_sec", "behavior_flags", "risk")


def _is_erased(value: Any) -> bool:
    """GT coi như đã xoá khi: mất key (None) hoặc dict/list rỗng (fact #8)."""
    return value is None or (isinstance(value, (dict, list)) and len(value) == 0)


def load_raw(trip_dir: Path) -> Dict[str, Any]:
    """Đọc JSON trip: ưu tiên bản giải nén, fallback .json.gz."""
    name = trip_dir.name
    for cand in (trip_dir / f"{name}.json", trip_dir / f"{name}.json.gz"):
        if cand.exists():
            path = cand
            break
    else:
        others = sorted(trip_dir.glob("*.json")) + sorted(trip_dir.glob("*.json.gz"))
        if not others:
            raise FileNotFoundError(f"Không tìm thấy JSON trong {trip_dir}")
        path = others[0]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        return json.load(f)  # `Infinity` trần → float('inf')


@dataclass
class Trip:
    """Một trip đã nạp: JSON thô + tiện ích đếm file, không giữ ảnh trong RAM."""

    trip_dir: Path
    raw: Dict[str, Any]

    # ---------------- thuộc tính cơ bản ----------------
    @property
    def trip_id(self) -> str:
        return self.raw.get("trip_id") or self.trip_dir.name

    @property
    def frames(self) -> List[Dict[str, Any]]:
        return self.raw.get("frames") or []

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.raw.get("metadata") or {}

    @property
    def fps(self) -> float:
        fps = self.metadata.get("fps")
        if fps:
            return float(fps)
        fr = self.frames
        if len(fr) >= 2:
            dt = fr[1].get("timestamp", 0.0) - fr[0].get("timestamp", 0.0)
            if dt > 0:
                return 1.0 / dt
        return 20.0

    @property
    def events_log(self) -> List[Dict[str, Any]]:
        return self.raw.get("events_log") or []

    @property
    def trip_aggregate(self) -> Optional[Dict[str, Any]]:
        v = self.raw.get("trip_aggregate")
        return None if _is_erased(v) else v

    @property
    def driver_summary(self) -> Optional[Dict[str, Any]]:
        v = self.raw.get("driver_summary")
        return None if _is_erased(v) else v

    @property
    def has_gt(self) -> bool:
        """Phát hiện GT theo HÌNH DẠNG dữ liệu (fact #8), không theo tên trip."""
        if self.trip_aggregate or self.driver_summary:
            return True
        fr = self.frames
        if not fr:
            return False
        return any(not _is_erased(fr[0].get(k)) for k in FRAME_GT_KEYS)

    # ---------------- đường dẫn / đếm file ----------------
    @property
    def kitti(self) -> Path:
        return self.trip_dir / "kitti"

    def _count(self, directory: Path, exts) -> int:
        if not directory.is_dir():
            return 0
        return sum(1 for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)

    def count_left(self) -> int:
        return self._count(self.kitti / "image_2", _IMG_EXTS)

    def count_right(self) -> int:
        return self._count(self.kitti / "image_3", _IMG_EXTS)

    def count_driver(self) -> int:
        return self._count(self.trip_dir / "driver", _IMG_EXTS)

    def count_depth(self) -> int:
        return self._count(self.kitti / "depth", {".npy"})

    def label_paths(self) -> List[Path]:
        d = self.kitti / "label_2"
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() == ".txt")

    def left_path(self, i: int) -> Optional[Path]:
        for e in _IMG_EXTS:
            p = self.kitti / "image_2" / f"{i:06d}{e}"
            if p.exists():
                return p
        return None

    def driver_path(self, i: int) -> Optional[Path]:
        for e in _IMG_EXTS:
            p = self.trip_dir / "driver" / f"frame_{i:06d}{e}"
            if p.exists():
                return p
        return None


def find_trip_dirs(data_dir: Path) -> List[Path]:
    """Mọi thư mục con chứa JSON trip. Không hardcode tên trip."""
    data_dir = Path(data_dir).resolve()
    if any(data_dir.glob("*.json")) or any(data_dir.glob("*.json.gz")):
        return [data_dir]
    out = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and (any(d.glob("*.json")) or any(d.glob("*.json.gz"))):
            out.append(d)
    return out


def load_trips(data_dir: Path) -> List[Trip]:
    return [Trip(d, load_raw(d)) for d in find_trip_dirs(data_dir)]


def iter_trips(data_dir: Path) -> Iterator[Trip]:
    """Nạp lần lượt — trip 1800 frame khá nặng, tránh giữ hết trong RAM."""
    for d in find_trip_dirs(data_dir):
        yield Trip(d, load_raw(d))
