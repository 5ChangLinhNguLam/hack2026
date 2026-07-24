"""Kiểm tra tính toàn vẹn trip: file theo đúng dải frame, kích thước ảnh,
calib đồng nhất.

    python -m tripkit.validate data/            # quét mọi trip trong data/
    python -m tripkit.validate data/T01-Sample  # 1 trip

Label rỗng 0 byte là BÌNH THƯỜNG (fact #7) — chỉ cảnh báo, không phải lỗi.
Trip chấm điểm không có GT (fact #8) — báo cột GT=không, không phải lỗi.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2

from .loader import DEPTH_KEYFRAME_STEP, TripLoader

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class TripValidation:
    trip_id: str
    trip_dir: Path
    n_frames: int = 0
    has_gt: bool = False
    counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_stems(
    v: TripValidation, name: str, directory: Path, exts: Set[str], expected: Set[str]
) -> None:
    """So khớp tập TÊN FILE kỳ vọng (không chỉ đếm — đếm suông sẽ bị file
    lạ che mất file thiếu)."""
    if not directory.is_dir():
        v.counts[name] = 0
        if expected:
            v.errors.append(f"{name}: thiếu thư mục ({0}/{len(expected)} file)")
        return
    actual = {
        p.stem for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    }
    v.counts[name] = len(actual & expected)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        v.errors.append(f"{name}: thiếu {len(missing)}/{len(expected)} file (vd {missing[0]})")
    if extra:
        v.warnings.append(f"{name}: {len(extra)} file lạ ngoài dải frame (vd {extra[0]})")


def validate_trip(trip_dir: str | Path) -> TripValidation:
    trip_dir = Path(trip_dir)
    v = TripValidation(trip_id=trip_dir.name, trip_dir=trip_dir)

    try:
        loader = TripLoader(trip_dir)
    except Exception as e:  # JSON hỏng/thiếu — lỗi chặn, các check sau vô nghĩa
        v.errors.append(f"Không load được trip: {e}")
        return v

    v.trip_id = loader.trip_id
    v.n_frames = n = loader.n_frames
    v.has_gt = loader.has_gt()
    kitti = trip_dir / "kitti"
    if n == 0:
        v.errors.append("JSON không có frame nào")

    # --- file từng modality theo đúng dải frame_id (fact #1, #4, #5) ---
    frame_stems = {f"{i:06d}" for i in range(n)}
    depth_stems = {f"{i:06d}" for i in range(0, n, DEPTH_KEYFRAME_STEP)}
    _check_stems(v, "image_2", kitti / "image_2", _IMG_EXTS, frame_stems)
    _check_stems(v, "image_3", kitti / "image_3", _IMG_EXTS, frame_stems)
    _check_stems(v, "driver", trip_dir / "driver", _IMG_EXTS,
                 {f"frame_{i:06d}" for i in range(n)})
    _check_stems(v, "depth", kitti / "depth", {".npy"}, depth_stems)
    _check_stems(v, "calib", kitti / "calib", {".txt"}, frame_stems)
    _check_stems(v, "label_2", kitti / "label_2", {".txt"}, frame_stems)

    # --- calib: mọi file phải giống hệt nhau (fact #6) ---
    calib_dir = kitti / "calib"
    if calib_dir.is_dir():
        digests = {
            hashlib.md5(p.read_bytes()).hexdigest()
            for p in calib_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        }
        if len(digests) > 1:
            v.errors.append(f"calib không đồng nhất: {len(digests)} md5 khác nhau")

    # --- calibration_info.txt + kích thước ảnh khớp calib ---
    try:
        calib = loader.calib
    except Exception as e:
        calib = None
        v.errors.append(f"calibration_info: {e}")
    if calib is not None and n > 0:
        for cam, path_fn in (("image_2", loader.left_path),
                             ("image_3", loader.right_path),
                             ("driver", loader.driver_path)):
            if not v.counts.get(cam):
                continue
            try:  # lỗi 1 camera không được nuốt check của camera còn lại
                img = cv2.imread(str(path_fn(0)))
                if img is None:
                    v.errors.append(f"{cam}: không đọc được ảnh frame 0")
                elif img.shape[:2] != (calib.height, calib.width):
                    v.errors.append(
                        f"{cam}: kích thước {img.shape[1]}x{img.shape[0]} "
                        f"≠ calib {calib.width}x{calib.height}"
                    )
            except Exception as e:
                v.errors.append(f"{cam}: {e}")

    # --- label rỗng: bình thường, chỉ cảnh báo (fact #7) ---
    label_dir = kitti / "label_2"
    if label_dir.is_dir():
        empty = sum(
            1 for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt" and p.stat().st_size == 0
        )
        v.counts["label_nonempty"] = v.counts.get("label_2", 0) - empty
        if empty:
            v.warnings.append(f"{empty} file label rỗng (bình thường — fact #7)")

    return v


def _has_trip_json(d: Path) -> bool:
    """Cùng tiêu chí với TripLoader._resolve_json_path (kể cả fallback)."""
    return any(d.glob("*.json")) or any(d.glob("*.json.gz"))


def _find_trip_dirs(root: Path) -> List[Path]:
    """``root`` là 1 trip, hoặc thư mục chứa nhiều trip con."""
    root = root.resolve()  # để "." cũng dùng được
    if _has_trip_json(root):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and _has_trip_json(d))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tripkit.validate",
        description="Kiểm tra tính toàn vẹn các trip trong 1 thư mục dataset.",
    )
    p.add_argument("data_dir", help="thư mục dataset (data/) hoặc 1 trip (data/T01-Sample)")
    args = p.parse_args(argv)

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"Lỗi: {root} không phải thư mục", file=sys.stderr)
        return 2
    trip_dirs = _find_trip_dirs(root)
    if not trip_dirs:
        print(f"Lỗi: không tìm thấy trip nào trong {root}", file=sys.stderr)
        return 2

    reports: List[TripValidation] = []
    for d in trip_dirs:
        try:
            reports.append(validate_trip(d))
        except Exception as e:  # 1 trip hỏng không được làm sập cả run
            broken = TripValidation(trip_id=d.name, trip_dir=d)
            broken.errors.append(f"validate crash: {type(e).__name__}: {e}")
            reports.append(broken)

    header = (f"{'Trip':<14} {'frames':>6} {'img2':>5} {'img3':>5} {'driver':>6} "
              f"{'depth':>5} {'calib':>5} {'label✓':>6} {'GT':>5}  status")
    print(header)
    print("-" * len(header))
    for v in reports:
        c = v.counts
        print(
            f"{v.trip_id:<14} {v.n_frames:>6} {c.get('image_2', 0):>5} "
            f"{c.get('image_3', 0):>5} {c.get('driver', 0):>6} {c.get('depth', 0):>5} "
            f"{c.get('calib', 0):>5} {c.get('label_nonempty', 0):>6} "
            f"{'có' if v.has_gt else 'không':>5}  {'OK' if v.ok else 'FAIL'}"
        )
        for e in v.errors:
            print(f"    [LỖI] {e}")
        for w in v.warnings:
            print(f"    [cảnh báo] {w}")

    n_fail = sum(1 for v in reports if not v.ok)
    print(f"\n{len(reports) - n_fail}/{len(reports)} trip đạt.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
