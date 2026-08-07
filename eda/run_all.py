"""Chạy toàn bộ EDA bằng 1 lệnh.

    python -m eda.run_all --data-dir data/ --out reports/eda/
    python -m eda.run_all --groups A B      # chạy chọn lọc khi phát triển
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from .common import Out
from .io import Trip, iter_trips

ALL_GROUPS = ["A", "B", "C", "D", "E"]


def load_split(data_dir: Path) -> tuple[List[Trip], List[Trip]]:
    """Chia trip theo HÌNH DẠNG dữ liệu: có GT (P) vs đã xoá GT (S)."""
    trips_p, trips_s = [], []
    for t in iter_trips(data_dir):
        (trips_p if t.has_gt else trips_s).append(t)
    return trips_p, trips_s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eda.run_all",
                                 description="EDA dataset hackathon (WBS 1.3)")
    ap.add_argument("--data-dir", default="data/", type=Path)
    ap.add_argument("--out", default="reports/eda/", type=Path)
    ap.add_argument("--groups", nargs="+", default=ALL_GROUPS,
                    choices=ALL_GROUPS, help="nhóm phân tích cần chạy")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    out = Out(args.out)
    print(f"Nạp trip từ {args.data_dir} ...")
    trips_p, trips_s = load_split(args.data_dir)
    print(f"  {len(trips_p)} trip CÓ GT (P): {', '.join(t.trip_id for t in trips_p)}")
    print(f"  {len(trips_s)} trip KHÔNG GT (S): {', '.join(t.trip_id for t in trips_s)}")
    if not trips_p and not trips_s:
        print(f"Lỗi: không tìm thấy trip nào trong {args.data_dir}", file=sys.stderr)
        return 2

    results = {}
    if "A" in args.groups:
        from . import group_a
        results["A"] = group_a.run(trips_p, trips_s, out)
    if "B" in args.groups:
        from . import group_b
        results["B"] = group_b.run(trips_p, trips_s, out)
    if "C" in args.groups:
        from . import group_c
        results["C"] = group_c.run(trips_p, out)
    if "D" in args.groups:
        from . import group_d
        results["D"] = group_d.run(trips_p, trips_s, out)
    if "E" in args.groups:
        from . import group_e
        results["E"] = group_e.run(trips_p, out)

    print(f"\nXong trong {time.perf_counter() - t0:.1f}s → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
