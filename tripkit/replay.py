"""CLI phát lại trip.

    python -m tripkit.replay data/T01-Sample --mode fast --limit 50 --stats
    python -m tripkit.replay data/T01-Sample --mode realtime --show
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from .loader import TripLoader
from .replayer import MODES, TripReplayer

WINDOW = "tripkit replay"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tripkit.replay",
        description="Phát lại 1 trip: ảnh stereo + driver + kinematics đồng bộ theo frame_id.",
    )
    p.add_argument("trip_dir", help="thư mục trip, vd data/T01-Sample")
    p.add_argument("--mode", choices=MODES, default="fast")
    p.add_argument("--speed", type=float, default=1.0, help="hệ số tốc độ realtime (vd 2.0)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="chỉ phát tối đa N frame kể từ --start")
    p.add_argument("--stats", action="store_true", help="in thống kê trip sau khi phát")
    p.add_argument("--show", action="store_true",
                   help="cửa sổ cv2: left|right|driver cạnh nhau + timestamp (q/ESC để thoát)")
    return p


def _compose_panel(bundle) -> np.ndarray:
    """Ghép 3 ảnh cạnh nhau + dòng chữ timestamp/frame/driver state."""
    panel = np.hstack([bundle.left(), bundle.right(), bundle.driver()])
    text = f"{bundle.trip_id}  t={bundle.timestamp:6.2f}s  frame={bundle.frame_id:4d}"
    state = ((bundle.gt or {}).get("driver") or {}).get("state")
    if state:
        text += f"  driver={state}"
    if bundle.events_active:
        text += "  event=" + ",".join(e.get("event_type", "?") for e in bundle.events_active)
    cv2.putText(panel, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return panel


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        loader = TripLoader(args.trip_dir)
    except Exception as e:  # JSON hỏng/gz cụt/thiếu file — báo gọn, không traceback
        print(f"Lỗi: không load được trip: {e}", file=sys.stderr)
        return 2

    end = loader.n_frames if args.end is None else args.end
    if args.limit is not None:
        end = min(end, args.start + args.limit)
    try:
        replayer = TripReplayer(
            loader, mode=args.mode, speed=args.speed, start=args.start, end=end
        )
    except ValueError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 2

    n_played = 0
    t0 = time.perf_counter()
    try:
        for bundle in replayer:
            n_played += 1
            if args.show:
                cv2.imshow(WINDOW, _compose_panel(bundle))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q hoặc ESC
                    break
    except cv2.error as e:
        print(f"Lỗi cv2 (môi trường không có GUI? bỏ --show): {e}", file=sys.stderr)
        return 3
    finally:
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # build headless: destroyAllWindows cũng raise — không được nuốt exit code
    elapsed = time.perf_counter() - t0

    read_fps = n_played / elapsed if elapsed > 0 else float("inf")
    print(
        f"Đã phát {n_played}/{loader.n_frames} frame của {loader.trip_id} "
        f"trong {elapsed:.2f}s ({read_fps:.1f} fps đọc, mode={args.mode})"
    )
    if args.stats:
        meta = loader.metadata
        print(f"  n_frames      : {loader.n_frames}")
        print(f"  fps (metadata): {loader.fps}")
        print(f"  có GT         : {'có' if loader.has_gt() else 'không'}")
        print(f"  map           : {meta.get('map', '?')}")
        print(f"  speed_limit   : {meta.get('speed_limit_kmh', '?')} km/h")
        events = ", ".join(e.get("type", "?") for e in loader.events_log) or "(không)"
        print(f"  events_log    : {events}")
        try:
            c = loader.calib
            print(f"  calib         : fx={c.fx:.0f} cx={c.cx:.0f} cy={c.cy:.0f} "
                  f"baseline={c.baseline_m}m ({c.width}x{c.height})")
        except Exception as e:
            print(f"  calib         : (không đọc được: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
