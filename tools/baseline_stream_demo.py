"""Demo streaming: baseline TTC của BTC chạy live trên luồng TripReplayer.

Minh hoạ đúng kiến trúc "hệ thống nhận tín hiệu streaming":
TripReplayer phát frame 20 FPS → mỗi frame gọi model (ở đây là baseline
stereo SGBM) → in/vẽ kết quả ngay khi frame "đến".

    python tools/baseline_stream_demo.py data/T04-Sample                # console
    python tools/baseline_stream_demo.py data/T04-Sample --show        # cửa sổ HUD
    python tools/baseline_stream_demo.py data/T04-Sample --mode fast   # không pacing
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2

from team_kit.baseline_ttc_predictor import BaselineTTCPredictor
from tripkit import TripLoader, TripReplayer

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fmt(ttc: float | None) -> str:
    if ttc is None:
        return "  (không GT)"
    return "   inf" if math.isinf(ttc) else f"{ttc:6.2f}s"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trip_dir")
    p.add_argument("--mode", choices=("fast", "realtime"), default="realtime")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--show", action="store_true", help="cửa sổ HUD (q/ESC thoát)")
    args = p.parse_args(argv)

    loader = TripLoader(args.trip_dir)
    c = loader.calib
    predictor = BaselineTTCPredictor({
        "K_left": [[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]],
        "baseline_m": c.baseline_m,
    })

    end = loader.n_frames if args.limit is None else min(loader.n_frames, args.limit)
    replayer = TripReplayer(loader, mode=args.mode, speed=args.speed, end=end)

    n, t0 = 0, time.perf_counter()
    try:
        for b in replayer:  # ◄◄◄ hệ thống nhận tín hiệu streaming tại đây
            n += 1
            ttc = predictor.predict(b.frame_id, b.timestamp, b.left(), b.right())
            gt = b.gt["min_ttc"] if b.gt else None
            danger = ttc < 2.0 or (gt is not None and gt < 2.0)

            if args.show:
                panel = b.left().copy()
                color = (0, 0, 255) if ttc < 2.0 else (0, 255, 0)
                cv2.putText(panel, f"t={b.timestamp:5.2f}s  frame={b.frame_id}",
                            (10, 25), FONT, 0.6, (255, 255, 255), 2)
                cv2.putText(panel, f"TTC du doan : {_fmt(ttc)}",
                            (10, 55), FONT, 0.7, color, 2)
                cv2.putText(panel, f"TTC that GT : {_fmt(gt)}",
                            (10, 85), FONT, 0.7, (0, 255, 255), 2)
                if danger:
                    cv2.putText(panel, "!! DANGER (<2s) !!", (10, 120),
                                FONT, 0.8, (0, 0, 255), 2)
                cv2.imshow("baseline streaming", panel)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif danger or b.frame_id % 40 == 0:
                mark = "  << DANGER" if danger else ""
                print(f"t={b.timestamp:5.2f}s  frame={b.frame_id:4d}  "
                      f"dự đoán={_fmt(ttc)}  GT={_fmt(gt)}{mark}")
    finally:
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.perf_counter() - t0
    print(f"\nStream {n} frame trong {elapsed:.1f}s "
          f"({n / elapsed:.1f} fps, mode={args.mode}, model=baseline SGBM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
