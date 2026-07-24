"""TripReplayer — iterator phát lại frame theo thứ tự timestamp.

Hai chế độ:
- ``fast`` (mặc định): chạy nhanh nhất có thể — batch inference & pipeline runner.
- ``realtime``: pacing đúng theo timestamp trong JSON (fact #10), nhân
  tốc độ qua ``speed`` — demo HUD.

Pacing realtime drift-free theo lịch TUYỆT ĐỐI, không cộng dồn
``sleep(1/fps)`` (sẽ trôi dần theo thời gian xử lý):

    deadline(i) = t0 + (ts_i − ts_start) / speed   với time.monotonic()

Frame bị chậm (load/consumer quá lâu) sẽ tự bắt kịp ở các frame sau vì
deadline không phụ thuộc thời điểm frame trước phát xong.
"""

from __future__ import annotations

import math
import time
from typing import Iterator, Optional

from .loader import TripLoader
from .types import FrameBundle

MODES = ("fast", "realtime")


class TripReplayer:
    """Phát lại ``loader`` từ frame ``start`` đến ``end`` (nửa hở, như range)."""

    def __init__(
        self,
        loader: TripLoader,
        mode: str = "fast",
        speed: float = 1.0,
        start: int = 0,
        end: Optional[int] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode phải là {MODES}, gặp {mode!r}")
        # not(>0) thay vì <=0 để chặn cả NaN (NaN so sánh gì cũng False —
        # lọt qua sẽ làm realtime âm thầm thành fast vì delay NaN không > 0)
        if not (math.isfinite(speed) and speed > 0):
            raise ValueError(f"speed phải là số dương hữu hạn, gặp {speed}")
        n = loader.n_frames
        end = n if end is None else end
        if not 0 <= start <= end <= n:
            raise ValueError(
                f"Khoảng phát không hợp lệ: start={start}, end={end}, n_frames={n}"
            )
        self.loader = loader
        self.mode = mode
        self.speed = speed
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self) -> Iterator[FrameBundle]:
        if self.start >= self.end:
            return
        if self.mode == "fast":
            for i in range(self.start, self.end):
                yield self.loader.frame(i)
            return

        # realtime — pacing theo timestamp trong JSON (fact #10)
        ts_start = self.loader.raw_frame(self.start).get(
            "timestamp", self.start / self.loader.fps
        )
        t0 = time.monotonic()
        for i in range(self.start, self.end):
            # load trước rồi mới chờ deadline — thời gian load nằm trong lịch
            bundle = self.loader.frame(i)
            deadline = t0 + (bundle.timestamp - ts_start) / self.speed
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            yield bundle
