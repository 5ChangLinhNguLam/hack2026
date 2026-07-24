"""Phase 3 — TripReplayer: fast + realtime pacing.

DoD (spec mục 4):
- fast trả đủ n_frames đúng thứ tự
- realtime 40 frame @ speed=4 chạy trong 0.5s ± 0.1s
"""

import time

import pytest

from conftest import make_redacted_frame, make_trip
from tripkit import TripLoader, TripReplayer


@pytest.fixture()
def fake40_loader(tmp_path):
    """Trip giả lập 40 frame @ 20 FPS (timestamp 0 → 1.95s), không ảnh."""
    frames = [make_redacted_frame(i) for i in range(40)]
    return TripLoader(make_trip(tmp_path, "T94d", frames))


# ---------------------------------------------------------------------- #
# fast
# ---------------------------------------------------------------------- #
def test_fast_full_order(fake40_loader):
    bundles = list(TripReplayer(fake40_loader))
    assert len(bundles) == fake40_loader.n_frames == 40
    assert [b.frame_id for b in bundles] == list(range(40))


def test_fast_no_pacing(fake40_loader):
    t0 = time.perf_counter()
    list(TripReplayer(fake40_loader, mode="fast"))
    assert time.perf_counter() - t0 < 0.2  # không được sleep trong mode fast


def test_fast_on_real_trip(t01_loader):
    bundles = list(TripReplayer(t01_loader, mode="fast"))
    assert len(bundles) == 600
    assert [b.frame_id for b in bundles] == list(range(600))
    assert bundles[0].depth is not None and bundles[3].depth is None


def test_start_end_slice(fake40_loader):
    rep = TripReplayer(fake40_loader, start=10, end=20)
    assert len(rep) == 10
    ids = [b.frame_id for b in rep]
    assert ids == list(range(10, 20))


def test_empty_range(fake40_loader):
    assert list(TripReplayer(fake40_loader, start=5, end=5)) == []


# ---------------------------------------------------------------------- #
# realtime — pacing theo lịch tuyệt đối
# ---------------------------------------------------------------------- #
def test_realtime_40_frames_speed4(fake40_loader):
    # 40 frame @ 20 FPS = 2.0s trip-time; speed 4 → lịch kết thúc ở 1.95/4 = 0.4875s
    rep = TripReplayer(fake40_loader, mode="realtime", speed=4.0)
    t0 = time.perf_counter()
    bundles = list(rep)
    elapsed = time.perf_counter() - t0
    assert len(bundles) == 40
    assert abs(elapsed - 0.4875) < 0.1  # DoD: 0.5s ± 0.1s


def test_realtime_catches_up_after_slow_consumer(fake40_loader):
    # Consumer chậm hơn lịch (60ms/frame > 50ms/frame @ 20FPS):
    # pacing tuyệt đối không cộng thêm sleep khi đã trễ deadline.
    # 10 frame: lịch 0.45s; consumer 0.6s; drift-free → ~0.6s (naive: ~1.05s)
    rep = TripReplayer(fake40_loader, mode="realtime", speed=1.0, start=0, end=10)
    t0 = time.perf_counter()
    for _ in rep:
        time.sleep(0.06)
    elapsed = time.perf_counter() - t0
    assert 0.55 < elapsed < 0.9


def test_realtime_per_frame_arrival_schedule(fake40_loader):
    # Từng frame phải đến KHÔNG SỚM HƠN deadline tuyệt đối của nó —
    # chặn degeneration kiểu "yield hết ngay rồi ngủ 1 cục cuối" (phá HUD).
    # time.sleep bảo đảm cận dưới một phía → test deterministic, không flaky.
    speed = 4.0
    rep = TripReplayer(fake40_loader, mode="realtime", speed=speed)
    t0 = time.perf_counter()
    for b in rep:
        arrival = time.perf_counter() - t0
        assert arrival >= b.timestamp / speed - 0.005, (
            f"frame {b.frame_id} đến lúc {arrival:.4f}s, sớm hơn deadline "
            f"{b.timestamp / speed:.4f}s"
        )


def test_realtime_paces_by_json_timestamp_not_fps(tmp_path):
    # fact #10: pacing phải theo timestamp trong JSON. Trip này cố tình khai
    # metadata.fps=10 lệch với timestamp (i/20) — impl tự tính i/fps sẽ chạy
    # ~0.475s thay vì ~0.24s.
    frames = [make_redacted_frame(i) for i in range(20)]  # ts: 0 → 0.95s
    trip = make_trip(
        tmp_path, "T93d", frames,
        metadata={"trip_id": "T93d", "fps": 10, "map": "Town04", "speed_limit_kmh": 60},
    )
    rep = TripReplayer(TripLoader(trip), mode="realtime", speed=4.0)
    t0 = time.perf_counter()
    assert len(list(rep)) == 20
    elapsed = time.perf_counter() - t0
    assert abs(elapsed - 0.95 / 4.0) < 0.08


def test_realtime_respects_start_offset(fake40_loader):
    # start=20 (ts=1.0s): lịch tính từ ts_start, không phải ts=0 —
    # nếu sai, replay sẽ ngủ thêm 1.0/speed giây ngay frame đầu
    rep = TripReplayer(fake40_loader, mode="realtime", speed=4.0, start=20, end=30)
    t0 = time.perf_counter()
    bundles = list(rep)
    elapsed = time.perf_counter() - t0
    assert [b.frame_id for b in bundles] == list(range(20, 30))
    assert elapsed < 0.25  # lịch: 0.45/4 ≈ 0.11s; sai ts_start sẽ ≥ 0.25s


# ---------------------------------------------------------------------- #
# validation
# ---------------------------------------------------------------------- #
def test_invalid_args(fake40_loader):
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, mode="turbo")
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, speed=0)
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, speed=float("nan"))
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, speed=float("inf"))
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, start=-1)
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, end=41)
    with pytest.raises(ValueError):
        TripReplayer(fake40_loader, start=30, end=20)
