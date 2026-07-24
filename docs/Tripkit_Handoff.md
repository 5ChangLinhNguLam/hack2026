# Bàn giao tripkit v0.1.0 — Trip Loader & Replayer (WBS 1.2)

> Interface **ĐÃ KHOÁ** theo contract ở `docs/Task_1.2_TripReplayer_Spec_ClaudeCode.md` mục 3.
> Muốn đổi tên field/chữ ký hàm → phải thống nhất lại với cả nhóm trước.
> Dành cho: **Hải (C1)**, **Thiện (C2)**, pipeline runner 1.4, HUD/dashboard (Sơn).

## Cài & kiểm tra

```bash
pip install -e ".[dev]"          # từ repo root
pytest                           # 63 test phải xanh
python -m tripkit.validate data/ # kiểm tra dataset trên máy bạn
```

## Dùng ngay — 10 dòng

```python
from tripkit import TripLoader, TripReplayer

loader = TripLoader("data/T01-Sample")          # hoặc data/T01d (trip chấm điểm)
print(loader.n_frames, loader.fps, loader.has_gt())   # 600/1800..., 20.0, True/False

for bundle in TripReplayer(loader, mode="fast"):           # batch inference
    left = bundle.left()          # np.ndarray BGR (360,640,3) — C1 dùng left()/right()
    face = bundle.driver()        # ảnh cabin — C2 dùng
    if bundle.gt:                 # LUÔN kiểm None trước — trip chấm điểm gt=None
        state = bundle.gt["driver"]["state"]

# Demo HUD: TripReplayer(loader, mode="realtime", speed=1.0) — pacing đúng 20 FPS
```

## Những điều PHẢI biết khi dùng

| # | Điều cần biết | Chi tiết |
|---|---|---|
| 1 | **Mọi field GT là Optional** | Trip chấm điểm: `bundle.gt is None`, `loader.trip_aggregate/driver_summary is None`, `has_gt() == False`. Code của bạn không được giả định GT tồn tại. |
| 2 | **Ảnh lazy** | `left()/right()/driver()` là **method**, chỉ đọc đĩa khi gọi, cache trong bundle (tắt: `loader.frame(i, cache_images=False)`). |
| 3 | **Depth chỉ có ở keyframe** `i%5==0` | `bundle.depth` là `None` ở frame khác. Cần depth mọi frame → `loader.depth_nearest(i)` trả `(depth tại i−i%5, is_exact)`. |
| 4 | **`min_ttc` = inf là bình thường** | `float('inf')` = không có target trong collision cone. Dùng `math.isinf()` để kiểm. |
| 5 | **Bundle an toàn để sửa tại chỗ** | `frame()` trả copy riêng — annotate `targets` khi tracking thoải mái, không làm bẩn loader. Cần zero-copy: `loader.raw_frame(i)` (đừng sửa dict này). |
| 6 | **KITTI label**: bbox 2D luôn = 0 | Chỉ `type` + dims 3D + xyz 3D là thật; file rỗng → `bundle.labels == []`. Trip chấm điểm xyz cũng bị zero. |
| 7 | **Calib load 1 lần/trip** | `loader.calib` → `Calib(fx=320, cx=320, cy=180, baseline_m=0.3, 640x360)`, `depth_factor == 96.0` (depth = 96/disparity). |
| 8 | **Không hardcode 600** | `loader.n_frames` là nguồn sự thật (600 mẫu / 1800 chấm điểm / bất kỳ). |
| 9 | **`target_id` ổn định** | Dùng trực tiếp cho tracking, không remap. |

## CLI

```bash
python -m tripkit.replay data/T01-Sample --limit 50 --stats        # thống kê nhanh
python -m tripkit.replay data/T01-Sample --mode realtime --show    # xem bằng mắt (cần GUI)
python -m tripkit.validate data/                                   # bảng toàn vẹn mọi trip
```

## Đã xác nhận chạy được trên

- 6 trip mẫu thật `T01..T06-Sample` (600 frame, đủ GT): validate 6/6 OK, replay đủ frame.
- Trip **1800 frame không GT** (giả lập đầy đủ modality đúng cấu trúc T0Xd): validate OK (depth 360/360, GT=không), replay fast 1800/1800, realtime đúng lịch, toàn bộ API không crash.

## Còn chờ (khi có trip T0Xd thật)

1. Chạy `python -m tripkit.validate data/` + `python -m tripkit.replay data/T01d --limit 50 --stats` để xác minh trên redaction thật của ban tổ chức.
2. Tài liệu kit đang **mâu thuẫn** về `behavior_flags` ở trip chấm điểm (README: bị xoá; HUONG_DAN: được giữ). `tripkit` đã chịu được cả 2 chiều; khi có data thật sẽ chốt lại để C3 biết đường dùng.
