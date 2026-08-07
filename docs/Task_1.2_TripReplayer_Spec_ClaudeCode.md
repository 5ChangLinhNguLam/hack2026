# Task 1.2 — Trip Loader & Replayer (20 FPS) — Spec cho Claude Code

> **WBS 1.2** · Owner: Tú · Support: Sơn · P0 · 22–25/07 · Milestone M1
> Mục tiêu: module nền tảng phát lại 1 trip theo timestamp, đồng bộ ảnh stereo + driver + kinematics theo frame, **dùng chung cho mọi module** (C1 của Hải, C2 của Thiện, pipeline runner 1.4, HUD/dashboard của Sơn).

---

## 1. Scope

### IN scope
1. **`tripkit/loader.py`** — bọc/thay `dataset_loader.py` của Team Kit: đánh index 1 thư mục trip, đọc JSON telemetry, tra cứu đường dẫn ảnh/depth/calib/label theo `frame_id`.
2. **`tripkit/replayer.py`** — iterator phát lại frame theo thứ tự timestamp, 2 chế độ:
   - `fast` (mặc định): chạy nhanh nhất có thể — dùng cho batch inference & pipeline runner.
   - `realtime`: pacing đúng 20 FPS (hoặc `--speed 2.0`…) — dùng cho demo HUD.
3. **`tripkit/validate.py`** — kiểm tra tính toàn vẹn trip (đếm file, kích thước ảnh, calib đồng nhất).
4. **CLI**: `python -m tripkit.replay <trip_dir>` và `python -m tripkit.validate <data_dir>`.
5. **Unit tests** (pytest) chạy được trên `T01-Sample`.

### OUT of scope (đừng để Claude Code lan man)
- Perception/model (C1, C2), ghi CSV predictions (đó là task 1.4), MQTT/Redis (đã cắt), CarSky, dashboard.
- Không sửa/copy dataset; **không commit dataset vào repo** (thêm `data/` vào `.gitignore` ngay từ đầu).

---

## 2. Facts về dataset mà code PHẢI tôn trọng (đã kiểm chứng)

| # | Fact | Hệ quả cho code |
|---|------|-----------------|
| 1 | Trip mẫu: 600 frame (30s @ 20FPS), trip chấm điểm `T01d–T10d`: **1800 frame (90s)** | Không hardcode 600; đọc `len(frames)` & `metadata.fps` |
| 2 | JSON chứa token **`Infinity` trần** (không phải JSON chuẩn) | Python `json.load` đọc được — GIỮ nguyên `float('inf')` trong bộ nhớ; nếu cần serialize cho JS thì thay bằng `null`/số lớn ở tầng khác |
| 3 | Chỉ `T01-Sample` có sẵn `.json` giải nén, còn lại chỉ có `.json.gz` | Ưu tiên `.json` nếu tồn tại, fallback `gzip.open(path, 'rt')` |
| 4 | Ảnh driver tên `driver/frame_{i:06d}.jpg`; ảnh kitti tên `kitti/image_2/{i:06d}.jpg` (không tiền tố) | 2 pattern tên file khác nhau — viết helper riêng, có test |
| 5 | Depth chỉ có ở **keyframe `i % 5 == 0`** — 120 file `.npy` float32 shape (360,640), đơn vị mét | `bundle.depth` là `Optional`; thêm helper `depth_nearest(i)` trả depth tại `i - i%5` kèm cờ `is_exact` |
| 6 | `calib/` 600 file **giống hệt nhau**; calib thật nằm ở `calibration_info.txt`: fx=fy=320, cx=320, cy=180, baseline 0.3m, `depth = 96/disparity` | Load calib **1 lần/trip** thành object `Calib`, đừng đọc 600 file |
| 7 | `label_2/` phần lớn file **rỗng 0 byte** (bình thường, không phải lỗi); chỉ `type` + dims 3D (h,w,l) + location 3D (x,y,z hệ cam trái) là thật; bbox 2D/alpha/rotation_y luôn = 0; ở trip chấm điểm x,y,z bị zero-out | Parser trả `[]` cho file rỗng; đừng validate "label phải có bbox" |
| 8 | Trip chấm điểm bị xoá GT: driver state, ttc_*, risk, trip_aggregate, driver_summary, vị trí ego, params event | **Mọi field GT là Optional** — loader không được crash vì thiếu key; ego `speed_kmh`/accel vẫn còn (là input C3) |
| 9 | `targets[].target_id` ổn định theo thời gian | Giữ nguyên, không remap — downstream dùng để tracking |
| 10 | `timestamp = frame_id / fps` | Replayer pacing dựa vào timestamp trong JSON, không tự tính lại tuỳ tiện |

---

## 3. API contract (downstream đã thống nhất — đừng đổi tên field)

```python
# tripkit/types.py
@dataclass(frozen=True)
class Calib:
    fx: float; fy: float; cx: float; cy: float
    baseline_m: float                      # 0.3
    width: int; height: int                # 640, 360
    @property
    def depth_factor(self) -> float:       # = fx * baseline = 96.0
        ...

@dataclass
class FrameBundle:
    trip_id: str
    frame_id: int
    timestamp: float                       # giây, = frame_id / fps
    # Ảnh: lazy-load, trả np.ndarray BGR (cv2.imread), cache tuỳ chọn
    def left(self)  -> np.ndarray: ...     # kitti/image_2
    def right(self) -> np.ndarray: ...     # kitti/image_3
    def driver(self) -> np.ndarray: ...    # driver/frame_*.jpg
    depth: np.ndarray | None               # CHỈ khi frame_id % 5 == 0
    ego: dict | None                       # speed_kmh, longitudinal/lateral_accel, ...
    targets: list[dict]                    # [] nếu không có / bị xoá
    labels: list[KittiLabel]               # [] nếu file rỗng
    gt: dict | None                        # driver/min_ttc/risk... — None ở trip chấm điểm
    events_active: list[dict]

class TripLoader:
    def __init__(self, trip_dir: str | Path): ...
    @property
    def trip_id(self) -> str: ...
    @property
    def n_frames(self) -> int: ...         # 600 hoặc 1800
    @property
    def fps(self) -> float: ...            # 20
    @property
    def calib(self) -> Calib: ...
    @property
    def metadata(self) -> dict: ...
    def frame(self, i: int) -> FrameBundle: ...
    def depth_nearest(self, i: int) -> tuple[np.ndarray, bool]: ...
    def has_gt(self) -> bool: ...          # tự phát hiện trip mẫu vs trip chấm điểm

class TripReplayer:
    def __init__(self, loader: TripLoader, mode: str = "fast",
                 speed: float = 1.0, start: int = 0, end: int | None = None): ...
    def __iter__(self) -> Iterator[FrameBundle]: ...
    # realtime: pacing drift-free theo lịch tuyệt đối
    # deadline(i) = t0 + (ts_i - ts_start)/speed, dùng time.monotonic()
```

---

## 4. Kế hoạch triển khai theo phase (mỗi phase: code → test → commit)

**Phase 0 — Skeleton (30')**
Cấu trúc `tripkit/` package, `requirements.txt` (`numpy`, `opencv-python-headless`, `pytest`), `.gitignore` có `data/`, `*.npy`, `predictions/`. Copy `dataset_loader.py` của Team Kit vào `third_party/` để tham chiếu (đọc, không sửa).
*DoD: `pip install -e .` + `pytest` (0 test) chạy xanh.*

**Phase 1 — TripLoader: JSON + index (nửa ngày)**
Đọc JSON (fact #2, #3), index đường dẫn ảnh theo 2 pattern (fact #4), load `Calib` từ `calibration_info.txt` (fact #6), auto-detect `has_gt()` (fact #8).
*DoD: test `loader.n_frames == 600` trên T01-Sample; test `math.isinf(frames[k]['min_ttc'])` với 1 frame biết trước; test thiếu key GT không crash (xoá key giả lập).*

**Phase 2 — FrameBundle: ảnh lazy + depth + labels (nửa ngày)**
Lazy-load 3 ảnh; `depth` chỉ keyframe + `depth_nearest` (fact #5); parser KITTI label chịu được file rỗng (fact #7).
*DoD: test shape ảnh (360,640,3); depth shape (360,640) float32; `frame(3).depth is None`; label frame rỗng → `[]`.*

**Phase 3 — TripReplayer (nửa ngày)**
Iterator `fast`; chế độ `realtime` pacing theo lịch tuyệt đối `time.monotonic()` (không `sleep(1/fps)` cộng dồn — sẽ trôi), hỗ trợ `speed`, `start/end`.
*DoD: test fast trả đủ n_frames đúng thứ tự; test realtime với 40 frame @ speed=4 chạy trong 0.5s ± 0.1s.*

**Phase 4 — CLI + validate + README (2–3h)**
- `python -m tripkit.replay data/T01-Sample --mode fast --limit 50 --stats` → in fps đọc được, số frame, có/không GT.
- `python -m tripkit.replay data/T01-Sample --mode realtime --show` → cửa sổ cv2 3 ảnh cạnh nhau + timestamp (tuỳ chọn, để smoke test bằng mắt).
- `python -m tripkit.validate data/` → bảng: mỗi trip đủ 600/1800 ảnh × 3 cam, 120/360 depth, calib md5 đồng nhất, cảnh báo (không phải lỗi) label rỗng.
*DoD: chạy validate trên cả 6 trip mẫu pass; README section "Quick start" 5 dòng.*

**Phase 5 — Bàn giao (1h)**
Chạy thử với 1 trip chấm điểm `T0Xd` (1800 frame, không GT) — xác nhận không crash. Thông báo data contract (mục 3) cho Hải, Thiện, và khoá interface.

---

## 5. Acceptance criteria (khớp DoD trong WBS)

- [ ] 1 lệnh phát lại trọn vẹn 1 trip: ảnh stereo + driver + kinematics **đồng bộ theo frame_id/timestamp**.
- [ ] Chạy được trên cả 6 trip mẫu **và** trip chấm điểm 1800 frame không GT.
- [ ] `pytest` xanh; không đọc lại calib/JSON nhiều lần trong vòng lặp (đọc 1 lần, lazy ảnh).
- [ ] Không có dataset trong git; clone sạch + trỏ `--data-dir` là chạy.
- [ ] Hải (C1) và Thiện (C2) import `TripLoader/TripReplayer` chạy được ngay, không cần hỏi lại.
