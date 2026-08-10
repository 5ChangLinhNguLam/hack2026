# Hackathon Starter Kit

Chào mừng đến với hackathon AI của FPT Automotive. Kit này cung cấp công cụ để bạn:

1. **Load** dataset trip (ảnh đường + ảnh driver + ground truth)
2. **Chạy baseline** TTC predictor làm điểm khởi đầu
3. **Đánh giá** dự đoán của bạn so với ground truth
4. **Khám phá** trip mẫu bằng Jupyter notebook

> **Ghi chú về tên gọi:** các field/folder liên quan driver trong kit này
> vẫn ghi "NTHU" (folder `driver/`, `frame.nthu_subject_id`,
> `T0Xd_nthu_mapping.json`) vì lý do lịch sử — dữ liệu driver thật lấy từ
> **DMD (Driver Monitoring Dataset)**, không phải dataset NTHU-DDD gốc.
> Điều này không ảnh hưởng gì đến cách bạn dùng kit; chỉ để tên field có
> nghĩa nếu bạn tự mở JSON gốc ra xem.

## Bạn nhận được gì

Mỗi "trip" là 1 mẫu tự chứa đầy đủ. Bạn nhận được **16 trip**, chia làm 2 bộ khác nhau:

- **10 trip chấm điểm — `T01d` .. `T10d`** (mỗi trip ~90 giây/1800 frame,
  trải dài từ lái xe cao tốc bình thường đến kịch bản nhiều sự kiện dồn
  dập lúc nửa đêm). **Cả 10 trip này đều đã bị xóa field ground truth**
  (driver state, TTC, risk score, aggregate trip — xem danh sách field
  bên dưới) — đây là những trip bạn nộp dự đoán để được chấm điểm; ban tổ
  chức chấm riêng bằng dữ liệu đầy đủ họ giữ. Việc thiếu `driver.state`,
  `min_ttc`, `risk`... hay không có file `T0Xd_nthu_mapping.json` ở các
  trip này **không phải bug** — đó là chủ đích.
- **6 trip luyện tập — `T01-Sample` .. `T06-Sample`** (mỗi trip ngắn hơn
  nhiều: ~30 giây/600 frame, chỉ 1-2 loại event kịch bản mỗi trip thay vì
  dồn dập nhiều event như trên). **Cả 6 trip này đều có full ground
  truth** — dùng để build, debug, và tự kiểm tra pipeline của bạn trước
  khi chạy trên 10 trip chấm điểm. Gộp lại 6 trip phủ đủ mọi driver state
  (`alert/drowsy/yawning/distracted/microsleep`) và mọi loại event kịch
  bản (`pedestrian_jaywalk/motorcycle_cut_in/lead_brake/stopped_vehicle_ahead`)
  có trong dataset.

Danh sách field bị xóa khỏi 10 trip chấm điểm (chỉ còn đầy đủ ở 6 trip `-Sample`):
- `frames[].driver.state / alertness_score / eye_state / head_pose / mouth_state / nthu_subject_id`
- `frames[].targets[].rel_pos / rel_velocity / longitudinal_distance / lateral_distance / closing_speed / ttc_simple / ttc_2d / in_collision_cone`
- `frames[].min_ttc`, `frames[].headway_sec`, `frames[].behavior_flags`, `frames[].risk`
- `trip_aggregate`, `driver_summary` (cấp top-level)
- `frames[].ego.location / rotation / geolocation` (giữ lại `speed_kmh`/`longitudinal_accel`/`lateral_accel`)
- `events_log[].params` (giữ lại `type` và `t` — biết "có event gì đó quanh giây X" nhưng không có thông số kịch bản chi tiết)
- toàn bộ file `T0Xd_nthu_mapping.json`
- `kitti/label_2/*.txt` — trường `location` (x, y, z) bị zero-out thành
  `0.00 0.00 0.00`. File `label_2` gốc mã hóa lại đúng vị trí 3D thật của
  target (bypass redaction ở JSON), nên phải xóa riêng ở đây; các trường
  khác trong `label_2` (`type`, `truncated`, `occluded`, `alpha`, bbox 2D,
  `dimensions`, `rotation_y`) vẫn giữ nguyên.

Mọi thứ khác (ảnh đường/ảnh driver, phần còn lại của KITTI label, kinematics
`ego.speed_kmh` + gia tốc, `target_id`/`target_class`, `events_log[].type`/`.t`)
đều có mặt đầy đủ ở **cả 16** trip — đây là input hợp lệ, không phải đáp án.
**Riêng `location` trong `label_2` chỉ có giá trị thật ở 6 trip `-Sample`**,
bị zero-out ở 10 trip `T0Xd` như liệt kê ở trên.

Cấu trúc từng trip (ví dụ với `T01-Sample`, 1 trong 6 trip full-GT):
```
data/T01-Sample/
├── kitti/
│   ├── image_2/         ảnh RGB trái (640×360 JPG — CARLA render ở 1280×720,
│   │                    resize xuống 640×360 cho bản phát hành; calib đã
│   │                    được đồng bộ lại theo đúng độ phân giải này)
│   ├── image_3/         ảnh RGB phải (cặp stereo, baseline 30cm)
│   ├── depth/           depth ground truth (keyframe, .npy)
│   ├── calib/           calibration KITTI theo từng frame
│   └── label_2/         annotation bounding box 2D + 3D
├── driver/              ảnh driver trong cabin (composite từ DMD, tên
│                        field/filename vẫn ghi "NTHU" — xem ghi chú trên)
└── T01-Sample.json.gz   dataset đầy đủ (frame + aggregate + driver state)
```
Với 10 trip `T0Xd` (bị chấm điểm), `driver/` và `kitti/` giống hệt, nhưng
`T0Xd.json.gz` đã bị xóa field ground truth và `T0Xd_nthu_mapping.json`
không tồn tại — xem mục "Bạn nhận được gì" ở trên để có danh sách field
chính xác.

## 3 challenge của hackathon

Dataset hỗ trợ 3 use case. Chọn 1 (hoặc kết hợp):

### Challenge 1 — Collision Risk Monitor (theo từng frame)
Dự đoán TTC (Time-To-Collision) từ ảnh camera. Vượt qua baseline ở:
- MAE trong vùng nguy hiểm (TTC < 3s)
- F1 cho việc phát hiện nguy hiểm (TTC < 2s)

### Challenge 2 — Driver Intelligence Platform
Phân loại trạng thái tài xế từ ảnh trong cabin:
- 5 lớp: `alert`, `drowsy`, `yawning`, `distracted`, `microsleep`
- Cộng thêm hồi quy alertness (0.0–1.0)

### Challenge 3 — Fleet Safe Driving Score (theo từng trip)
Tổng hợp dự đoán per-frame thành 1 điểm số cấp trip (0–100). Bao gồm:
- Tái tạo lại đúng luật rule-based đã dùng làm ground truth
- Hoặc xây model học máy vượt qua rule đơn giản đó

## Cài đặt

```bash
pip install -r requirements.txt
# Tối thiểu cần: opencv-python, numpy, pandas, matplotlib, pyyaml
```

## Quick start — tripkit (Task 1.2: Trip Loader & Replayer)

```bash
pip install -e ".[dev]" && pytest                                  # cài package + chạy test
python -m tripkit.replay data/T01-Sample --limit 50 --stats        # phát nhanh + thống kê trip
python -m tripkit.replay data/T01-Sample --mode realtime --show    # demo HUD 20 FPS (q/ESC thoát)
python -m tripkit.validate data/                                   # kiểm tra toàn vẹn mọi trip
# API cho C1/C2/pipeline:  from tripkit import TripLoader, TripReplayer
```

Contract API (tên field đã khoá với downstream): xem `docs/Task_1.2_TripReplayer_Spec_ClaudeCode.md` mục 3.

## SafeLoop telemetry — thin slice CarSky REST

Phát ego telemetry hợp lệ (không lẫn ground truth/event label) dưới dạng
NDJSON để kiểm tra contract cục bộ:

```bash
python3 -m safeloop.replay_telemetry data/T01-Sample --limit 5
```

Sau khi deploy KUKSA signal node và discovery đúng `roomId`, `nodeKey`, signal
path từ CarSky:

```bash
export A8_API_KEY='<key chỉ lưu local>'
export A8_ROOM_ID='<device id>'
export A8_NODE_KEY='<signal node key>'

python3 -m safeloop.replay_telemetry data/T01-Sample \
  --sink carsky-rest --start 10 --limit 5
```

REST sink dùng `X-API-Key`, validate signal path trước khi replay và không cần
`a8_pin`. Các bước CarSky/curl: `docs/CarSky_REST_Telemetry_Quickstart.md`.
Wire contract và quy tắc chống GT leakage: `docs/SafeLoop_Telemetry_Contract.md`.

Đo live ngày 03/08/2026 cho thấy REST chỉ đạt khoảng `3,3 message/s`, không
phù hợp làm data plane 20 Hz. Integration Lab dùng Script Node timer 50 ms đã
phát 20 frame và observer độc lập nhận đủ 60/60 signal update. Xem kết quả,
failure paths và kiến trúc tiếp theo tại
`reports/CarSky_Integration_Report_20260803.md`.

### Mock C1 + C2 + Risk Fusion trên CarSky

Trong lúc chờ model thật, chạy pipeline deterministic có gắn cờ `mock=true`:

```bash
python3 -m safeloop.replay_mock data/T01-Sample --limit 3

# REST smoke test lên deployment đang chạy
python3 -m safeloop.replay_mock data/T01-Sample \
  --sink carsky-rest --start 0 --limit 40
```

Bản chạy native 20 Hz cho CarSky Script Node nằm tại
`carsky/scripts/safeloop_mock_pipeline.lua`; hướng dẫn gắn node và tạo Signal
Watch: `docs/CarSky_SafeLoop_Mock_Demo.md`.


## Bắt đầu nhanh (5 phút)

> **Không cần copy dataset vào thư mục `data/`.** `./data/T01-Sample` dưới
> đây chỉ là ví dụ đường dẫn — `TripDataset`/`HackathonDataset` nhận
> **bất kỳ đường dẫn nào**, kể cả đường dẫn tuyệt đối trỏ thẳng đến nơi
> bạn đã tải/giải nén dataset (ví dụ `TripDataset(r"D:\hackathon\T01-Sample")`
> trên Windows hoặc `TripDataset("/home/user/dataset/T01-Sample")` trên
> Linux/Mac). Copy vào `data/` chỉ là 1 lựa chọn cho gọn (đường dẫn ngắn,
> khớp đúng ví dụ trong tài liệu) — không phải yêu cầu bắt buộc của code.

```python
from team_kit.dataset_loader import TripDataset

# Load 1 trip -- dùng 1 trong 6 trip T0X-Sample để thấy đủ ground truth
# (10 trip T0Xd vẫn load được bình thường, nhưng field GT sẽ là None/thiếu)
# Đường dẫn dưới đây chỉ là ví dụ -- thay bằng đường dẫn thật tới nơi bạn
# đã giải nén dataset, không cần đúng "./data/..."
ds = TripDataset("./data/T01-Sample")
print(ds.summary())

# Duyệt qua từng frame
for frame in ds.iter_frames():
    left_img = ds.load_left(frame.frame_id)         # H×W×3 BGR
    right_img = ds.load_right(frame.frame_id)
    driver_img = ds.load_driver(frame.frame_id)
    depth = ds.load_depth(frame.frame_id)           # có thể là None (chỉ có ở keyframe)

    # Nhãn ground truth có sẵn:
    print(frame.min_ttc, frame.driver_state, frame.final_risk_score)
```

## Chạy baseline

Baseline dùng **stereo SGBM + tracking depth trung vị trong ROI** — không dùng deep learning. Đây là mức sàn để bạn vượt qua.

```bash
# 1. Chạy baseline trên T01-Sample (1 trong 6 trip full-GT -- tự đánh giá
#    local chỉ hoạt động ở nơi có ground truth; với 10 trip T0Xd, vẫn
#    sinh predictions y hệt cách này nhưng nộp để ban tổ chức chấm thay vì
#    tự chạy evaluation.py cục bộ)
python team_kit/baseline_ttc_predictor.py \
    --trip-dir ./data/T01-Sample \
    --output ./predictions/T01-Sample.csv \
    --verbose

# 2. Đánh giá
python team_kit/evaluation.py \
    --predictions ./predictions/T01-Sample.csv \
    --trip-dir ./data/T01-Sample
```

Output mẫu (baseline chỉ làm Challenge 1 nên chỉ có 1 phần báo cáo):
```
==============================================================================
EVALUATION REPORT - Challenge 1: Collision Risk Monitor (TTC)
==============================================================================
Trips evaluated:        1
Overall MAE (critical): 1.420s
Overall F1:             0.480
Overall composite:      52.3 / 100

Trip         n_crit  MAE-crit   F1      FPR     Composite
------------------------------------------------------------------------------
T01-Sample   62      1.420s     0.480   0.080   52.3
```

Baseline chỉ đạt composite ~50 nghĩa là còn **rất nhiều khoảng trống** để vượt qua.

## SafeLoop C1 — một camera, TTC + lane + matrix-light mô phỏng

Chạy C1 physics v2 từ đúng `image_2`; mặc định detector stride 3 và range-TTC
đã hiệu chỉnh được bật:

```bash
python3 -m safeloop.c1.replay data/T02-Sample \
  --detector-stride 3 --video predictions/T02-c1.mp4 --evaluate

# HUD tổng hợp C1 + lane + virtual matrix-light
python3 -m safeloop.c1.perception_replay data/T02-Sample \
  --video predictions/T02-perception.mp4 --evaluate
```

Sinh pseudo-label có confidence cho 10 trip bị che TTC, rồi chuyển sang schema
submission. Pseudo-label không phải ground truth:

```bash
python3 -m safeloop.c1.cache_detections data/T01d --stride 3
python3 -m safeloop.c1.pseudo_label data/T01d
python3 -m safeloop.c1.submission \
  predictions/c1_pseudo_labels/T01d.csv \
  predictions/c1_submission/T01d.csv \
  --minimum-confidence 0.30
```

Đánh giá offline các phần có thể kiểm chứng:

```bash
python3 tools/evaluate_c1_matrix_light.py
python3 tools/evaluate_c1_lane.py
```

Matrix-light hiện chỉ xuất grid-space request cho mô phỏng/HMI, không có CAN,
GPIO hoặc actuator. Báo cáo và giới hạn đo lường nằm ở
`docs/C1_Optimization_Matrix_Lane_Report.md`.

**`evaluation.py` chấm được cả 3 challenge**, tự động — không cần cờ nào thêm. Nếu file CSV bạn nộp có cột
`predicted_driver_state` và/hoặc `predicted_risk_score` (không chỉ `predicted_ttc`), báo cáo sẽ in
thêm phần Challenge 2 / Challenge 3 tương ứng; thiếu cột nào thì bỏ qua phần đó (không bị trừ điểm vì
"không làm"). Ví dụ nộp đủ cả 3 cột:
```
==============================================================================
Challenge 2: Driver Intelligence Platform (driver state)
==============================================================================
Overall composite:      73.4 / 100

Trip           n_scored   Accuracy   Macro-F1   Composite
------------------------------------------------------------------------------
T01-Sample     600        0.688      0.780      73.4

==============================================================================
Challenge 3: Fleet Safe Driving Score
==============================================================================
Overall composite:      100.0 / 100

Trip           Predicted  True       AbsErr     Composite
------------------------------------------------------------------------------
T01-Sample     0.0        0.0        0.0        100.0

Breakdown (deterministic from trip facts, except near_miss = your own predicted_ttc):
Trip           near_miss  harsh_brk  harsh_acc  harsh_crn  speeding%
------------------------------------------------------------------------------
T01-Sample     3          14         53         2          0.0
```

**Công thức từng challenge:**

| Challenge | Composite (0–100) |
|---|---|
| 1 — TTC | `40%×MAE-critical + 30%×F1 + 30%×inverse-TTC MAE` |
| 2 — Driver state | `50%×accuracy + 50%×macro-F1` (macro-F1 chỉ tính trên các lớp thật sự xuất hiện trong trip đó) |
| 3 — Fleet Safe Driving Score | Tái tạo đúng công thức nội bộ (trừ dần từ 100) — xem bên dưới |

**Challenge 3 tái tạo đúng công thức gốc** ở
`package_organizer-v3-remote/src/analytics/behavior.py` (`BehaviorScorer.aggregate()`), gần như chính
xác 100%:

```
predicted_safe = 100 − (harsh_brake×3.0 + harsh_accel×2.0 + harsh_corner×2.0 + near_miss×5.0 + speeding%×0.15)
composite = 100 − 2×|predicted_safe − true_safe|
```

- `harsh_brake`/`harsh_accel`/`harsh_corner` (số lần) và `speeding%` (% thời gian) được tính **thẳng từ
  dữ liệu trip** (`ego.longitudinal_accel`/`lateral_accel`/`speed_kmh` so với `metadata.speed_limit_kmh`)
  — các field này **không bị redact** ở bất kỳ trip nào kể cả `T0Xd`, và không phụ thuộc model của bạn,
  nên giống nhau cho mọi team trên cùng 1 trip.
- `near_miss` (số frame TTC thật < 1.5s) dùng thẳng cột `predicted_ttc` bạn đã nộp cho Challenge 1 —
  không cần nộp thêm cột nào mới cho phần này.
- Cột `predicted_risk_score` vẫn là điều kiện để báo cáo Challenge 3 xuất hiện (bạn "đăng ký" làm
  challenge này), nhưng giá trị số trong đó **không được đọc** vào công thức trên nữa.

> **1 khoảng trống còn lại:** công thức gốc còn 1 số hạng `tailgating%×0.10` (bám đuôi quá gần) mà
> `evaluation.py` **chưa tái tạo được**, vì nó phụ thuộc `headway_sec` (đã bị xóa) và hiện chưa có cột
> nào trong CSV nộp bài để thí sinh dự đoán khoảng cách/thời gian bám đuôi. Vì vậy composite có thể hơi
> cao hơn 1 chút so với model hoàn hảo ở các trip có bám đuôi thật — nhỏ hơn nhiều so với sai số của cách
> tính cũ (gộp trung bình risk). Muốn khớp 100% cần thêm cột `predicted_headway_sec` vào format nộp bài.

> **Lưu ý bảo mật:** `evaluation.py` **luôn luôn** tự load ground truth từ
> `--data-dir`/`--trip-dir` đáng tin cậy, **không bao giờ** tin vào bất kỳ
> cột "ground_truth" nào bạn tự ghi trong file predictions.csv của mình —
> kể cả khi `baseline_ttc_predictor.py` có ghi sẵn 1 cột như vậy để bạn
> tham khảo cục bộ, cột đó hoàn toàn bị bỏ qua khi chấm điểm thật.

## Khám phá trip trực quan

```bash
jupyter notebook team_kit/explore_trip.ipynb
```

Notebook này hiển thị:
- Timeline driver state + đường cong alertness
- Tốc độ/gia tốc theo thời gian
- TTC ground truth + risk score (có vạch đánh dấu event)
- 4 frame lấy mẫu (ảnh đường + ảnh driver đặt cạnh nhau)
- Tổng kết aggregate của trip

## Định dạng nộp bài

Với mỗi trip bạn làm (trong số 10 trip chấm điểm `T01d`..`T10d`), nộp 1 file CSV:

```csv
frame_id,timestamp,predicted_ttc,predicted_driver_state,predicted_risk_score
0,0.000,inf,alert,5
1,0.050,inf,alert,5
...
1798,89.900,2.3,drowsy,67
1799,89.950,2.1,drowsy,68
```

(số dòng = số frame của trip — 1800 dòng cho các trip `T0Xd`, 600 dòng nếu
bạn cũng xuất CSV cho 1 trip luyện tập `T0X-Sample`)

- `predicted_ttc`: đơn vị giây, dùng `inf` nếu không phát hiện vật cản
- `predicted_driver_state`: 1 trong các giá trị `alert|drowsy|yawning|distracted|microsleep`
- `predicted_risk_score`: số thực 0–100

Chỉ làm 1 challenge? Bỏ hẳn cột tương ứng khỏi CSV (đừng để trống hoặc
điền số bừa) — `evaluation.py` tự phát hiện cột nào có mặt để chỉ chấm
đúng challenge bạn đã làm, xem mục "Chạy baseline" ở trên.

Quy ước tên file: `predictions/<tên_team>/<trip_id>.csv` (ví dụ `T01d.csv`).

## Tham chiếu API

### `TripDataset(trip_dir)`

| Method/Property | Trả về | Mô tả |
|---|---|---|
| `len(ds)` | int | Số lượng frame |
| `ds[idx]` | `FrameRecord` | Frame tại vị trí idx |
| `ds.iter_frames()` | iterator | Duyệt qua toàn bộ frame |
| `ds.frames_df` | `pd.DataFrame` | DataFrame phẳng phục vụ phân tích |
| `ds.load_left(frame_id)` | `np.ndarray` | Ảnh RGB trái (HxWx3 BGR) |
| `ds.load_right(frame_id)` | `np.ndarray` | Ảnh RGB phải |
| `ds.load_depth(frame_id)` | `np.ndarray \| None` | Depth GT tính bằng mét (chỉ có ở keyframe) |
| `ds.load_driver(frame_id)` | `np.ndarray` | Ảnh driver (NTHU) |
| `ds.load_calibration()` | dict | Intrinsics camera + baseline (global, không đổi theo frame — dùng cho baseline) |
| `ds.load_frame_calibration(frame_id)` | dict | Calib KITTI chuẩn theo từng frame: `P0`-`P3` (3×4), `R0_rect` (3×3), `Tr_velo_to_cam`/`Tr_imu_to_velo` (3×4), parse từ `kitti/calib/{frame_id:06d}.txt` — dùng khi cần tái sử dụng code KITTI 3D detection có sẵn |
| `ds.summary()` | dict | Tổng quan cấp trip |
| `ds.metadata` | dict | Metadata config |
| `ds.trip_aggregate` | dict | Safe Driving Score + số đếm |
| `ds.driver_summary` | dict | Phân bố driver state + fatigue |
| `ds.events_log` | list | Các event đã fire trong trip |

### Thuộc tính `FrameRecord`

```python
frame.frame_id              # int
frame.timestamp             # float, giây
frame.speed_kmh             # float
frame.longitudinal_accel    # float, m/s²
frame.lateral_accel         # float, m/s²
frame.driver_state          # str: alert|drowsy|yawning|distracted|microsleep
frame.alertness_score       # float 0-1
frame.eye_state             # str: open|partial|closed
frame.head_pose             # str: normal|down|side
frame.mouth_state           # str: normal|yawning|talking
frame.nthu_subject_id       # str
frame.min_ttc               # float, giây (inf nếu không có target)
frame.headway_sec           # float
frame.base_risk             # float 0-100
frame.driver_factor         # float, hệ số nhân
frame.final_risk_score      # float 0-100
frame.is_harsh_brake        # bool
frame.is_harsh_accel        # bool
frame.is_harsh_corner       # bool
frame.is_speeding           # bool
frame.is_tailgating         # bool
frame.targets               # list các actor phát hiện được, kèm thông tin TTC
frame.events_active         # list event đang fire tại thời điểm đó
```

### `HackathonDataset(data_root)` — wrapper nhiều trip

`data_root` cũng là đường dẫn bất kỳ, không bắt buộc là `./data` — trỏ
vào thư mục **chứa trực tiếp** các thư mục trip (`T01-Sample/`, `T01d/`...)
ở đâu cũng được, miễn tất cả 16 trip nằm chung 1 thư mục cha.

```python
from team_kit.dataset_loader import HackathonDataset

all_trips = HackathonDataset("./data")  # thay bằng đường dẫn thật của bạn
print(all_trips.trip_ids)              # ['T01-Sample', ..., 'T06-Sample', 'T01d', ..., 'T10d']
print(all_trips.summary_table())       # DataFrame: mỗi dòng 1 trip

for trip in all_trips:                  # duyệt qua toàn bộ trip
    print(trip.summary())
```

## Ý tưởng để vượt qua baseline

Baseline *chỉ* tính trung vị disparity stereo trong 1 ROI cố định. Vài hướng dễ cải thiện:

| Cách tiếp cận | Kỳ vọng tăng điểm |
|---|---|
| Thêm object detection (YOLOv8) → ROI bám đúng xe gần nhất | +20-30 composite |
| Multi-object tracking (DeepSORT) → làm mượt theo thời gian cho từng object | +5-10 |
| Depth đơn mắt (MiDaS, ZoeDepth) làm tín hiệu dự phòng cho stereo | +5 |
| Driver state qua CNN (ResNet, ViT) trên ảnh driver | trọn Challenge 2 |
| Kết hợp alertness driver vào risk score (drowsy → hạ ngưỡng TTC) | trọn Challenge 3 |
| Temporal modeling (LSTM/Transformer) qua chuỗi frame | +5-15 |

## Mẹo

- **Frame rate**: 20 FPS. Tận dụng ngữ cảnh thời gian — TTC ở frame hiện tại phụ thuộc lịch sử chuyển động.
- **Stereo baseline**: 30 cm. `ds.load_calibration()` cho bạn `fx`, `K`, `P2`, `P3`.
- **Determinism**: Cùng input → cùng output kỳ vọng. Đừng dựa vào random augmentation lúc inference.
- **Vùng nguy hiểm quan trọng nhất**: lỗi TTC khi GT < 3s được tính trọng số gấp 10× khi chấm điểm. Đừng tối ưu quá đà cho case `inf`.
- **Driver/road tách biệt**: driver state và event CARLA **độc lập với nhau** — đừng train với giả định "tài xế buồn ngủ luôn xảy ra trước khi có xe cắt ngang". Dữ liệu được sinh ra có chủ đích tách biệt như vậy.

## Tùy chọn: truy vấn dataset qua HTTP thay vì đọc file trực tiếp

Nếu stack training của bạn không phải Python, hoặc muốn có 1 dashboard
local nhanh, `team_kit/local_stream_server.py` wrap `TripDataset`/
`HackathonDataset` thành 1 API HTTP nhỏ — **hoàn toàn tùy chọn**, chạy
trên máy bạn, trên dữ liệu bạn đã tải sẵn, không cần cài thêm dependency
nào (chỉ dùng thư viện chuẩn Python):

```bash
python team_kit/local_stream_server.py --data-dir ./data_public --port 8765
curl http://127.0.0.1:8765/trips
curl http://127.0.0.1:8765/trips/T01-Sample/frames/0
curl http://127.0.0.1:8765/trips/T01-Sample/frames/0/left --output frame0.jpg
```

Đây là pass-through thuần túy qua đúng file/redaction bạn đã có sẵn — xem
docstring trong file để có đầy đủ danh sách route. Hầu hết các team không
cần dùng đến; đọc file trực tiếp qua `dataset_loader.py` đơn giản hơn nếu
stack của bạn là Python.

## Có câu hỏi?

Hỏi trong kênh Slack của hackathon hoặc xem README chính của dự án (ở thư mục cha).
