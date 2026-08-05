# Giải thích toàn bộ Label & Đặc trưng của Dataset

> Tài liệu tra cứu nhanh, ngôn ngữ đơn giản. Mọi con số/enum trong đây đã được đối chiếu trực tiếp với `team_kit/evaluation.py`, `team_kit/dataset_loader.py` và dữ liệu thật trong `data/T01-Sample`..`T06-Sample`.

---

## 1. Tổng quan trong 1 phút

Dataset mô phỏng **an toàn lái xe đa phương thức**. Mỗi mẫu là 1 **trip** (chuyến đi) dài **30 giây, 20 FPS = 600 frame** (đánh số `frame_id` từ 0 đến 599).

Có **4 nguồn dữ liệu** ghi đồng thời trong mỗi trip:

| Nguồn | Là gì | Xuất xứ |
|---|---|---|
| Camera đường (stereo trái/phải) | Ảnh RGB nhìn về phía trước | Mô phỏng CARLA 0.9.15 |
| Camera cabin (`driver/`) | Ảnh khuôn mặt tài xế thật | Bộ dữ liệu DMD (tên trường ghi "nthu" chỉ là tên cũ, **không phải** NTHU-DDD) |
| Telemetry + Ground truth (`.json.gz`) | Vận tốc, vật thể, TTC, trạng thái tài xế, điểm rủi ro, sự kiện | Engine mô phỏng |
| Nhãn KITTI (`kitti/`) | Calib, nhãn 3D, ảnh depth | Xuất theo chuẩn thư mục KITTI |

**Quy tắc liên kết:** mọi thứ ghép với nhau bằng `frame_id`. Frame thứ `i` tương ứng:

```
frame i  ⇔  driver/frame_{i:06d}.jpg          (mặt tài xế)
         ⇔  kitti/image_2/{i:06d}.jpg          (camera trái)
         ⇔  kitti/image_3/{i:06d}.jpg          (camera phải)
         ⇔  kitti/calib/{i:06d}.txt            (thông số camera)
         ⇔  kitti/label_2/{i:06d}.txt          (nhãn 3D — nhiều file rỗng)
         ⇔  frames[i]  trong file JSON          (toàn bộ ground truth)
```

`timestamp = frame_id / 20` (giây). Ảnh **depth** chỉ có ở frame chia hết cho 5 (`i % 5 == 0`).

**Cây thư mục 1 trip:**

```
T01-Sample/
├── driver/                  600 ảnh mặt tài xế: frame_000000.jpg … frame_000599.jpg
├── kitti/
│   ├── image_2/             600 ảnh camera trái
│   ├── image_3/             600 ảnh camera phải (baseline 0.3 m)
│   ├── depth/               120 file .npy (chỉ frame chia hết 5)
│   ├── calib/               600 file thông số camera (nội dung giống hệt nhau)
│   ├── label_2/             600 file nhãn 3D KITTI (đa số RỖNG)
│   └── calibration_info.txt  thông số ống kính chung cho cả trip
├── T01-Sample.json.gz       telemetry + ground truth (nén gzip)
└── T01-Sample.json          chỉ T01 mới có bản giải nén (nội dung giống .gz)
```

Có **3 bài toán (challenge)** dùng chung dataset này:
- **C1 — Collision Risk Monitor:** từ ảnh camera, dự đoán TTC (thời gian đến va chạm) mỗi frame.
- **C2 — Driver Intelligence:** từ ảnh cabin, phân loại trạng thái tài xế (5 lớp).
- **C3 — Fleet Safe Driving Score:** tính điểm lái xe an toàn 0–100 cho cả trip.

---

## 2. Label chính theo từng Challenge (phần quan trọng nhất)

### 2.1. C1 — `min_ttc` (Time-To-Collision)

`min_ttc` = thời gian ngắn nhất (giây) đến khi va chạm, xét trên các vật thể phía trước.

- **Quan trọng:** chỉ tính những target có `in_collision_cone = true` (nằm trong "nón va chạm" phía trước xe). Nếu không có target nào trong nón → `min_ttc = Infinity` (vô cực = an toàn, không có nguy cơ).
- Nó **không phải** là min của tất cả `ttc_simple` — chỉ xét vật trong nón.

Các ngưỡng dùng để chấm điểm (`evaluation.py:117-133`):

| Ngưỡng | Giá trị | Ý nghĩa |
|---|---|---|
| `CRITICAL_TTC_SEC` | < 3.0 s | Vùng "nguy cấp" — MAE ở vùng này mới được tính điểm |
| `DANGER_TTC_SEC` | < 2.0 s | "Cần phanh khẩn cấp (AEB)" — đây là mốc để tính lớp dương/âm cho F1 |
| `NEAR_MISS_TTC_SEC` | < 1.5 s | 1 frame "suýt va chạm" (near-miss) — dùng cho điểm C3 |

### 2.2. C2 — `driver.state` (trạng thái tài xế, 5 lớp)

Đây là label chính của C2. **5 lớp** (`evaluation.py:121`):

`alert` · `distracted` · `drowsy` · `yawning` · `microsleep`

Điểm mấu chốt: **điểm tỉnh táo (`alertness_score`) và các đặc điểm khuôn mặt được suy ra CỐ ĐỊNH từ `state`** — biết `state` là biết hết. Bảng ánh xạ (xác minh từ dữ liệu):

| `state` | Nghĩa tiếng Việt | `alertness_score` | `eye_state` | `head_pose` | `mouth_state` |
|---|---|---|---|---|---|
| `alert` | Tỉnh táo | 0.95 | open (mở) | normal | normal |
| `distracted` | Mất tập trung | 0.45 | open (mở) | side (nghiêng) | normal |
| `drowsy` | Buồn ngủ | 0.35 | partial (lim dim) | down (cúi) | normal |
| `yawning` | Đang ngáp | 0.55 | partial (lim dim) | normal | yawning (ngáp) |
| `microsleep` | Ngủ gật (nguy hiểm nhất) | 0.05 | closed (nhắm) | down (cúi) | normal |

- `alertness_score`: thang **0–1** (0 = ngủ gật/nguy hiểm, 1 = tỉnh táo hoàn toàn).
- ⚠️ **Chỉ `predicted_driver_state` được chấm điểm.** Alertness **không** được chấm (dù README có nhắc "hồi quy alertness"). Bộ chấm chỉ đọc lớp trạng thái.
- Trạng thái đổi theo **đoạn thời gian dài**, không nhấp nháy từng frame (ví dụ T01: 0–15s là `distracted`, 15–30s là `alert`; điểm chuyển đúng ở frame 300).
- (Enum phụ đầy đủ: `eye_state ∈ {open, partial, closed}`, `head_pose ∈ {normal, down, side}`, `mouth_state ∈ {normal, yawning, talking}` — riêng `talking` không xuất hiện trong 6 trip mẫu.)

### 2.3. C3 — `safe_driving_score` (điểm lái xe an toàn, 0–100)

Điểm cấp-trip, tính bằng **công thức phạt cố định** (không cần ML), càng nhiều hành vi xấu càng bị trừ:

```
safe_driving_score = max(0, 100 − (
      harsh_brake_count   × 3.0     (mỗi lần phanh gấp)
    + harsh_accel_count   × 2.0     (mỗi lần tăng tốc gấp)
    + harsh_corner_count  × 2.0     (mỗi lần vào cua gắt)
    + near_miss_count     × 5.0     (mỗi frame suýt va chạm)
    + speeding_pct_time   × 0.15    (% thời gian chạy quá tốc độ)
    + tailgating_pct_time × 0.10    (% thời gian bám đuôi quá gần)
))
```

Các ngưỡng phát hiện hành vi (`evaluation.py:129-134`, `G = 9.81 m/s²`):

| Cờ hành vi | Điều kiện |
|---|---|
| `harsh_brake` (phanh gấp) | `longitudinal_accel < −0.40 × G` |
| `harsh_accel` (tăng tốc gấp) | `longitudinal_accel > +0.35 × G` |
| `harsh_corner` (vào cua gắt) | `\|lateral_accel\| > 0.30 × G` |
| `speeding` (quá tốc độ) | `speed_kmh > speed_limit_kmh + 5` |
| `near_miss` (suýt va chạm) | `min_ttc < 1.5 s` |

> ⚠️ **Lưu ý:** Cả 6 trip mẫu đều có `safe_driving_score = 0` vì tổng phạt vượt quá 100 (ví dụ T01: 100 − 187 ≈ −87 → kẹp về 0). Đây là **đúng thiết kế, không phải lỗi**. Nghĩa là C3 không có tín hiệu phân biệt trên tập mẫu.

---

## 3. Giải thích chi tiết các trường trong file JSON

File JSON có 6 khóa cấp cao nhất: `trip_id`, `metadata`, `driver_summary`, `trip_aggregate`, `events_log`, `frames`.

> ⚠️ **Cảnh báo khi đọc file:** JSON chứa token `Infinity` trần (không phải chuỗi). Python `json.load()` đọc được, nhưng JavaScript `JSON.parse()` sẽ **lỗi** — phải thay `Infinity` trước khi parse.

### 3.1. `metadata` — cấu hình mô phỏng

| Trường | Ý nghĩa |
|---|---|
| `trip_id`, `description` | Mã và mô tả trip |
| `duration_sec`, `fps` | 30 giây, 20 FPS |
| `map` | Bản đồ CARLA (Town10HD, Town04…) |
| `weather` | Thời tiết: `cloudiness`, `precipitation` (mưa), `sun_altitude_angle` (độ cao mặt trời, âm = tối), `fog_density`, `wetness`… |
| `driver_profile` | Luôn là `normal` |
| `speed_limit_kmh` | Giới hạn tốc độ — dùng để tính cờ `speeding` |
| `random_seed`, `carla_version` | Seed và phiên bản |

### 3.2. `frames[]` — bản ghi mỗi frame (600 phần tử)

Mỗi frame gồm các nhóm sau.

**Nhóm `ego`** (xe của mình):

| Trường | Ý nghĩa |
|---|---|
| `speed_kmh` | Tốc độ (km/h) |
| `longitudinal_accel` | Gia tốc dọc (m/s²) — dùng cho phanh/tăng tốc gấp |
| `lateral_accel` | Gia tốc ngang (m/s²) — dùng cho vào cua gắt |
| `location {x,y,z}` | Toạ độ thế giới CARLA (m) |
| `rotation {yaw,pitch,roll}` | Góc quay (độ) |
| `geolocation {lat,lon,alt}` | Toạ độ GPS mô phỏng |

**Nhóm `targets[]`** (danh sách vật thể xung quanh mà xe cảm nhận được):

| Trường | Ý nghĩa |
|---|---|
| `target_id` | ID vật thể, **ổn định theo thời gian** → dùng để tracking |
| `target_class` | `vehicle` (ô tô) · `walker` (người đi bộ) · `bike` (xe máy trong tình huống cut-in) |
| `rel_pos {x,y}` | Vị trí tương đối so với ego (m): x = dọc/phía trước, y = ngang |
| `rel_velocity {x,y}` | Vận tốc tương đối (m/s) |
| `longitudinal_distance` / `lateral_distance` | Khoảng cách dọc / ngang (= rel_pos.x / rel_pos.y) |
| `closing_speed` | Tốc độ tiếp cận theo phương dọc (>0 = đang lại gần) |
| `ttc_simple` | TTC ước lượng theo phương dọc (`Infinity` nếu không lại gần) |
| `ttc_2d` | TTC xét quỹ đạo 2D (`Infinity` nếu quỹ đạo không cắt nhau) |
| `in_collision_cone` | `true` nếu vật nằm trong nón va chạm phía trước — chỉ những vật này mới tính vào `min_ttc` |

**Nhóm `driver`** (ground truth cho C2): 6 trường `state`, `alertness_score`, `eye_state`, `head_pose`, `mouth_state`, `nthu_subject_id` — xem [Mục 2.2](#22-c2--driverstate-trạng-thái-tài-xế-5-lớp).

**Nhóm `behavior_flags`** — 5 cờ boolean cho mỗi frame: `harsh_brake`, `harsh_accel`, `harsh_corner`, `speeding`, `tailgating` (điều kiện ở [Mục 2.3](#23-c3--safe_driving_score-điểm-lái-xe-an-toàn-0100)).

**Nhóm `risk`** — điểm rủi ro mỗi frame:

| Trường | Ý nghĩa |
|---|---|
| `base_risk` | Rủi ro tình huống, 0–100 (từ TTC, headway, event…) |
| `driver_factor` | Hệ số nhân theo trạng thái tài xế (ví dụ `distracted` → 2.2; `alert` → 1.0) |
| `final_risk_score` | `= min(100, base_risk × driver_factor)` |

**Các trường mức frame khác:**

| Trường | Ý nghĩa |
|---|---|
| `frame_id`, `timestamp`, `world_frame` | Chỉ số frame, thời gian (giây), số frame nội bộ của CARLA (offset ngẫu nhiên — **không** phải tín hiệu) |
| `min_ttc` | TTC nhỏ nhất, chỉ xét vật trong nón va chạm — xem [Mục 2.1](#21-c1--min_ttc-time-to-collision) |
| `headway_sec` | Khoảng cách thời gian tới xe phía trước (không có thành phần closing-speed như TTC) |
| `events_active[]` | Các sự kiện đang diễn ra ở frame này: `event_id`, `event_type`, `age_sec` (số giây kể từ khi kích hoạt), `actor_ids` |

### 3.3. `events_log[]` — danh sách sự kiện kịch bản (4 loại)

Mỗi phần tử: `{ t: <giây kích hoạt>, type: <1 trong 4>, params: {...} }`.

| `type` | Nghĩa | Tham số (`params`) |
|---|---|---|
| `pedestrian_jaywalk` | Người băng qua đường ẩu | `side`, `walk_speed_mps`, `distance_ahead_m`, `crossing_distance_m` |
| `motorcycle_cut_in` | Xe máy tạt đầu | `side`, `lateral_speed_mps`, `gap_m`, `target_speed_kmh` |
| `lead_brake` | Xe trước phanh gấp | `target_deceleration_g`, `duration_sec`, `detection_radius_m` |
| `stopped_vehicle_ahead` | Xe đỗ chắn phía trước | `vehicle_bp`, `distance_ahead_m` |

> Thời gian sống của sự kiện khác nhau theo loại: `lead_brake` chỉ ~2.5s (~50 frame); `motorcycle_cut_in` ~15s (~300 frame); `pedestrian_jaywalk` và `stopped_vehicle_ahead` kéo dài đến hết trip. **Đừng** giả định mọi sự kiện đều kéo dài đến frame 599.

### 3.4. `driver_summary` — tổng kết tài xế cả trip (ground truth cấp-trip, C2)

| Trường | Ý nghĩa |
|---|---|
| `subject_id` | ID người trong DMD |
| `condition_subset` | Điều kiện ghi hình: `Sunny` · `Cloudy` · `Rainy` |
| `state_distribution_pct` | % thời lượng mỗi trạng thái, ví dụ `{distracted: 50, alert: 50}` |
| `longest_drowsy_episode_sec` | Đợt buồn ngủ dài nhất (giây) |
| `microsleep_count` | Số lần ngủ gật |
| `average_alertness_score` | Điểm tỉnh táo trung bình (0–1) |
| `fatigue_score` | Điểm mệt mỏi (0–100) |

### 3.5. `trip_aggregate` — chỉ số an toàn cả trip (ground truth cho C3)

| Trường | Ý nghĩa |
|---|---|
| `safe_driving_score` | Điểm an toàn 0–100 (xem [Mục 2.3](#23-c3--safe_driving_score-điểm-lái-xe-an-toàn-0100)) |
| `harsh_brake_count` / `harsh_accel_count` / `harsh_corner_count` | Số frame có cờ phanh/tăng tốc/vào cua gấp |
| `near_miss_count` | Số frame có `min_ttc < 1.5s` |
| `speeding_pct_time` / `tailgating_pct_time` | % thời gian quá tốc độ / bám đuôi quá gần |
| `avg_headway_sec` | Khoảng cách thời gian trung bình tới xe trước |
| `max_risk_score` / `avg_risk_score` | Rủi ro lớn nhất / trung bình |
| `risk_classification` | Phân loại rủi ro: `low` · `moderate` · … (theo `avg_risk_score`) |

---

## 4. Nhãn KITTI (thư mục `kitti/`)

### 4.1. `label_2/{frame_id}.txt` — nhãn vật thể 3D

Mỗi dòng có đúng **15 trường** theo chuẩn KITTI:

```
type  truncated  occluded  alpha  bbox_left bbox_top bbox_right bbox_bottom  height width length  x y z  rotation_y
```

Ví dụ thật (T01, frame 323):
```
Pedestrian 0.00 0 0.00 0.00 0.00 0.00 0.00 1.70 0.60 0.60 -2.42 1.50 10.18 0.00
```

⚠️ **Những điều cần biết về nhãn này:**
1. **Chỉ 3 nhóm trường có giá trị thật:** `type`, kích thước 3D (`height width length`, mét) và vị trí 3D (`x y z` trong hệ camera trái: x-phải, y-xuống, z-tiến, mét). Tất cả trường còn lại (`truncated`, `occluded`, `alpha`, **cả 4 giá trị bbox 2D**, `rotation_y`) **luôn = 0**. Muốn có bbox 2D phải tự chiếu 3D→2D qua ma trận P2.
2. **Đa số file RỖNG (0 byte)** — chỉ frame có actor của sự kiện (hoặc thi thoảng 1 xe dẫn đầu) mới có dòng nhãn. Đây là bình thường.
3. Nhãn có thể xuất hiện **TRƯỚC khi event kích hoạt** → **đừng** dùng sự hiện diện của nhãn làm tín hiệu sự kiện.
4. Ánh xạ lớp: `Pedestrian` ↔ `walker`, `Car` ↔ `vehicle`, `Cyclist` ↔ `bike` (xe máy).

### 4.2. `calib/` và `calibration_info.txt` — thông số camera

- `calibration_info.txt` (chung cả trip): `fov_deg=90`, `baseline_m=0.3`, ảnh 640×360, `fx=fy=320`, `cx=320`, `cy=180`.
- Công thức stereo: **`depth = fx × baseline / disparity = 96 / disparity`**.
- `calib/{frame}.txt` (mỗi frame): định dạng KITTI chuẩn `P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo` — nội dung mọi frame giống hệt nhau (chỉ để tương thích code KITTI có sẵn).

### 4.3. `depth/{frame_id}.npy` — ground truth độ sâu

- Chỉ có ở frame chia hết cho 5 (`i % 5 == 0`) → 120 file/trip (tần suất 4 Hz).
- Mỗi file: mảng NumPy `float32`, kích thước `(360, 640)` (cao × rộng), đơn vị **mét**, khớp pixel với ảnh camera trái (`image_2`).
- Code phải chấp nhận **không có depth** ở các frame còn lại.

---

## 5. Bảng tra cứu 6 trip mẫu

Số liệu phân bố label đếm trực tiếp từ dữ liệu (mỗi trip = 600 frame).

| Trip | Kịch bản | Map | Thời tiết | Giới hạn | Sự kiện (giây) | Trạng thái tài xế (phân bố thực tế) |
|---|---|---|---|---|---|---|
| **T01** | Đô thị, ban ngày | Town10HD | Nắng | 40 | `pedestrian_jaywalk` @15s | `distracted` 0–15s → `alert` 15–30s (mỗi bên 300 frame) |
| **T02** | Cao tốc, chạng vạng | Town04 | Sương mù nhẹ | 80 | `motorcycle_cut_in` @15s | `drowsy` 100% (600 frame) |
| **T03** | Cao tốc, đêm, mưa to | Town06 | Mưa 80, đêm | 70 | `lead_brake` @15s | `yawning` 100% (600 frame) |
| **T04** | Đường hỗn hợp, ban ngày | Town05 | Nắng, mây 50 | 40 | `stopped_vehicle_ahead` @10s | `distracted`/`alert` 50/50 (mỗi bên 300 frame) |
| **T05** | Nông thôn, đơn điệu | Town07 | Nắng đẹp | 60 | `pedestrian_jaywalk` @10s **+** `lead_brake` @22s | `microsleep` 100% (600 frame) — fatigue 95 |
| **T06** | Ngã tư, trưa, mưa vừa | Town06 | Mưa 50, ướt 50 | 45 | `motorcycle_cut_in` @6s **+** `stopped_vehicle_ahead` @19s | `drowsy`/`distracted` 50/50 — rủi ro `moderate` |

- **T05, T06** là 2 trip có **2 sự kiện**; các trip khác chỉ 1 sự kiện.
- 6 trip mẫu cộng lại phủ **đủ 5 trạng thái tài xế** và **đủ 4 loại sự kiện**.

---

## 6. Những lưu ý quan trọng khi sử dụng

1. **JSON có `Infinity` trần** → không phải JSON chuẩn; JavaScript `JSON.parse` sẽ lỗi, Python `json.load` thì đọc được.
2. **Trạng thái tài xế và sự kiện CARLA độc lập có chủ đích** — đừng học tương quan kiểu "tài xế buồn ngủ → sắp có xe tạt đầu".
3. **10 trip chấm điểm thật (`T01d`..`T10d`, dài ~90s/1800 frame) bị XOÁ ground truth**, chỉ 6 trip `-Sample` có nhãn đầy đủ. Trên trip chấm điểm, những phần sau bị xoá/che:
   - Toàn bộ nhóm `driver` (state, alertness, eye/head/mouth, subject_id)
   - Toàn bộ trường con của `targets[]` (rel_pos, closing_speed, ttc_simple, ttc_2d, in_collision_cone…)
   - `min_ttc`, `headway_sec`, `behavior_flags`, `risk`
   - `trip_aggregate` và `driver_summary`
   - `ego.location / rotation / geolocation` (giữ lại speed và accel)
   - `events_log[].params` (giữ lại `type` và `t`)
   - Trường `x y z` trong nhãn `label_2` bị đưa về 0
4. **Định dạng nộp bài** — file CSV tên `<trip_id>.csv`, các cột: `frame_id, timestamp, predicted_ttc, predicted_driver_state, predicted_risk_score`. Chỉ `frame_id` là bắt buộc; thiếu cột nào thì coi như không làm challenge đó (không bị phạt). Cột `ground_truth_ttc` (nếu tự ghi) bị **bỏ qua** — bộ chấm luôn tự nạp ground truth từ thư mục trip.

---

## Nguồn tham chiếu

- Enum & ngưỡng: `team_kit/evaluation.py:116-134`
- Schema mỗi frame: `team_kit/dataset_loader.py:88-121` (`FrameRecord`)
- Đặc tả chi tiết: `docs/Practice_Dataset_Architecture.md`
- Đính chính & kiểm toán: `docs/TeamKit_Audit_Report.md`
