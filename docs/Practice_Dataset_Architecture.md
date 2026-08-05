# Practice_Dataset — Mô tả kiến trúc & định dạng dữ liệu

> Tài liệu này mô tả **đầy đủ và tự chứa** cấu trúc của thư mục `Practice_Dataset/`
> (6 trip luyện tập của hackathon FPT Automotive 2026). Mọi con số, tên trường và
> ngữ nghĩa dưới đây đều được kiểm chứng trực tiếp từ dữ liệu thật.

---

## 1. Tổng quan

Dataset mô phỏng bài toán **an toàn lái xe đa phương thức**: mỗi mẫu là một
"**trip**" (chuyến đi) 30 giây, ghi đồng thời:

| Nguồn dữ liệu | Nội dung | Xuất xứ |
|---|---|---|
| Camera đường (stereo trái/phải) | Ảnh RGB nhìn ra phía trước xe | Mô phỏng **CARLA 0.9.15** (render 1280×720, resize còn 640×360) |
| Camera cabin (driver) | Ảnh người lái thật trong xe | Footage thật từ **DMD (Driver Monitoring Dataset)** — các field/tên thư mục vẫn ghi "nthu" vì lý do lịch sử, KHÔNG phải dataset NTHU-DDD |
| Telemetry + Ground truth | JSON: động học ego, targets + TTC, trạng thái tài xế, risk score, sự kiện | Sinh từ engine mô phỏng |
| Nhãn KITTI | calib, label 3D, depth ground truth | Xuất theo chuẩn thư mục KITTI |

**Thông số chung (giống nhau cho cả 6 trip):**
- Thời lượng: **30 giây** @ **20 FPS** → **600 frame/trip** (frame_id 0–599, `timestamp = frame_id / 20`)
- Ảnh: **640×360 JPG** (cả camera đường lẫn camera driver)
- Stereo baseline: **0.3 m**, FOV **90°** → fx = fy = 320, cx = 320, cy = 180
- Driver profile: `normal`; mỗi trip 1 random seed riêng (2001–2006)

**Bối cảnh hackathon:** bộ dữ liệu đầy đủ có 16 trip. `Practice_Dataset/` chứa
**6 trip luyện tập** (`T01-Sample` … `T06-Sample`) có **đầy đủ ground truth**.
10 trip chấm điểm còn lại (`T01d` … `T10d`, ~90s/1800 frame, KHÔNG có trong thư mục
này) đã bị xóa các field ground truth (driver state, TTC, risk, trip_aggregate,
driver_summary, vị trí ego, params của event, `location` trong label KITTI…).
Gộp lại, 6 trip mẫu phủ đủ **cả 5 driver state** và **cả 4 loại event** của dataset.

Dataset phục vụ 3 challenge:
1. **Collision Risk Monitor** — dự đoán TTC theo từng frame từ ảnh camera.
2. **Driver Intelligence** — phân loại driver state (5 lớp) + hồi quy alertness (0–1).
3. **Fleet Safe Driving Score** — tổng hợp điểm an toàn 0–100 cấp trip.

---

## 2. Cây thư mục

```
Practice_Dataset/                     (~908 MB, 6 trip)
├── T01-Sample/
│   ├── driver/                       600 ảnh cabin: frame_000000.jpg … frame_000599.jpg (640×360)
│   ├── kitti/
│   │   ├── image_2/                  600 ảnh RGB camera TRÁI:  000000.jpg … 000599.jpg (640×360)
│   │   ├── image_3/                  600 ảnh RGB camera PHẢI:  000000.jpg … 000599.jpg (640×360)
│   │   ├── depth/                    120 file .npy — CHỈ keyframe mỗi 5 frame: 000000, 000005, …, 000595
│   │   ├── calib/                    600 file calib KITTI theo frame: 000000.txt … 000599.txt
│   │   ├── label_2/                  600 file nhãn KITTI theo frame (nhiều file RỖNG — xem §5.4)
│   │   └── calibration_info.txt      1 file JSON: intrinsics + baseline (chung cho cả trip)
│   ├── T01-Sample.json.gz            Telemetry + ground truth đầy đủ (nén gzip)
│   └── T01-Sample.json               (chỉ riêng T01 có sẵn bản giải nén; nội dung y hệt file .gz)
├── T02-Sample/    … cấu trúc y hệt (chỉ có .json.gz, không có bản .json giải nén)
├── T03-Sample/    …
├── T04-Sample/    …
├── T05-Sample/    …
└── T06-Sample/    …
```

**Quy ước đồng bộ:** mọi modality khớp nhau qua **frame_id**.
Frame thứ `i` ⇔ `driver/frame_{i:06d}.jpg` ⇔ `kitti/image_2/{i:06d}.jpg` ⇔
`kitti/image_3/{i:06d}.jpg` ⇔ `kitti/calib/{i:06d}.txt` ⇔ `kitti/label_2/{i:06d}.txt`
⇔ phần tử `frames[i]` trong JSON. Depth chỉ có khi `i % 5 == 0`.

---

## 3. Sáu trip: kịch bản & khác biệt

| Trip | Kịch bản | Map CARLA | Thời tiết | Giới hạn tốc độ | Event (thời điểm t giây) | Driver state (subject DMD, điều kiện) |
|---|---|---|---|---|---|---|
| **T01** | Đô thị ban ngày | Town10HD | Nắng nhẹ (cloud 20) | 40 km/h | `pedestrian_jaywalk` @ t=15 | `distracted` (0–15s) → `alert` (15–30s) — subject 14, Sunny |
| **T02** | Cao tốc chiều tối | Town04 | Sương mù nhẹ, mặt trời lặn (sun_alt −15°) | 80 km/h | `motorcycle_cut_in` @ t=15 | `drowsy` 100% — subject 10, Cloudy |
| **T03** | Cao tốc ban đêm, mưa lớn | Town06 | Mưa 80, ướt 80, đêm (sun_alt −90°) | 70 km/h | `lead_brake` @ t=15 | `yawning` 100% — subject 1, Rainy |
| **T04** | Đường hỗn hợp ban ngày | Town05 | Nắng, mây 50 | 40 km/h | `stopped_vehicle_ahead` @ t=10 | `distracted`/`alert` 50/50 — subject 6, Sunny |
| **T05** | Nông thôn đơn điệu | Town07 | Nắng đẹp | 60 km/h | `pedestrian_jaywalk` @ t=10 **+** `lead_brake` @ t=22 | `microsleep` 100% (fatigue 95) — subject 36, Sunny |
| **T06** | Giao lộ giữa trưa, mưa vừa | Town06 | Mưa 50, ướt 50 | 45 km/h | `motorcycle_cut_in` @ t=6 **+** `stopped_vehicle_ahead` @ t=19 | `drowsy`/`distracted` 50/50 — subject 23, Cloudy |

- Random seed: T01=2001 … T06=2006. `driver_profile` đều là `normal`.
- T05, T06 là 2 trip **đa event** (2 event/trip); các trip còn lại 1 event.
- Đúng dụng ý thiết kế: **driver state và event CARLA độc lập nhau** — không nên
  học tương quan kiểu "tài xế buồn ngủ thì sắp có xe cắt ngang".

**4 loại event và tham số (`events_log[].params`):**

| `type` | Tham số | Ví dụ thực tế |
|---|---|---|
| `pedestrian_jaywalk` | `side`, `walk_speed_mps`, `distance_ahead_m`, `crossing_distance_m` | T01: left, 1.5 m/s, 19 m, 10 m |
| `motorcycle_cut_in` | `side`, `lateral_speed_mps`, `gap_m`, `target_speed_kmh` | T02: right, 2.5 m/s, 8 m, 70 km/h |
| `lead_brake` | `target_deceleration_g`, `duration_sec`, `detection_radius_m` | T03: 0.4 g, 2.5 s, 40 m |
| `stopped_vehicle_ahead` | `vehicle_bp`, `distance_ahead_m` | T04: vehicle.audi.a2, 30 m |

---

## 4. File JSON telemetry (`T0X-Sample.json.gz`)

File gzip chứa 1 object JSON duy nhất, ~2–4 MB sau giải nén. **6 khóa cấp cao nhất:**

```
{ "trip_id", "metadata", "driver_summary", "trip_aggregate", "events_log", "frames" }
```

> ⚠️ **Cảnh báo parse:** file chứa token `Infinity` trần (không phải chuỗi) cho các
> giá trị TTC/headway vô hạn — đây **không phải JSON chuẩn** (RFC 8259).
> `json.load()` của Python đọc được mặc định; `JSON.parse()` của JavaScript và
> nhiều parser strict sẽ **lỗi**. Với JS: thay `Infinity` trước khi parse hoặc
> dùng parser hỗ trợ JSON5.

### 4.1 `trip_id` — string
`"T01-Sample"`, … trùng tên thư mục.

### 4.2 `metadata` — cấu hình mô phỏng

```jsonc
{
  "trip_id": "T01-Sample",
  "description": "Urban daytime, pedestrian jaywalk only",
  "duration_sec": 30,
  "fps": 20,
  "map": "Town10HD",              // map CARLA
  "weather": {                     // đúng bộ tham số WeatherParameters của CARLA
    "cloudiness": 20.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
    "sun_altitude_angle": 45.0, "sun_azimuth_angle": 90.0,
    "fog_density": 0.0, "fog_distance": 100.0,
    "wind_intensity": 5.0, "wetness": 0.0
  },
  "driver_profile": "normal",
  "carla_version": "0.9.15",
  "random_seed": 2001,
  "speed_limit_kmh": 40            // dùng để tính cờ speeding
}
```

### 4.3 `driver_summary` — tóm tắt trạng thái tài xế cả trip

```jsonc
{
  "subject_id": "14",                    // ID subject trong DMD (tên field lịch sử ghi "nthu")
  "condition_subset": "Sunny",           // Sunny | Cloudy | Rainy — điều kiện quay footage driver
  "state_distribution_pct": {"distracted": 50.0, "alert": 50.0},  // % thời lượng từng state
  "longest_drowsy_episode_sec": 0.0,
  "microsleep_count": 0,
  "average_alertness_score": 0.7,        // trung bình alertness (0–1)
  "fatigue_score": 30.0                  // 0–100
}
```

### 4.4 `trip_aggregate` — chỉ số an toàn cấp trip (đáp án Challenge 3)

```jsonc
{
  "safe_driving_score": 0,        // 0–100, xem công thức bên dưới
  "harsh_brake_count": 14,        // số frame có cờ harsh_brake
  "harsh_accel_count": 53,
  "harsh_corner_count": 2,
  "near_miss_count": 7,           // = số frame có min_ttc < 1.5 s (đã kiểm chứng)
  "speeding_pct_time": 0.0,       // % thời gian vượt speed_limit_kmh
  "tailgating_pct_time": 0.0,     // % thời gian bám đuôi quá gần
  "avg_headway_sec": 1.46,        // trung bình headway trên các frame có lead
  "max_risk_score": 60.0,
  "avg_risk_score": 2.3,
  "risk_classification": "low"    // low | moderate | ... theo avg_risk_score
}
```

**Công thức Safe Driving Score** (rule-based, đã kiểm chứng khớp dữ liệu):

```
safe = max(0, 100 − (harsh_brake×3.0 + harsh_accel×2.0 + harsh_corner×2.0
                     + near_miss×5.0 + speeding_pct×0.15 + tailgating_pct×0.10))
```

Lưu ý: cả 6 trip mẫu đều có mức phạt vượt 100 nên `safe_driving_score = 0` ở
tất cả (ví dụ T01: 100 − 187 = −87 → clamp 0). Đây là hành vi đúng, không phải lỗi.

### 4.5 `events_log` — danh sách event đã kích hoạt

Mảng (1–2 phần tử/trip):

```jsonc
[{ "t": 15.0,                       // giây kích hoạt
   "type": "pedestrian_jaywalk",    // 1 trong 4 loại ở §3
   "params": { ... } }]             // tham số theo loại (bảng §3)
```

### 4.6 `frames` — mảng 600 bản ghi per-frame (phần lớn dung lượng)

Mỗi phần tử có **cấu trúc cố định** (mọi khóa luôn hiện diện, cả 6 trip):

```jsonc
{
  "frame_id": 300,          // 0–599, khớp tên file ảnh/label
  "world_frame": 96563,     // số frame nội bộ của CARLA server
  "timestamp": 15.0,        // giây, = frame_id / 20

  "ego": {                  // trạng thái xe của mình
    "speed_kmh": 29.04,
    "longitudinal_accel": 0.063,   // m/s², dọc trục xe
    "lateral_accel": -0.0,         // m/s², ngang
    "location": {"x": 59.028, "y": 137.681, "z": 0.002},   // tọa độ world CARLA (m)
    "rotation": {"yaw": 0.32, "pitch": 0.0, "roll": -0.0}, // độ
    "geolocation": {"lat": -0.001237, "lon": 0.00053, "alt": 0.0}  // GNSS giả lập
  },

  "targets": [              // các actor xung quanh mà hệ thống "cảm nhận" được
    {
      "target_id": 296,               // actor ID CARLA, ổn định theo thời gian → dùng để tracking
      "target_class": "vehicle",      // vehicle | walker | bike (bike = mô tô trong event cut-in)
      "rel_pos": {"x": 32.496, "y": -22.591},        // m, hệ quy chiếu ego: x = dọc (trước), y = ngang
      "rel_velocity": {"x": -12.243, "y": 6.557},    // m/s, vận tốc tương đối
      "longitudinal_distance": 32.496,   // = rel_pos.x
      "lateral_distance": -22.591,       // = rel_pos.y
      "closing_speed": 12.243,           // m/s, tốc độ tiếp cận theo phương dọc (>0 = đang lại gần)
      "ttc_simple": 2.254,               // s, TTC ước lượng theo phương dọc (Infinity nếu không tiếp cận)
      "ttc_2d": 3.064,                   // s, TTC xét quỹ đạo 2D (Infinity nếu quỹ đạo không giao)
      "in_collision_cone": false         // true nếu target nằm trong "nón va chạm" phía trước ego
    }
  ],

  "driver": {               // ground truth trạng thái tài xế (đáp án Challenge 2)
    "state": "alert",       // alert | distracted | drowsy | yawning | microsleep
    "alertness_score": 0.95,// 0–1, CỐ ĐỊNH theo state (bảng bên dưới)
    "eye_state": "open",    // open | partial | closed
    "head_pose": "normal",  // normal | side | down
    "mouth_state": "normal",// normal | yawning (giá trị "talking" tồn tại trong spec nhưng không xuất hiện ở 6 mẫu)
    "nthu_subject_id": "14" // subject DMD (tên field lịch sử)
  },

  "events_active": [        // event kịch bản đang hiệu lực tại frame này
    { "event_id": 0, "event_type": "pedestrian_jaywalk",
      "age_sec": 0.0,       // số giây kể từ lúc event kích hoạt
      "actor_ids": [383] }  // actor ID của đối tượng do event sinh ra
  ],

  "min_ttc": Infinity,      // s — TTC nhỏ nhất CHỈ TÍNH các target có in_collision_cone=true
  "headway_sec": Infinity,  // s — longitudinal_distance của lead trong cone / tốc độ ego

  "behavior_flags": {       // cờ hành vi tại frame (đếm tổng khớp trip_aggregate)
    "harsh_brake": false, "harsh_accel": false, "harsh_corner": false,
    "speeding": false, "tailgating": false
  },

  "risk": {                 // điểm rủi ro tại frame
    "base_risk": 0.0,       // 0–100, rủi ro tình huống (TTC, headway, event…)
    "driver_factor": 1.0,   // hệ số nhân theo trạng thái tài xế (ví dụ distracted → 2.2)
    "final_risk_score": 0.0 // = min(100, base_risk × driver_factor) — đã kiểm chứng
  }
}
```

**Các ngữ nghĩa quan trọng (đều kiểm chứng bằng dữ liệu):**

- `min_ttc` **không phải** min của mọi `ttc_simple` trong `targets` — chỉ tính các
  target `in_collision_cone == true`. Không có target nào trong cone ⇒ `Infinity`.
- `final_risk_score = base_risk × driver_factor`, chặn trần 100.
- `near_miss_count` (trip_aggregate) = số frame có `min_ttc < 1.5`.
- `events_active` xuất hiện từ frame kích hoạt (ví dụ t=15 → frame 300), liên tục
  với `age_sec` tăng dần, nhưng **thời gian sống phụ thuộc loại event**:
  `lead_brake` chỉ hiệu lực đúng `duration_sec` (~2.5 s ≈ 50 frame),
  `motorcycle_cut_in` trong cửa sổ ~15 s (~300 frame), còn `pedestrian_jaywalk`
  và `stopped_vehicle_ahead` tồn tại đến hết trip. Đừng giả định mọi event còn
  active tới frame 599.
- Driver state đổi **theo phân đoạn thời gian dài** (T01: distracted suốt 0–15s rồi
  alert 15–30s), không nhấp nháy từng frame.
- `alertness_score` là hàm cố định của state: `alert`=0.95, `yawning`=0.55,
  `distracted`=0.45, `drowsy`=0.35, `microsleep`=0.05.
- Số target/frame thay đổi (0–~15); danh sách chỉ gồm actor trong phạm vi cảm biến.

### 4.7 Bảng driver state ↔ biểu hiện khuôn mặt (quan sát từ 6 trip)

| `state` | `alertness` | `eye_state` | `head_pose` | `mouth_state` |
|---|---|---|---|---|
| `alert` | 0.95 | open | normal | normal |
| `distracted` | 0.45 | open | side | normal |
| `drowsy` | 0.35 | partial | down | normal |
| `yawning` | 0.55 | partial | normal | yawning |
| `microsleep` | 0.05 | closed | down | normal |

---

## 5. Thư mục `kitti/` — dữ liệu thị giác

### 5.1 `image_2/` & `image_3/` — cặp stereo

- `image_2` = camera **trái** (ảnh chính, trùng hệ quy chiếu label), `image_3` = camera **phải**.
- 640×360 JPG, 600 ảnh/thư mục, tên `{frame_id:06d}.jpg`.
- Render gốc CARLA 1280×720 rồi resize 640×360; calib đã đồng bộ theo độ phân giải này.

### 5.2 `calibration_info.txt` — calib toàn cục (JSON)

```jsonc
{
  "fov_deg": 90, "baseline_m": 0.3,
  "image_width": 640, "image_height": 360,
  "K_left":  [[320, 0, 320], [0, 320, 180], [0, 0, 1]],
  "P2_left": [[320, 0, 320, 0], [0, 320, 180, 0], [0, 0, 1, 0]],
  "P3_right":[[320, 0, 320, -96], [0, 320, 180, 0], [0, 0, 1, 0]]  // -96 = -fx × baseline
}
```

Giống hệt nhau ở cả 6 trip (đã so md5). Công thức stereo: `depth = fx × baseline / disparity = 96 / disparity`.

### 5.3 `calib/` — calib KITTI theo từng frame

600 file `{frame_id:06d}.txt`, đúng format calib của KITTI object detection:

```
P0: 320 0 320 0   0 320 180 0   0 0 1 0        # 3×4, ghi phẳng 12 số
P1: (như P0 nhưng phần tử [0][3] = -96)
P2: (camera trái — trùng P0)
P3: (camera phải — trùng P1)
R0_rect: ma trận đơn vị 3×3
Tr_velo_to_cam: [I | 0] 3×4
Tr_imu_to_velo: [I | 0] 3×4
```

Mọi file giống hệt nhau (không có LiDAR/IMU thật nên các ma trận Tr là đơn vị);
tồn tại theo từng frame chỉ để **tương thích code KITTI 3D detection có sẵn**.

### 5.4 `label_2/` — nhãn object KITTI (⚠️ nhiều điểm đặc thù)

600 file `{frame_id:06d}.txt`. Mỗi dòng đúng 15 trường theo thứ tự KITTI:

```
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
height width length x y z rotation_y
```

Ví dụ thật (T01, frame 323):
```
Pedestrian 0.00 0 0.00 0.00 0.00 0.00 0.00 1.70 0.60 0.60 -2.42 1.50 10.18 0.00
```

**Đặc thù quan trọng:**

1. **Chỉ gán nhãn rất chọn lọc, KHÔNG phải mọi phương tiện trong cảnh:** gồm actor
   của event kịch bản (người đi bộ băng đường, mô tô cắt ngang, xe phanh gấp, xe
   đỗ chắn đường) **cộng thêm**, ở T02/T03/T04/T06, 1 xe giao thông thường phía
   trước (lead car trong collision cone) không thuộc event nào. Vì vậy nhãn có thể
   xuất hiện **từ frame 0, trước cả khi event kích hoạt** (T04, T06) — đừng dùng
   sự xuất hiện của label làm tín hiệu event. Tối đa 2 dòng/file — xảy ra khi 2
   actor được gán nhãn cùng lúc trong tầm nhìn (thấy ở T02, T06; riêng T05 dù có
   2 event nhưng 2 actor không bao giờ cùng trong tầm nên tối đa 1 dòng/file).
2. **Phần lớn file RỖNG** (0 byte) — chỉ frame nào actor của event nằm trong tầm
   phát hiện mới có dòng nhãn. Số file có nhãn: T01=57, T02=~190, T03=~81,
   T04=490, T05=~126, T06=600.
3. **Chỉ 3 nhóm trường mang giá trị thật:** `type`, kích thước 3D
   (`height width length`, mét) và vị trí 3D (`x y z` — hệ tọa độ camera trái:
   x phải, y xuống, z sâu về phía trước, mét). Các trường còn lại
   (`truncated`, `occluded`, `alpha`, **bbox 2D**, `rotation_y`) đều bằng **0** ở
   toàn bộ 6 trip mẫu — muốn có bbox 2D phải tự chiếu 3D→2D qua P2 hoặc tự chạy detector.
4. Lớp xuất hiện: `Pedestrian` (walker), `Car` (vehicle), `Cyclist` (mô tô của
   event cut-in — trong JSON `target_class` của nó là `bike`).
5. Ở 10 trip chấm điểm (không thuộc thư mục này), trường `x y z` bị zero-out —
   chỉ 6 trip mẫu có vị trí 3D thật.

### 5.5 `depth/` — depth ground truth (keyframe)

- 120 file `.npy`/trip, **chỉ tại frame chia hết cho 5** (000000, 000005, …, 000595) = 4 Hz.
- Mỗi file: mảng NumPy `float32`, shape `(360, 640)` (H×W), đơn vị **mét**,
  khớp pixel với `image_2` (camera trái).
- Giá trị từ ~1 m (min quan sát được: 0.93 m ở T03) đến ~1000 m (bầu trời/xa vô
  cực bị chặn ở ~1000).
- Đây là depth dày đặc (dense) từ sensor depth của CARLA — dùng làm ground truth
  cho stereo/monocular depth.

---

## 6. Thư mục `driver/` — camera cabin

- 600 ảnh JPG 640×360/trip, tên `frame_{frame_id:06d}.jpg` (lưu ý có tiền tố
  `frame_`, khác với ảnh kitti).
- Là **footage người thật** ghép từ DMD (Driver Monitoring Dataset), đồng bộ 1-1
  với frame mô phỏng: ảnh tại frame `i` thể hiện đúng `frames[i].driver.state`
  trong JSON (ví dụ T05 frame 300: mắt nhắm — microsleep).
- Mỗi trip dùng 1 subject cố định (`nthu_subject_id`) và 1 điều kiện ánh sáng
  (`condition_subset`: Sunny/Cloudy/Rainy).
- Đây là input cho Challenge 2 (phân loại driver state); nhãn nằm trong JSON.

---

## 7. Lưu ý khi sử dụng / parse

1. **`Infinity` trong JSON** — không phải JSON chuẩn; Python `json` đọc được,
   `JSON.parse` (JS) thì không (xem §4).
2. **Giải nén:** chỉ T01 có sẵn `.json`; các trip khác đọc thẳng `.json.gz`
   (Python: `gzip.open(path, 'rt')` rồi `json.load`).
3. **Depth chỉ có ở keyframe** (i % 5 == 0) — code phải chịu được `None` ở frame khác.
4. **Label rỗng** là bình thường (không có event actor trong tầm) — không phải lỗi dữ liệu.
5. **bbox 2D trong label luôn = 0** — đừng dùng trực tiếp làm nhãn detection 2D.
6. **`safe_driving_score = 0` ở cả 6 trip** do mức phạt vượt 100 (clamp) — hợp lệ.
7. **`target_id` ổn định theo thời gian** — dùng được cho tracking/temporal model.
8. **Tên "nthu" chỉ là di sản** — dữ liệu driver thật lấy từ DMD.
9. File `T0X_nthu_mapping.json` được nhắc trong tài liệu kit **không có mặt**
   trong 6 thư mục mẫu này.

---

## 8. Đoạn code đọc nhanh (tham khảo)

```python
import gzip, json, glob
import numpy as np
import cv2

trip = "Practice_Dataset/T03-Sample"
tid  = trip.split("/")[-1]

# 1) Telemetry + ground truth
with gzip.open(f"{trip}/{tid}.json.gz", "rt") as f:
    data = json.load(f)          # Python chấp nhận token Infinity

meta   = data["metadata"]        # fps=20, map, weather, speed_limit...
frames = data["frames"]          # 600 bản ghi

fr = frames[300]                 # frame tại t = 15s
print(fr["ego"]["speed_kmh"], fr["driver"]["state"], fr["min_ttc"])

# 2) Ảnh 3 camera cùng frame
i = fr["frame_id"]
left   = cv2.imread(f"{trip}/kitti/image_2/{i:06d}.jpg")
right  = cv2.imread(f"{trip}/kitti/image_3/{i:06d}.jpg")
driver = cv2.imread(f"{trip}/driver/frame_{i:06d}.jpg")

# 3) Depth ground truth (chỉ keyframe i % 5 == 0)
depth = np.load(f"{trip}/kitti/depth/{i - i % 5:06d}.npy")   # (360, 640) float32, mét

# 4) Nhãn KITTI của frame
for line in open(f"{trip}/kitti/label_2/{i:06d}.txt"):
    t, *v = line.split()         # type + 14 số (chỉ dims/location có giá trị thật)
    h, w, l, x, y, z = map(float, v[7:13])
    print(t, "cách", z, "m phía trước")
```

---

## 9. Tóm tắt số liệu mỗi trip

| Thành phần | Số lượng / trip | Định dạng |
|---|---|---|
| `driver/*.jpg` | 600 | JPG 640×360, footage DMD thật |
| `kitti/image_2/*.jpg` | 600 | JPG 640×360, camera trái CARLA |
| `kitti/image_3/*.jpg` | 600 | JPG 640×360, camera phải (baseline 0.3 m) |
| `kitti/depth/*.npy` | 120 | float32 (360, 640), mét, mỗi 5 frame |
| `kitti/calib/*.txt` | 600 | KITTI calib (giống hệt nhau) |
| `kitti/label_2/*.txt` | 600 | KITTI 15 trường (nhiều file rỗng) |
| `calibration_info.txt` | 1 | JSON intrinsics + baseline |
| `T0X-Sample.json.gz` | 1 | Telemetry + GT, ~40–210 KB nén |
| **Tổng frame JSON** | 600 | 30 s × 20 FPS |

Toàn bộ 6 trip: ~908 MB, 18.7 nghìn file.
