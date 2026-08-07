# Báo cáo Audit Team-Kit — đối chiếu với Practice_Dataset

> **Phạm vi:** kiểm tra toàn bộ starter kit (`dataset_loader.py`, `evaluation.py`,
> `baseline_ttc_predictor.py`, `README.md`, `HUONG_DAN_NGUOI_MOI.md`,
> `explore_trip.ipynb`) đối chiếu với 6 trip thật trong `Practice_Dataset/` và file
> `Practice_Dataset_Architecture.md`.
> **Phương pháp:** 5 nhánh audit song song, mỗi phát hiện được **chạy lại code thật**
> để xác minh, rồi **1 agent độc lập kiểm chứng đối kháng** (cố gắng bác bỏ).
> **Kết quả:** 32 phát hiện — 30 confirmed, 2 partially-correct (đã tích hợp đính chính).
> Không phát hiện nào bị bác bỏ.

---

## 0. TL;DR — 5 điều quan trọng nhất

1. **⚠️ Số liệu "output mẫu" của baseline trong README/HƯỚNG DẪN sai ~40 lần.** Tài liệu ghi
   composite 52.3, MAE-critical 1.420s, n_crit 62; chạy đúng lệnh tài liệu trên T01-Sample
   thật ra được **composite ~30.6, MAE-critical 58.6s, n_crit 12**. Đừng lấy "vượt 52.3"
   làm mốc — mốc thật là ~30. `n_crit` là thuộc tính thuần ground-truth (số frame min_ttc<3s)
   nên chắc chắn là 12, không phải 62 → số liệu README sinh từ bản dataset cũ.

2. **⚠️ Challenge 3 luôn = 100 trên cả 6 trip luyện tập — không có tín hiệu gì.** Mọi trip có
   `safe_driving_score = 0` (bị clamp), và riêng phần phạt harsh (tính từ ego kinematics) đã
   vượt 100. Nên C3 composite = 100 với **bất kỳ** bài nộp nào. Đừng tưởng "C3 = 100 nghĩa là
   đã xong". Thuật ngữ `near_miss` (đòn bẩy duy nhất bạn điều khiển được) **bất hoạt hoàn toàn**
   khi test local.

3. **💰 Đòn bẩy dễ nhất để vượt baseline: đừng bao giờ trả `inf` trong vùng nguy hiểm.** Baseline
   trả `inf` ở 7/12 frame critical → **mất trắng toàn bộ 40% điểm MAE-critical** (điểm = 0).
   ~88% điểm 30.6 của baseline đến từ term inv-TTC trên các frame dễ. Chỉ cần emit một TTC hữu
   hạn hợp lý ở 2 cửa sổ nguy hiểm là lấy lại được tới ~40 điểm composite.

4. **💰 Challenge 3 KHÔNG cần model, Challenge 2 alertness là tra bảng miễn phí.** C3 hoàn toàn
   xác định từ ego kinematics (giống nhau mọi team) + cột `predicted_ttc` của bạn. C2: `alertness`
   là hàm cố định của `state`, và bộ chấm **không hề chấm alertness**; bộ ba (eye, head, mouth)
   ánh xạ **song ánh** với state. Chỉ cần 1 classifier driver-state là ra tất cả.

5. **🐛 `evaluation.py` thưởng điểm cho việc XÓA dòng.** Không có kiểm tra coverage: xóa 12 frame
   critical khỏi CSV làm composite tăng +20.8 điểm, không cảnh báo. Nếu đây là bộ chấm thật cho
   10 trip T0Xd thì đây là lỗ hổng gian lận điểm — nhưng đừng dựa vào nó (ban tổ chức có thể vá).

---

## 1. Bảng tổng hợp 32 phát hiện

| # | Khu vực | Loại | Mức | Phát hiện |
|---|---|---|---|---|
| E0 | evaluation | bug | **cao** | Không kiểm tra coverage → xóa dòng được thưởng điểm (+20.8) |
| B0/D2 | baseline/docs | inconsistency | **cao** | Output mẫu baseline sai ~40x (composite 52.3→30.6, n_crit 62→12) |
| B2 | baseline | opportunity | **cao** | Baseline mất trắng 40% điểm do trả `inf` ở 7/12 frame critical |
| B3 | baseline | risk | **cao** | Fixed ROI + median bỏ sót pedestrian jaywalk (recall 0.100) |
| S0 | strategy | opportunity | **cao** | Challenge 3 deterministic — không cần model |
| S1 | strategy | opportunity | **cao** | Challenge 2 alertness = tra bảng; evaluator bỏ qua alertness; (eye,head,mouth)→state song ánh |
| S2 | strategy | opportunity | **cao** | Depth GT 4Hz là đòn bẩy lớn nhất cho Challenge 1 |
| S4 | strategy | risk | **cao** | C3 clamp về 0 trên mọi practice trip → composite local luôn 100, vô nghĩa |
| S5 | strategy | risk | **cao** | Regularities practice-set (event time, flip @15s, world_frame) không generalize |
| L0 | loader | bug | trung | `frames_df` sentinel min_ttc=99.0 thấp hơn giá trị finite thật (tới 389.6s) |
| L1 | loader | bug | trung | `headway_sec` cũng sentinel 99.0 nhưng KHÔNG có cột `*_inf` → avg sai 67x |
| L2 | loader | gap | trung | Không có label loader dù đã định nghĩa `label_dir` |
| E1 | evaluation | inconsistency | trung | Challenge 1 luôn được chấm dù thiếu cột `predicted_ttc` |
| E2 | evaluation | gap | trung | C3 = 100 cố định trên practice → không validate được gì |
| E3 | evaluation | bug | trung | `--trip-dir` bị bỏ qua; tên file CSV mới chọn ground-truth |
| B1 | baseline | gap | trung | `evaluation.py` chết với lỗi khó hiểu nếu CSV không đặt đúng tên `<trip>.csv` |
| B4 | baseline | bug | trung | Cold-start: baseline báo TTC < 2s trên cảnh trống trong ~0.2s đầu |
| D0 | docs | inconsistency | trung | README nói label_2 có bbox 2D nhưng mọi trường bbox/trunc/occ/alpha/rot_y đều = 0 |
| D1/E4/S6 | nhiều | bug | trung/thấp | Tái tạo harsh_corner off-by-one trên T03 — không "bit-for-bit" như tài liệu |
| S3 | strategy | opportunity | trung | label_2 3D + P2 tự sinh bbox 2D (tight cho pedestrian, cần chỉnh cho car) |
| L3 | loader | gap | thấp | `load_depth` trả None âm thầm cho frame_id sai/ngoài phạm vi |
| L4 | loader | risk | thấp | `_parse_frame` crash nếu GT sub-object có giá trị `null` |
| L5 | loader | gap | thấp | Mất `ego.location`/`world_frame`, không có accessor public |
| E5 | evaluation | bug | thấp | frame_id không tồn tại được chấp nhận & chấm với GT bịa |
| E6 | evaluation | inconsistency | thấp | Ví dụ output trong README không tái tạo được (n_crit 62) |
| D3 | docs | inconsistency | thấp | README ghi "tối thiểu cần pyyaml" (thừa), thiếu jupyter |
| D4 | docs | inconsistency | thấp | HƯỚNG DẪN có anchor link hỏng (`...3-cách` vs `...2-cách`) |
| E7 | evaluation | ✅ positive | — | Xác nhận: cột `ground_truth` bị bỏ qua, cell rác không crash |
| B5 | baseline | ✅ positive | — | Xác nhận: CSV baseline đúng format & an toàn |
| — | loader | ✅ positive | — | Core loader đúng 100%: 3600 frame khớp raw JSON, xử lý `Infinity` ổn |

---

## 2. Chiến lược thi đấu (validated) — làm gì để ăn điểm

### 💰 Challenge 1 (TTC) — ưu tiên cao nhất

**Đòn bẩy #1: đừng bao giờ trả `inf` khi đang tiến gần vật cản.** Composite =
`40% MAE-critical + 30% F1 + 30% inv-TTC-MAE`. Metric dùng `INF_CLIP_VALUE = 99.0`, nên một
`inf` sai ở frame critical tốn ~97–98s error. Chỉ cần MAE-critical > 5s là **mất sạch khối 40%**
(điểm term = 0). Baseline chính là chết vì điều này. Với team: luôn phát ra một TTC hữu hạn khi
có tín hiệu tiếp cận, kể cả detector thô cũng hơn `inf` ~97s/frame.

**Đòn bẩy #2: depth GT dày 4 Hz là tài nguyên giá trị nhất.** Mỗi trip có 120 keyframe depth
`(360,640) float32` mét, khớp pixel với `image_2`, sai số hình học chỉ vài cm (chiếu điểm 3D của
label lên depth khớp trong ~0.05m). Đây là **720 frame supervised** (6 trip) để distill/fine-tune
mô hình monocular depth (MiDaS/ZoeDepth) rồi đổi depth→TTC qua closing-rate liên frame.
⚠️ **Depth GT CHỈ có trên 6 practice trip, không có trên 10 scored trip** → dùng làm dữ liệu
train/validate, mô hình cuối phải chạy chỉ bằng ảnh.

**Đòn bẩy #3: tự sinh nhãn 2D từ label_2 3D.** Trường bbox 2D trong `label_2` toàn bằng 0, nhưng
`dimensions (h,w,l)` và `location (x,y,z)` là thật; chiếu 8 góc hộp qua
`P2 = [[320,0,320,0],[0,320,180,0],[0,0,1,0]]` sinh ra bbox 2D. **Lưu ý:** box **pedestrian tight,
dùng được ngay**; box **car bị rộng ~50%** (vì `rotation_y` bị zero-out nên chiều dài 4m thành bề
ngang) và depth label của xe đỗ có thể xa hơn mặt xe thật → cần chỉnh hướng + kiểm tra depth. Chỉ
event actor + đôi khi 1 xe dẫn đầu được gán nhãn (KHÔNG phải mọi xe), phần lớn file rỗng → đây là
công cụ train practice-only, không phải input được chấm.

### 💰 Challenge 2 (Driver state) — gần như miễn phí

- `alertness_score` là hàm cố định của `state` (alert=0.95, yawning=0.55, distracted=0.45,
  drowsy=0.35, microsleep=0.05). Bộ ba `(eye_state, head_pose, mouth_state)` ánh xạ **song ánh**
  với 5 state (không va chạm). → Chỉ cần **1 classifier driver-state** trên ảnh cabin là suy ra
  được tất cả bằng tra bảng.
- Bộ chấm **không hề chấm alertness** (schema chỉ có `predicted_driver_state`), dù README nhắc
  "hồi quy alertness". Đừng xây regressor riêng cho alertness.
- Composite = `50% accuracy + 50% macro-F1` chỉ trên các lớp **xuất hiện trong trip đó**. State
  đổi theo phân đoạn dài (không nhấp nháy) → **temporal smoothing mạnh** là đòn bẩy lớn để đạt
  gần tuyệt đối.

### 💰 Challenge 3 (Safe Driving Score) — KHÔNG cần ML

Toàn bộ điểm tái tạo từ: (a) đếm harsh_brake/accel/corner + speeding% tính từ ego kinematics
**không bị redact** (giống nhau mọi team); (b) `near_miss_count` = số frame có **`predicted_ttc`
của bạn** < 1.5s. Giá trị số trong `predicted_risk_score` **không được đọc**. Ngưỡng đã tái tạo
khớp dữ liệu:

```
harsh_brake:  longitudinal_accel < -0.40 × 9.81
harsh_accel:  longitudinal_accel >  0.35 × 9.81
harsh_corner: |lateral_accel|    >  0.30 × 9.81   (xem cảnh báo off-by-one ở §4)
speeding:     speed_kmh > speed_limit_kmh + 5
```

→ Đừng xây model C3. Chỉ cần thêm 1 cột `predicted_risk_score` bất kỳ để "đăng ký", rồi dồn sức
làm TTC chính xác quanh dải **1.5s** — đó là đòn bẩy C3 duy nhất bạn kiểm soát. **`near_miss` đếm
FRAME, không phải episode** → mỗi frame near-miss sai (thừa hoặc thiếu) trên scored trip tốn tới
**10 điểm composite** (5 điểm safe-score × hệ số lỗi 2×).

---

## 3. Rủi ro — điều KHÔNG được làm

- **Đừng tin composite Challenge 3 = 100 khi test local** (S4). Trên practice nó luôn 100 vì bị
  clamp; scored trip (~90s highway bình thường) mật độ harsh thấp hơn nhiều nên safe-score sẽ
  **không** clamp, và mọi near-miss sai sẽ bị tính điểm thật.
- **Đừng overfit vào regularity của practice set** (S5): event chỉ rơi vào {6,10,15,19,22}s;
  driver-state chỉ đổi tại t=0 và t=15s (split 50/50 giữa trip); mỗi trip chỉ 1–2 event. Scored
  trip dài 90s/1800 frame, đa event dày đặc, timing driver độc lập. Đừng hardcode mốc thời gian,
  mốc 15s, hay cửa sổ near-miss cố định. `world_frame` offset là ngẫu nhiên per-trip (96k … 2.64M)
  — **không phải tín hiệu**.
- **Driver state và event CARLA độc lập nhau có chủ đích** — đừng học tương quan kiểu "buồn ngủ →
  sắp có xe cắt ngang".
- Train mô hình **bất biến theo thời gian, dựa trên ảnh**, với smoothing không giả định điểm đổi
  cố định; kiểm tra không có gì key theo `frame_id`/`world_frame`/mốc 15s.

---

## 4. Bug trong `evaluation.py` (ảnh hưởng chấm điểm)

Các mục này quan trọng nếu `evaluation.py` là bộ chấm thật, hoặc nếu bạn dùng nó để tự đánh giá.

- **E0 (cao) — Không kiểm tra coverage.** Chỉ chấm các `frame_id` có trong CSV; frame thiếu không
  bị phạt, không cảnh báo. Xóa 12 frame critical → composite 28.8 lên 49.6 (+20.8). CSV nửa độ phủ
  (300/600 dòng, giữ đúng các frame critical) vẫn ra 100/100/100. **Khuyến nghị nếu bạn sửa bộ
  chấm:** dựng cặp từ tập frame GT, frame thiếu gán prediction xấu nhất (`ttc=inf`), in % coverage.
- **E1 (trung) — Challenge 1 luôn bị chấm.** Docstring và README nói "chỉ chấm challenge có cột
  tương ứng", nhưng chỉ C2/C3 được gate; C1 tính vô điều kiện, `predicted_ttc` thiếu → mặc định
  `inf`. Bài chỉ làm C2 vẫn nhận điểm C1 ~29 làm headline + vào overall composite. Nếu bạn chỉ làm
  1 challenge khác C1, đừng hoảng khi thấy điểm C1 thấp.
- **E3 (trung) — `--trip-dir` bị bỏ qua.** `main()` chỉ dùng thư mục **cha** của `--trip-dir`, rồi
  chọn trip theo **tên file CSV** (`csv_path.stem`). Trỏ `--trip-dir T03-Sample` nhưng CSV tên
  `T01-Sample.csv` → chấm nhầm T01, không cảnh báo.
- **E4 / D1 / S6 (trung–thấp) — Tái tạo harsh_corner off-by-one, KHÔNG "bit-for-bit".** Docstring
  của `evaluation.py` (dòng 124–128) tự nhận "exact copies … bit-for-bit". Nhưng dùng so sánh
  **ngặt** `|lat| > 0.30×9.81`. T03 frame 507 có `lateral_accel = -2.943 = đúng -0.30×9.81`;
  generator gốc gắn cờ `harsh_corner=True` (aggregate=41) nhưng `>` loại nó ra → eval tính 40.
  Bị che trên practice do clamp; trên scored trip không clamp thì mỗi frame biên như vậy làm lệch
  điểm C3 kể cả predictor hoàn hảo. *(README bản tiếng Việt mềm hơn: "gần như chính xác 100%" — chỉ
  docstring code mới khẳng định "bit-for-bit".)*
- **E5 (thấp) — frame_id lạ được chấm với GT bịa.** frame_id không tồn tại → gán `gt_ttc=inf`,
  `state='unknown'` thay vì loại bỏ. Pad CSV lên 1800 dòng cho trip 600 frame → bị chấm trên 1200
  frame ma, không cảnh báo (C1 F1 rớt 1.0→0.286, C3 near_miss phồng 7→57).
- **B1 (trung) — Bẫy đặt tên file CSV.** `evaluation.py` suy `trip_id` từ **tên file CSV** và yêu
  cầu khớp regex `^T\d+d?(-Sample)?$` **và** trùng tên thư mục trip. Sai tên → CSV bị skip (chỉ log
  WARNING) rồi chết bằng `RuntimeError: No valid trip evaluations produced.` không rõ nguyên nhân.
  README (dòng 147) dùng đúng quy ước `T01-Sample.csv`, nhưng **docstring của chính
  `baseline_ttc_predictor.py` (dòng 34) ví dụ `predictions.csv`** — làm theo là dính bẫy ngay.

**✅ E7 (positive, tin được):** Cột `ground_truth_ttc` tự ghi trong CSV **bị bỏ qua hoàn toàn** —
bộ chấm luôn load GT từ thư mục trip đáng tin (CSV có cột `ground_truth_ttc=0.4` khớp
`predicted_ttc=0.4` vẫn bị chấm 31.7, không phải 100). Cell rác (`nan`/`abc`/trống → `inf`,
frame_id không phải số → bỏ dòng, state lạ → tính sai) không làm crash. Hai rìa cần biết:
`predicted_ttc` âm được nhận như dự đoán nguy hiểm hợp lệ (không validate); CSV rỗng/chỉ header →
traceback thay vì báo lỗi thân thiện.

---

## 5. Bug/Gap trong `dataset_loader.py` (ảnh hưởng phân tích của BẠN)

Core loader đã được xác minh **đúng 100%**: 23 thuộc tính `FrameRecord` khớp raw JSON cho cả
3.600 frame của 6 trip; xử lý `Infinity` (cả `.json.gz` lẫn `.json` token trần), khám phá
`HackathonDataset` (tìm đúng 6 trip -Sample, không lỗi khi thiếu T0Xd), thông báo lỗi rõ, decompress
JSON 1 lần lúc init (không có bẫy hiệu năng). Các vấn đề nằm ở tiện ích phân tích:

- **L0 (trung) — Sentinel `min_ttc = 99.0` thấp hơn giá trị finite thật.** `frames_df` thay `inf`
  bằng 99.0, nhưng T03 có 22 frame `min_ttc` finite > 99 (max **389.6s** ở frame 557), T04/T06 mỗi
  trip 1 frame (129.6/140.1). → Cột **không đơn điệu theo mức nguy hiểm**: frame không có mối đe dọa
  (sentinel 99) lại xếp "nguy hiểm hơn" frame TTC thật 389s. Lọc `df.min_ttc < 100` hay sort/regress
  mà không xem cột `min_ttc_inf` → sai âm thầm. **Fix (1 dòng, `dataset_loader.py:183`):** giữ
  `np.inf` trong DataFrame, hoặc nâng sentinel lên trên max quan sát (vd 1000.0).
- **L1 (trung) — `headway_sec` cũng sentinel 99.0 nhưng KHÔNG có cột `headway_sec_inf`.** T01 có
  595/600 frame headway vô hạn → cột ~99% là sentinel không phân biệt được với dữ liệu thật.
  `df.headway_sec.mean() = 98.19` trong khi `avg_headway_sec` thật = 1.46 (**sai 67 lần**); lọc
  `< 99.0` mới ra 1.462. **Fix:** thêm cột `headway_sec_inf` đối xứng với `min_ttc_inf`
  (`dataset_loader.py:198`) hoặc lưu thẳng `np.inf`.
- **L2 (trung) — Không có label loader.** `self.label_dir` được gán (`:152`) nhưng **không hàm nào
  dùng** — labels là modality duy nhất không có loader. **Fix:** thêm `load_labels(frame_id) ->
  List[dict]`, trả `[]` cho file rỗng, parse 15 trường KITTI, docstring cảnh báo bbox 2D toàn 0.
- **L3 (thấp) — `load_depth` nuốt lỗi.** Trả `None` cho mọi file `.npy` thiếu — đúng cho non-keyframe,
  nhưng cũng nuốt luôn typo/ngoài phạm vi: `load_depth(10000) → None` thay vì raise (khác
  `load_left` có báo `FileNotFoundError`). **Fix:** raise khi `frame_id % 5 == 0` mà file thiếu,
  hoặc khi `frame_id >= len(self)`.
- **L4 (thấp, rủi ro cho scored trip) — Crash nếu GT sub-object = `null`.** Xóa hẳn key GT thì loader
  chịu tốt (default kích hoạt). Nhưng nếu ban tổ chức để `"driver": null`, `raw.get('driver', {})`
  trả `None` → `AttributeError`. **Fix (4 token, `:297-300`):** `raw.get('driver') or {}` (tương tự
  ego/risk/behavior_flags).
- **L5 (thấp) — Mất `ego.location/rotation/geolocation` và `world_frame`.** `FrameRecord` không expose,
  `frames_df` không có cột vị trí; chỉ tới được qua thuộc tính private `ds._raw_frames`. Quỹ đạo ego
  là GT thật, hữu ích cho feature engineering C1/C3. **Fix:** thêm field vào `FrameRecord` (+ cột x/y
  vào `frames_df`) hoặc expose property `raw_frames` công khai.

*(Các fix trên áp dụng cho **bản copy cục bộ** của bạn — không nên sửa file kit gốc mà ban tổ chức
phát hành.)*

---

## 6. Sai sót tài liệu (sửa để không mất thời gian)

- **D0 / D2 / B0 (trung) — Số liệu sai:**
  - README dòng 68 mô tả `label_2` là "bounding box 2D + 3D" và liệt kê bbox 2D / truncated /
    occluded / alpha / rotation_y là các trường "vẫn giữ nguyên" → **sai**, chúng bằng 0 trên cả 6
    trip. Chỉ `type` + dims 3D + location 3D là thật. (File `Practice_Dataset_Architecture.md §5.4`
    đã ghi đúng; `HUONG_DAN` không lặp lại lỗi này.)
  - "Output mẫu" của baseline (README 162–169, HƯỚNG DẪN 421–423, và bản .html): composite 52.3 /
    MAE-crit 1.420s / F1 0.480 / **n_crit 62** — thực chạy ra **30.6 / 58.6s / 0.125 / n_crit 12**.
    `n_crit` là số frame GT có min_ttc<3s (thuần GT, độc lập model) = 12 cho mọi trip → số 62 chắc
    chắn từ bản dataset cũ. Nên **chạy lại và dán số thật**, hoặc đánh dấu rõ khối này là minh họa
    không tái tạo được.
- **D3 (thấp) — requirements sai:** README dòng 101 ghi "tối thiểu cần … pyyaml" (không file nào
  `import yaml`) và **thiếu jupyter** (cần để chạy notebook). `requirements.txt` mới đúng
  (opencv/numpy/pandas/matplotlib/jupyter, không pyyaml).
- **D4 (thấp) — Link hỏng trong HƯỚNG DẪN:** dòng 602 trỏ anchor `#…chọn-1-trong-3-cách` nhưng
  heading thật là "…chọn 1 trong **2** cách" → slug không khớp, link chết. (Chuỗi "3 cách" chỉ nằm
  trong href, không hiển thị cho người đọc.)
- **E6 (thấp) — Ví dụ output README không tái tạo được** (cùng gốc với B0/D2: n_crit 62 vs 12).
  *Đính chính: khối CSV ví dụ ở README 258–263 hiển thị 1800 dòng/90s là **đúng chủ đích** cho scored
  trip T0Xd, không phải dấu hiệu bản cũ.*

---

## 7. Xác nhận positive (phần chắc chắn, tin được)

- **Core `dataset_loader.py` đúng 100%** — 3.600 frame khớp raw JSON, xử lý `Infinity`, khám phá
  đa trip, thông báo lỗi rõ, không bẫy hiệu năng.
- **E7** — Bảo mật `evaluation.py` đúng: cột `ground_truth` tự ghi bị bỏ qua; cell rác không crash.
- **B5** — CSV baseline đúng format: 600 dòng + header, cột `ground_truth_ttc` khớp JSON (0 sai lệch),
  `inf` parse chuẩn, và `evaluation.py` không hề đọc/ghi đè cột GT tự ghi (md5 CSV không đổi trước/sau
  chấm).
- Notebook `explore_trip.ipynb`, baseline, evaluation **đều chạy sạch** trên T01-Sample thật; bảng
  API trong README khớp đúng `dataset_loader.py`.

---

## 8. Việc nên làm — theo thứ tự ưu tiên

**Nếu bạn là đội thi (dùng kit):**
1. Xây classifier driver-state (C2) — ăn trọn C2 + alertness free (S1). Temporal smoothing mạnh.
2. Xây monocular depth từ 720 keyframe depth GT → TTC (S2, B2). **Tuyệt đối tránh `inf` trong vùng
   tiếp cận** (B2). Detector crop VRU + near-min depth thay median-over-ROI (B3).
3. C3: chỉ thêm 1 cột `predicted_risk_score` hằng, dồn sức tinh chỉnh TTC quanh 1.5s (S0). Tự tái tạo
   safe-score bằng công thức §2 để sanity-check.
4. Đặt tên file nộp đúng `<trip_id>.csv` (B1). Xuất đủ 1800 dòng/scored trip, không thừa/thiếu (E0/E5).
5. Không overfit regularity practice-set (S5); đừng tin C3=100 local (S4).

**Nếu muốn vá bản kit cục bộ cho gọn:** áp các fix 1 dòng ở §5 (L0/L1/L4 đáng nhất), thêm
`load_labels()` (L2). Đây là bản copy của bạn, không đụng file gốc ban tổ chức.

**Đáng báo cho ban tổ chức:** số liệu output mẫu sai ~40x (B0/D2), C3 clamp làm practice không có tín
hiệu (S4/E2), off-by-one harsh_corner (E4), và lỗ hổng coverage (E0).

---

*Báo cáo sinh từ audit đa tác nhân có kiểm chứng đối kháng. Mọi con số đều được chạy lại trên dữ liệu
thật; các phát hiện partially-correct đã tích hợp đính chính của agent kiểm chứng. Tham chiếu dòng
theo bản kit tại `team-kit/Package_starterkit/package_starterkit/`.*
