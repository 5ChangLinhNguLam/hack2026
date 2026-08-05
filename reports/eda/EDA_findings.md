# EDA findings — 6 trip practice + 10 trip chấm điểm (WBS 1.3)

> Draft từ Claude Code theo **số liệu thật**; Tú review chốt.
> Sinh lại toàn bộ: `python -m eda.run_all --data-dir data/ --out reports/eda/` (13.3s, 16/16 trip).
> Mọi kết luận đều trỏ về bảng trong `tables/` hoặc figure trong `figures/`.

---

## 0. TÓM TẮT CHO NGƯỜI BẬN

| # | Kết luận | Nguồn |
|---|---|---|
| 1 | **C3 miễn phí trên 10/10 trip chấm điểm** — sàn phạt deterministic 441–1050 ≫ 100, `safe_driving_score` chắc chắn = 0 bất kể TTC dự đoán. Không cần đầu tư gì cho C3. | T-D5 |
| 2 | Công thức flags chốt được **duy nhất 1 tổ hợp**: đơn vị **m/s²**, biên **`>=`**, đếm theo **frame**. Sai số 0 tuyệt đối trên cả 6 trip. | T-D3 |
| 3 | ⚠️ Scorer `evaluation.py` dùng biên `>` — **lệch với generator** đúng 1 frame (T03 f507). | T-D3 |
| 4 | ⚠️ **Fact #6 (driver ⊥ event) KHÔNG kiểm chứng được** — số liệu cho liên hệ MẠNH (V=0.55 gộp; T01 V=0.997) do trùng điểm đổi, không phải nhân quả. Rủi ro model học tương quan giả. | T-E1 |
| 5 | ⚠️ Nhãn C2 có dấu hiệu **không khớp ảnh** (T06 đoạn `drowsy`: mắt mở to, đầu ngẩng). | F-A1 |
| 6 | C2 chỉ có **9 đoạn / 3 chuyển trạng thái** trên toàn bộ practice — gần như không có tín hiệu temporal, và mọi đoạn đều đúng 15s/30s (không tổng quát hoá). | T-C2, T-C3 |
| 7 | Distribution shift C1: trip chấm điểm **dài gấp 3, nhiều event hơn** (2.8 vs 1.33 instance/trip) nhưng **loãng hơn** (41.4% vs 53.6% frame có event). | T-B5 |
| 8 | `ttc_2d` **hoàn toàn trùng** `ttc_simple` trên target trong cone (lệch max = 0.0000s) → chỉ cần học 1 đại lượng. | F-B3 |

---

## 1. Tình trạng dữ liệu & hình dạng che GT

**Toàn vẹn 16/16 trip sạch** (`integrity.csv`): ảnh ×3 cam = `n_frames`, depth = `ceil(n/5)`, label = `n_frames`, không trip nào lệch alignment.

| | 6 trip mẫu (P) | 10 trip chấm điểm (S) |
|---|---|---|
| n_frames | 600 (30s) | **1800 (90s)** |
| depth `.npy` | 120 | 360 |
| label khác rỗng | 57–600 | **332–1302** |
| `speed_limit_kmh` | 40–80 | 40–**90** |
| map | Town04/05/06/07/10HD | + **Town01, Town03** |

**Hình dạng che GT** (`redaction_shape.csv`) — **10 trip chấm điểm đồng nhất 100%**, xác nhận đúng fact #8:

- **Mất hẳn key**: `trip_aggregate`, `driver_summary`, `frames.min_ttc`, `frames.headway_sec`, `frames.behavior_flags`, `frames.risk`, `ego.location/rotation/geolocation`, `targets[].rel_pos/closing_speed/ttc_simple/ttc_2d/in_collision_cone`, `events_log[].params`
- **Rỗng (không mất key)**: `frames.driver = {}` ← chính là hình dạng mà loader phải phòng
- **Còn nguyên**: `ego.speed_kmh/longitudinal_accel/lateral_accel`, `targets[].target_id/target_class`, `events_active[]` đủ 4 key, `events_log[].t/type`, `metadata.speed_limit_kmh/weather`
- **`label_2` x/y/z**: ZERO hết ở 10/10 trip S; còn nguyên ở 6/6 trip P

**Bổ sung fact #8**: `events_active[].params` **không tồn tại ngay cả ở 6 trip mẫu** → đây không phải "bị xoá" mà là schema vốn không có. Params chỉ từng nằm ở `events_log[].params` (và bị xoá ở S).

---

## 2. C1 — phân bố TTC, target, tuổi thọ event, cảnh báo distribution shift

### 2.1. TTC cực kỳ mất cân bằng (`ttc_summary.csv`, `ttc_hist.png`, `min_ttc_timeline_*.png`)

| Trip | %inf | %<3s | %<1.5s | TTC min |
|---|---|---|---|---|
| T01 | 90.5 | 2.00 | 1.17 | 1.034 |
| T02 | 94.2 | 3.17 | 3.00 | 0.172 |
| T03 | 89.8 | 5.33 | 4.33 | **0.007** |
| T04 | 79.8 | 12.67 | 5.00 | 0.913 |
| T05 | 89.0 | 7.83 | 2.17 | 1.427 |
| T06 | 56.3 | 16.17 | 7.00 | 0.082 |

**56–94% frame là `inf`**; vùng critical (<3s) chỉ 2–16%. Hệ quả cho Hải: bài toán cực kỳ imbalanced, và metric chấm điểm chỉ tính MAE trong vùng critical — **phải tách nhóm `inf` riêng**, đừng để `inf`→99 làm loãng loss.

T03 có `ttc_max = 389.6s` — trùng đúng phát hiện L0 của audit report (sentinel 99.0 của loader gốc BTC sai ở đây).

### 2.2. Target & collision cone (`targets_summary.csv`, `relpos_scatter.png`, `ttc_simple_vs_2d.png`)

- Mật độ target rất khác nhau: T03 chỉ 0.96 target/frame (138 frame trống), T01 tới 11.88.
- **Chỉ 0.8–27.4% target nằm trong cone** → phần lớn target là nhiễu nền.
- **Cone rất hẹp**: `|lateral y|` của target trong cone có **max = 2.43m**, p99 = 2.15m, median = 0.04m — thực chất là một **dải dọc ~±2.4m**, không phải hình nón mở rộng theo khoảng cách (`relpos_scatter.png`).
- **`ttc_2d` ≡ `ttc_simple` trên target trong cone**: lệch max = **0.0000s** trên toàn bộ 624 cặp hữu hạn (`ttc_simple_vs_2d.png`). → Không cần mô hình hoá riêng `ttc_2d`.
- Track sống rất lâu (median 81–170 frame, max 600) → `target_id` dùng tracking tốt như fact #10 nói.

### 2.3. Tuổi thọ event — bác bỏ giả định "kéo dài đến hết trip" (`events_lifetime.csv`)

Đo trực tiếp qua `age_sec` trên **36 instance** (P+S), khớp fact #9:

| event_type | Instance (P/S) | Tuổi thọ đo được | Số frame active |
|---|---|---|---|
| `lead_brake` | 2 / 7 | **1.95–2.95s** (biến thiên = `duration_sec` khôi phục được) | 40–60 |
| `motorcycle_cut_in` | 2 / 8 | **14.95s hằng số** | 300 |
| `pedestrian_jaywalk` | 2 / 6 | 14.95s hoặc 19.95s | 300–400 |
| `stopped_vehicle_ahead` | 2 / 7 | **5.95–29.95s** (biến thiên mạnh nhất) | 120–600 |

**Bằng chứng bác bỏ giả định cũ**: trên trip 600 frame, `jaywalk` 2/2 và `stopped` 2/2 chạm frame cuối → dễ tưởng "kéo dài đến hết trip". Trên trip 1800 frame, **`cut_in`/`jaywalk`/`lead_brake` 0/21 instance chạm frame cuối**; chỉ `stopped` 6/7. Giả định cũ là **ảo giác do trip mẫu quá ngắn**.

`age_sec` chỉ khôi phục được **thời lượng**, không khôi phục được **cường độ** (`target_deceleration_g`) — đúng fact #9.

### 2.4. Label KITTI KHÔNG phải tín hiệu event (`labels_vs_events.csv`)

Xác nhận fact #7 bằng số:
- **T06: 100% frame có label kể cả khi không có event** — label vô dụng làm tín hiệu event.
- **T04: 45.5%** frame không-event vẫn có label; **91 label xuất hiện TRƯỚC event đầu tiên**.
- T02: 26 label trước event đầu.
- Ngược lại T01 chỉ 19% frame có event là có label → label cũng không phủ hết event.

→ Không dùng sự hiện diện của label làm feature/nhãn phụ cho event.

### 2.5. ⚠️ Distribution shift practice → scoring (`event_density_compare.csv`, `event_density.png`)

| | P (6 trip mẫu) | S (10 trip chấm điểm) |
|---|---|---|
| % frame có ≥1 event (TB) | **53.6%** | **41.4%** |
| Instance event/trip (TB) | 1.33 | **2.80** |
| Khoảng biến thiên %frame | 8.3 – 80.0 | 18.9 – 66.7 |

Kỳ vọng trong spec **được xác nhận đúng**: T04d = 66.67% frame có event, T10d = 3 instance.

**Chiều shift ngược với trực giác**: trip chấm điểm có **nhiều event hơn** nhưng **loãng hơn** (vì dài gấp 3). Nghĩa là model sẽ gặp **nhiều đoạn "yên tĩnh" dài hơn hẳn** so với lúc train trên practice → cẩn thận với model có trạng thái nội tại/temporal window ngắn, và với việc calibrate ngưỡng cảnh báo trên tỉ lệ dương của practice.

---

## 3. C2 — cân bằng lớp & temporal

### 3.1. Phân bố lớp (`state_distribution.csv`, `state_timeline_*.png`)

| state | frame | % |
|---|---|---|
| distracted | 900 | 25.0 |
| drowsy | 900 | 25.0 |
| alert | 600 | 16.7 |
| yawning | 600 | 16.7 |
| microsleep | 600 | 16.7 |

**Imbalance ratio toàn tập chỉ 1.5** — cân bằng tốt bất thường. Nhưng **mỗi trip chỉ có 1–2 lớp**: T02/T03/T05 chỉ 1 lớp duy nhất suốt 600 frame. → **Không được chia train/val theo frame**; phải chia theo trip, và với 6 trip thì mỗi lớp chỉ có 1–3 trip → CV rất mong manh.

### 3.2. Temporal gần như không có tín hiệu (`state_segments.csv`, `state_transition_matrix.csv`)

- Toàn bộ practice chỉ có **9 đoạn** và **3 lần chuyển trạng thái** (distracted→alert ×2, drowsy→distracted ×1).
- **Mọi đoạn dài đúng 15.0s hoặc 30.0s**, mọi điểm đổi đều ở **frame 300 (t=15s)**.
- Xác suất tự-chuyển 0.9978–1.0.

→ Không đủ dữ liệu để học mô hình chuỗi (HMM/LSTM state transition). Và quy luật "đổi tại t=15s" là **artifact của practice set**, chắc chắn không giữ trên trip 1800 frame — đừng hard-code.

### 3.3. Alertness verify ĐẠT (`alertness_check.csv`)

**9/9 cặp (trip, state) khớp 100% fact #4** — mỗi state đúng 1 giá trị alertness: alert 0.95, yawning 0.55, distracted 0.45, drowsy 0.35, microsleep 0.05. Không có ngoại lệ.

### 3.4. Face features suy ra từ state, không phải quan sát độc lập (`face_features_by_state.csv`)

Mỗi state ứng với **đúng 1 tổ hợp** `(eye_state, head_pose, mouth_state)` chiếm **100%** frame, và mỗi tổ hợp ánh xạ ngược về **đúng 1 state** (song ánh):

| state | eye | head | mouth |
|---|---|---|---|
| alert | open | normal | normal |
| distracted | open | side | normal |
| drowsy | partial | down | normal |
| yawning | partial | normal | yawning |
| microsleep | closed | down | normal |

**Hệ quả quan trọng cho 3.1 (FaceMesh)**: 3 field này **không mang thông tin gì thêm** ngoài `state`, và **không dùng để kiểm tra ảnh có khớp nhãn hay không** (chúng được sinh ra từ nhãn, không phải từ ảnh). Muốn kiểm alignment ảnh↔nhãn phải dùng mắt hoặc model thị giác.

---

## 4. C3 — công thức chốt, error budget, phân loại 10 trip chấm điểm

### 4.1. ✅ T-D3 chốt công thức: **m/s² + biên `>=` + đếm theo frame** (`flag_recipe_verify.csv`)

Duyệt đủ 8 tổ hợp {g, m/s²} × {`>`, `>=`} × {frame, event}. **Đúng 1 tổ hợp cho sai số 0 tuyệt đối** trên cả 2 phép đối chiếu (trip_aggregate và behavior_flags từng frame), trên cả 6 trip:

| unit | boundary | count | err vs trip_aggregate | err vs behavior_flags/frame | đạt |
|---|---|---|---|---|---|
| **m/s²** | **`>=`** | **frame** | **0.00** | **0** | ✅ |
| m/s² | `>` | frame | 1.00 | 1 | ✗ |
| g | `>` | event | 614.17 | 1101 | ✗ |
| g | `>=` | event | 615.17 | 1102 | ✗ |
| m/s² | `>=` | event | 659.17 | 0 | ✗ |
| m/s² | `>` | event | 659.17 | 1 | ✗ |
| g | `>` | frame | 1101.00 | 1101 | ✗ |
| g | `>=` | frame | 1102.00 | 1102 | ✗ |

**3 câu hỏi của v2 đều có đáp án dứt khoát:**
1. **Đơn vị**: `longitudinal_accel`/`lateral_accel` trong JSON là **m/s²** (không phải g). Ngưỡng phải nhân 9.81.
2. **Biên**: **`>=`** (bao gồm điểm bằng đúng ngưỡng).
3. **Đếm**: theo **frame**, không phải theo event/đợt.

### 4.2. ⚠️ Scorer lệch với generator đúng 1 frame

`evaluation.py:129-134` dùng biên **`>`** (strict), trong khi generator dùng **`>=`**. Điểm khác biệt duy nhất trong toàn bộ practice:

> **T03-Sample frame 507**: `|lateral_accel| = 2.943 = 0.30 × 9.81` **đúng bằng ngưỡng**. Generator gán `harsh_corner = True` (aggregate = 41); scorer với `>` sẽ đếm 40.

Xác suất gặp lại điểm bằng-đúng-ngưỡng trên trip chấm điểm là thấp nhưng khác 0. Vì C3 miễn phí (§4.4) nên **không ảnh hưởng điểm số**; chỉ cần biết khi đối chiếu số liệu với BTC.

### 4.3. Error budget (`error_budget.csv`, `safe_score_check.csv`)

**T-D2 khớp tuyệt đối**: safe tính lại = JSON, lệch **0.0** trên 6/6 trip. Cả 6 trip đều bị clamp; giá trị **trước clamp**: −87 (T01), −129 (T02), −244 (T04), −268 (T05), −448 (T06), −498 (T03).

Tỷ trọng phạt trung bình:

| Hạng mục | Tỷ trọng TB | Nhận xét |
|---|---|---|
| harsh_accel | **43.5%** | áp đảo |
| near_miss | **29.9%** | phần duy nhất phụ thuộc perception |
| harsh_brake | 21.5% | |
| harsh_corner | 5.1% | |
| speeding | 0.1% | không đáng kể |
| **tailgating** | **0.0%** | **gần như bằng 0 ở mọi trip** |

**Độ nhạy**: sai 1 near_miss → Δsafe 5 điểm → **ΔC3 = 10 điểm**; sai 1 harsh_brake → ΔC3 = 6 điểm; sai 1% tailgating → ΔC3 chỉ **0.2 điểm**.

**Trả lời câu hỏi v2 "tailgating từ perception cần chính xác đến mức nào?"** → **Không cần chút nào.** Trên practice, để đẩy safe thoát khỏi clamp 0 cần sai `tailgating_pct` từ **870% đến 4983%** (bất khả thi, vì tối đa là 100%). Tailgating **không bao giờ ảnh hưởng điểm**. Không đầu tư ước lượng `headway` cho C3.

### 4.4. 🎯 T-D5 — **10/10 trip chấm điểm là "trip miễn phí"** (`clamp_floor_scoring.csv`)

Áp công thức đã chốt (m/s², `>=`, frame) lên ego kinematics **không bị che** của 10 trip chấm điểm:

| Trip | hb | ha | hc | speed% | **Sàn phạt deterministic** | safe cận trên | Kết luận |
|---|---|---|---|---|---|---|---|
| T01d | 106 | 234 | 113 | 2.50 | **1012.4** | 0.0 | MIỄN PHÍ |
| T02d | 118 | 242 | 105 | 11.00 | **1049.7** | 0.0 | MIỄN PHÍ |
| T03d | 43 | 109 | 47 | 0.00 | **441.0** | 0.0 | MIỄN PHÍ |
| T04d | 106 | 224 | 22 | 4.06 | **810.6** | 0.0 | MIỄN PHÍ |
| T05d | 121 | 228 | 0 | 1.72 | **819.3** | 0.0 | MIỄN PHÍ |
| T06d | 54 | 126 | 92 | 0.00 | **598.0** | 0.0 | MIỄN PHÍ |
| T07d | 55 | 129 | 213 | 0.00 | **849.0** | 0.0 | MIỄN PHÍ |
| T08d | 51 | 145 | 42 | 0.00 | **527.0** | 0.0 | MIỄN PHÍ |
| T09d | 55 | 153 | 68 | 0.00 | **607.0** | 0.0 | MIỄN PHÍ |
| T10d | 139 | 255 | 18 | 1.28 | **963.2** | 0.0 | MIỄN PHÍ |

Sàn thấp nhất là **441.0** (T03d) — vẫn **gấp 4.4 lần** biên 100. Vì `near_miss` và `tailgating` chỉ **cộng thêm** phạt, `safe_driving_score` thật của cả 10 trip **chắc chắn = 0**.

**Chiến lược cho Hải:**
- Scorer tính `predicted_safe` từ ego kinematics (giống nhau cho mọi đội) + `near_miss` từ `predicted_ttc` của đội. Vì phần kinematics một mình đã ≥ 441, `predicted_safe` = 0 **bất kể TTC dự đoán ra sao**; `true_safe` cũng = 0 → **C3 composite = 100 tuyệt đối**.
- ⇒ **Không cần đầu tư một giờ nào cho C3.** Chỉ cần cột `predicted_risk_score` có mặt trong CSV để kích hoạt chấm C3 (giá trị số không được đọc).
- ⇒ Toàn bộ công sức dồn cho **C1 (TTC)** và **C2 (driver state)**.

*(Kiểm chứng độ bền: nếu dùng biên `>` của scorer thay vì `>=`, sàn thay đổi tối đa vài đơn vị — kết luận không đổi.)*

### 4.5. Risk decomposition (`risk_decomposition.png`)

`final_risk_score = min(100, base_risk × driver_factor)` — xác nhận đúng, không có ngoại lệ. Số frame bị clamp: T01 0, T02 18, T03 20, T04 18, T05 37, T06 32.

---

## 5. Kiểm định độc lập driver ⊥ event

### ⚠️ KHÔNG kiểm chứng được fact #6 — số liệu cho kết quả NGƯỢC LẠI (`independence_test.csv`)

| Phạm vi | Cramér's V | p-value | Kết luận thống kê |
|---|---|---|---|
| **Gộp 6 trip** | **0.549** | 1.1e-233 | liên hệ **MẠNH** |
| T01-Sample | **0.997** | 1.2e-131 | liên hệ **gần như hoàn hảo** |
| T04-Sample | 0.704 | 1.5e-66 | MẠNH |
| T06-Sample | 0.496 | 6.1e-34 | trung bình |
| T02/T03/T05 | — | — | không kiểm được (chỉ 1 state) |

**Nguyên nhân là cấu trúc, không phải nhân quả**: cả `state` lẫn `event` đều là hàm bậc thang, và điểm đổi rơi gần/trùng nhau trong trip 30s. Riêng **T01: state đổi tại frame 300 và event bật cũng tại frame 300 — trùng khít**, nên V ≈ 1.

| Trip | Frame đổi state | Frame bật event | Trùng? |
|---|---|---|---|
| T01 | 300 | 300 | **TRÙNG KHÍT** |
| T04 | 300 | 200 | không |
| T06 | 300 | 120 | không |

**Cảnh báo hành động**: dù thiết kế có ý định độc lập, **dữ liệu practice DẠY model tương quan giả** "có event → alert" (T01). Vì vậy:
- Model C2 **chỉ được nhìn ảnh cabin**, tuyệt đối không đưa feature event/đường vào.
- Không làm multi-modal fusion C1↔C2 dựa trên practice set.
- Không dùng tương quan này để "cải thiện" điểm — nó sẽ biến mất trên trip chấm điểm.

### F-E1 `drowsy_vs_ttc.png`
Boxplot `min_ttc` hữu hạn theo state — chỉ mang tính mô tả, **không suy ra nhân quả** vì mỗi trip gần như chỉ 1 state (biến state gần như trùng biến trip).

---

## 6. ⚠️ Bất thường cần hỏi BTC / cần Tú quyết

| # | Vấn đề | Bằng chứng | Đề xuất |
|---|---|---|---|
| **1** | **T-D1 lệch ≠ 0 ở `avg_risk_score`** (0.01–0.05). Gate của Tú yêu cầu DỪNG báo cáo. **Đã kiểm: 100% do JSON làm tròn 1 chữ số thập phân** — `round(full_precision, 1) == json` đúng 6/6 trip (vd T02: full 5.246667 → JSON 5.2). **Mọi trường còn lại lệch đúng 0.0**, kể cả toàn bộ count dùng cho C3. Tôi **không sửa gì** để ép khớp. | `trip_aggregate.csv` | Tú xác nhận đây là artifact làm tròn (không phải sai công thức) → coi như PASS. `avg_risk_score` không nằm trong công thức safe score nên không ảnh hưởng C3. |
| **2** | **Fact #6 bị số liệu bác bỏ**: driver ⊥ event cho V=0.549 (p<1e-233), T01 V=0.997. | `independence_test.csv` | Hỏi BTC: có chủ đích trùng điểm đổi ở T01 không? Dù sao vẫn phải cách ly C2 khỏi feature event. |
| **3** | **Nhãn C2 nghi không khớp ảnh**: T06 đoạn `drowsy` (f=173, f=244) tài xế mắt mở to, đầu ngẩng, đang cử động — trái fact #4 (`drowsy` → mắt partial, đầu down). T01/T04 đoạn `alert` (f=576/f=384) tài xế quay đầu nhìn sang bên, giống `distracted`. | `contact_sheet_T06-Sample.png`, `contact_sheet_T01-Sample.png`, `contact_sheet_T04-Sample.png` | Hỏi BTC nhãn được gán theo đoạn video DMD hay theo từng frame. Ảnh hưởng trực tiếp trần accuracy của C2. Lưu ý `eye/head/mouth` trong JSON **không dùng kiểm được** (§3.4). |
| **4** | **Scorer `>` vs generator `>=`** — lệch 1 frame (T03 f507, `|lat_accel|` = 2.943 đúng bằng ngưỡng). | `flag_recipe_verify.csv` | Ghi nhận; không ảnh hưởng điểm vì C3 miễn phí. |
| **5** | **`events_active[].params` không tồn tại kể cả ở trip mẫu** — fact #8 mô tả nó như thứ "còn nguyên 4 key, KHÔNG params", nhưng thực ra schema vốn chưa bao giờ có params ở đây. | `redaction_shape.csv` | Chỉ là làm rõ câu chữ của fact #8, không phải lỗi dữ liệu. |
| **6** | **Quy luật practice không tổng quát hoá**: mọi đoạn state đúng 15s/30s, mọi điểm đổi tại t=15s; `stopped_vehicle_ahead` chạm frame cuối 2/2 ở P nhưng chỉ 6/7 ở S; `cut_in`/`jaywalk`/`lead_brake` **0/21 instance** chạm frame cuối ở S. | `state_segments.csv`, `events_lifetime.csv` | Cấm hard-code các quy luật này vào model/pipeline. |

---

## 7. Phụ lục — danh mục output

**20 bảng** trong `tables/`, **30 figure** trong `figures/` (figures đã gitignore theo spec).

| Nhóm | Bảng | Figure |
|---|---|---|
| A | `integrity.csv`, `redaction_shape.csv` | `contact_sheet_T0X.png` ×6 |
| B | `ttc_summary.csv`, `targets_summary.csv`, `events_lifetime.csv`, `events_lifetime_by_type.csv`, `labels_vs_events.csv`, `event_density_compare.csv` | `min_ttc_timeline_T0X.png` ×6, `ttc_hist.png`, `ttc_simple_vs_2d.png`, `relpos_scatter.png`, `event_density.png` |
| C | `state_distribution.csv`, `state_segments.csv`, `state_segments_by_state.csv`, `state_transition_matrix.csv`, `alertness_check.csv`, `face_features_by_state.csv` | `state_timeline_T0X.png` ×6 |
| D | `trip_aggregate.csv`, `safe_score_check.csv`, `flag_recipe_verify.csv`, `error_budget.csv`, `clamp_floor_scoring.csv` | `ego_kinematics_T0X.png` ×6, `risk_decomposition.png` |
| E | `independence_test.csv` | `drowsy_vs_ttc.png` |
