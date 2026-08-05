# Task 1.3 — EDA 6 trip practice + 10 trip chấm điểm — Spec cho Claude Code (v2)

> **WBS 1.3** · Owner: Tú · Support: Hải (C1/C3), Thiện (C2) · P0 · 24–26/07 · Milestone M1
> Mục tiêu: hiểu dữ liệu bằng số liệu & biểu đồ TRƯỚC khi train; sản phẩm là **bộ figure/bảng + báo cáo findings** cho cả 3 nhánh model. Không phải code production.

## ⚠ CHANGELOG v2 (24/07 — sau khi kiểm tra thực tế 10 trip T01d–T10d)

| # | Thay đổi so với v1 |
|---|---|
| 1 | Scope mở rộng: 10 trip chấm điểm chạy ĐẦY ĐỦ nhóm B-rút-gọn + D (không còn là "1 lần kiểm tra không crash") |
| 2 | T-D3 đổi bản chất: từ *ước lượng* ngưỡng → **VERIFY** ngưỡng đã biết trong `evaluation.py:129-134` (−0.40g / +0.35g / 0.30g / speed_limit+5km/h) với 3 câu hỏi: đơn vị, biên >/>=, đếm frame-hay-event |
| 3 | MỚI **T-D4** (error budget) và **T-D5** (clamp floor trên 10 trip chấm điểm) — T-D5 là deliverable chiến lược nhất |
| 4 | T-B3 đo tuổi thọ event trực tiếp qua `age_sec` (params đã mất); BỎ giả định "jaywalk/stopped kéo dài đến hết trip" — SAI với trip 1800 frame (thực đo: cut_in 15s hằng số; jaywalk 15/20s; lead_brake 2–3s; stopped 6–30s) |
| 5 | MỚI **T-B5/F-B5**: so mật độ event practice vs scoring (distribution shift — T04d: 1200/1800 frame có event) |
| 6 | Fact mới về hình dạng che GT: `driver = {}` dict rỗng (không phải mất key); `targets[]` chỉ còn `target_id + target_class`; `headway_sec` xoá; `events_active` còn nguyên 4 key `{event_id, event_type, age_sec, actor_ids}`, KHÔNG params |
| 7 | Tailgating không tái tạo trực tiếp được → findings phải trả lời "cần headway từ perception chính xác đến mức nào" (nối với T-D4) |

---

## 1. Scope

### IN scope
1. **`eda/` package** (script, không phải notebook khổng lồ): chạy toàn bộ bằng 1 lệnh
   `python -m eda.run_all --data-dir data/ --out reports/eda/`
   Tự phát hiện trip có GT (6 mẫu) vs không GT (10 chấm điểm) và chạy nhóm phân tích phù hợp cho từng loại.
2. Output chuẩn hoá vào `reports/eda/`: `figures/*.png` (matplotlib Agg), `tables/*.csv`, `EDA_findings.md` (Claude Code draft theo số liệu thật, Tú review chốt).
3. `explore_trip.ipynb` của Team Kit chỉ là **tham chiếu cách đọc dữ liệu** (không bắt buộc nộp — đã xác nhận với BTC).

### OUT of scope
- Train model, inference từ ảnh — EDA chỉ dùng JSON + metadata ảnh/depth/label.
- Sửa/copy dataset; không commit dataset; chỉ commit findings + tables (không commit figures nặng).

### Phụ thuộc mềm vào task 1.2
Ưu tiên `tripkit.TripLoader` (đã xử lý đúng `driver={}` = GT xoá từ Phase 2); fallback tự đọc raw nếu chưa merge. Adapter: `eda/io.py`. **Không dùng loader gốc BTC cho trip chấm điểm** — nó không phòng dict rỗng.

---

## 2. Facts PHẢI tôn trọng (v2 — đã đối chiếu dữ liệu thật)

1. JSON có token `Infinity` trần → tách nhóm `inf` trước khi vẽ histogram.
2. Chỉ T01-Sample có `.json` giải nén; còn lại `gzip.open(path,'rt')`.
3. `min_ttc` CHỈ tính target `in_collision_cone=true`; không có → `Infinity`. (Field này KHÔNG còn ở trip chấm điểm.)
4. Alertness cố định theo state: alert 0.95 / yawning 0.55 / distracted 0.45 / drowsy 0.35 / microsleep 0.05 — verify 6/6 trip mẫu.
5. Công thức C3: `safe = max(0, 100 − (hb×3 + ha×2 + hc×2 + near_miss×5 + speeding%×0.15 + tailgating%×0.10))`; near_miss = số frame `min_ttc<1.5s`; 6 trip mẫu đều safe=0 do clamp. Ngưỡng flags: `evaluation.py:129-134` — harsh_brake −0.40g, harsh_accel +0.35g, harsh_corner 0.30g (lateral), speeding > speed_limit+5km/h. **Chưa chắc chắn**: đơn vị accel trong JSON (g hay m/s²), biên `>` vs `>=`, đếm frame hay event → T-D3 phải chốt cả 3.
6. Driver state ⊥ event CARLA — kiểm định bằng số trên 6 trip mẫu (trip chấm điểm không kiểm được: driver bị xoá).
7. `label_2` gán nhãn chọn lọc, có thể xuất hiện TRƯỚC event → chỉ mô tả, không kết luận "label = tín hiệu event".
8. Hình dạng che GT ở T01d–T10d (đã kiểm thực tế): `behavior_flags` XOÁ hẳn; `driver` = `{}`; `headway_sec` xoá; `targets[]` chỉ còn `target_id + target_class`; `ego` còn speed_kmh + longitudinal/lateral_accel; `metadata.speed_limit_kmh` còn; `events_active` còn nguyên `{event_id, event_type, age_sec, actor_ids}` không params; `events_log` còn t + type.
9. Tuổi thọ event (đo thực từ age_sec trên 10 trip): lead_brake 2.0–3.0s (biến thiên — chính là duration_sec khôi phục được), motorcycle_cut_in 15.0s hằng số, pedestrian_jaywalk 15.0/20.0s, stopped_vehicle_ahead 6–30s. **KHÔNG code theo giả định "active tới frame cuối".** Cường độ (target_deceleration_g) không khôi phục được.
10. `actor_ids` ghép được với `targets[].target_id` — dùng cho T-B3/T-B5.

---

## 3. Danh mục phân tích (đầu ra cố định — bám đúng danh sách)

Ký hiệu cột "Tập": **P** = 6 trip mẫu, **S** = 10 trip chấm điểm, **P+S** = cả hai.

### Nhóm A — Toàn vẹn & alignment
- **T-A1** `integrity.csv` (P+S): mỗi trip × {n_frames, n ảnh ×3 cam, n depth, n label, n label khác rỗng, fps, map, weather, speed_limit, has_gt}.
- **F-A1** `contact_sheet_T0X.png` (P, 6 file): 5 frame ngẫu nhiên/trip, [road trái + driver] + caption JSON (state, min_ttc, events_active, speed) — kiểm bằng mắt.
- **T-A2** `redaction_shape.csv` (S): mỗi trip × field nào còn/mất/rỗng — bằng chứng máy móc cho fact #8, phát hiện nếu 10 trip không đồng nhất với nhau.

### Nhóm B — C1: TTC, targets, events (cho Hải)
- **T-B1** `ttc_summary.csv` (P): %inf, %<3s, %<1.5s, min/p25/median TTC hữu hạn — theo trip.
- **F-B1** `min_ttc_timeline_T0X.png` (P, 6): min_ttc theo thời gian, nền tô khoảng event theo loại, vạch 1.5s/3s.
- **F-B2** `ttc_hist.png` (P): histogram TTC hữu hạn + vạch 1.5s, 3s.
- **T-B2** `targets_summary.csv` (P): target/frame, phân bố class, tỷ lệ in_cone, số track, tuổi thọ track.
- **F-B3** `ttc_simple_vs_2d.png` (P): scatter, chỉ target trong cone.
- **F-B4** `relpos_scatter.png` (P): rel_pos tô màu theo in_cone — hình dạng cone thực tế.
- **T-B3** `events_lifetime.csv` (P+S): mỗi instance event × {trip, type, frame bắt đầu, tuổi thọ = age_sec max, số frame active, actor_ids, actor_class (join qua target_id)}. Thống kê theo type, đối chiếu fact #9.
- **T-B4** `labels_vs_events.csv` (P): tỷ lệ label≠rỗng khi event active vs không; số frame label xuất hiện TRƯỚC event đầu.
- **T-B5** `event_density_compare.csv` + **F-B5** `event_density.png` (P+S): mỗi trip × {%frame có ≥1 event active, số instance theo type, số event đồng thời max}; bar chart P cạnh S — **định lượng distribution shift**. Kỳ vọng xác nhận: T04d ~67% frame có event, T10d 3 event.

### Nhóm C — C2: driver state (cho Thiện — chỉ chạy trên P)
- **T-C1** `state_distribution.csv`: đếm frame 5 state × trip + imbalance ratio.
- **F-C1** `state_timeline_T0X.png` (6): dải màu state theo thời gian.
- **T-C2** `state_segments.csv`: run-length phân đoạn; min/median/max độ dài theo state.
- **T-C3** `state_transition_matrix.csv`.
- **T-C4** `alertness_check.csv`: verify fact #4 — kỳ vọng đúng 1 giá trị/state, 6/6 trip.
- **T-C5** `face_features_by_state.csv`: eye/head/mouth theo state — preview cho FaceMesh 3.1.

### Nhóm D — C3: kinematics & score (cho Hải) — TRỌNG TÂM v2
- **T-D1** `trip_aggregate.csv` (P): trip_aggregate đặt cạnh số đếm tái tạo từ frames — chênh lệch phải = 0.
- **T-D2** `safe_score_check.csv` (P): safe tính lại vs JSON (6/6 khớp, đều 0) + **giá trị TRƯỚC clamp** từng trip.
- **T-D3** `flag_recipe_verify.csv` (P): với MỖI tổ hợp {đơn vị: g | m/s²} × {biên: > | >=} × {đếm: frame | event}, áp ngưỡng evaluation.py lên ego kinematics rồi so với behavior_flags/trip_aggregate thật → tổ hợp nào cho sai số = 0 trên cả 6 trip là **công thức chốt**. Nếu không tổ hợp nào = 0 tuyệt đối → DỪNG, báo cáo bảng sai số, không tự chọn "gần đúng nhất".
- **T-D4** `error_budget.csv` (P): đóng góp từng hạng mục phạt vào tổng phạt mỗi trip + độ nhạy (sai 1 đơn vị đếm / sai 1% tailgating → Δsafe → ΔC1-điểm-C3 = 2×Δsafe). Trả lời: *tailgating từ perception cần chính xác đến mức nào thì đáng công?*
- **T-D5** `clamp_floor_scoring.csv` (S) — **deliverable chiến lược nhất**: mỗi trip chấm điểm × {harsh_brake/accel/corner + speeding% tính bằng công thức đã chốt ở T-D3, sàn phạt deterministic = hb×3+ha×2+hc×2+speeding%×0.15, kết luận: sàn ≥100 → safe chắc chắn = 0 ("trip miễn phí") | sàn <100 → cần near_miss+tailgating, ghi khoảng cách tới biên}. Chỉ chạy SAU khi T-D3 chốt được công thức 0-sai-số.
- **F-D1** `ego_kinematics_T0X.png` (P, 6): speed + accel theo thời gian, đánh dấu frame flag bật — nhìn ngưỡng bằng mắt, đối chiếu T-D3.
- **F-D2** `risk_decomposition.png` (P): final_risk_score vs base×driver_factor.

### Nhóm E — Kiểm định độc lập (P)
- **T-E1** `independence_test.csv`: contingency state × event-active, Cramér's V + p-value.
- **F-E1** `drowsy_vs_ttc.png`: boxplot min_ttc hữu hạn theo state.

### Findings — `EDA_findings.md` cấu trúc cố định
(1) Tình trạng dữ liệu & hình dạng che GT (T-A2); (2) C1: phân bố + tuổi thọ event + **cảnh báo distribution shift** (T-B5); (3) C2: cân bằng lớp + temporal; (4) C3: công thức flags đã chốt (T-D3) + error budget (T-D4) + **danh sách trip "miễn phí" và trip cần perception** (T-D5) + yêu cầu độ chính xác headway cho tailgating; (5) Độc lập driver ⊥ event; (6) Bất thường cần hỏi BTC. Mỗi kết luận trỏ về figure/table.

---

## 4. Kế hoạch phase

- **P0 (1h):** skeleton `eda/`, `eda/io.py` (tripkit-hoặc-raw, Infinity + gzip + `driver={}` handling), deps: pandas numpy matplotlib scipy. Test đọc đủ 16 trip, in n_frames + has_gt.
- **P1 (2h):** Nhóm A (gồm T-A2 trên 10 trip S). Dừng cho Tú xem contact sheet.
- **P2 (3h):** Nhóm B — gồm T-B3/T-B5 chạy cả P+S.
- **P3 (2h):** Nhóm C.
- **P4 (3h):** Nhóm D theo đúng thứ tự T-D1 → T-D2 → **T-D3 (chốt công thức — nếu không tổ hợp nào 0-sai-số thì DỪNG báo cáo)** → T-D4 → T-D5. Không tự "sửa cho khớp".
- **P5 (1.5h):** Nhóm E + draft findings + `run_all` một lệnh chạy sạch từ đầu.

## 5. Acceptance criteria

- [ ] 1 lệnh sinh lại toàn bộ output cho cả 16 trip.
- [ ] T-D3 chốt được đúng 1 tổ hợp công thức 0-sai-số trên 6 trip mẫu (hoặc báo cáo DỪNG có bảng sai số).
- [ ] T-D5 phân loại đủ 10 trip chấm điểm: "miễn phí" vs "cần perception" kèm khoảng cách biên.
- [ ] T-B3 tuổi thọ event khớp bảng thực đo (fact #9); T-B5 định lượng được shift.
- [ ] T-C4 alertness, T-D1/T-D2 công thức C3, T-E1 độc lập — đều verify; lệch = escalate, không im.
- [ ] `EDA_findings.md` được Hải & Thiện đọc và xác nhận đủ để ra quyết định thiết kế.