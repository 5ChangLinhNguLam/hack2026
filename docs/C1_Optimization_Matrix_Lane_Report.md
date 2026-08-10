# C1 một camera: kết quả tối ưu, pseudo-label, matrix-light và lane

Ngày đo: 10/08/2026. Toàn bộ inference C1 trong phần này chỉ dùng camera trái
`image_2` và ego kinematics được phép. Không dùng `image_3`, depth, camera tài
xế, `min_ttc`, `targets`, `events_active` hoặc lịch sự kiện làm input.

## 1. Protocol và môi trường

- 6 trip có nhãn, tổng 3.600 frame: development và leave-one-trip-out.
- 10 trip T01d–T10d, tổng 18.000 frame: chỉ sinh prediction/pseudo-label;
  không tự chấm accuracy vì TTC ground truth bị che.
- Máy hiện tại: 12 CPU thread, OpenCV DNN CPU; không có CUDA device,
  `nvidia-smi`, PyTorch hay ONNX Runtime lúc audit.
- Detector: `models/yolo11s.onnx`, confidence cache 0,20, runtime mặc định
  confidence 0,25 và stride 3.

## 2. C1 TTC

### Các mốc đã đo

| Phiên bản | Cách đo | Điểm trung bình /100 | Ghi chú |
|---|---:|---:|---|
| Monocular baseline ban đầu | 6 sample | 37,6 | Scale/lateral TTC, stride 3 |
| Sweep heuristic tốt nhất | Development fit | 53,2 | Không phải held-out |
| Physics v2 đang chạy | 6 sample, evaluator BTC | 55,2 | Range slope + safety envelope |
| ExtraTrees calibrator nghiên cứu | Leave-one-trip-out | 61,2 | Chưa đưa vào edge runtime |
| Pseudo teacher, confidence ≥0,30 | 6 sample | 56,5 | Dùng để chọn pseudo-label |

Physics v2 đạt MAE critical trung bình 4,254 s, inverse-TTC MAE 0,2821 và
F1 trung bình 0,442. Theo trip: T01 33,0; T02 68,3; T03 65,6; T04 78,7;
T05 67,1; T06 18,7. T06 vẫn yếu do detector đổi `motorcycle/person/car`
và reset track trong đoạn cold-start của vật thể cắt vào.

Mục tiêu 85–90 chưa đạt. Con số kiểm chứng tốt nhất về khả năng tổng quát hóa
hiện là 61,2, không phải 85–90. Muốn tiến gần mục tiêu cần train temporal
monocular model trên GPU với nhiều trip có TTC thật hơn; self-training từ
pseudo-label hiện tại không thể thay cho nhãn độc lập.

Inference hiện tại không cần GPU: smoke 200 frame C1 + lane + matrix-light đạt
23,39 FPS trên CPU (mean 42,26 ms; p95 122,95 ms) với detector stride 3. Cho
bước train temporal tiếp theo, cấu hình thực dụng là GPU 24 GB như NVIDIA L4
hoặc tương đương. Edge target đề xuất để benchmark INT8 là Jetson Orin NX
16 GB; đây là đề xuất phần cứng, chưa phải kết quả benchmark của repo.

### Thay đổi vật lý

- Giữ track qua một số lần detector đổi class.
- Range-TTC dùng median pairwise slope của khoảng cách đơn camera.
- Trừ safety envelope trước khi chia closing speed, phù hợp cách TTC CARLA
  tiến gần contact hơn công thức `distance / speed` đơn giản.
- Hiệu chỉnh effective bbox height từ KITTI của 6 sample; các hệ số không đọc
  annotation ở runtime.
- Kiểm tra tỷ lệ sample giảm liên tục và MAD của slope để chặn bbox jitter.
- Xử lý detector blink, border truncation, cold-start cut-in và short coast.

## 3. T01d–T10d và pseudo-label

Đã tạo 18.000 dòng ở `predictions/c1_pseudo_labels/` và 10 CSV đúng schema ở
`predictions/c1_submission/`. Mỗi pseudo-label có confidence, số teacher đồng
thuận, danger vote và `label_source=pseudo_rgb_physics_ensemble_v1`.

Quy tắc bắt buộc: không gọi đây là ground truth. Confidence chỉ đo mức đồng
thuận nội bộ giữa các tracker; các tracker có thể cùng sai. Ngưỡng 0,30 được
chọn trên 6 sample và dùng để suppress prediction yếu trước khi tạo submission.

## 4. Virtual matrix-light

Luồng: C1 risk track → kiểm tra TTC/evidence → kiểm tra ego lane → ánh xạ bbox
sang grid 32×18 → tính azimuth/elevation → xuất request mô phỏng và overlay.
Day, night và low-visibility dùng intensity khác nhau.

Module không điều khiển đèn thật. Cố tình chiếu vào xe/người có thể gây chói và
nguy hiểm; cần hệ thống eye-safety, luật ADB, fail-safe độc lập và homologation
trước khi nối CAN/GPIO.

Kết quả độc lập trên 6 sample bằng bbox KITTI chiếu sang camera:

- Frame precision: 46,30%.
- Frame recall: 81,04%.
- Target precision: 40,65%.
- Tỷ lệ GT object được phủ ≥95%: 57,31%.
- Coverage của bbox do chính model chọn đạt 100%, nhưng đây chỉ là geometric
  actuator coverage, không phải target-selection accuracy.

Do đó yêu cầu 95% coverage chính xác trên GT chưa đạt. Code giữ
`simulation_only=true` và report `selection_accuracy=null` nếu không có nhãn
độc lập.

## 5. Lane detection

Lane detector dùng đúng camera trái: white/yellow mask, Canny, Hough geometry,
robust fit, causal smoothing, suy ra một phía khi phía còn lại tạm mất và cảnh
báo lệch làn theo normalized center offset.

Dataset không có lane ground truth. Công cụ `tools/evaluate_c1_lane.py` chỉ báo
`temporal_geometry_proxy_not_lane_accuracy`, gồm valid fraction, độ hợp lý lane
width và temporal jitter. Không được trình bày proxy này thành lane accuracy.

Trên toàn bộ 16 trip/21.600 frame, macro proxy score là 91,4/100 và valid
fraction là 89,68%. Hai edge case yếu nhất là T05d (60,4; valid 36,06%) và
T07d (71,1; valid 61,17%), nên lane output ở hai trip này phải được gate bằng
confidence. Lane-only throughput theo trip nằm khoảng 96–455 FPS trên CPU.

## 6. Lệnh tái lập

```bash
pytest -q tests/test_c1_*.py

python3 -m safeloop.c1.cache_detections data/T01d --stride 3
python3 -m safeloop.c1.pseudo_label data/T01d
python3 -m safeloop.c1.submission \
  predictions/c1_pseudo_labels/T01d.csv \
  predictions/c1_submission/T01d.csv

python3 -m safeloop.c1.replay data/T02-Sample \
  --detector-stride 3 --video predictions/T02-c1.mp4 --evaluate

python3 -m safeloop.c1.perception_replay data/T02-Sample \
  --video predictions/T02-perception.mp4 --evaluate

python3 tools/evaluate_c1_matrix_light.py
python3 tools/evaluate_c1_lane.py
```

Artifacts trong `predictions/` bị `.gitignore`; model/code/test/docs vẫn được
version-control bình thường.
