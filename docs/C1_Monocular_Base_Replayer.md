# C1 monocular baseline chạy bằng TripReplayer

Pipeline này dùng đúng một camera đường phía trước:

```text
TripReplayer -> FrameBundle.left() -> YOLO11s ONNX -> IoU tracker
             -> scale-rate TTC -> CSV + HUD MP4
```

Inference không gọi `right()`, `driver()` và không đọc `depth`. Đây là baseline
chưa train TTC: detector dùng weight COCO có sẵn, còn TTC được tính từ tốc độ
phóng to và chuyển động ngang của bounding box qua thời gian. Tracker cho phép
một vật thể đổi nhãn `car/person/motorcycle` mà không mất lịch sử, đồng thời giữ
dự đoán ngắn khi detector chớp mất object. Nhánh fast-attack dùng ego speed và
kích thước ảnh để cảnh báo sớm một vật thể lớn vừa đi vào hành lang xe.

## Smoke test 50 frame

```bash
python3 -m safeloop.c1.replay data/T01-Sample \
  --limit 50 \
  --video predictions/c1_monocular_base/T01-preview.mp4
```

## Chạy và chấm toàn bộ T01-Sample

```bash
python3 -m safeloop.c1.replay data/T01-Sample \
  --output predictions/c1_monocular_base/T01-Sample.csv \
  --video predictions/c1_monocular_base/T01-Sample.mp4 \
  --evaluate
```

Muốn xem cửa sổ trực tiếp thay vì ghi video:

```bash
python3 -m safeloop.c1.replay data/T01-Sample --mode realtime --show
```

Chế độ CPU tiết kiệm chạy detector khoảng 6,7 Hz nhưng vẫn xuất TTC ở 20 Hz:

```bash
python3 -m safeloop.c1.replay data/T01-Sample \
  --mode realtime --detector-stride 3 --show
```

Đây là chế độ demo CPU. Frame có detector vẫn có latency cao; bản production
phải dùng GPU/model tiny để mọi frame đạt latency dưới 50 ms.

Máy có OpenCV DNN CUDA:

```bash
python3 -m safeloop.c1.replay data/T01-Sample --device cuda --evaluate
```

`--ego-fallback` bật TTC dự phòng từ speed của xe và kích thước vật thể.
Mặc định nó tắt để baseline RGB thuần không cảnh báo nhầm xe đang chạy cùng tốc
độ. Đây là nhánh phải đo bằng ablation, không nên bật mặc định trước khi đánh giá.

## Output

- CSV chuẩn: `frame_id,timestamp,predicted_ttc`.
- `predicted_ttc=inf` khi chưa có vật thể tiến vào vùng chuyển động của xe hoặc
  chưa đủ lịch sử để tính tốc độ phóng to.
- HUD hiển thị track id, class, confidence, TTC và latency từng frame.
- Màu đỏ: TTC dưới 2 giây; cam: TTC hữu hạn; vàng: vật thể nằm trong vùng theo
  dõi nhưng chưa đủ bằng chứng TTC; xanh: ngoài vùng va chạm.

## Giới hạn của baseline chưa train

- Bounding box rung có thể làm TTC dao động.
- Chưa học quỹ đạo cắt ngang và chưa có scene-level fail-safe.
- Physical-size fallback không biết vận tốc thật của vật thể phía trước.
- YOLO11s CPU có thể chưa đạt 20 FPS; GPU và model tiny/TensorRT là bước tối ưu
  tiếp theo.
