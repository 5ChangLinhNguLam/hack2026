# SafeLoop combined C1 + C2 + C3 replay

## Mục đích

Một lệnh chạy đồng thời:

- C1 từ `C1/student_ttc.pth` trên camera đường `kitti/image_2`;
- C2 từ `models/driver_state_phase_2_v13` trên camera tài xế `driver/`;
- C3 tích lũy điểm an toàn cấp trip từ ego telemetry và raw TTC của C1;
- contextual risk kết hợp C1+C2 cho cảnh báo sản phẩm tức thời;
- một `TripLoader` và một `TripReplayer` duy nhất cho mỗi trip.

```text
TripLoader -> TripReplayer 20 Hz
                  |
                  +-- image_2 -> C1 StudentTTC update 10 Hz
                  |               -> giữ kết quả gần nhất ở frame xen kẽ
                  |
                  +-- driver  -> C2 GeneralDMS update 20 Hz
                  |
                  +-- ego + raw TTC -> C3 safe-score accumulator 20 Hz
                  |
                  +-- C1 + C2 -> contextual warning policy 20 Hz
                                  |
                                  v
            combined CSV: TTC + driver state + finite C3 opt-in risk
```

Không dùng camera phải, depth, event hoặc ground truth cho inference C1. C1
physics trong `safeloop/c1/` vẫn được giữ để nghiên cứu nhưng không nằm trên
đường chạy chung này.

## Cài đặt

```bash
python3 -m venv .venv-runtime
source .venv-runtime/bin/activate
pip install -e ".[c1-runtime,dms-phase2]"
```

`timm>=1.0` là dependency bắt buộc vì checkpoint dùng backbone
`mobileone_s2`. Runtime kiểm tra đúng thứ tự 11 scalar feature ghi trong
checkpoint và dừng ngay nếu contract bị lệch.

## Chạy một trip

```bash
python -m safeloop.replay_models \
  --dataset data \
  --trip T01-Sample \
  --device cuda \
  --output-dir predictions/safeloop_models
```

Smoke test nhanh trên CPU:

```bash
python -m safeloop.replay_models \
  --dataset data --trip T01-Sample --device cpu --limit 5
```

`--limit` chỉ tạo output kiểm tra dưới `partial/`, không ghi đè CSV hoàn chỉnh
ở thư mục gốc. Tương tự, nếu nhấn `q`/`ESC`, CSV/video đang chạy được giữ dưới
`partial/` và command trả mã khác 0 để tránh nộp nhầm file thiếu frame.
Output cũ của đúng trip được chuyển an toàn sang `archive/<run-id>/` trước lượt
mới, nên file hoàn chỉnh cũ không thể bị hiểu nhầm là kết quả vừa chạy.

Xem dashboard live đúng timestamp của trip:

```bash
python -m safeloop.replay_models \
  --dataset data --trip T01-Sample --device auto \
  --mode realtime --show
```

Nhấn `q` hoặc `ESC` để dừng. `--show` cần desktop display. Trên GPU server
headless, ghi video rồi tải/mở file MP4:

```bash
python -m safeloop.replay_models \
  --dataset data --trip T01-Sample --device cuda \
  --write-video --output-dir predictions/safeloop_models

# File kết quả:
# predictions/safeloop_models/videos/T01-Sample.mp4
```

Video dashboard có kích thước 1280×432, 20 FPS: camera đường và TTC C1 ở bên
trái; camera tài xế, trạng thái và DMS warning C2 ở bên phải. Ghi MP4 không làm
thay đổi raw TTC hoặc CSV dùng để chấm. Thanh dưới tách rõ `C3 SAFE EST.`
(cao là an toàn) và `CONTEXT RISK` (cao là nguy hiểm).

Omit `--trip` để chạy `T01d..T10d`; có thể lặp `--trip` để chạy nhiều trip.
Mỗi trip tạo runtime C2 mới để state face detector/temporal không rò sang trip
tiếp theo.

## Nhịp xử lý

C1 đã train ở 10 Hz với receptive field 31 mẫu, tương đương khoảng 3,1 giây.
Nguồn hackathon là 20 Hz nên mặc định `--c1-stride 2`:

- frame 0, 2, 4, ...: decode `image_2` và chạy C1;
- frame 1, 3, 5, ...: C1 không decode `image_2`, giữ raw TTC gần nhất;
- C2 vẫn xử lý mọi driver frame.

Khi bật `--show` hoặc `--write-video`, dashboard vẫn decode `image_2` ở frame
xen kẽ để vẽ hình; model C1 không chạy lại ở các frame đó.

Không đổi C1 sang 20 Hz nếu chưa train lại; làm vậy sẽ rút ngữ cảnh của model
còn khoảng 1,55 giây. Acceleration và jerk của C1 được tính bằng sai phân lùi
sau khi lấy mẫu 10 Hz, đúng như lúc train.

## Output

`predictions/safeloop_models/<trip>.csv` có schema chung để evaluator đọc:

```text
frame_id,timestamp,predicted_ttc,predicted_driver_state,predicted_risk_score
```

`predicted_ttc` là raw TTC của model, không phải số đã EMA. File
`diagnostics/<trip>.csv` có thêm:

- collision probability, raw/display TTC và warning của C1;
- `c1_model_updated` để phân biệt inference và forward-fill;
- toàn bộ probability/PERCLOS/VSS diagnostics hiện có của C2;
- C3 safe-score estimate, penalty, năm bộ đếm và cờ theo từng frame;
- contextual risk level/action/reasons riêng, không gọi nhầm là C3 score;
- latency riêng của từng model.

`predicted_risk_score` là contextual product risk 0–100, cao là nguy hiểm.
Evaluator hiện chỉ dùng giá trị hữu hạn này làm công tắc đăng ký C3; nó tự tái
tạo C3 từ raw TTC và ego telemetry. Policy `safeloop-context-risk-v1` lấy mức
nguy cơ va chạm dạng piecewise từ TTC, lấy driver risk bằng giá trị lớn nhất
giữa thiếu chú ý/phân tâm/mệt mỏi, rồi fusion có giới hạn:

```text
context_risk = 100 * min(1, 0.75*collision_risk + 0.45*driver_risk)
```

Đây là policy HMI có version, không phải metric chính thức và không được dùng
để tuyên bố độ chính xác C3.

### C3 evaluator-compatible

Runtime đếm trên từng frame nguồn 20 Hz, dùng đúng biên strict của evaluator:

```text
harsh brake  = longitudinal_accel < -0.40 * 9.81
harsh accel  = longitudinal_accel >  0.35 * 9.81
harsh corner = abs(lateral_accel)  >  0.30 * 9.81
speeding     = speed_kmh > speed_limit_kmh + 5
near miss    = finite raw predicted_ttc < 1.5 s

safe estimate = clamp(100 - (
    brake*3 + accel*2 + corner*2 + near_miss*5 + speeding_pct*0.15
), 0, 100)
```

C2 không tham gia công thức C3. Công thức gốc còn `tailgating_pct*0.10`, nhưng
submission không có predicted headway và evaluator cũng bỏ hạng mục đó. Report
vì vậy luôn ghi `tailgating_penalty_omitted=true`; không được mô tả estimate này
là công thức organizer đầy đủ. Nếu dùng `--limit` hoặc dừng sớm, điểm hiển thị
là `PREFIX ONLY`, chưa phải điểm cuối trip.

C3 là phép tính rule-based O(1) theo frame, không cần train và không cần GPU.
GPU/CPU trong lệnh replay chỉ ảnh hưởng hai model C1 và C2.

`replay_report.json` tổng hợp số frame, cadence, state counts, latency, C3
estimate cuối và contextual-risk breakdown. C1
không xuất bbox, collision target hoặc obstacle distance; không được suy diễn
`speed × TTC` thành khoảng cách để phát lên CarSky.

## API orchestration

`safeloop.combined_replay.CombinedModelReplay` chỉ làm một việc: với mỗi
`FrameBundle`, gọi C1, C2, C3 accumulator và contextual-risk policy theo đúng
thứ tự. Nó không tạo replayer thứ hai và không đọc lại trip. C3 production không
dùng `C3Accumulator` trong `mock_pipeline.py`; class đó chỉ phục vụ kịch bản mock
16 giây và có công thức khác evaluator.
