# SafeLoop combined C1 + C2 replay

## Mục đích

Một lệnh chạy đồng thời:

- C1 từ `C1/student_ttc.pth` trên camera đường `kitti/image_2`;
- C2 từ `models/driver_state_phase_2_v13` trên camera tài xế `driver/`;
- một `TripLoader` và một `TripReplayer` duy nhất cho mỗi trip.

```text
TripLoader -> TripReplayer 20 Hz
                  |
                  +-- image_2 -> C1 StudentTTC update 10 Hz
                  |               -> giữ kết quả gần nhất ở frame xen kẽ
                  |
                  +-- driver  -> C2 GeneralDMS update 20 Hz
                                  |
                                  v
                 combined CSV: C1 TTC + C2 driver state
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

Video dashboard có kích thước 1280×396, 20 FPS: camera đường và TTC C1 ở bên
trái; camera tài xế, trạng thái và DMS warning C2 ở bên phải. Ghi MP4 không làm
thay đổi raw TTC hoặc CSV dùng để chấm.

Omit `--trip` để chạy `T01d..T10d`; có thể lặp `--trip` để chạy nhiều trip.
Mỗi trip tạo runtime C2 mới để state face detector/temporal không rò sang trip
tiếp theo.

## Nhịp xử lý

C1 đã train ở 10 Hz với receptive field 31 mẫu, tương đương khoảng 3,1 giây.
Nguồn hackathon là 20 Hz nên mặc định `--c1-stride 2`:

- frame 0, 2, 4, ...: decode `image_2` và chạy C1;
- frame 1, 3, 5, ...: không decode `image_2`, giữ raw TTC gần nhất;
- C2 vẫn xử lý mọi driver frame.

Không đổi C1 sang 20 Hz nếu chưa train lại; làm vậy sẽ rút ngữ cảnh của model
còn khoảng 1,55 giây. Acceleration và jerk của C1 được tính bằng sai phân lùi
sau khi lấy mẫu 10 Hz, đúng như lúc train.

## Output

`predictions/safeloop_models/<trip>.csv` có schema chung để evaluator đọc:

```text
frame_id,timestamp,predicted_ttc,predicted_driver_state
```

`predicted_ttc` là raw TTC của model, không phải số đã EMA. File
`diagnostics/<trip>.csv` có thêm:

- collision probability, raw/display TTC và warning của C1;
- `c1_model_updated` để phân biệt inference và forward-fill;
- toàn bộ probability/PERCLOS/VSS diagnostics hiện có của C2;
- latency riêng của từng model.

`replay_report.json` tổng hợp số frame, cadence, state counts và latency. C1
không xuất bbox, collision target hoặc obstacle distance; không được suy diễn
`speed × TTC` thành khoảng cách để phát lên CarSky.

## API orchestration

`safeloop.combined_replay.CombinedModelReplay` chỉ làm một việc: với mỗi
`FrameBundle`, gọi `C1.runtime.StudentTTCRuntime.process_bundle()` rồi gọi
`GeneralDMS.process_bgr()` trên driver frame. Nó không tạo replayer thứ hai và
không đọc lại trip.
