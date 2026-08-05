# CarSky SafeLoop Mock Demo

Mục tiêu của mock này là dựng trước luồng `Telemetry → C1 → C2 → Risk Fusion
→ CarSky`. Đây không phải kết quả AI: JSON luôn có `"mock": true`, log luôn có
`[SafeLoop MOCK]`, không đọc ground truth và không điều khiển actuator thật.

## Chạy ngay trong CarSky ở 20 Hz

Nếu `A8_API_KEY` đang có trong environment, repo có lệnh clone Blueprint gốc,
thêm một mock Script Node bằng atomic batch, validate và deploy. Blueprint sạch
không chứa replay probe nên không có hai publisher chạy đồng thời. Đây là
workaround cho hai drift của tenant: route cập nhật node trả `404`, còn import
từ chính portable export bị validator từ chối các pin legacy:

```bash
python3 tools/carsky_mock_ctl.py install --deploy
```

Sau khi deployment `RUNNING`, sample ba lần để xác nhận TTC và speed thực sự
thay đổi trên live broker:

```bash
python3 tools/carsky_mock_ctl.py verify
```

Chỉ chốt runtime pass khi output có `"mock_live": true` và
`"missing_paths": []`.

Nếu muốn thao tác trên UI, làm các bước sau:

1. Mở Blueprint **SafeLoop Integration Lab**.
2. Tắt hoặc xóa node **SafeLoop 20Hz Replay Probe** để tránh hai publisher ghi
   đồng thời vào ba ego signal.
3. Tạo **Script Node**, đặt tên `SafeLoop Mock C1 C2 Fusion`.
4. Thêm KUKSA pin tên chính xác `kuksa`, direction **Output**.
5. Nối pin đó vào pin `kuksa` của **Central Broker (VSS)**.
6. Chọn inline script và dán toàn bộ nội dung
   `carsky/scripts/safeloop_mock_pipeline.lua`.
7. Validate Blueprint, deploy vào Device lab, chờ toàn bộ node `Running`.

Script chạy liên tục một scenario 16 giây ở timer 50 ms:

- `0–4s`: tài xế alert, TTC 8 → 5 giây.
- `4–8s`: distracted, TTC 5 → 2,2 giây.
- `8–12s`: drowsy/microsleep, TTC 2,2 → 0,8 giây.
- `12–16s`: yawning rồi alert, TTC hồi phục 0,8 → 8 giây.

Sau 16 giây scenario tự lặp. Script chỉ publish sensor simulation và log hành
động đề xuất; `EMERGENCY_BRAKE_REQUEST` không được gửi tới actuator.

## Signal Watch cần tạo

Do Signal Watch nhiều signal từng có hiện tượng chỉ signal đầu tiếp tục cập
nhật, tạo một widget cho mỗi signal quan trọng:

- C1 TTC: `Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap` — đơn vị ms.
- C1 distance: `Vehicle.ADAS.ObstacleDetection.Front.Center.Distance` — m.
- C1 warning: `Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning`.
- C2 attention: `Vehicle.Driver.AttentiveProbability` — %.
- C2 distraction: `Vehicle.Driver.DistractionLevel` — %.
- C2 fatigue: `Vehicle.Driver.FatigueLevel` — %.
- C2 eyes: `Vehicle.Driver.IsEyesOnRoad`.
- DMS warning: `Vehicle.ADAS.DMS.IsWarning`.

Giữ ba widget ego đã có cho Speed, Longitudinal Acceleration và Lateral
Acceleration. Mở log node để xem `driver state`, `risk score`, `level` và
`action` vì VSS starter chưa có path chuẩn cho bốn field SafeLoop này.

## Chạy mock bằng TripReplayer qua REST

REST giúp kiểm tra model adapter với trip thật, nhưng chỉ khoảng 3 fps trên
tenant hiện tại:

```bash
export A8_URL='https://hackathon-1.carsky.io'
export A8_API_KEY='<key local>'
export A8_ROOM_ID='ehlbxyyc4px2it-cip50w'
export A8_NODE_KEY='central-broker-vss'

python3 -m safeloop.replay_mock data/T01-Sample \
  --sink carsky-rest --start 0 --limit 40
```

Smoke test local không cần credential:

```bash
python3 -m safeloop.replay_mock data/T01-Sample --limit 3
```

Khi C1/C2 thật được bàn giao, thay `predict_c1()` và `predict_c2()` trong
`safeloop/mock_pipeline.py` bằng adapter model; contract CarSky và Risk Fusion
được giữ nguyên.
