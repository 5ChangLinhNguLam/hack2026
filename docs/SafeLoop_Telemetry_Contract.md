# SafeLoop Telemetry Contract v1

## Mục tiêu

`TripReplayer` phát telemetry theo cùng thứ tự/timestamp của dataset để các
module C1, C2, Risk Fusion và CarSky dùng chung một định danh frame. Contract
v1 chỉ chứa input hợp lệ có mặt ở cả practice và scoring trip; không phát
ground truth hay event label.

## Schema

```json
{
  "schema_version": "safeloop.telemetry.v1",
  "message_id": "T01-Sample:320",
  "source": "trip-replayer",
  "trip_id": "T01-Sample",
  "frame_id": 320,
  "timestamp_ms": 16000,
  "emitted_monotonic_ns": 1234567890,
  "ego": {
    "speed_kmh": 27.35,
    "longitudinal_accel_mps2": -0.42,
    "lateral_accel_mps2": 0.18
  }
}
```

### Ý nghĩa timestamp

- `timestamp_ms`: thời gian deterministic trong trip, dùng để align C1/C2.
- `emitted_monotonic_ns`: đồng hồ runtime tại publisher, dùng đo latency
  trong cùng host/pod. Không dùng làm business timestamp hoặc so giữa máy.
- `message_id = <trip_id>:<frame_id>`: trace key xuyên suốt pipeline.

## Tín hiệu bị cấm trong stream này

- `min_ttc`, `risk`, `behavior_flags`, driver ground truth.
- `events_active`/`events_log` và mọi event label.
- Practice-only `location`, `rotation`, `geolocation`.
- Target ground truth hoặc depth ground truth.

Oracle/test bench có thể đọc các field trên bằng một đường riêng và phải ghi
rõ là evaluation-only; chúng không được đi vào inference core.

## Chạy local

```bash
python3 -m safeloop.replay_telemetry data/T01-Sample --limit 5

# Phát realtime 20 Hz và lưu NDJSON
python3 -m safeloop.replay_telemetry data/T01-Sample \
  --mode realtime --output /tmp/safeloop-telemetry.ndjson
```

Summary được in ra `stderr`; `stdout` chỉ chứa NDJSON nên có thể pipe sang
consumer khác.

## Phát lên CarSky qua REST Signal API

Không dùng `a8_pin`: starter pack và CarSky OpenAPI không cung cấp helper đó.
Replayer gọi endpoint chính thức
`POST /api/v1/signals/{roomId}/{nodeKey}/actuate`, mỗi frame là một batch ba
KUKSA sensor signal. Request không gửi thuộc tính `actuate`, đúng quy ước
direct-write cho sensor/state simulation trong OpenAPI.

Trước khi chạy, cần deploy Blueprint có KUKSA signal-source node và xác nhận
ba path tồn tại bằng `GET /api/v1/signals/{roomId}/{nodeKey}`. Mặc định:

- `Vehicle.Speed` — dữ liệu đầu vào là km/h.
- `Vehicle.Acceleration.Longitudinal` — dữ liệu đầu vào là m/s².
- `Vehicle.Acceleration.Lateral` — dữ liệu đầu vào là m/s².

Nếu VSS artifact dùng tên khác, truyền ba cờ `--carsky-*-path` tương ứng.
Không bỏ qua bước kiểm tra `unit`, `dataType` và miền giá trị trong metadata.

```bash
export A8_API_KEY='<key chỉ lưu local>'
export A8_ROOM_ID='<device id>'
export A8_NODE_KEY='<key từ signal discovery>'

python3 -m safeloop.replay_telemetry /data/T01d \
  --mode realtime \
  --sink carsky-rest
```

Mỗi frame tạo payload dạng:

```json
{
  "signals": [
    {"path": "Vehicle.Speed", "value": 27.35},
    {"path": "Vehicle.Acceleration.Longitudinal", "value": -0.42},
    {"path": "Vehicle.Acceleration.Lateral", "value": 0.18}
  ]
}
```

Ba metadata `trip_id`, `frame_id`, `timestamp_ms` có thể phát thêm nếu VSS
artifact khai báo custom path; dùng `--carsky-trip-id-path`,
`--carsky-frame-id-path`, `--carsky-timestamp-path`. Hướng dẫn discovery và
xử lý lỗi: `docs/CarSky_REST_Telemetry_Quickstart.md`.
