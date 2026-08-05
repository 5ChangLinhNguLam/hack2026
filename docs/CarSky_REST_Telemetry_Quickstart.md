# CarSky REST Telemetry — Quickstart

Hướng dẫn này bám theo `docs/carsky-openapi.json` (A8 API 1.0.0). Mục tiêu là
phát ego telemetry từ `TripReplayer` vào một signal-source node đang chạy trên
CarSky, không phụ thuộc helper `a8_pin`.

## 1. Chuẩn bị trên giao diện CarSky

1. Đăng nhập CarSky Rework UI.
2. Bấm biểu tượng bánh răng góc dưới trái → **Credentials** → **+ New**.
3. Nhập Credential name → **Create** → **Copy key** ngay khi hiện. Key chỉ
   được hiển thị một lần; không commit key.
4. Trong Nydus, mở Blueprint có **KUKSA Broker/Signal Source** với VSS artifact
   chứa các ego signal cần phát.
5. Ở Blueprint Inspector → **Deployments** → **New Deployment** → chọn Device
   hoặc **Create new device** → **Deploy**.
6. Chờ deployment hiện `Running` và `n/n nodes ready` trước khi discovery.

> **Hai “starter pack” khác nhau:** Blueprint **Started pack** thấy trong Nydus
> là demo platform của CarSky, có sẵn `Central Broker (VSS)` và nhiều node xe.
> Nó không phải repo Hackathon Starter Kit đang chứa `tripkit`. Có thể dùng
> Blueprint đó để khảo sát signal, nhưng artifact/deployment nộp bài nên thuộc
> quyền quản lý của đội và ghi rõ phần nào kế thừa.

Ba path mặc định của replayer là:

```text
Vehicle.Speed
Vehicle.Acceleration.Longitudinal
Vehicle.Acceleration.Lateral
```

Nếu Blueprint dùng path khác thì giữ nguyên path của Blueprint; ta sẽ cấu hình
replayer theo kết quả discovery ở bước 3.

## 2. Đặt credential an toàn

Chạy trong terminal hiện tại. Không đặt key trực tiếp trong tham số CLI vì có
thể lưu vào shell history.

```bash
export A8_URL='https://hackathon-1.carsky.io'
export A8_API_KEY='<API key vừa tạo>'
```

Credential được phân vùng theo tenant. Key tạo tại Hackathon 1 không xác thực
được với `https://carsky.io`; dùng sai domain trả `401 Invalid or revoked
credential` dù key vẫn còn hiệu lực.

Kiểm tra xác thực:

```bash
curl --fail-with-body --silent --show-error \
  -H "X-API-Key: ${A8_API_KEY}" \
  "${A8_URL}/api/v1/healthz"
```

## 3. Discovery, không đoán ID/path

Liệt kê Device; theo OpenAPI lấy `data[].id` của đúng Device đã deploy làm
`roomId`. Trên UI, cùng ID này nằm ngay dưới tên Device và tại
**Deployment Inspector → Device ID**:

```bash
curl --fail-with-body --silent --show-error \
  -H "X-API-Key: ${A8_API_KEY}" \
  "${A8_URL}/api/v1/devices?limit=100"

export A8_ROOM_ID='<id của Device đã deploy>'
```

Liệt kê signal-source node trong room. Lấy `nodes[].key` của node có
`kind: "kuksa"`; không dùng node name từ deployment-node API:

```bash
curl --fail-with-body --silent --show-error \
  -H "X-API-Key: ${A8_API_KEY}" \
  "${A8_URL}/api/v1/signals/${A8_ROOM_ID}"

export A8_NODE_KEY='<nodes[].key của KUKSA node>'
```

Signal Widget có thể giúp nhận diện live part, ví dụ ảnh cho thấy
`central-broker-vss-signal`, `drive-controls-signal`. Tuy nhiên không tự cắt
hậu tố để đoán `nodeKey`: dùng đúng `nodes[].key` do API trả về. OpenAPI cũng
cho phép truyền full part name khi một node có nhiều signal source.

Liệt kê metadata signal và xác nhận chính xác `path`, `dataType`, `unit`,
`min/max/allowed`:

```bash
curl --fail-with-body --silent --show-error \
  -H "X-API-Key: ${A8_API_KEY}" \
  "${A8_URL}/api/v1/signals/${A8_ROOM_ID}/${A8_NODE_KEY}"
```

Dataset dùng km/h cho speed và m/s² cho acceleration. Nếu metadata CarSky dùng
đơn vị khác, phải chuyển đổi trong mapping trước khi phát.

### Kết quả discovery của 5ChangLinhNguLam — 03/08/2026

- Tenant: `https://hackathon-1.carsky.io`.
- Device: `5ChangLinhNguLam`.
- `roomId`: `ehlbxyyc4px2it-cip50w`.
- KUKSA `nodeKey`: `central-broker-vss`.
- Live part: `central-broker-vss-signal`.
- Tổng VSS signal: 1.272.
- `Vehicle.Speed`: `float`, `km/h`, `sensor`.
- `Vehicle.Acceleration.Longitudinal`: `float`, `m/s^2`, `sensor`.
- `Vehicle.Acceleration.Lateral`: `float`, `m/s^2`, `sensor`.

Snapshot evidence không chứa credential:
`reports/evidence/carsky-discovery-20260803.json`.

## 4. Smoke test một signal

Chỉ làm sau khi path đã xuất hiện trong response discovery:

```bash
curl --fail-with-body --silent --show-error \
  -X POST \
  -H "X-API-Key: ${A8_API_KEY}" \
  -H 'Content-Type: application/json' \
  "${A8_URL}/api/v1/signals/${A8_ROOM_ID}/${A8_NODE_KEY}/actuate" \
  -d '{"path":"Vehicle.Speed","value":20.0}'
```

Với KUKSA sensor/state simulation, không gửi `"actuate": true`. Theo OpenAPI,
cờ đó chỉ dùng để điều khiển actuator target thông qua provider đã đăng ký.

Đọc lại giá trị:

```bash
curl --fail-with-body --silent --show-error \
  -X POST \
  -H "X-API-Key: ${A8_API_KEY}" \
  -H 'Content-Type: application/json' \
  "${A8_URL}/api/v1/signals/${A8_ROOM_ID}/${A8_NODE_KEY}/values" \
  -d '{"paths":["Vehicle.Speed"]}'
```

### Kết quả smoke test — 03/08/2026

Đã chạy trên Device `5ChangLinhNguLam`, node `central-broker-vss`:

1. Read-before: `Vehicle.Speed = 0`, HTTP 200.
2. Direct-write `20.0 km/h`: `{"ok":true,"sent":1}`, HTTP 200.
3. Readback: `Vehicle.Speed = 20`, assertion pass.
4. Restore `0 km/h`: `{"ok":true,"sent":1}`, HTTP 200.
5. Readback sau restore: `Vehicle.Speed = 0`, assertion pass.

Evidence: `reports/evidence/carsky-signal-smoke-20260803.json`. Đây là bằng
chứng đường REST → live KUKSA broker, chưa phải latency E2E của SafeLoop.

## 5. Phát ba ego signal từ trip

Chạy 5 frame nhanh trước:

```bash
python3 -m safeloop.replay_telemetry data/T01-Sample \
  --sink carsky-rest \
  --limit 5
```

Không dùng REST sink làm đường 20 FPS production. Có thể chạy `realtime` để
kiểm tra pacing/behavior trên một đoạn ngắn, nhưng mỗi frame vẫn là một HTTP
request và runtime thực tế phải được đo:

```bash
python3 -m safeloop.replay_telemetry data/T01-Sample \
  --mode realtime \
  --sink carsky-rest
```

Nếu path khác mặc định:

```bash
python3 -m safeloop.replay_telemetry data/T01-Sample \
  --mode realtime \
  --sink carsky-rest \
  --carsky-speed-path '<speed path từ discovery>' \
  --carsky-longitudinal-accel-path '<longitudinal path>' \
  --carsky-lateral-accel-path '<lateral path>'
```

Mỗi frame là một request batch ba signal, dưới giới hạn 64 signal/request của
OpenAPI. Sink kiểm tra cả ba path tồn tại trước khi phát frame đầu tiên và xác
nhận response `{"ok":true,"sent":3}`.

### Kết quả replay live — 03/08/2026

Đã phát frame `10..14` của `T01-Sample` vào deployment chính:

- 5/5 message thành công, mỗi message 3 signal.
- Publish elapsed `1,537 s`, chỉ khoảng `3,3 message/s`.
- Tổng CLI wall time kể cả signal discovery/validation: `3,79 s`.
- Readback frame cuối khớp source trong sai số float32:
  `9,19 km/h`, `8,24 m/s²`, `-0,091 m/s²`.
- Yêu cầu dataset là 20 FPS nên REST sink **không đạt throughput data plane**.

Evidence: `reports/evidence/carsky-replay-5frames-20260803.json`.

### Restore float: không dùng `null`

OpenAPI mô tả signal scalar có thể là `null`, nhưng KUKSA float parser trên
tenant này từ chối `null`. Batch restore đã áp dụng phần tử đầu rồi mới lỗi ở
phần tử thứ hai và server ghi rõ batch không atomic. Vì vậy:

1. Dùng safe-state tường minh như `0.0` cho float.
2. Không coi batch actuation là transaction.
3. Nếu nhiều output có quan hệ safety, thêm sequence/trace ID, validity và TTL
   ở consumer thay vì dựa vào tính atomic của request.

Sau test, ba ego signal đã được xác minh về `0/0/0`.

## 6. Trace frame tùy chọn

Để giữ `trip_id`, `frame_id`, `timestamp_ms` trên CarSky, VSS artifact phải có
ba custom signal phù hợp. Sau đó thêm:

```bash
--carsky-trip-id-path 'SafeLoop.TripId' \
--carsky-frame-id-path 'SafeLoop.FrameId' \
--carsky-timestamp-path 'SafeLoop.TimestampMs'
```

Không bật các cờ trên nếu custom path chưa có trong signal metadata.

## 7. Lỗi thường gặp

- `401`: key sai/hết hạn/đã revoke; tạo lại trong **Credentials**.
- `404`: thường chọn sai `roomId` hoặc dùng deployment node name thay vì
  `nodes[].key` từ Signal discovery.
- `Signal path chưa có`: VSS artifact không có path hoặc path khác tên; đọc lại
  metadata và truyền `--carsky-*-path`.
- `400 unknown path`: node đang chạy nhưng payload dùng path không thuộc node.
- `502 ... batch is not atomic`: một phần batch có thể đã được áp dụng; đọc
  lại từng path rồi đưa về safe-state tường minh.
- `400 Periodic ... not supported`: endpoint cyclic-send tồn tại nhưng signal
  source hiện tại không triển khai; dùng Script Node timer hoặc Container.
- Log Script Node báo phải chọn container: thêm query
  `container=script-node`, vì pod có cả `script-node` và `sidecar`.
- Connection timeout: deployment chưa ready hoặc endpoint không reachable.
- Không dùng `--carsky-skip-signal-validation` trong demo/chấm điểm; cờ này chỉ
  dành cho chẩn đoán.

## 8. Đường 20 Hz đã kiểm chứng trong room

Để tách khỏi giới hạn public REST, đội đã tạo blueprint
`SafeLoop Integration Lab` với hai Script Node nối cùng KUKSA broker:

- `SafeLoop 20Hz Replay Probe`: dùng `timer.periodic(50, callback)`, phát 20
  frame T01-Sample (`10..29`) rồi neutralize một lần.
- `SafeLoop Telemetry Observer`: subscribe ba ego path và log thay đổi.

Deployment lab đạt 23/23 node Running. Publisher ghi đủ 20 frame/60 signal
update; observer độc lập nhận đủ 60/60 update và thêm đúng 3 update neutralize.
Readback cuối là `0/0/0`. Sau capture, deployment lab được teardown để giải
phóng slot; blueprint và Device vẫn được giữ để redeploy.

Source và evidence:

- `carsky/scripts/safeloop_20hz_replay_probe.lua`
- `carsky/scripts/safeloop_telemetry_observer.lua`
- `carsky/blueprints/SafeLoop-Integration-Lab-20260803.json`
- `reports/evidence/carsky-script-node-20hz-probe-20260803.json`

Đây là bằng chứng timer 50 ms và delivery đủ/đúng thứ tự, chưa phải đo latency
p95/p99 vì timestamp log hiện chỉ có độ phân giải giây. Đường production tiếp
theo nên là Container Node chạy replayer/model qua KUKSA/a8_pin nội bộ; REST
giữ vai trò control plane, smoke test và evidence automation.
