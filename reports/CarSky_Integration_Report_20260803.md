# Báo cáo tích hợp CarSky — 5ChangLinhNguLam / SafeLoop

Ngày kiểm chứng: 03/08/2026  
Tenant: `https://hackathon-1.carsky.io`  
Phạm vi: discovery, telemetry REST, KUKSA, widget, blueprint export, Script Node, deployment lab, observability, failure paths và giới hạn runtime.

## 1. Kết luận điều hành

Đường CarSky đầu tiên của SafeLoop đã chạy thật, có log và có thể tái tạo:

1. Repo phát telemetry `safeloop.telemetry.v1` vào KUKSA của deployment chính qua REST.
2. Một Integration Lab riêng được clone từ blueprint của đội, bổ sung hai Script Node do đội sở hữu.
3. `SafeLoop 20Hz Replay Probe` dùng `timer.periodic(50, callback)` phát 20 frame liên tiếp, tổng cộng 60 signal update.
4. `SafeLoop Telemetry Observer` độc lập nhận đủ 60/60 update, sau đó nhận thêm đúng 3 update neutralize.
5. Lab đạt `RUNNING 23/23`, bằng chứng được chụp rồi deployment lab được teardown để giải phóng slot. Deployment chính vẫn `RUNNING`.

Mức claim hợp lệ hiện tại là **CarSky platform thin-slice end-to-end**: nguồn team-owned → KUKSA → consumer team-owned, có replay và observability. Chưa được gọi là full SafeLoop core vì chưa có perception/risk policy, command gateway/back-to-car và đo latency p95/p99 trong cùng trace.

REST không phù hợp làm data plane 20 Hz. Replay 5 frame đạt khoảng `3,3 message/s`; API cyclic-send có trong OpenAPI nhưng KUKSA source đang chạy trả `400 not supported`. Hướng đúng là chạy replayer/policy trong Script Node hoặc Container Node, còn REST dùng cho discovery, smoke test, bounded control và thu evidence.

## 2. Trạng thái cuối sau thao tác

| Đối tượng | Trạng thái cuối |
|---|---|
| Device chính | `5ChangLinhNguLam` — `PUBLISHED`, `PRIVATE` |
| Deployment chính | `5ChangLinhNguLam-deploy` — `RUNNING`, 21/21 node |
| Namespace chính | `room-v63yvkik` |
| Blueprint chính | 21 node / 21 edge; đã export vào repo |
| KUKSA chính | `central-broker-vss`, 1.272 VSS signal |
| Widget chính | `SafeLoop Ego Telemetry`, read-only Signal Widget |
| Integration Lab blueprint | 23 node / 23 edge; validate pass; vẫn được giữ |
| Integration Lab deployment | Đã teardown sau evidence; 0 active deployment |
| Slot deployment thứ hai | Đã được giải phóng |
| Ba ego signal sau test | `0 / 0 / 0` |
| Credential | Không nằm trong repo/evidence |

## 3. Inventory hệ thống chính

### 3.1 Device và deployment

- Device name: `5ChangLinhNguLam`.
- `roomId`: `ehlbxyyc4px2it-cip50w`.
- Deployment ID: `7KpCh37iq_1A-vkxJf91n`.
- Blueprint ID: `wVMseHi7XtUhuMIn_1D5e`.
- Deployment status sau toàn bộ kiểm chứng: `RUNNING`.
- 21/21 node ở phase `Running`.

Phân bố node:

| Node type | Số lượng |
|---|---:|
| Script Node | 8 |
| GPIO Panel | 4 |
| CAN Bus | 2 |
| Ethernet Bridge | 2 |
| Container | 2 |
| KUKSA Databroker | 1 |
| Device Proxy | 1 |
| Skycraft | 1 |

Hai Container Node có sẵn đang dùng image `localhost:5000/tcu-demo/tcu-nad:demo`; đây không phải image SafeLoop.

### 3.2 KUKSA / VSS

- Node key dùng với Signal API: `central-broker-vss`.
- Live part dùng cho Signal Widget: `central-broker-vss-signal`.
- KUKSA Databroker log báo version `0.7.1-dev.0`.
- Metadata được nạp từ `/vss/vss.json`.
- Runtime broker công bố 1.272 signal qua API; Script Node client lấy 1.267 signal metadata. Chênh lệch này được giữ nguyên như quan sát runtime, không tự quy đồng.
- Kết nối nội bộ của broker dùng Unix/gRPC sidecar; TLS và auth nội bộ tắt trong room. Xác thực REST bên ngoài vẫn dùng API key của tenant.

Ba path SafeLoop đã xác nhận:

| Path | Type | Unit | Entry type |
|---|---|---|---|
| `Vehicle.Speed` | float | km/h | sensor |
| `Vehicle.Acceleration.Longitudinal` | float | m/s² | sensor |
| `Vehicle.Acceleration.Lateral` | float | m/s² | sensor |

## 4. Các kiểm chứng đã thực hiện

### 4.1 Discovery và smoke test một signal

Đã chạy discovery hoàn toàn từ API, không đoán `roomId`, node key hoặc signal path. Smoke test `Vehicle.Speed`:

1. Read-before: `0`.
2. Write: `20.0 km/h`, response `ok=true, sent=1`.
3. Readback: `20`, assertion pass.
4. Restore: `0`, response pass.
5. Readback: `0`, assertion pass.

Evidence: `reports/evidence/carsky-discovery-20260803.json` và `reports/evidence/carsky-signal-smoke-20260803.json`.

### 4.2 Replay 5 frame từ repo vào deployment chính

Đã chạy:

```bash
python3 -m safeloop.replay_telemetry data/T01-Sample \
  --sink carsky-rest --start 10 --limit 5
```

Kết quả:

- 5 message, frame `10..14`.
- Mỗi message là batch 3 signal.
- Publish elapsed `1,537 s`, khoảng `3,3 message/s`.
- Tổng wall time kể cả metadata validation: `3,79 s`.
- Readback frame cuối trùng dữ liệu nguồn trong sai số float32:
  - Speed: `9.1899995803833` so với `9.19`.
  - Longitudinal: `8.239999771118164` so với `8.24`.
  - Lateral: `-0.09099999815225601` so với `-0.091`.

Kết luận: mapping, write và readback đúng; throughput REST không đạt 20 Hz.

Evidence: `reports/evidence/carsky-replay-5frames-20260803.json`.

### 4.3 Script Node observer trong Integration Lab

Đã clone blueprint chính thành `SafeLoop Integration Lab` và thêm Script Node read-only:

- Node: `SafeLoop Telemetry Observer`.
- Source: `carsky/scripts/safeloop_telemetry_observer.lua`.
- KUKSA pin nối vào Central Broker.
- Runtime log xác nhận:
  - metadata fetch thành công;
  - VSS tree được tạo;
  - script loaded;
  - subscribe đúng ba path;
  - observer ready.

Khi phát 5 frame REST vào lab, observer ghi đủ 15 thay đổi, gồm chính xác ba giá trị của từng frame.

### 4.4 Probe 20 frame trong room

Đã thêm Script Node thứ hai:

- Node: `SafeLoop 20Hz Replay Probe`.
- Source: `carsky/scripts/safeloop_20hz_replay_probe.lua`.
- Timer: `timer.periodic(50, callback)`.
- Input: T01-Sample frame `10..29`.
- Sau frame cuối, probe publish `0/0/0` một lần rồi mọi timer callback sau là no-op.

Kết quả runtime:

| Check | Quan sát | Kết quả |
|---|---:|---|
| Blueprint validation | 23 node / 23 edge, 0 error | Pass |
| Deployment readiness | 23/23 Running | Pass |
| Publisher frame log | 20/20, đúng thứ tự 10..29 | Pass |
| Signal update phát | 60 | Pass |
| Observer replay update | 60/60 | Pass |
| Observer neutralize update | 3/3 | Pass |
| Readback cuối | 0 / 0 / 0 | Pass |
| Teardown lab | `ok=true`, active deployment còn 0 | Pass |
| Deployment chính sau cleanup | `RUNNING` | Pass |

Đây là bằng chứng strongest hiện có cho platform integration. Timer được cấu hình 50 ms và 20 frame hoàn tất trong khoảng log giây tương ứng, nhưng timestamp log chỉ có độ phân giải giây; chưa được dùng để claim jitter hoặc p95/p99 latency.

Evidence: `reports/evidence/carsky-script-node-20hz-probe-20260803.json`.

## 5. Failure paths và contract drift

### 5.1 Unknown signal path

Write vào `SafeLoop.DoesNotExist` trả `HTTP 400 VALIDATION_ERROR`. Ba signal hợp lệ vẫn giữ nguyên `0/0/0`. Đây là fail-closed ở bước validation.

### 5.2 `null` trên float và batch không atomic

OpenAPI cho phép scalar `null`, nhưng KUKSA float parser đang chạy từ chối `null`. Khi restore batch `[Speed=0, Longitudinal=null, Lateral=null]`, server đã áp dụng Speed rồi mới fail ở phần tử thứ hai và ghi rõ `1 of 3 already applied — batch is not atomic`.

Quy tắc vận hành rút ra:

- Không dùng `null` để neutralize float signal.
- Dùng giá trị safe-state tường minh như `0.0` theo contract.
- Không coi batch actuation là transaction.
- Nếu nhiều output liên quan safety, consumer phải dùng sequence/trace ID, validity và TTL; không dựa vào tính atomic của REST batch.

### 5.3 Periodic-send không hỗ trợ KUKSA source này

`POST .../periodic/start` với `Vehicle.Speed=12.34`, `intervalMs=50` trả `HTTP 400`: `Periodic (cyclic) send is not supported by this signal source`. Danh sách active path vẫn rỗng và giá trị cuối đã được neutralize.

### 5.4 Drift giữa OpenAPI và runtime route

Route được mô tả `POST /api/v1/nodes/{nodeId}/pins` trả runtime `404 Route not found`. Fallback qua atomic blueprint batch API hoạt động và đã được dùng để thêm pin/edge.

### 5.5 Log của pod nhiều container

Log API không truyền container trả upstream `500/502` vì Script Node pod có hai container `script-node` và `sidecar`. Thêm query `container=script-node` đọc log thành công.

### 5.6 Container execution/build

- `container-exec` trên Container Node có sẵn trả `503 SERVICE_UNAVAILABLE: Conduit service not configured`.
- Máy làm việc có Docker client `29.6.1` nhưng không có quyền vào `/var/run/docker.sock`.
- Không có Podman, Buildah, nerdctl hoặc công cụ build thay thế.

Vì vậy phiên làm việc này không thể build/push SafeLoop Container image. Không có claim giả rằng Container core đã chạy.

Evidence tổng hợp: `reports/evidence/carsky-failure-paths-20260803.json`.

## 6. Tài nguyên và giới hạn tenant

| Limit | Giá trị |
|---|---:|
| Nodes / blueprint | 30 |
| Blueprints / account | 20 |
| Concurrent deployments / account | 2 |
| Devices | 5 |
| Skycraft / blueprint | 2 |
| CPU warning threshold | 80% |
| CPU sample interval | 5.000 ms |

Lab dùng 23/30 node và trong lúc test chiếm deployment slot thứ hai. Sau capture evidence, lab được teardown nên hiện chỉ deployment chính chiếm slot.

## 7. Artifact được tạo

### 7.1 Runtime/source

- `safeloop/telemetry.py`: contract và CarSky REST sink.
- `safeloop/replay_telemetry.py`: CLI phát telemetry.
- `carsky/scripts/safeloop_telemetry_observer.lua`: observer KUKSA read-only.
- `carsky/scripts/safeloop_20hz_replay_probe.lua`: probe 20 frame dùng timer 50 ms.

### 7.2 Blueprint

- `carsky/blueprints/5ChangLinhNguLam-base-20260803.json`: export blueprint chính 21/21.
- `carsky/blueprints/SafeLoop-Integration-Lab-20260803.json`: export lab 23/23, đã validate.

### 7.3 Evidence

- `carsky-discovery-20260803.json`.
- `carsky-signal-smoke-20260803.json`.
- `carsky-replay-5frames-20260803.json`.
- `carsky-script-node-20hz-probe-20260803.json`.
- `carsky-failure-paths-20260803.json`.
- `carsky-platform-inventory-20260803.json`.
- `carsky-artifact-manifest-20260803.sha256`: SHA-256 của source, blueprint,
  evidence và bản báo cáo tiến độ v2.

Tất cả evidence đều loại credential.

## 8. Kiến trúc nên chốt cho SafeLoop

```text
Trip / live sensors
      │
      ▼
Team Container Replayer ──20 Hz local KUKSA/a8_pin──► Central Broker
                                                      │
                         ┌────────────────────────────┴─────────────────────┐
                         ▼                                                  ▼
              Team Risk/Policy Container                         Evidence Observer
                         │                                                  │
                         ▼                                                  ▼
              Command Gateway + TTL                              trace/log/dashboard
                         │
                         ▼
               simulated back-to-car
```

Phân vai transport:

- **REST/OpenAPI:** discovery, metadata validation, deployment automation, smoke test, bounded actuation, status/log/evidence.
- **Script Node:** bridge nhỏ, observer, fault injection, deterministic test probe, logic không cần thư viện nặng.
- **Container Node:** production replayer, model inference, policy/oracle, latency instrumentation và service/dashboard.
- **KUKSA/VSS:** contract trung tâm giữa source, policy, command gateway và observer.

Không nên đưa inference hoặc 20 Hz trip replay qua public REST từng frame.

## 9. Việc tiếp theo để đạt barem Platform/E2E

### P0 — Team-owned Container data plane

1. Dùng máy có Docker daemon hoặc CI để build image từ repo.
2. Login Zot Registry bằng registry API key riêng, không dùng CarSky OpenAPI key làm registry password.
3. Push image và pin immutable digest trong blueprint.
4. Container nối KUKSA pin, phát đủ trip theo timestamp tuyệt đối ở 20 Hz.

### P0 — Full core trace

1. Thêm Risk/Policy node team-owned.
2. Input không chứa ground truth.
3. Output tối thiểu gồm `trace_id`, `frame_id`, `risk`, `action`, `valid_until` và reason code.
4. Command Gateway phải fail-silent khi mất heartbeat/TTL hết hạn.
5. Observer/oracle phải chứng minh normal, late, duplicate, out-of-order và disconnect.

### P0 — Latency evidence

Đo clock monotonic tại ingest, policy start/end, command publish và consumer receive. Báo cáo n, p50, p95, p99, max và warm/cold run. Probe hiện tại không thay thế latency test này.

### P1 — Demo và report

- Đưa Signal Widget hiện có vào kịch bản demo.
- Lưu log của publisher, policy, gateway và observer theo cùng trace ID.
- Quay demo gồm happy path và ít nhất một failure path.
- Chỉ nâng claim C2 từ platform thin-slice lên full E2E sau khi command consumer và oracle đã pass.

## 10. Claim được phép dùng trong báo cáo

Được phép:

- “Đã discovery, write/read/restore live KUKSA trên Device của đội.”
- “Đã deploy Integration Lab 23 node và chạy hai Script Node team-owned.”
- “Probe cấu hình timer 50 ms phát 20 frame; observer độc lập nhận đủ 60/60 update.”
- “Đã kiểm chứng unknown path, null-float partial batch và periodic unsupported.”
- “Blueprint, source và evidence có thể tái tạo; lab được teardown sau test.”

Chưa được phép:

- “SafeLoop core đã chạy full E2E.”
- “Back-to-car dưới 200 ms.”
- “REST đạt 20 Hz.”
- “Container SafeLoop đã build/deploy.”
- “Đã đạt production safety/AEB compliance.”

## 11. Cách redeploy lab

Blueprint `SafeLoop Integration Lab` và Device `Inspiring Spence` vẫn tồn tại. Khi cần xem lại:

1. Mở Nydus → Blueprint `SafeLoop Integration Lab`.
2. Validate topology.
3. New Deployment → chọn Device `Inspiring Spence`.
4. Chờ 23/23 node Running.
5. Probe tự chạy một lần, phát frame 10..29 và neutralize.
6. Đọc log hai Script Node; với OpenAPI log route nhớ dùng `container=script-node`.
7. Teardown deployment sau khi capture evidence để trả slot thứ hai.

Không cần tạo lại node, pin hoặc edge.

## 12. Addendum 04/08/2026 — Mock C1 + C2 + Risk Fusion live

Đã dựng và deploy một mock product slice riêng trong Blueprint
`SafeLoop Mock Integration Lab`:

- Blueprint ID: `f0975e76-6c6b-40b5-9441-8a2dd0edcd5d`.
- Node team-owned: `SafeLoop Mock C1 C2 Fusion`.
- Deployment ID: `e691c66a-8837-4512-9133-6e4f63c88271`.
- Device/room: `h5sjc3jtzl8vzl9wyqokv`.
- Namespace: `room-nckuqalz`.
- Trạng thái: `RUNNING`.
- Blueprint validation: pass, mock KUKSA output đã nối Central Broker.

Script Node chạy scenario deterministic 16 giây, timer cấu hình 50 ms, gồm
normal → distracted → drowsy/microsleep → recovery. Mọi log/contract đều đánh
dấu `MOCK`; core không đọc ground truth và không gửi lệnh tới actuator thật.

Live readback ba mẫu cách nhau 600 ms tại pha microsleep:

| Signal | Mẫu 1 | Mẫu 2 | Mẫu 3 |
|---|---:|---:|---:|
| Speed (km/h) | 40,0347 | 40,0154 | 40,0039 |
| C1 TimeGap (ms) | 905 | 870 | 835 |
| C1 Distance (m) | 10,0643 | 9,6704 | 9,2787 |
| C1 IsWarning | true | true | true |
| C2 AttentiveProbability (%) | 5 | 5 | 5 |
| C2 FatigueLevel (%) | 98 | 98 | 98 |
| C2 IsEyesOnRoad | false | false | false |
| DMS IsWarning | true | true | true |

Toàn bộ 11 signal dự kiến đều hiện diện; `TimeGap`, `Distance` và ba ego
kinematic signal thay đổi giữa các mẫu. Runtime verifier kết luận
`mock_live=true`, `missing_paths=[]`.

Claim mới được phép dùng: **mock platform E2E của C1 + C2 + Risk Fusion đã
chạy live qua KUKSA trên CarSky**. Vẫn chưa được gọi là inference model thật,
back-to-car thật hoặc latency p95/p99 đã kiểm chứng.

Evidence:
`reports/evidence/carsky-mock-c1-c2-runtime-20260804.json`.
