# Chuẩn bị triển khai SafeLoop native trên CarSky

Tài liệu này mô tả phase 1 an toàn: clone blueprint sạch, giữ nguyên topology có sẵn, nối một SafeLoop addon vào `scriptContent` của node `IVI Gateway` trên clone, rồi tạo deployment trên đúng device trống đã khóa. Công cụ không tự tạo node/pin/edge, không dừng hoặc thay thế deployment đang chạy.

## Phạm vi phase 1

Topology được giữ nguyên:

```text
REAL MODEL hoặc MODEL REPLAY
  -> IVI Gateway script-node
       -> KUKSA pin có sẵn -> Central Broker -> standard VSS
       -> ETHERNET pin có sẵn -> IVI Switch -> UDP 10.99.0.14:48100
                                             -> IVI - Android HMI
```

- Base sạch: `rEg5AvGuzcKQHtihPiW0q`.
- Device đích đã được duyệt: `h5sjc3jtzl8vzl9wyqokv`.
- KUKSA reference: chính room `h5sjc3jtzl8vzl9wyqokv`, node key `central-broker-vss`. Khi room chưa có deployment để đọc metadata live, gate bootstrap bên dưới được áp dụng; schema không bị bỏ qua.
- Phase 1 không dùng container hoặc registry, vì vậy không có image để pull. Đây là lựa chọn có chủ đích, không phải image ngầm định. Nếu mở phase container sau này, image phải được khai báo rõ bằng immutable digest; công cụ phase 1 này sẽ không deploy container.
- HMI transport là UDP trên Ethernet. Screen Widget không được xem là đường truyền dữ liệu.

Các dữ kiện live dùng để khóa thiết kế nằm trong [evidence audit](../reports/evidence/carsky-live-audit-20260810T214419+0700/findings.json).

## Vì sao phải dùng lại IVI Gateway

Blueprint export có pin `ETHERNET`, nhưng OpenAPI hiện tại không liệt kê `ETHERNET` trong enum của thao tác add-pin/import. Do đó không có đường API được chứng minh để tạo node runtime mới rồi nối node đó vào IVI Switch. Không được đổi pin này thành `GENERIC`, vì như vậy không chứng minh được đường truyền HMI thật.

Phase 1 tránh giới hạn API bằng cách dùng lại cả hai edge đã có trên `IVI Gateway`: KUKSA tới broker và Ethernet tới IVI Switch. Addon runtime gọi `pins.kuksa` cho signal và `nydus.net.udp(...)` cho HMI. Pin Ethernet chỉ cấu hình NIC ở tầng kernel, không tạo `pins.eth` trong Lua. Preflight vẫn yêu cầu đúng một KUKSA OUTPUT tên `kuksa` và đúng một ETHERNET OUTPUT tên `eth` đã nối đúng switch. Nếu clone đổi tên pin hoặc làm mất một trong hai edge, preflight từ chối PATCH và deploy.

## Ranh giới dữ liệu của addon

File đưa qua `--gateway-addon` chỉ là phần nối thêm, không chứa/copy script nền khoảng 30 KB. Công cụ GET `scriptContent` hiện tại của node trên clone rồi nối addon sau marker có hash.

Addon bắt buộc có:

```lua
-- SAFELOOP_NATIVE_GATEWAY_ADDON_V1
-- SAFELOOP_SOURCE_MODE: REAL_MODEL
-- hoặc chính xác một marker MODEL_REPLAY
```

File do `tools/generate_carsky_native_replay.py` sinh được nhận diện bằng cặp marker `SAFELOOP_NATIVE_REPLAY_ADDON_V1_BEGIN/END` và được khóa là `MODEL_REPLAY`; không cần chép thêm marker bằng tay.

Generator kiểm kích thước từng JSONL envelope và từng packet sẽ được Lua tạo
ra. Nếu bất kỳ frame nào vượt 1.472 byte, lệnh dừng trước khi tạo addon. Addon
cũng có assertion cùng giới hạn ngay trước `send_to` để fail closed ở runtime.

Ngoài ra addon phải thể hiện KUKSA, UDP, `10.99.0.14` và port `48100`. Các chuỗi cho thấy dùng ground truth, trip aggregate hoặc mock bị từ chối. `MODEL_REPLAY` phải replay output thật đã sinh từ model, không phải nhãn đáp án.

KUKSA contract phase 1 chỉ chấp nhận 10 standard VSS đã có trên stock broker:

- `Vehicle.Speed`
- `Vehicle.Acceleration.Longitudinal`
- `Vehicle.Acceleration.Lateral`
- `Vehicle.ADAS.ObstacleDetection.Front.Center.TimeGap`
- `Vehicle.ADAS.ObstacleDetection.Front.Center.IsWarning`
- `Vehicle.Driver.AttentiveProbability`
- `Vehicle.Driver.DistractionLevel`
- `Vehicle.Driver.FatigueLevel`
- `Vehicle.Driver.IsEyesOnRoad`
- `Vehicle.ADAS.DMS.IsWarning`

Envelope đầy đủ cho dashboard đi qua UDP; không giả vờ rằng custom C3/Drive Quality paths đã được đăng ký trong broker.

### Gate KUKSA khi room đang trống

Room đích phải trống trước khi deploy, nên API signal không có deployment đang
chạy để trả metadata. Trong trường hợp đó, preflight chỉ chấp nhận evidence cục
bộ đã khóa tại
`reports/evidence/carsky-live-audit-20260810T214419+0700/deployments/uQQBHr256M5GEdNzq2GuQ/signals/broker-metadata.json`.
Đây là metadata 1.272 signal đã đọc từ stock broker trong audit chỉ đọc ngày
2026-08-10.

Gate kiểm đồng thời:

- SHA-256 của toàn file phải đúng `a203d81c0bb3a7c03edbe4c2c7e5d1562d76c2f086bee59750d38372f1eaf8c1`;
- JSON không được có key trùng và response envelope phải đúng GET/status/path/node key đã khóa;
- broker trong base phải dùng đúng artifact `zsJexrIgGwIeyk3ODyYoh`, version ID `Cpi8WkMz_bH35uIQFviim`;
- mỗi một trong 10 standard VSS phải xuất hiện đúng một lần và có đúng data type.

Thiếu file, sai một byte, sai format, sai artifact/version, thiếu signal hoặc sai
type đều tạo blocker `KUKSA_BOOTSTRAP_ATTESTATION_INVALID`; công cụ không tiếp
tục bằng cách bỏ qua kiểm schema. Evidence này chứng minh metadata runtime đã
quan sát trước đó gắn với artifact/version mà blueprint khai báo; nó không phải
hash nội dung file artifact VSS. Nếu có reference deployment live, preflight vẫn
ưu tiên discovery và metadata live như trước. Sau khi deployment mới chạy, bắt
buộc đọc lại metadata live trước khi tuyên bố demo end-to-end hoàn tất.

Mỗi envelope UDP phải không quá **1.472 byte UTF-8**. Đây là phần payload còn
lại của Ethernet MTU 1.500 byte sau IPv4 (20 byte) và UDP (8 byte), giúp tránh
phân mảnh IP. `UdpDecisionPublisher`, generator native replay và parser Android
đều từ chối packet vượt giới hạn. Android nhận vào buffer 1.473 byte chỉ để
phát hiện packet quá cỡ rồi loại bỏ, không parse dữ liệu đã bị cắt.

Để giữ đủ contract trong giới hạn này, `health.model_versions` dùng ID rút gọn
gồm 12 ký tự hex đầu của SHA-256 cho C1, C2, C3, Drive Quality và Contextual
Risk. File `carsky_edge_report.json` vẫn lưu SHA-256 đầy đủ, loại digest và
đường dẫn nguồn tương ứng để truy vết; không bỏ metadata khỏi contract.

## Quy trình chỉ đọc

Tạo config làm việc ở thư mục đã được Git-ignore:

```bash
mkdir -p .carsky-build
cp carsky/native-deployment.example.json .carsky-build/native-deployment.json
```

Kiểm tra offline, không cần credential:

```bash
python3 tools/carsky_native_ctl.py validate-config \
  .carsky-build/native-deployment.json \
  --gateway-addon .carsky-build/native-replay/safeloop_t01_native_addon.lua
```

Sau đó nạp credential từ environment và chạy audit GET-only:

```bash
source .carsky.env
python3 tools/carsky_native_ctl.py preflight \
  .carsky-build/native-deployment.json \
  --gateway-addon .carsky-build/native-replay/safeloop_t01_native_addon.lua

python3 tools/carsky_native_ctl.py plan \
  .carsky-build/native-deployment.json \
  --gateway-addon .carsky-build/native-replay/safeloop_t01_native_addon.lua
```

Config không được chứa API key, token, password hoặc credential trong URL. Công cụ chỉ đọc `A8_API_KEY` từ environment và không in key/response body lỗi.

Khi `candidate_blueprint_id` còn `null`, preflight phải báo `CANDIDATE_BLUEPRINT_REQUIRED`. Đây là trạng thái đúng trước khi clone. `plan` chỉ in các request dự kiến; không gửi mutation.

## Clone và PATCH có kiểm soát

Không clone bằng tay. Lệnh sau chỉ chấp nhận đúng base ID đã khóa, chạy preflight mới rồi gửi duy nhất `POST /api/v1/blueprints/rEg5AvGuzcKQHtihPiW0q/clone` với `isSnapshot=false`:

```bash
python3 tools/carsky_native_ctl.py clone-candidate \
  .carsky-build/native-deployment.json \
  --gateway-addon .carsky-build/native-replay/safeloop_t01_native_addon.lua \
  --confirm-base-id 'rEg5AvGuzcKQHtihPiW0q'
```

Lệnh clone từ chối trước khi gửi POST nếu config đã có candidate, base đang live, tên blueprint bị trùng, quota không đủ hoặc device đích đang bận. Kết quả clone phải là direct child editable của base và có cùng hash topology/config sau khi bỏ các ID do CarSky sinh lại. ID mới chỉ được ghi vào `candidate_blueprint_id` sau khi tất cả kiểm tra đạt.

Để tránh sửa nhầm file được track, `clone-candidate` chỉ cập nhật config thật nằm bên trong `.carsky-build/`, từ chối symlink hoặc `carsky/native-deployment.example.json`, rồi dùng atomic replace. Nếu POST clone đã thành công nhưng không xác minh duy nhất được ID/topology, lệnh dừng, không tự clone lần hai và không ghi candidate chưa kiểm chứng.

Sau khi clone thành công:

1. Kiểm tra ID mà lệnh vừa ghi vào working config.
2. Chạy lại `preflight`.
3. Chỉ tiếp tục khi `patch_ready=true` và blocker duy nhất cho deploy là `GATEWAY_ADDON_NOT_APPLIED`.
4. Xác nhận đúng chính ID clone trong lệnh PATCH.

```bash
python3 tools/carsky_native_ctl.py patch-gateway \
  .carsky-build/native-deployment.json \
  --gateway-addon .carsky-build/native-replay/safeloop_t01_native_addon.lua \
  --confirm-candidate-id '<candidate-blueprint-id>'
```

Lệnh này từ chối nếu candidate:

- trùng base;
- là blueprint của một deployment đang chạy;
- không phải direct clone của base;
- là snapshot/locked;
- khác hash script nền của base;
- broker KUKSA không còn đúng cùng artifact ID và version ID với base;
- thiếu node/edge KUKSA hoặc Ethernet;
- đã có marker addon khác/hỏng;
- có node mock hoặc replay probe.

Trước PATCH, toàn bộ config hiện tại của `IVI Gateway` được sao lưu nguyên trạng tại `.carsky-build/carsky-native-backups/`. Payload PATCH chỉ có top-level `config`; các field config không liên quan được giữ nguyên và chỉ `scriptContent` thay đổi. Preflight yêu cầu gateway vốn đã ở chế độ `script=inline`. Sau PATCH, công cụ GET lại clone, so hash rồi gọi validate blueprint.

Nếu server trả `404` cho route PATCH node, công cụ dừng và không tự thử route khác. Bên cạnh file backup, công cụ tạo nguyên tử một file `*.manual.lua` quyền riêng tư `0600`, chứa chính xác **toàn bộ** script nền + ranh giới/hash + addon. Công cụ đọc lại file này, xác minh SHA-256, rồi báo đầy đủ đường dẫn file, candidate blueprint ID, IVI Gateway node ID và hash dự kiến.

Khi fallback này xảy ra, trên CarSky UI chỉ mở đúng candidate blueprint ID được báo, chọn đúng node `IVI Gateway`, rồi thay **toàn bộ Script Content** bằng nội dung file `*.manual.lua`. Không dán riêng addon, không sửa base và không sửa blueprint/snapshot đang chạy. Sau khi Save, chạy lại `preflight`; chỉ deploy khi `addon_applied=true`, hash readback trùng `expected_script_sha256` và `deploy_ready=true`. File manual dùng cùng stem duy nhất với backup và công cụ từ chối ghi đè nếu artifact đó đã tồn tại.

Chạy lại cùng addon sau khi đã áp dụng chính xác là idempotent và tạo `mutation_calls=0`.

## Gate triển khai

`deploy_ready=true` chỉ có khi:

- platform/CPU và quota đủ;
- device đích published, unlocked và chưa có deployment;
- candidate clone hợp lệ, inactive và editable;
- topology KUKSA + IVI Ethernet còn nguyên;
- KUKSA reference live hoặc bootstrap attestation đã khóa hash chứng minh đúng artifact/version và kiểu của 10 standard VSS;
- addon đúng hash đã xuất hiện trên candidate.

Sau khi review kết quả preflight mới nhất, triển khai bằng hai xác nhận độc lập:

```bash
python3 tools/carsky_native_ctl.py deploy \
  .carsky-build/native-deployment.json \
  --gateway-addon .carsky-build/native-replay/safeloop_t01_native_addon.lua \
  --confirm-candidate-id '<candidate-blueprint-id>' \
  --confirm-device-id 'h5sjc3jtzl8vzl9wyqokv'
```

`deploy` không tin kết quả preflight cũ: nó tự chạy lại toàn bộ GET gate, đọc target thêm một lần ngay trước mutation, POST-validate candidate, rồi mới gửi đúng một `POST /api/v1/deployments` với candidate/device/name trong config. Tên deployment trùng, candidate đang live, target bận hoặc quota hết đều làm lệnh dừng trước create. Không có code path DELETE, stop, restart hoặc replace deployment.

Sau create, lệnh poll bounded endpoint trạng thái của đúng device (mặc định tối đa 30 lần, cách 2 giây) và trả `deployment_id`, `candidate_blueprint_id`, `target_device_id`, trạng thái cuối cùng cùng namespace. Có thể đổi giới hạn bằng `--poll-attempts` và `--poll-interval`; tối đa lần poll bị khóa ở 120. Exit code `0` chỉ khi đã thấy `RUNNING`; exit code `6` nghĩa là deployment đã được tạo nhưng poll kết thúc khi chưa thấy `RUNNING`. Không chạy lại create một cách mù quáng: lượt chạy sau sẽ thấy target bận và từ chối tạo duplicate.

ADB shell/container exec hiện còn bị chặn bởi Conduit `502 SERVICE_UNAVAILABLE`. UDP tới Android qua IVI Ethernet là data path thiết kế; vẫn cần chứng minh receiver HMI đang lắng nghe port `48100` trong lượt triển khai thật trước khi tuyên bố demo end-to-end hoàn tất.
