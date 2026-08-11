# Luồng SafeLoop suy luận trực tiếp trên CarSky

## Kết luận hiện tại

Bản đang chạy trên CarSky **chưa phải suy luận trực tiếp**. Nó phát lại theo
thời gian thực các kết quả C1, C2 và C3 đã được tính trước từ `T01-Sample`.
Các giá trị đó là kết quả thật của pipeline, không phải ground truth và không
phải số ngẫu nhiên, nhưng model không xử lý ảnh ngay trong lúc CarSky phát
stream.

Mục tiêu đúng là: CarSky nhận ảnh và telemetry, SafeLoop xử lý từng frame ngay
lúc đó, rồi gửi kết quả mới tính tới ứng dụng Android native. KUKSA chỉ là
đường lưu và quan sát tín hiệu phụ; lỗi hoặc chậm ở KUKSA không được làm chậm
cảnh báo trên Android.

## Phân biệt bản hiện tại và bản mục tiêu

| Nội dung | Bản đang chạy | Bản mục tiêu |
|---|---|---|
| Nguồn | `T01-Sample` | Camera thật, simulator hoặc nguồn test đã ghi |
| Ảnh trên CarSky | Không truyền qua pin VIDEO | Hai stream VIDEO: camera trước và camera tài xế |
| C1/C2/C3 | Đã chạy trước, sau đó phát lại output | Chạy ngay khi frame tới SafeLoop Container |
| Android | Nhận UDP từ danh sách output có sẵn | Nhận UDP từ kết quả vừa được tính |
| KUKSA | Phát các signal mirror có sẵn | Nhận ego input và ghi output mirror bất đồng bộ |
| Tuyên bố hợp lệ | `MODEL_OUTPUT_REPLAY` | `LIVE_MODEL` |

`LIVE_MODEL` có nghĩa là model chạy tại thời điểm nhận dữ liệu. Nó không tự
động có nghĩa là camera vật lý. Chế độ nguồn phải được hiển thị riêng:

- `LIVE_CAMERA + LIVE_MODEL`: camera vật lý và suy luận trực tiếp.
- `SIMULATOR + LIVE_MODEL`: simulator và suy luận trực tiếp.
- `RECORDED_SOURCE + LIVE_MODEL`: phát stream ảnh đã ghi nhưng model vẫn chạy
  trực tiếp. Đây chỉ là nguồn kiểm thử, không được gọi là demo camera thật.
- `MODEL_OUTPUT_REPLAY`: chỉ phát lại kết quả đã tính trước, chính là bản hiện
  tại.

## Topology mục tiêu

```text
Nguồn camera trước ── VIDEO 20 Hz ─────┐
                                       │
Nguồn camera tài xế ─ VIDEO 20 Hz ─────┼──► SafeLoop Container
                                       │      ├─ C1: 10 Hz, dùng camera trước
Ego telemetry ─────── KUKSA 20 Hz ─────┘      ├─ C2: 20 Hz, dùng camera tài xế
                                              ├─ C3 + Drive Quality: 20 Hz
                                              └─ tạo một decision envelope/frame
                                                         │
                               đường chính, ưu tiên ─────┼──► UDP :48100
                                                         │    qua IVI Switch
                                                         │    tới Android native
                               đường phụ, bất đồng bộ ───└──► KUKSA output mirror
```

Mỗi chu kỳ 50 ms có cùng `session_id`, `sequence`, `capture_timestamp` và ego
telemetry. C1 chạy cách một frame nên đạt 10 Hz; ở frame xen giữa, C3 dùng kết
quả C1 mới nhất còn hợp lệ. C2, C3, Drive Quality và gói UDP chạy 20 Hz. Không
được tạo hàng đợi dài: khi xử lý chậm, runtime giữ frame mới nhất và báo số
frame đã bỏ thay vì tiếp tục cảnh báo bằng dữ liệu cũ. Một khoảng trống trong
`sequence` không được đưa thẳng vào temporal model: runtime chuyển sang
degraded và bắt đầu phiên model mới trước khi xử lý tiếp.

Kết quả gửi Android trước. Việc ghi các signal chuẩn sang KUKSA chạy ở luồng
riêng, có timeout và tự kết nối lại. Central Broker dừng không được làm dừng
UDP tới Android.

## Ranh giới không dùng đáp án

Container suy luận chỉ được đọc:

- ảnh camera trước;
- ảnh camera tài xế;
- `frame_id`/`sequence`, timestamp và thông tin phiên;
- ego speed, longitudinal acceleration và lateral acceleration;
- checkpoint, cấu hình và class map đã khóa của C1/C2.

Container không được chứa, mount hoặc đọc:

- TTC, risk, driver-state ground truth;
- file event/label;
- target/depth ground truth;
- CSV prediction đã sinh trước;
- kết quả tổng hợp theo trip dùng để chấm điểm.

C3, Drive Quality và Contextual Risk phải được tính từ output C1/C2 và ego của
chính frame hiện tại. Dữ liệu ground truth chỉ được dùng ở môi trường đánh giá
offline, tách khỏi deployment.

Nếu dùng trip đã ghi để kiểm thử, image/container nguồn chỉ được đóng gói ảnh,
timestamp và ego telemetry hợp lệ. Không đóng gói label hoặc prediction.

## Các điểm đang chặn đã kiểm tra

| Điểm chặn | Trạng thái đã xác minh | Ảnh hưởng |
|---|---|---|
| VIDEO pin | Deployment hiện tại có **0 VIDEO pin** | Chưa có đường camera trước/camera tài xế vào container |
| SDK VIDEO | Ví dụ `a8_pin` trong tài liệu chỉ có tính minh họa; starter pack và môi trường hiện tại không có package/helper này | Chưa thể viết transport VIDEO thật chỉ bằng cách đoán API |
| Zot registry | Đường registry hiện gặp HTTP 502 và chưa có credential push/pull đã kiểm chứng | Chưa thể đưa image model lên CarSky một cách lặp lại |
| Docker local | Có Docker CLI nhưng user hiện tại không có quyền Docker socket; `sudo` cần mật khẩu | Chưa thể build image tại máy này bằng lệnh tự động |
| GPU và kiến trúc máy chạy | Tài liệu có cờ `gpu: true`, nhưng chưa chứng minh device được cấp GPU nào, CUDA/driver nào và container worker dùng `amd64` hay `arm64` | Chưa thể chọn base image và tuyên bố đạt 20 Hz |
| Đường UDP từ container | Android/UDP hiện đã chạy qua IVI Gateway và IVI Switch, nhưng quyền gắn Ethernet trực tiếp cho Container Node chưa được chứng minh qua OpenAPI | Cần BTC xác nhận cách nối container vào đường IVI hiện có |

`aarch64` đã quan sát trên Android AAOS không đủ để kết luận Container Node cũng
chạy `arm64`. `virglrenderer` là GPU đồ họa của máy ảo Android, không phải bằng
chứng rằng container AI có NVIDIA CUDA.

Deployment đang chạy là bản dự phòng để demo. Không dừng, xóa hoặc sửa trực
tiếp deployment đó. Bản suy luận trực tiếp phải được dựng trên blueprint/device
thử riêng; chỉ chuyển sang bản mới sau khi toàn bộ gate bên dưới đạt.

## Gate nghiệm thu

Chỉ được gọi là end-to-end realtime khi tất cả mục sau đều đạt:

1. **Input thật trên CarSky:** có hai VIDEO pin đang tăng sequence ở 20 Hz và
   ego KUKSA ở 20 Hz. Road/cabin cùng chu kỳ hoặc lệch timestamp không quá
   25 ms; quá ngưỡng phải đánh dấu degraded, không ghép âm thầm.
2. **Suy luận tại thời điểm nhận frame:** runtime image không có prediction
   CSV/JSON; log cùng một trace ID cho input, C1, C2, C3 và UDP output. Thay đổi
   ảnh đầu vào phải tạo thay đổi output model, không chỉ đổi đồng hồ.
3. **Đúng nhịp:** C1 đạt 10 Hz; C2, C3 và decision envelope đạt 20 Hz liên tục
   ít nhất 10 phút. Không có hàng đợi quá một frame và không xử lý lại sequence
   cũ.
4. **Độ trễ:** p95 từ capture tới lúc gửi UDP không quá 150 ms; tỷ lệ thiếu
   decision envelope không quá 0,5%. Báo cáo phải tách thời gian decode, C1,
   C2, C3 và publish.
5. **Android native:** app `com.fptautomotive.safeloop` nhận UDP port `48100`,
   sequence tăng theo stream, hiển thị source mode, tuổi dữ liệu và C1/C2/C3.
   Không dùng HTML/WebView để thay thế gate này.
6. **An toàn khi mất dữ liệu:** nếu input cũ quá 250 ms, Android phải hiện
   `INPUT STALE`; runtime không được tiếp tục nâng mức cảnh báo từ kết quả cũ.
   Đây là demo cảnh báo, không gửi lệnh phanh/actuator thật.
7. **KUKSA không chặn HMI:** ngắt broker trong một lượt thử không làm dừng UDP;
   mirror tự kết nối lại và bắt kịp bằng giá trị mới nhất, không phát lại hàng
   đợi cũ.
8. **Khởi động lại được:** restart source, runtime và deployment ba lần; model
   version/digest, cấu hình, nhịp và output contract không thay đổi ngoài giá
   trị dự đoán theo input.
9. **Bằng chứng:** lưu topology, image digest, model digest, 10 phút log,
   latency/drop report và video Android cùng trace ID. Không dùng điểm practice
   dataset làm bằng chứng cho độ trễ realtime.

Nếu nguồn là trip đã ghi, gate 1--9 chỉ chứng minh
`RECORDED_SOURCE + LIVE_MODEL`. Muốn tuyên bố `LIVE_CAMERA + LIVE_MODEL` phải
chạy lại gate với camera vật lý.

## Thông tin và tài nguyên cần có

### Cần BTC/CarSky cung cấp hoặc xác nhận

1. SDK VIDEO chính thức hoặc sample project chạy được, gồm package/version của
   `a8_pin` (hoặc API thay thế), format frame, header, pixel format, cách đồng
   bộ timestamp và thứ tự khởi động publisher/subscriber.
2. Zot registry URL đang hoạt động, credential có quyền push/pull và cách gắn
   image bằng immutable digest. HTTP 502 cần được xử lý trước khi deploy.
3. Kiến trúc Container Node (`amd64` hay `arm64`), hệ điều hành/base image được
   hỗ trợ, GPU thực tế, CUDA/driver/runtime và xác nhận `gpu: true` cấp GPU cho
   container chứ không chỉ GPU đồ họa của Android VM.
4. Quota và một device/blueprint thử riêng có thể thêm hai VIDEO pin, KUKSA và
   đường Ethernet/IVI Switch tới Android mà không đụng deployment hiện tại.
5. Cách cấu hình Container Node gửi UDP tới `10.99.0.14:48100`, hoặc bridge
   chính thức tương đương nếu container không được gắn Ethernet pin trực tiếp.
6. Tên, kiểu, đơn vị và nhịp của ba ego signal KUKSA; quyền publish output
   mirror và danh sách custom VSS được phép nếu cần.

### Cần phía đội/người vận hành chuẩn bị

1. Quyền Docker trên máy build, hoặc chạy lệnh build/push trên GPU server đã có
   Docker; credential registry chỉ để trong environment/secret, không commit.
2. Chọn nguồn nghiệm thu đầu tiên: simulator, camera vật lý, hay một trip đã
   ghi. Nếu dùng trip, chỉ đóng gói road image, cabin image, timestamp và ego.
3. Cấp đúng checkpoint/config C1 và C2 đã khóa; lưu SHA-256 trong manifest.
4. Giữ Android app lắng nghe UDP `48100` và giữ địa chỉ IVI đã xác minh trong
   suốt lượt thử.

Khi chưa đủ các tài nguyên trên, có thể tiếp tục phát triển và kiểm thử runtime
model ở local/GPU server, nhưng không được báo rằng CarSky đã chạy suy luận
VIDEO end-to-end.
