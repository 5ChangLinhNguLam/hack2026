# tools/

## carsky_recorded_stream_sender.py — nguồn test realtime, không dùng đáp án

Script chỉ nhận bundle `safeloop.carsky.demo-bundle.v1` đã tạo bởi
`prepare_carsky_demo.py`; thư mục dataset `data/Txx-Sample`, file prediction,
label, target, event và depth bị từ chối. Mỗi tick giữ nguyên
`source_sequence`/timestamp media gốc, nhưng lấy `capture_timestamp_ms` epoch
ngay lúc phát, với nhãn bắt buộc `RECORDED_STREAM + LIVE_MODEL`.
`metadata.description` được nhận diện khi xác minh bundle nhưng bị loại trước
runtime vì nội dung mô tả có thể làm lộ annotation kịch bản/event.

```bash
python tools/prepare_carsky_demo.py /path/to/T02-Sample \
  --output-root .carsky-build/demo
python tools/carsky_recorded_stream_sender.py \
  .carsky-build/demo/T02-Sample --verify-only
python tools/carsky_recorded_stream_sender.py \
  .carsky-build/demo/T02-Sample \
  --session-id p2-t02-boot-1 --generation 0 --limit 20
```

CLI mặc định dùng adapter JSONL local, không kết nối AWS và không in bytes ảnh.
Adapter vẫn nhận đủ payload road+cabin+ego. `TransportAdapter` là ranh giới để
nối WebRTC/SRT/Kinesis sau này; khóa mapping chính xác là
`(session_id, generation, source_sequence)`, cộng `media_id` và RTP timestamp
90 kHz riêng cho từng stream.

CLI này không gọi model hoặc `LiveInferenceController`; vì vậy riêng output
JSONL không phải bằng chứng inference/soak `LIVE_MODEL`. Harness soak phải nối
adapter vào decode BGR + controller và đo model độc lập, nhưng vẫn dùng đúng
identity/timestamp của sender này.

`PersistentMediaTransform` và `TransformingTransportAdapter` là hook cho hai
pipeline codec H.264 chạy xuyên suốt session. Hook mở đúng một lần, không chạy
FFmpeg theo từng frame, và từ chối output làm đổi identity/RTP mapping. Repo
hiện chưa tuyên bố H.264 encode/decode soak: raw pipe của FFmpeg 4.4 không cung
cấp side channel đủ để chứng minh mapping; không được ghép output bằng thứ tự
callback decoder.

## carsky_live_soak.py — soak local T4

Harness chạy C1+C2+C3/DQ/context thật trên inference thread một-slot, build
decision-v2 và serialize bằng publisher thread tới mock sink local. Model chỉ
load một lần; T01/T02 luân phiên theo session.

```bash
# smoke 30 giây
.venv-dms2/bin/python tools/carsky_live_soak.py --duration-seconds 30 --device cuda

# soak T4 10 phút (ít nhất 600 giây, cộng model load/publisher drain)
.venv-dms2/bin/python tools/carsky_live_soak.py --duration-seconds 600 --device cuda
```

Report strict JSON nằm trong `.carsky-build/soak/`. Exit `0` chỉ khi `PASS`;
`1` cho `DEGRADED`/`INCONCLUSIVE`/`FAIL`, `2` nếu không khởi chạy được. Report
luôn ghi `media_codec=IMAGE_FILE_DECODE`, `h264_included=false` và
`aws_connectivity=false`: đây không phải bằng chứng H.264, WebRTC/MQTT, AWS hay
Android network latency. Có thể mô phỏng broker bằng `--publisher-delay-ms` và
`--publisher-fail-every` mà không gọi mạng.

## safe_extract.py — Giải nén zip lớn không bị fail giữa chừng

**Vấn đề:** `unzip` của Ubuntu (và trình giải nén GUI) sẽ **dừng toàn bộ** khi
gặp 1 file hỏng CRC. Với `data/Hackathon_Dataset_Redacted.zip`, file hỏng nằm ở
offset ~1.67 GB (`T08d/kitti/image_2/001615.jpg`) nên giải nén luôn fail ở mốc
~1.7 GB — dù 93,700 file còn lại đều tốt.

**Giải pháp:** script này giải nén từng entry độc lập, bỏ qua/cứu file hỏng, và
báo cáo rõ ràng. Chỉ cần `python3` (không cần cài thêm gì).

### Dùng nhanh
```bash
# Giải nén + cố cứu file hỏng CRC (khuyến nghị)
python3 tools/safe_extract.py data/Hackathon_Dataset_Redacted.zip --salvage

# Chỉ kiểm tra xem có file nào hỏng (không giải nén)
python3 tools/safe_extract.py data/Hackathon_Dataset_Redacted.zip --list-bad
```

Kết quả mặc định vào `data/Hackathon_Dataset_Redacted/` (tự nhận thư mục gốc
trong zip nên không bị lồng 2 lần). Danh sách file lỗi ghi ra
`<dest>/_extract_errors.log`.

### Tuỳ chọn
| Cờ | Ý nghĩa |
|----|---------|
| `-d, --dest DIR` | Chỉ định thư mục đích |
| `--salvage` | Cứu file hỏng CRC bằng cách ghi bytes giải nén được (bỏ kiểm CRC) |
| `--list-bad` | Chỉ liệt kê file hỏng, không giải nén |
| `--no-resume` | Không bỏ qua file đã có (giải nén lại từ đầu) |

### Đặc điểm
- **Resume:** chạy lại sẽ bỏ qua file đã giải nén đúng kích thước → an toàn khi
  bị ngắt giữa chừng.
- **Chống Zip-Slip:** chặn entry có `../` thoát ra ngoài thư mục đích.
- **Zip64:** xử lý được archive/entry lớn.

### Kết quả đã kiểm chứng với dataset hiện tại
- 93,620 file giải nén thành công; 1 file (`001615.jpg`) hỏng CRC đã salvage
  (đủ 39,378 bytes, JPEG hợp lệ, mở xem được — một vài pixel có thể sai).
- 0 file thất bại. Thời gian ~75s.
