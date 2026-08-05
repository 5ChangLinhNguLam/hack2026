# tools/

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
