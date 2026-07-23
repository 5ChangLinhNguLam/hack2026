# Hướng dẫn dùng Team Kit (dành cho người mới, không cần biết sâu về kỹ thuật)

Tài liệu này viết cho người **chưa quen với Python/dòng lệnh** nhưng cần tự
chạy được `team_kit` để khai thác dataset hackathon. Nếu bạn đã quen thuộc
với Python, hãy đọc `README.md` ở thư mục cha (`package_starterkit/README.md`)
— tài liệu đó ngắn gọn hơn và có đầy đủ tham chiếu API. Tài liệu này đi
chậm hơn, giải thích từng bước và từng khái niệm.

**Cách đọc tài liệu này:** đọc tuần tự từ trên xuống nếu bạn mới bắt đầu.
Nếu chỉ cần tra cứu 1 phần cụ thể, dùng mục lục bên dưới để nhảy tới.

> **Bản dễ đọc hơn:** mở file [`HUONG_DAN_NGUOI_MOI.html`](./HUONG_DAN_NGUOI_MOI.html)
> cùng thư mục bằng trình duyệt (double-click) để có mục lục điều hướng,
> checklist tick được, và giao diện dễ đọc hơn bản Markdown thuần này —
> nội dung giống hệt, không cần mạng.

## Mục lục

1. [Team kit gồm những gì](#1-team-kit-gồm-những-gì)
2. [Chuẩn bị môi trường — chọn 1 trong 2 cách](#2-chuẩn-bị-môi-trường--chọn-1-trong-2-cách)
3. [Trích xuất dữ liệu bằng Dataset Loader](#3-trích-xuất-dữ-liệu-bằng-dataset-loader)
4. [(Tùy chọn) Trích xuất qua Local Stream Server](#4-tùy-chọn-trích-xuất-qua-local-stream-server)
5. [Đọc hiểu explore_trip.ipynb](#5-đọc-hiểu-explore_tripipynb)
6. [Chạy & hiểu output của baseline_ttc_predictor.py](#6-chạy--hiểu-output-của-baseline_ttc_predictorpy)
7. [Chạy & hiểu report của evaluation.py](#7-chạy--hiểu-report-của-evaluationpy)
8. [Tham chiếu API đầy đủ (giải thích từng thuộc tính dùng để làm gì)](#8-tham-chiếu-api-đầy-đủ)
9. [Bảng thuật ngữ (Glossary)](#9-bảng-thuật-ngữ-glossary)
10. [Lỗi thường gặp & cách xử lý](#10-lỗi-thường-gặp--cách-xử-lý)
11. [Checklist trước khi nộp bài](#11-checklist-trước-khi-nộp-bài)

---

## 1. Team kit gồm những gì

Thư mục `team_kit/` có 5 file, mỗi file làm 1 việc:

| File | Việc nó làm | Chạy trực tiếp bằng dòng lệnh? | Bạn có cần sửa code không? |
|---|---|---|---|
| `dataset_loader.py` | Đọc 1 trip (ảnh, JSON, calib) thành các đối tượng Python dễ dùng | **Không** — đây là thư viện thuần, không có lệnh CLI. Kiểm tra nhanh bằng: `python -c "from team_kit.dataset_loader import TripDataset; ds = TripDataset('./data/T01-Sample'); print(ds.summary())"` | Không — chỉ import và gọi |
| `baseline_ttc_predictor.py` | Model mẫu dự đoán TTC (Time-To-Collision) bằng OpenCV, không dùng AI | Có — `python team_kit/baseline_ttc_predictor.py --trip-dir ... --output ...` | Có thể sửa để cải thiện, hoặc bỏ qua và tự viết model riêng |
| `evaluation.py` | Chấm điểm dự đoán của bạn so với ground truth thật | Có — `python team_kit/evaluation.py --predictions ... --trip-dir ...` | Không — chỉ chạy như công cụ |
| `explore_trip.ipynb` | Notebook trực quan hoá 1 trip (biểu đồ, ảnh mẫu) | Không phải file `.py` — mở bằng Jupyter/Colab (xem mục 2) | Không cần, chỉ đổi 1 dòng path |
| `local_stream_server.py` | Server HTTP tùy chọn, để lấy dữ liệu qua URL thay vì đọc file trực tiếp | Có — `python team_kit/local_stream_server.py --data-dir ... --port ...` | Không cần nếu bạn code bằng Python |

Bạn **không cần hiểu code bên trong** các file này để dùng — chỉ cần biết
gọi hàm nào, khi nào, và output trả về có nghĩa là gì. Đó là mục tiêu của
tài liệu này.

> **Vì sao không có lệnh `python dataset_loader.py` để chạy?** Vì file này
> là **thư viện** (library) thuần, khác với 3 file `baseline_ttc_predictor.py`
> / `evaluation.py` / `local_stream_server.py` — 3 file đó có sẵn phần
> "CLI" (`if __name__ == "__main__"` + `argparse`) nên chạy trực tiếp được
> bằng `python <tên_file>.py --tham-số`. `dataset_loader.py` **không có**
> phần đó — nó chỉ định nghĩa sẵn các class (`TripDataset`, `HackathonDataset`)
> để bạn `import` vào code Python của chính mình rồi gọi hàm, giống ví dụ ở
> [mục 3](#3-trích-xuất-dữ-liệu-bằng-dataset-loader) bên dưới. Chạy thẳng
> `python team_kit/dataset_loader.py` (không tham số) sẽ không báo lỗi
> nhưng cũng không làm gì cả — vì không có code nào được gọi khi chạy trực tiếp kiểu đó.

**Nhắc lại cấu trúc dataset (đã có trong `README.md`):** bạn nhận 16 trip,
chia 2 nhóm:
- **10 trip chấm điểm — `T01d`..`T10d`**: đây là trip bạn nộp dự đoán. Các
  field đáp án (TTC thật, driver state thật, risk score thật...) đã bị xoá.
- **6 trip luyện tập — `T01-Sample`..`T06-Sample`**: có đầy đủ đáp án, dùng
  để bạn tự kiểm tra model trước khi nộp bài. **Luôn bắt đầu học/thử nghiệm
  với nhóm này**, vì mọi biểu đồ/số liệu ở đây đều có dữ liệu thật để đối
  chiếu — nhóm `T0Xd` sẽ hiện giá trị rỗng/mặc định ở các cell liên quan
  đáp án (đây là chủ đích, không phải lỗi).

---

## 2. Chuẩn bị môi trường — chọn 1 trong 2 cách

Bạn chỉ cần làm **1 trong 2 cách** dưới đây, tùy máy tính/thói quen của bạn.

### Cách 1 — Chạy trên máy tính cá nhân (khuyên dùng nếu máy bạn đã có Python)

1. **Cài Python** (nếu chưa có): tải tại [python.org/downloads](https://www.python.org/downloads/),
   chọn bản 3.10 trở lên. Khi cài trên Windows, nhớ tick vào ô
   "Add Python to PATH" ở màn hình cài đặt đầu tiên.
2. **Mở terminal** tại đúng thư mục `package_starterkit/` (thư mục chứa
   `team_kit/`, `data/`, `requirements.txt`):
   - Windows: mở thư mục đó trong File Explorer → gõ `cmd` vào thanh địa
     chỉ rồi Enter, hoặc chuột phải → "Open in Terminal".
   - macOS/Linux: chuột phải → "Open Terminal Here", hoặc dùng lệnh `cd`.
3. **Cài thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```
   Lệnh này cài `opencv-python`, `numpy`, `pandas`, `matplotlib`, `jupyter`
   — đủ để chạy mọi thứ trong `team_kit/`.
4. **Cho code biết dataset đang ở đâu** — có 2 cách, chọn 1:
   - **Cách A (không cần copy gì cả — khuyên dùng):** cứ để dataset ở
     chỗ bạn đã tải/giải nén (ví dụ `D:\hackathon\T01-Sample` hay
     `~/Downloads/T01-Sample`), rồi dùng **đúng đường dẫn đó** khi gọi
     `TripDataset(...)` ở bước 5 — không cần đúng chữ `data/` gì cả.
   - **Cách B (copy vào `data/` cho đường dẫn gọn hơn):** giải nén trip
     bạn muốn dùng (ví dụ `T01-Sample`) vào trong thư mục `data/` đi kèm
     `package_starterkit/`, sao cho có đường dẫn
     `package_starterkit/data/T01-Sample/...`. Xem ghi chú trong
     `data/PUT_DATASET_HERE.txt`. Đây chỉ là tiện lợi để các ví dụ lệnh
     trong tài liệu này chạy đúng y hệt (`./data/T01-Sample`) mà không
     cần sửa đường dẫn — **không phải yêu cầu bắt buộc**.
5. **Kiểm tra mọi thứ hoạt động** — thay `./data/T01-Sample` bằng đường
   dẫn thật của bạn nếu bạn chọn Cách A ở trên:
   ```bash
   python -c "from team_kit.dataset_loader import TripDataset; ds = TripDataset('./data/T01-Sample'); print(ds.summary())"
   ```
   Nếu lệnh này in ra 1 khối JSON (map, số frame, safe_driving_score...)
   nghĩa là môi trường đã sẵn sàng. Nếu báo lỗi, xem [mục 10](#10-lỗi-thường-gặp--cách-xử-lý).

### Cách 2 — Chạy trên Google Colab (không cần cài gì trên máy)

Dùng cách này nếu máy bạn không có Python, hoặc bạn muốn tận dụng GPU miễn
phí của Google (dù baseline hiện tại không cần GPU — chỉ cần khi bạn tự
train model AI).

1. **Nén dữ liệu cần dùng thành 2 file zip riêng** (để nhẹ, không cần zip
   cả 16 trip 1 lúc):
   - `team_kit.zip` — nén toàn bộ thư mục `team_kit/` (rất nhẹ, chỉ vài
     trăm KB, toàn mã nguồn `.py`).
   - `T01-Sample.zip` — nén 1-2 trip bạn muốn khám phá trước (mỗi trip
     full-GT nặng vài trăm MB tùy độ phân giải).
2. **Upload 2 file zip lên Google Drive** — vào [drive.google.com](https://drive.google.com),
   kéo thả file vào, đợi upload xong. Ghi nhớ đường dẫn (ví dụ
   `MyDrive/hackathon/team_kit.zip`).
3. **Tạo notebook mới** tại [colab.research.google.com](https://colab.research.google.com)
   → File → New notebook.
4. **Mount Google Drive** (dán vào ô code đầu tiên, bấm nút ▶ hoặc
   Shift+Enter để chạy):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Colab sẽ hiện cửa sổ đăng nhập Google và xin quyền truy cập Drive —
   đăng nhập rồi bấm "Allow".
5. **Giải nén vào Colab** (ô code tiếp theo, sửa đường dẫn cho khớp với
   Drive của bạn):
   ```python
   !mkdir -p /content/project/data
   !unzip -q "/content/drive/MyDrive/hackathon/team_kit.zip" -d /content/project
   !unzip -q "/content/drive/MyDrive/hackathon/T01-Sample.zip" -d /content/project/data
   ```
6. **Cài thư viện còn thiếu** — Colab đã có sẵn numpy/pandas/matplotlib,
   nhưng nên đổi `opencv-python` → `opencv-python-headless` (bản không cần
   giao diện đồ họa, tránh lỗi trên server đám mây):
   ```python
   !pip install -q opencv-python-headless
   ```
7. **Chạy thử**:
   ```python
   import sys
   sys.path.insert(0, '/content/project')
   from team_kit.dataset_loader import TripDataset
   ds = TripDataset('/content/project/data/T01-Sample')
   print(ds.summary())
   ```

**Chạy `explore_trip.ipynb` trên Colab:** File → Upload notebook → chọn
file `explore_trip.ipynb` từ máy bạn. Sau khi mở, sửa dòng `TRIP_DIR = ...`
ở Cell đầu tiên thành `Path('/content/project/data/T01-Sample')`, rồi chạy
lần lượt từng cell (Shift+Enter).

---

## 3. Trích xuất dữ liệu bằng Dataset Loader

Đây là cách **chính, được khuyên dùng** để lấy dữ liệu ra khỏi trip
(ảnh, thông số xe, đáp án...) — không cần tự viết code đọc JSON hay tự mở
file ảnh.

**Lưu ý:** `dataset_loader.py` không có lệnh chạy riêng kiểu
`python dataset_loader.py ...` (xem giải thích ở [mục 1](#1-team-kit-gồm-những-gì)) —
mọi ví dụ dưới đây đều là code Python bạn tự chạy (trong 1 file `.py` riêng,
trong Python REPL, hoặc trong `explore_trip.ipynb`), trong đó dòng đầu tiên
luôn là `from team_kit.dataset_loader import ...` để import.

```python
from team_kit.dataset_loader import TripDataset

# Bước 1: Load 1 trip — trỏ vào thư mục trip, không phải file JSON
ds = TripDataset("./data/T01-Sample")

# Bước 2: Xem tổng quan trip (map, thời lượng, điểm số...)
print(ds.summary())

# Bước 3: Duyệt qua từng frame (1 frame = 1 khoảnh khắc, ~1/20 giây)
for frame in ds.iter_frames():
    left_img = ds.load_left(frame.frame_id)     # ảnh camera trái (đường)
    right_img = ds.load_right(frame.frame_id)    # ảnh camera phải (đường)
    driver_img = ds.load_driver(frame.frame_id)  # ảnh trong cabin (tài xế)
    depth = ds.load_depth(frame.frame_id)        # bản đồ độ sâu (có thể None)

    # Đáp án thật có sẵn ở đây (rỗng/mặc định nếu là trip T0Xd chấm điểm):
    print(frame.min_ttc, frame.driver_state, frame.final_risk_score)
```

**Giải thích từng dòng, chậm hơn:**

- `TripDataset("./data/T01-Sample")` — mở 1 trip. Bên trong, nó tự đọc
  file `T01-Sample.json.gz` (hoặc `.json`), và ghi nhớ đường dẫn tới các
  thư mục ảnh (`kitti/image_2`, `kitti/image_3`, `driver/`, `kitti/depth`)
  — chưa đọc ảnh nào cả, chỉ khi bạn gọi `load_left()`/`load_right()`/... nó
  mới thực sự mở file ảnh đó.
- `ds.summary()` — trả về 1 dict (kiểu như 1 bảng key–value) tóm tắt trip:
  bao nhiêu frame, dài bao nhiêu giây, điểm an toàn... Đây là cách nhanh
  nhất để "liếc" 1 trip mà không cần duyệt hết dữ liệu.
- `frame.frame_id` — số thứ tự frame, bắt đầu từ 0. Dùng số này để load
  đúng ảnh tương ứng (`load_left(frame_id)`).
- `frame.min_ttc` — TTC nhỏ nhất tại thời điểm đó (giây). Nếu không có xe
  nào phía trước đáng lo, giá trị là `inf` (vô cực = an toàn).
- Nếu bạn thấy `frame.driver_state == "unknown"` hoặc `frame.min_ttc == inf`
  luôn ở **mọi** frame, và bạn đang mở 1 trip `T0Xd` — đó là do đáp án đã
  bị xoá để chấm điểm, **không phải bug**. Dùng trip `T0X-Sample` để thấy
  dữ liệu thật.

**Nếu bạn muốn phân tích bằng bảng (Excel-style)** thay vì duyệt từng
frame một, dùng `frames_df` — trả về 1 bảng pandas, mỗi hàng là 1 frame:

```python
df = ds.frames_df
print(df.head())                              # xem 5 hàng đầu
drowsy_frames = df[df["driver_state"] == "drowsy"]   # lọc các frame tài xế buồn ngủ
print(f"{len(drowsy_frames)} frame tài xế buồn ngủ trên tổng {len(df)} frame")
```

**Muốn duyệt nhiều trip cùng lúc** (ví dụ để so sánh cả 6 trip luyện tập)?
Dùng `HackathonDataset`:

```python
from team_kit.dataset_loader import HackathonDataset

all_trips = HackathonDataset("./data")   # trỏ vào thư mục CHỨA các trip, không phải 1 trip
print(all_trips.trip_ids)                # ['T01-Sample', 'T02-Sample', ..., 'T01d', ...]
print(all_trips.summary_table())         # 1 bảng, mỗi hàng = 1 trip — tiện để so sánh nhanh
```

---

## 4. (Tùy chọn) Trích xuất qua Local Stream Server

**Bạn có cần dùng phần này không?** Hầu hết mọi người **không cần** — nếu
bạn code bằng Python, đọc trực tiếp qua `dataset_loader.py` (mục 3) đơn
giản và nhanh hơn. Chỉ dùng server này nếu:

- Model/pipeline của bạn viết bằng ngôn ngữ khác Python (Java, Node.js,
  C#...) và bạn muốn lấy dữ liệu qua URL thay vì viết lại logic đọc JSON.
- Bạn muốn xây 1 dashboard riêng và gọi dữ liệu qua HTTP cho tiện.

Server này **chạy hoàn toàn trên máy bạn**, đọc dữ liệu bạn đã tải sẵn —
không phải server của ban tổ chức, không ai khác cần nó ngoài bạn.

**Cách chạy:**

```bash
python team_kit/local_stream_server.py --data-dir ./data --port 8765
```

Sau khi thấy dòng `Local dataset server running at http://127.0.0.1:8765`,
mở 1 terminal khác (hoặc trình duyệt) để gọi thử:

```bash
curl http://127.0.0.1:8765/trips
curl http://127.0.0.1:8765/trips/T01-Sample/summary
curl http://127.0.0.1:8765/trips/T01-Sample/frames/0
curl http://127.0.0.1:8765/trips/T01-Sample/frames/0/left --output frame0.jpg
```

Bạn cũng có thể dán thẳng các URL trên (trừ URL trả ảnh) vào thanh địa chỉ
trình duyệt để xem kết quả JSON.

**Bảng route đầy đủ:**

| Route | Trả về |
|---|---|
| `GET /trips` | Danh sách tất cả trip_id tìm thấy trong `--data-dir` |
| `GET /trips/{trip_id}/summary` | Giống hệt `ds.summary()` |
| `GET /trips/{trip_id}/calibration` | Giống hệt `ds.load_calibration()` |
| `GET /trips/{trip_id}/events` | Giống hệt `ds.events_log` |
| `GET /trips/{trip_id}/aggregate` | `trip_aggregate` + `driver_summary` |
| `GET /trips/{trip_id}/frames` | Số lượng frame + danh sách frame_id |
| `GET /trips/{trip_id}/frames/{frame_id}` | 1 frame đầy đủ, dạng JSON |
| `GET /trips/{trip_id}/frames/{frame_id}/left` | Ảnh camera trái (bytes ảnh gốc, PNG/JPG) |
| `GET /trips/{trip_id}/frames/{frame_id}/right` | Ảnh camera phải |
| `GET /trips/{trip_id}/frames/{frame_id}/driver` | Ảnh trong cabin |
| `GET /trips/{trip_id}/frames/{frame_id}/depth` | File depth `.npy` (404 nếu frame đó không phải keyframe — không phải mọi frame đều có depth) |

**Lưu ý an toàn:** server này mặc định chỉ nhận kết nối từ chính máy bạn
(`127.0.0.1`), không có xác thực (không cần mật khẩu) — **không** đưa cờ
`--host 0.0.0.0` trừ khi bạn thực sự muốn máy khác trong cùng mạng LAN
truy cập được, và tuyệt đối không public ra Internet.

---

## 5. Đọc hiểu `explore_trip.ipynb`

Đây là notebook Jupyter để **xem** dữ liệu trực quan (biểu đồ, ảnh) trước
khi viết model — nên chạy phần này đầu tiên để hiểu dataset trông như thế
nào, trước khi lao vào code.

**Trước khi chạy:** sửa dòng `TRIP_DIR = PROJECT_ROOT / 'data' / 'T01-Sample'`
ở Cell đầu tiên nếu bạn muốn xem trip khác. Khuyên dùng 1 trong 6 trip
`T0X-Sample` lúc đầu, vì có đủ đáp án để đối chiếu.

Notebook có 8 phần, chạy tuần tự từ trên xuống (mỗi cell là 1 khối code,
bấm Shift+Enter để chạy từng cell):

| # | Phần | Bạn sẽ thấy gì | Ghi chú |
|---|---|---|---|
| Setup | Cell 1 | Load thư viện + load trip, in số frame và thời lượng | Chạy cell này trước tiên, luôn luôn |
| 1 | Trip overview | In ra 1 khối JSON tóm tắt: map, safe_driving_score, driver_subject... | Ở trip `T0Xd`, các field đáp án sẽ là `null` — chủ đích |
| — | Events log | Danh sách sự kiện kịch bản đã xảy ra (loại + thời điểm) | Có ở **mọi** trip kể cả trip bị redact (chỉ mất phần "thông số chi tiết") |
| 2 | Driver state timeline | Biểu đồ dải màu theo thời gian — mỗi màu là 1 trạng thái tài xế (xanh lá=tỉnh táo, đỏ=ngủ gật...) | Ở trip full-GT sẽ có màu đa dạng; ở trip `T0Xd` sẽ toàn "unknown" |
| — | Alertness score | Đường cong 0→1, càng gần 0 càng nguy hiểm (ngủ gật) | Đường kẻ ngang 0.5 chỉ là ngưỡng tham khảo, không phải luật chính thức |
| 3 | Ego dynamics | 2 biểu đồ: tốc độ xe theo thời gian, và gia tốc dọc/ngang | Dữ liệu này **luôn có thật** ở mọi trip (không bị xoá) — đây là input, không phải đáp án |
| 4 | TTC + Risk score | Biểu đồ TTC thật và risk score, có vạch tím đánh dấu mỗi sự kiện | **Đây chính là đáp án của Challenge 1** — chỉ có dữ liệu thật ở trip full-GT (`T0X-Sample`) |
| 5 | Driver state distribution | Biểu đồ cột: % thời gian ở mỗi trạng thái tài xế trong cả trip | |
| 6 | Sample frames | 4 cặp ảnh (đường + driver) lấy mẫu trải đều trong trip, kèm số liệu (tốc độ, TTC, risk, alertness) ngay trên mỗi ảnh | Cách nhanh nhất để "nhìn thấy" dữ liệu thay vì chỉ đọc số |
| 7 | Behavior events | Biểu đồ dạng vạch: khi nào phanh gấp/tăng tốc gấp/cua gấp/quá tốc độ/bám đuôi xảy ra | Các flag này **không bị xoá** ở bất kỳ trip nào |
| 8 | Trip aggregate | Điểm Safe Driving Score cuối trip + số lần các hành vi nguy hiểm | **Đây chính là đáp án của Challenge 3** — chỉ có ở trip full-GT |


**Ý nghĩa của phần "Next steps" ở cuối notebook:** đây là gợi ý 3 hướng
làm bài (tương ứng 3 challenge), và nhắc lại nguyên tắc quan trọng nhất:
khi tự chấm điểm ở nhà bằng `evaluation.py`, chỉ dùng được với trip có
GT thật (`T0X-Sample`) — trip `T0Xd` phải nộp lên để ban tổ chức chấm.

---

## 6. Chạy & hiểu output của `baseline_ttc_predictor.py`

Baseline là 1 model **cố tình đơn giản** (không dùng AI/deep learning) để
làm mức sàn — mục tiêu của bạn là vượt qua nó.

**Baseline hoạt động thế nào (giải thích không cần đọc code):**
1. So sánh ảnh trái/phải để tính "độ lệch" (disparity) giữa 2 camera —
   vật càng gần thì lệch càng nhiều (giống 2 mắt người ước lượng khoảng
   cách).
2. Lấy 1 vùng cố định ở giữa-dưới ảnh (ROI — nơi xe phía trước thường xuất
   hiện) và tính "độ sâu" (khoảng cách, tính bằng mét) trung vị trong vùng đó.
3. Theo dõi độ sâu này qua vài frame gần nhất để ước lượng xe đang tiến
   lại gần với tốc độ bao nhiêu (m/s).
4. TTC = khoảng cách ÷ tốc độ tiến lại gần.

Vì ROI là **vùng cố định** (không biết xe thật sự ở đâu — không có object
detection), baseline sẽ sai nhiều khi xe không nằm giữa ảnh. Đây chính là
chỗ dễ cải thiện nhất.

**Cách chạy:**

```bash
python team_kit/baseline_ttc_predictor.py \
    --trip-dir ./data/T01-Sample \
    --output ./predictions/T01-Sample.csv \
    --verbose
```

- `--trip-dir`: trip nào để chạy.
- `--output`: nơi ghi file CSV kết quả.
- `--verbose`: in tiến trình ra màn hình (khuyên bật lúc mới thử, để biết
  script còn chạy hay bị treo).

**Đọc hiểu output khi chạy `--verbose`:** mỗi vài chục frame sẽ in 1 dòng
dạng:
```
14:32:07 | INFO | Frame 150/1799  pred=4.20  gt=3.85
```
- `Frame 150/1799`: đang xử lý frame 150 trên tổng 1800 frame (frame_id
  bắt đầu từ 0 nên số cuối là 1799).
- `pred=4.20`: TTC baseline dự đoán (giây). Nếu là `99.00` nghĩa là baseline
  không phát hiện gì nguy hiểm (giá trị `inf` được hiển thị thành 99 cho dễ đọc).
- `gt=3.85`: TTC thật (ground truth) tại frame đó, **chỉ có ý nghĩa** nếu
  bạn đang chạy trên trip full-GT (`T0X-Sample`) — trên trip `T0Xd` số này
  sẽ luôn là `99.00` vì không có đáp án.

**Đọc hiểu file CSV output** (`predictions/T01-Sample.csv`):

| Cột | Ý nghĩa |
|---|---|
| `frame_id` | Số thứ tự frame |
| `timestamp` | Thời điểm (giây) tính từ đầu trip |
| `predicted_ttc` | TTC baseline dự đoán — số giây, hoặc chữ `inf` nếu không phát hiện nguy hiểm |
| `ground_truth_ttc` | TTC thật, **chỉ để bạn tham khảo cục bộ** — cột này **hoàn toàn bị bỏ qua** khi ban tổ chức chấm điểm thật, kể cả khi bạn để nguyên hoặc chỉnh sửa nó |

> **Vì sao có cột `ground_truth_ttc` nếu nó bị bỏ qua?** Để bạn tự so sánh
> `predicted_ttc` với đáp án ngay trong Excel/pandas khi đang thử nghiệm ở
> nhà (trên trip full-GT) — tiện hơn phải chạy `evaluation.py` mỗi lần chỉ
> để nhìn nhanh 1 vài dòng. Khi nộp file dự đoán thật để nộp bài, format
> submission (xem [mục 11](#11-checklist-trước-khi-nộp-bài)) không có cột
> này.

---

## 7. Chạy & hiểu report của `evaluation.py`

Script này chấm điểm dự đoán của bạn — dùng để **tự kiểm tra ở nhà** trên
6 trip full-GT trước khi nộp bài (10 trip `T0Xd` phải nộp để ban tổ chức
chấm bằng đáp án họ giữ riêng). Script chấm được **cả 3 challenge**, tự
động phát hiện challenge nào bạn đã làm dựa trên cột nào có mặt trong CSV
— không cần cờ gì thêm, và **không bị trừ điểm** nếu chỉ làm 1 challenge.

```bash
python team_kit/evaluation.py \
    --predictions ./predictions/T01-Sample.csv \
    --trip-dir ./data/T01-Sample
```

### 7.1. Challenge 1 — Collision Risk Monitor (TTC)

Luôn được chấm (cần cột `predicted_ttc`, mặc định `inf` nếu bỏ trống).

```
==============================================================================
EVALUATION REPORT - Challenge 1: Collision Risk Monitor (TTC)
==============================================================================
Trips evaluated:        1
Overall MAE (critical): 1.420s
Overall F1:             0.480
Overall composite:      52.3 / 100

Trip         n_crit  MAE-crit   F1      FPR     Composite
------------------------------------------------------------------------------
T01-Sample   62      1.420s     0.480   0.080   52.3

RANKING (by composite score, higher = better):
  #1  T01-Sample  ->  52.3
```

| Chỉ số | Ý nghĩa | Càng cao hay càng thấp là tốt? |
|---|---|---|
| `n_crit` (n_critical_zone) | Số frame mà TTC thật < 3 giây (vùng nguy hiểm) | Chỉ là số đếm, không phải điểm |
| `MAE-crit` (mae_critical) | Sai số trung bình (giây) giữa TTC dự đoán và TTC thật, **chỉ tính trên các frame nguy hiểm** — đây là chỉ số quan trọng nhất vì hệ thống phanh khẩn cấp chỉ cần chính xác lúc sắp va chạm | Càng **thấp** càng tốt (0 = hoàn hảo) |
| `F1` | Coi "TTC < 2 giây" là 1 câu hỏi Có/Không (có nguy hiểm hay không); F1 đo việc bạn phát hiện đúng các trường hợp nguy hiểm mà không báo động giả quá nhiều | Càng **cao** càng tốt (1.0 = hoàn hảo) |
| `FPR` (false_positive_rate) | Tỷ lệ báo động giả — dự đoán "nguy hiểm" trong khi thực tế an toàn | Càng **thấp** càng tốt |
| `Composite` | Điểm tổng hợp 0–100, kết hợp MAE-crit (40%) + F1 (30%) + inv-TTC MAE (30%) — đây là điểm dùng để **xếp hạng** | Càng **cao** càng tốt |

Các chỉ số khác có trong file JSON (`--output`, nếu bạn dùng cờ này) nhưng
không in ra bảng trên:

| Chỉ số | Ý nghĩa |
|---|---|
| `mae_overall` | Giống MAE-crit nhưng tính trên **toàn bộ** frame có đáp án, không chỉ vùng nguy hiểm |
| `rmse_critical` | Giống MAE-crit nhưng phạt nặng hơn với các lỗi lớn (bình phương sai số rồi khai căn) |
| `inv_ttc_mae` | Sai số trên **1/TTC** thay vì TTC trực tiếp — vì TTC có thể tiến tới vô cực (an toàn), lấy nghịch đảo giúp mọi giá trị nằm trong khoảng hữu hạn, dễ so sánh và không bị 1 giá trị `inf` làm lệch trung bình |
| `precision` | Trong số các frame bạn dự đoán "nguy hiểm", bao nhiêu % thực sự nguy hiểm |
| `recall` | Trong số các frame thực sự nguy hiểm, bạn phát hiện được bao nhiêu % |
| `accuracy` | % tổng số frame bạn phân loại đúng (nguy hiểm/an toàn) |

### 7.2. Challenge 2 — Driver Intelligence Platform (driver state)

Chỉ xuất hiện trong báo cáo nếu CSV có cột `predicted_driver_state` với ít
nhất 1 dòng không để trống.

```
==============================================================================
Challenge 2: Driver Intelligence Platform (driver state)
==============================================================================
Overall composite:      73.4 / 100

Trip           n_scored   Accuracy   Macro-F1   Composite
------------------------------------------------------------------------------
T01-Sample     600        0.688      0.780      73.4
```

| Chỉ số | Ý nghĩa | Càng cao hay càng thấp là tốt? |
|---|---|---|
| `n_scored` | Số frame có prediction hợp lệ (không để trống) cho cột này | Chỉ là số đếm |
| `Accuracy` | % frame bạn đoán đúng 1-trong-5 trạng thái (`alert/drowsy/yawning/distracted/microsleep`) | Càng **cao** càng tốt |
| `Macro-F1` | F1 trung bình qua các lớp **thật sự xuất hiện** trong trip đó (không tính lớp không xuất hiện) — tránh việc 1 lớp hiếm chiếm áp đảo accuracy | Càng **cao** càng tốt |
| `Composite` | `50%×Accuracy + 50%×Macro-F1`, thang 0–100 | Càng **cao** càng tốt |

### 7.3. Challenge 3 — Fleet Safe Driving Score

Chỉ xuất hiện nếu CSV có cột `predicted_risk_score` với ít nhất 1 dòng
không để trống (cột này là "công tắc" bật challenge — xem giải thích bên
dưới). Script **tái tạo gần đúng 100%** công thức nội bộ ban tổ chức
dùng để tạo `safe_driving_score`:

```
==============================================================================
Challenge 3: Fleet Safe Driving Score
==============================================================================
Overall composite:      100.0 / 100

Trip           Predicted  True       AbsErr     Composite
------------------------------------------------------------------------------
T01-Sample     0.0        0.0        0.0        100.0

Breakdown (deterministic from trip facts, except near_miss = your own predicted_ttc):
Trip           near_miss  harsh_brk  harsh_acc  harsh_crn  speeding%
------------------------------------------------------------------------------
T01-Sample     3          14         53         2          0.0
```

| Chỉ số | Ý nghĩa | Càng cao hay càng thấp là tốt? |
|---|---|---|
| `Predicted` | `100 − (harsh_brake×3.0 + harsh_accel×2.0 + harsh_corner×2.0 + near_miss×5.0 + speeding%×0.15)` | Càng gần `True` càng tốt |
| `True` | `trip_aggregate.safe_driving_score` thật (0–100, cao = an toàn) | — |
| `AbsErr` | `\|Predicted − True\|` | Càng **thấp** càng tốt |
| `Composite` | `100 − 2×AbsErr` | Càng **cao** càng tốt |
| `near_miss` (breakdown) | Số frame trong `predicted_ttc` bạn nộp (Challenge 1) có giá trị < 1.5s | Phần **duy nhất** thực sự phụ thuộc model của bạn |
| `harsh_brk`/`harsh_acc`/`harsh_crn`/`speeding%` (breakdown) | Tính thẳng từ `ego.longitudinal_accel`/`lateral_accel`/`speed_kmh` so với `metadata.speed_limit_kmh` của chính trip đó | Giống nhau cho **mọi** team trên cùng 1 trip — không phải dự đoán |

**Muốn chấm nhiều trip cùng lúc** (ví dụ cả 6 trip luyện tập)?

```bash
python team_kit/evaluation.py \
    --predictions ./predictions/ \
    --data-dir ./data \
    --output ./evaluation_report.json
```
(`--predictions` trỏ vào **thư mục** chứa nhiều file CSV, mỗi file tên
đúng `<trip_id>.csv`, ví dụ `T01-Sample.csv`)

> **Lưu ý bảo mật (quan trọng để hiểu, không cần làm gì):** `evaluation.py`
> **luôn luôn** tự đọc đáp án thật từ `--trip-dir`/`--data-dir`, **không
> bao giờ** tin vào cột `ground_truth_ttc` trong file CSV bạn nộp — kể cả
> nếu bạn (vô tình hay cố ý) sửa cột đó thành số đẹp. Vì vậy đừng cố "làm
> đẹp" cột đó — nó không ảnh hưởng gì đến điểm số thật.

---

## 8. Tham chiếu API đầy đủ

### `TripDataset(trip_dir)` — mở 1 trip

| Gọi hàm/thuộc tính | Trả về | Dùng để làm gì |
|---|---|---|
| `len(ds)` | int | Biết trip có bao nhiêu frame |
| `ds[idx]` | `FrameRecord` | Lấy 1 frame theo vị trí (giống lấy 1 phần tử trong list) |
| `ds.iter_frames()` | iterator | Duyệt qua **toàn bộ** frame theo thứ tự — dùng trong vòng lặp `for` |
| `ds.frames_df` | `pandas.DataFrame` | Xem toàn bộ trip dạng bảng — tiện để lọc/thống kê (vd đếm số frame buồn ngủ) |
| `ds.load_left(frame_id)` | ảnh (H×W×3, BGR) | Lấy ảnh camera trái tại 1 frame cụ thể — dùng làm input cho model TTC |
| `ds.load_right(frame_id)` | ảnh (H×W×3, BGR) | Ảnh camera phải — ghép cặp với ảnh trái để tính stereo depth |
| `ds.load_depth(frame_id)` | ảnh độ sâu (mét) hoặc `None` | Đáp án độ sâu thật — chỉ có ở 1 số frame nhất định (keyframe), không phải mọi frame |
| `ds.load_driver(frame_id)` | ảnh (H×W×3, BGR) | Ảnh trong cabin — dùng làm input cho model phân loại trạng thái tài xế |
| `ds.load_calibration()` | dict | Thông số camera (tiêu cự, khoảng cách 2 mắt camera) — cần để tính depth từ disparity, dùng cho baseline |
| `ds.load_frame_calibration(frame_id)` | dict | Ma trận calib KITTI chuẩn theo từng frame — chỉ cần nếu bạn muốn tái sử dụng code KITTI 3D-detection có sẵn trên mạng |
| `ds.summary()` | dict | Xem nhanh tổng quan 1 trip mà không cần duyệt hết dữ liệu |
| `ds.metadata` | dict | Thông tin cấu hình trip: `fps`, `duration_sec`, `map`, `speed_limit_kmh`... |
| `ds.trip_aggregate` | dict | Điểm Safe Driving Score + số đếm hành vi — rỗng `{}` ở trip `T0Xd` |
| `ds.driver_summary` | dict | Phân bố trạng thái tài xế + điểm mệt mỏi — rỗng `{}` ở trip `T0Xd` |
| `ds.events_log` | list | Danh sách sự kiện kịch bản đã xảy ra (loại + thời điểm) |

### Thuộc tính của `FrameRecord` (1 frame)

| Thuộc tính | Kiểu | Dùng để làm gì |
|---|---|---|
| `frame.frame_id` | int | Số thứ tự frame — dùng để load ảnh tương ứng |
| `frame.timestamp` | float (giây) | Thời điểm frame này trong trip |
| `frame.speed_kmh` | float | Tốc độ xe ego — input hợp lệ, không bị xoá ở trip nào |
| `frame.longitudinal_accel` | float (m/s²) | Gia tốc dọc (phanh/ga) — dùng để phát hiện phanh gấp |
| `frame.lateral_accel` | float (m/s²) | Gia tốc ngang — dùng để phát hiện cua gấp |
| `frame.driver_state` | str | 1 trong `alert/drowsy/yawning/distracted/microsleep` — đáp án Challenge 2 |
| `frame.alertness_score` | float 0–1 | Phiên bản liên tục của driver_state (0=ngủ gật, 1=tỉnh táo hoàn toàn) |
| `frame.eye_state` | str | `open/partial/closed` |
| `frame.head_pose` | str | `normal/down/side` |
| `frame.mouth_state` | str | `normal/yawning/talking` |
| `frame.min_ttc` | float (giây) | TTC nhỏ nhất tại frame này — đáp án chính của Challenge 1, `inf` = an toàn |
| `frame.headway_sec` | float | Khoảng cách thời gian tới xe phía trước (khác TTC — không tính tốc độ tiến lại gần) |
| `frame.final_risk_score` | float 0–100 | Điểm rủi ro tổng hợp (base_risk × driver_factor) tại frame này |
| `frame.is_harsh_brake` / `is_harsh_accel` / `is_harsh_corner` | bool | Cờ hành vi lái xe gấp — không bị xoá ở trip nào |
| `frame.is_speeding` / `is_tailgating` | bool | Cờ quá tốc độ / bám đuôi quá gần — không bị xoá ở trip nào |
| `frame.targets` | list[dict] | Danh sách các xe/vật thể phát hiện được, mỗi phần tử có `rel_pos`, `rel_velocity`, `closing_speed`, `ttc_simple`... (bị xoá ở trip `T0Xd`) |
| `frame.events_active` | list[dict] | Sự kiện kịch bản đang diễn ra ngay tại frame này |

### `HackathonDataset(data_root)` — làm việc với nhiều trip cùng lúc

| Gọi hàm/thuộc tính | Trả về | Dùng để làm gì |
|---|---|---|
| `hd.trip_ids` | list[str] | Danh sách trip tìm thấy trong thư mục |
| `hd.get_trip(trip_id)` | `TripDataset` | Load 1 trip cụ thể theo tên |
| `for trip in hd:` | iterator | Duyệt qua tất cả trip |
| `hd.summary_table()` | `pandas.DataFrame` | Bảng so sánh nhanh mọi trip cùng lúc |

---

## 9. Bảng thuật ngữ (Glossary)

| Thuật ngữ | Nghĩa |
|---|---|
| **TTC** (Time-To-Collision) | Thời gian còn lại (giây) trước khi xe ego va chạm với vật cản, nếu tốc độ hiện tại giữ nguyên |
| **GT** (Ground Truth) | Đáp án thật, do hệ thống mô phỏng sinh ra — dùng để chấm điểm |
| **Trip bị redact** | Trip đã bị xoá field đáp án (10 trip `T0Xd`) — để bạn nộp dự đoán mà không nhìn thấy đáp án |
| **Frame** | 1 khoảnh khắc trong trip, chụp tại 20 khung hình/giây (fps) |
| **ROI** (Region of Interest) | Vùng ảnh được chọn để phân tích (ở baseline: 1 hình chữ nhật cố định giữa-dưới ảnh) |
| **Disparity** | Độ lệch vị trí của cùng 1 điểm giữa ảnh trái và ảnh phải — vật gần thì lệch nhiều, vật xa thì lệch ít |
| **Stereo baseline** | Khoảng cách vật lý giữa 2 camera (trái/phải) — ở dataset này là 30cm, cần để tính depth từ disparity |
| **KITTI** | 1 định dạng chuẩn phổ biến trong nghiên cứu xe tự hành (dùng cho `calib/`, `label_2/`) — dùng chuẩn này để bạn dễ tái sử dụng code/model có sẵn trên mạng |
| **Composite score** | Điểm tổng hợp 0–100 dùng để xếp hạng, kết hợp nhiều chỉ số theo trọng số |
| **AEB** | Automatic Emergency Braking — hệ thống phanh khẩn cấp tự động, lý do vùng TTC < 2-3 giây được coi trọng khi chấm điểm |
| **Headway** | Khoảng thời gian giữa xe ego và xe phía trước, tính theo giây (khác TTC vì không tính tốc độ đang tiến lại gần) |
| **Fatigue score** | Điểm mệt mỏi tổng hợp của tài xế trong cả trip (0–100) |

---

## 10. Lỗi thường gặp & cách xử lý

| Lỗi/Thông báo | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'team_kit'` | Đang chạy Python từ sai thư mục, hoặc chưa thêm project root vào `sys.path` | Chạy lệnh từ đúng thư mục `package_starterkit/` (chứa cả `team_kit/` và `data/`), hoặc thêm `sys.path.insert(0, "đường/dẫn/tới/package_starterkit")` ở đầu script |
| `FileNotFoundError: Trip directory not found` | Đường dẫn bạn truyền vào `TripDataset(...)` không đúng | Kiểm tra lại đường dẫn — phải trỏ vào **thư mục trip** (ví dụ `./data/T01-Sample`), không phải file JSON bên trong. Đường dẫn có thể là **bất kỳ nơi nào** bạn đã giải nén dataset — **không bắt buộc** phải là `./data/...` hay đã copy vào `data/` (xem [mục 2](#2-chuẩn-bị-môi-trường--chọn-1-trong-3-cách), Cách A) |
| `FileNotFoundError: No image for '000123' ... (tried .png/.jpg/.jpeg)` | Ảnh thật sự bị thiếu ở frame đó, hoặc trip tải về bị lỗi | Thử tải lại trip đó; nếu vẫn thiếu, báo qua kênh hỗ trợ hackathon |
| `OpenCV failed to read: ...` | File ảnh tồn tại nhưng bị hỏng (0 byte hoặc copy dở dang) | Tải lại file/trip đó |
| Mọi field đáp án đều là `unknown`/`inf`/`0` dù bạn không code sai gì | Bạn đang mở 1 trong 10 trip `T0Xd` — đáp án đã bị xoá theo thiết kế | Đổi sang trip `T0X-Sample` để thấy dữ liệu thật; dùng `T0Xd` chỉ để lấy ảnh làm input và nộp dự đoán |
| `invalid keyword argument` khi gọi `cv2.StereoSGBM_create` | Bạn (hoặc ai đó) đã đổi tên tham số sang snake_case khi sửa `baseline_ttc_predictor.py` | Giữ nguyên tên tham số camelCase gốc của OpenCV (`minDisparity`, `numDisparities`...) — xem chú thích ngay trong code |
| Kết quả `evaluation.py` không đổi dù bạn sửa cột `ground_truth_ttc` trong CSV | Đúng như thiết kế — cột đó luôn bị bỏ qua, đáp án luôn đọc từ `--trip-dir`/`--data-dir` | Không cần sửa gì — muốn cải thiện điểm thì phải cải thiện `predicted_ttc`, không phải cột đáp án |

---

## 11. Checklist trước khi nộp bài

- [ ] Đã chạy thử trên **ít nhất 1 trip full-GT** (`T0X-Sample`) và thấy
      `evaluation.py` chạy ra điểm số hợp lý (không phải toàn `-1`/`nan`).
- [ ] File CSV nộp bài có đúng số dòng = số frame của trip (1800 dòng cho
      mọi trip `T0Xd`, kể cả `T02d` — dù mô tả cấu hình của trip này có ghi
      "DEBUG 30s (compressed)", số frame thực tế vẫn là 1800/90 giây như
      các trip khác).
- [ ] Cột `predicted_ttc` dùng đơn vị **giây**, và ghi `inf` (không phải
      để trống) khi không phát hiện vật cản.
- [ ] Tên file đúng quy ước: `predictions/<tên_team>/<trip_id>.csv` (ví dụ
      `predictions/team_abc/T01d.csv`).
- [ ] **Không** để cột `ground_truth_ttc` (hoặc bất kỳ cột đáp án nào) lẫn
      trong file nộp — không sai quy tắc gì nếu để lại, nhưng nó sẽ bị bỏ
      qua và không giúp ích gì, chỉ khiến file nặng hơn.
- [ ] Đã đọc mục "3 challenge của hackathon" trong `README.md` để chắc
      chắn bạn đang làm đúng loại dự đoán ban tổ chức yêu cầu chấm.
- [ ] Chỉ làm Challenge 1? **Bỏ hẳn** 2 cột `predicted_driver_state` và
      `predicted_risk_score` khỏi CSV thay vì để trống hoặc điền số bừa —
      `evaluation.py` chỉ chấm challenge nào có cột tương ứng trong file
      (xem [mục 7](#7-chạy--hiểu-report-của-evaluationpy)).
- [ ] Nếu có làm Challenge 2, `predicted_driver_state` phải đúng 1 trong 5
      giá trị `alert|drowsy|yawning|distracted|microsleep` (không viết
      hoa/thường lẫn lộn tuỳ ý, không dùng nhãn khác) — sai chính tả bị
      tính là dự đoán sai.
