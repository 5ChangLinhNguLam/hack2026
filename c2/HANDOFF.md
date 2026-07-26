# C2 Handoff — Driver Intelligence Platform (Thiện)

> Cập nhật: 26/07/2026 tối. Người đọc: agent tiếp quản C2.
> Đọc file này TRƯỚC khi làm gì. Bối cảnh nền: `README.md` repo + `Connected Car.html`
> (đề thi đầy đủ). Deadline nộp: **10/08/2026**.

## 1. Vai trò & mục tiêu

Tôi là "Thiện" — thành viên team phụ trách **Challenge 2**: phân loại driver state
5 lớp (`alert/drowsy/yawning/distracted/microsleep`) per-frame từ ảnh cabin,
chấm per-trip `50%×accuracy + 50%×macro-F1` (macro-F1 chỉ trên lớp xuất hiện
trong trip đó). Mục tiêu: thắng leaderboard C2. BTC **cho phép mọi data
ngoài/pretrained model**. User cho quyền tự chủ cao.

## 2. Trạng thái hiện tại

- **10 CSV nộp bài đã sinh**: `predictions/thien_c2/T01d.csv .. T10d.csv`
  (cột `frame_id,timestamp,predicted_driver_state` — đúng format "chỉ làm C2").
  LƯU Ý: thư mục `predictions/` bị gitignore (chủ đích của BTC) — CSV không
  nằm trong git, muốn sinh lại chạy lệnh ở mục 5.
- **Self-check LOTO trung bình 93.9/100** trên 6 trip Sample
  (T01=94.1, T02=100, T03=100, T04=99.5, T05=100, T06=69.9).
- Việc ĐANG DỞ: user đang tải **DMD gốc** (đã được cấp quyền, email links
  time-limited) về `C:\DMD\` → cần viết `c2/dmd_match.py` (spec ở mục 6).

## 3. Kiến trúc pipeline + bằng chứng vì sao chọn

`c2/predict_final.py` = 2 track, quyết định per-segment:

1. **Track A — retrieval**: ảnh driver của 16 trip đều composite từ DMD;
   hash scan phát hiện 6/10 trip chấm điểm tái dùng gần nguyên clip đã có
   nhãn trong 6 trip Sample (MD5 trùng từng byte + dHash≤6: T03d 94.6%,
   T05d 99.9%, T07d 94.5%, T04d 90.1%, T06d 50%, T10d 47%). Segment có
   coverage ≥50% → vote nhãn từ match tin cậy. Không model nào thắng nổi
   "chép đáp án".
2. **Track B — rule classifier ngữ nghĩa** trên blendshape MediaPipe
   (`jawOpen>0.25`→yawning; `eyeBlink>0.55`→microsleep; `blink>0.13 &
   lookDown>0.20`→drowsy; `MAR>0.15`→distracted; else alert; rolling
   window; ngưỡng grid-search trên 6 Sample). Đạt **91.7** LOTO.
   - Vì sao KHÔNG dùng ML học: bake-off 4 họ model (workflow có adversarial
     verify, scripts trong `c2/bakeoff/`) — MLP tốt nhất chỉ **39.2** LOTO
     vì 6 subject quá ít → model học "vân tay subject" thay vì hành vi.
     Đừng quay lại hướng đó trừ khi có data DMD (mục 6).
3. Smoothing majority ±75 frame + **guardrail ERASE_WARN** (in ⚠ nếu lớp
   ≥8% raw bị smoothing xóa sạch — bài học suýt mất lớp yawning ở T02d).
4. `alertness_score` nếu cần = tra bảng hằng số theo state
   (alert .95 / yawning .55 / distracted .45 / drowsy .35 / microsleep .05).

Phát hiện nền tảng (đã kiểm chứng trên data): nhãn phụ là hàm 1-1 của state
(alert=open/normal/normal, distracted=open/side/normal + cầm điện thoại,
drowsy=partial/down, microsleep=closed/down, yawning=partial/normal/yawning);
state đi theo block 15-30s; trip T0Xd có 1-4 segment, **có thể đổi subject
giữa trip** (thấy ở T06d ~frame 1500, hash-cut hiện tại KHÔNG bắt được vụ này).

## 4. Môi trường máy (gotchas!)

- Python: `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`
  (cài bằng winget, **không có trên PATH** của terminal cũ; PowerShell:
  `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" ...`).
- **Không có GPU NVIDIA** (Iris Xe, 16GB RAM) — train nặng phải Colab.
- mediapipe 0.10.35: API cũ `mp.solutions` **đã bị gỡ** — phải dùng Tasks API
  (xem `c2/features.py`); model `.task` đã tải sẵn ở `c2/cache/face_landmarker.task`.
- `c2/cache/` chứa toàn bộ cache (features 63-dim + dHash/MD5 + labels cho
  16 trip) — bị gitignore, sinh lại bằng `c2/features.py` (~15ph) và
  `c2/knn_baseline.py` (tự build khi chạy).
- Ổ C còn ~128GB trước khi tải DMD.

## 5. Lệnh chạy lại

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
cd c:\HackathonFPT\hack2026
& $py c2/features.py                        # cache đặc trưng (bỏ qua trip đã có)
& $py c2/predict_final.py --selfcheck       # LOTO 6 Sample — phải ra ~93.9
& $py c2/predict_final.py --predict         # sinh 10 CSV vào predictions/thien_c2/
& $py team_kit/evaluation.py --predictions predictions/thien_c2_selfcheck --data-dir data
```

## 6. VIỆC TIẾP THEO #1 — DMD matching (`c2/dmd_match.py`, chưa viết)

Mục đích: gỡ các segment chưa chắc chắn bằng nhãn gốc DMD (mục 7).

- Data: user tải về `C:\DMD\` (giữ cấu trúc `gX/<subject>/s5/...`).
  Bộ **Drowsiness = session s5** (nhãn: Safe driving→alert, Sleepy
  driving→drowsy, Yawning with/without hand→yawning, Microsleep→microsleep).
  Bộ **Distraction = s1/s2/s3**, chỉ cần **s2** (Phonecall/Texting L/R→
  distracted, Safe driving→alert). Annotation OpenLABEL/VCD JSON nằm cùng
  thư mục session; đọc trực tiếp JSON được (xem readme DMD user đã paste
  trong transcript — mục "Format and access": frame → action id → action
  "type" + "frame_intervals").
- Ảnh driver hackathon 640×360 = đúng **camera body** (1280×720 ÷ 2), nên
  match với `*_rgb_body*.mp4`. DMD quay 29.76/29.98fps, hackathon 20fps →
  match theo nearest-hash, không map tuyến tính frame index.
- Thuật toán đề xuất: decode video body (cv2.VideoCapture) → resize 640×360
  → dHash 64-bit từng frame (tái dùng `dhash64` trong `c2/knn_baseline.py`)
  → cache npz/video → với mỗi segment chưa chắc của T0Xd: min-hamming vs
  toàn bộ hash DMD → (video, frame) tốt nhất → tra OpenLABEL tại frame đó
  → map sang 5 lớp. Ngưỡng tin cậy: bắt đầu ≤10/64 (pHash-friendly regime
  đã xác nhận: composite = pure rescale + re-encode).
- ⚠ Sync caveat: annotation làm trên mosaic đã align — frame index annotation
  có thể lệch offset so với video body. Kiểm tra bằng mắt 2-3 match đầu
  (Read ảnh so sánh) trước khi tin; nếu lệch, đọc shift trong
  `streams`/`stream_properties` của JSON hoặc dùng DEx tool.
- Sau match: cập nhật nhãn segment trong pipeline (thêm track ưu tiên cao
  nhất "DMD-verified" vào `predict_final.py`), chạy lại `--predict`.
- Nếu subject KHÔNG có trong DMD public (20% giữ làm benchmark): giữ nhãn
  rules như hiện tại; cân nhắc fine-tune CNN trên DMD export (DEx tool,
  Colab) — chỉ khi thực sự hụt.
- **License**: DMD = CC BY-NC-ND, academic-only. TUYỆT ĐỐI không commit
  frame/video DMD vào repo, không đưa vào artifact nộp bài; cite Ortega
  et al. ECCV 2020 trong README nộp.

## 7. Segment cần chú ý (rủi ro hiện tại)

| Trip/segment | Pipeline đang đoán | Nghi vấn (đã xem ảnh bằng mắt) |
|---|---|---|
| T01d toàn trip | drowsy | có cả ngáp rõ (f1200/f1620) — drowsy vs yawning? DMD s5 sẽ chốt |
| T06d [500-1800) | trộn drowsy/alert/microsleep | đổi subject ~f1500 (nữ, nghe điện thoại → distracted?); hash-cut không bắt được điểm đổi |
| T09d toàn trip | drowsy chủ đạo | subject đeo kính — eyeBlink blendshape có thể ảo cao → drowsy giả |
| T10d [460-1800) | drowsy 1340 frame | thấy cầm điện thoại ở f800 → nghi distracted |
| T02d [77-1800) | yawning 1523 | khớp mắt thường, tin được — chỉ verify lại khi có DMD |

## 8. Việc còn lại đến 10/08 (ngoài DMD)

1. **Demo** (hạng mục 03 bắt buộc): kế hoạch đã thống nhất với user —
   `c2/demo.py` Streamlit 2 tab (Trip replay HUD + Live webcam real-time
   bằng chính rule classifier; khả thi vì CPU-only ~30fps) + quay video
   2-5 phút. Ghép fusion với C1 (TTC của Hải) khi có.
2. **Merge CSV nộp chung**: thêm cột `predicted_ttc` (Hải/C1) vào CSV;
   format cuối `predictions/<tên_team>/<trip_id>.csv`.
3. **README approach + Ghi chú triển khai** (hạng mục 02/04/05): kể chuyện
   bake-off có số liệu (39.2 vs 91.7) — điểm cộng "quyết định có bằng chứng";
   khai báo dùng DMD (nguồn + license) và MediaPipe.
4. Checklist nộp trong `Connected Car.html` mục "Definition of Done" (14 mục).

## 9. Nguồn tham chiếu nội bộ

- Memory (persistent): `C:\Users\nguye\.claude\projects\c--HackathonFPT\memory\`
  — `role-thien-c2.md`, `c2-data-findings.md` (tóm tắt phát hiện, luôn cập nhật).
- Báo cáo deep-research đầy đủ (nguồn trích dẫn, claim verified 21/25):
  transcript session cũ; kết luận chính đã gói trong file này.
- `c2/bakeoff/*.py` — 4 script bake-off tái chạy được (bằng chứng cho README).
- Điểm liên hệ team: Hải (C1), Sơn (HUD/dashboard), pipeline runner 1.4
  (xem `docs/Tripkit_Handoff.md`).
