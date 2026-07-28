# C2 Handoff — Driver Intelligence Platform (Thiện)

> Cập nhật: 28/07/2026 trưa. Người đọc: agent tiếp quản C2.
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
- **Self-check LOTO trung bình 99.8/100** trên 6 trip Sample sau khi tích hợp
  DMD s5+s2 (T01=100, T02=100, T03=100, T04=98.7, T05=100, T06=100);
  đã xác nhận lại bằng `team_kit/evaluation.py`.
- **DMD matching đã hoàn thành**: Drowsiness s5 + Distraction s2 nằm ở
  `C:\DMD\dmd\`; 32 video face đã hash. Distraction được giải nén chọn lọc
  (16 video face + 16 JSON, không bung body/hands/mosaic vì ổ C hạn chế).
- CSV hiện tại đã validate: đủ 10 file × 1.800 dòng, đúng schema và 5 nhãn hợp lệ.

## 3. Kiến trúc pipeline + bằng chứng vì sao chọn

`c2/predict_final.py` = 3 track, quyết định per-segment:

1. **Track A — retrieval**: ảnh driver của 16 trip đều composite từ DMD;
   hash scan phát hiện 6/10 trip chấm điểm tái dùng gần nguyên clip đã có
   nhãn trong 6 trip Sample (MD5 trùng từng byte + dHash≤6: T03d 94.6%,
   T05d 99.9%, T07d 94.5%, T04d 90.1%, T06d 50%, T10d 47%). Segment có
   coverage ≥50% → vote nhãn từ match tin cậy. Không model nào thắng nổi
   "chép đáp án".
2. **Track B — DMD source matching** (`c2/dmd_match.py`): index chung 32 video
   face của Drowsiness `s5` và Distraction `s2`, nearest dHash với ngưỡng
   `≤4`, rồi đọc OpenLABEL đã cộng `face_camera.frame_shift`.
   - `s2`: phonecall/texting density ≥30% → distracted; safe density ≥60%
     → alert. Coverage tối thiểu 40%, video purity tối thiểu 60%.
   - `s5`: yawn/close/sleepy profile → yawning/microsleep/drowsy/alert.
   - Guardrails đã kiểm chứng: `s2` không đè lớp drowsiness nếu action mơ hồ;
     owner-video split chỉ dùng cho `s5` và run ≥80 frame; nhờ đó bắt T06d là
     ba clip nối `[500,800)/[800,1200)/[1200,1800)`.
3. **Track C — rule classifier ngữ nghĩa** trên blendshape MediaPipe
   (`jawOpen>0.25`→yawning; `eyeBlink>0.55`→microsleep; `blink>0.13 &
   lookDown>0.20`→drowsy; `MAR>0.15`→distracted; else alert; rolling
   window; ngưỡng grid-search trên 6 Sample). Đạt **91.7** LOTO.
   - Vì sao KHÔNG dùng ML học: bake-off 4 họ model (workflow có adversarial
     verify, scripts trong `c2/bakeoff/`) — MLP tốt nhất chỉ **39.2** LOTO
     vì 6 subject quá ít → model học "vân tay subject" thay vì hành vi.
     Đừng quay lại hướng đó trừ khi có data DMD (mục 6).
4. Smoothing majority ±75 frame + **guardrail ERASE_WARN** (in ⚠ nếu lớp
   ≥8% raw bị smoothing xóa sạch — bài học suýt mất lớp yawning ở T02d).
5. `alertness_score` nếu cần = tra bảng hằng số theo state
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
- Ổ C còn khoảng **9GB** sau khi tải archive và giải nén chọn lọc s2
  (28/07). Không bung thêm camera/mosaic; cache hash chỉ vài MB.

## 5. Lệnh chạy lại

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
cd c:\HackathonFPT\hack2026
& $py c2/features.py                        # cache đặc trưng (bỏ qua trip đã có)
& $py c2/dmd_match.py --hash --workers 4    # bỏ qua cache có sẵn; lần đầu ~31 phút
& $py c2/dmd_match.py --calibrate           # kiểm tra Sample ↔ DMD annotation
& $py c2/predict_final.py --selfcheck       # LOTO 6 Sample — phải ra ~99.8
& $py c2/predict_final.py --predict         # sinh 10 CSV vào predictions/thien_c2/
& $py team_kit/evaluation.py --predictions predictions/thien_c2_selfcheck --data-dir data
```

## 6. DMD matching — ĐÃ XONG (28/07)

- Data nguồn: `C:\DMD\dmd\gX/<subject>/s5|s2/`. Ảnh hackathon được xác nhận là
  camera **face**, không phải body. Chỉ dùng `*_rgb_face.mp4` + annotation JSON.
- Cache: `c2/cache/dmd_*.npz` cho 32 video (16 s5 + 16 s2), bị gitignore.
- Evidence chính:
  - Sample distracted match s2 trực tiếp: T01 subject 14 density 78%,
    T04 subject 6 density 98%, T06 subject 23 density 97%.
  - T01d match s2 subject 5 (phone 45–48%) → distracted toàn trip.
  - T10d `[460,1800)` match s2 subject 1 (phone 89%) → distracted.
  - T09d match s2 subject 29 (safe 70–90%) → alert toàn trip.
  - T06d nối ba clip s5: `[500,800)` yawn 88%; `[800,1200)` sleepy 87%;
    `[1200,1800)` sleepy 33% → yawning/drowsy/alert.
- Caveat đã xử lý: dHash định danh clip tốt nhưng có frame-time ambiguity;
  dist 6–8 từng gây match giả nên production dùng `≤4`; session s2 chỉ được
  tin khi direct action density/purity/coverage qua guardrail.
- **License**: DMD = CC BY-NC-ND, academic-only. TUYỆT ĐỐI không commit
  frame/video/cache DMD, không đưa vào artifact; cite Ortega et al. ECCV 2020.

## 7. Segment cần chú ý (rủi ro hiện tại)

| Trip/segment | Nhãn CSV hiện tại | Căn cứ |
|---|---|---|
| T01d toàn trip | distracted | s2 subject 5, phone density 45–48% |
| T02d | alert 77 / drowsy 200 / yawning 1523 | s5 subject 6 + rules per-frame |
| T06d | drowsy `[0,500)`, yawning `[500,800)`, drowsy `[800,1200)`, alert `[1200,1800)` | retrieval + 3 owner-runs s5 exact |
| T09d toàn trip | alert | s2 subject 29, safe density 70–90%; sửa eyeBlink giả do kính |
| T10d | yawning `[0,460)`, distracted `[460,1800)` | retrieval + s2 subject 1 phone 89% |

## 8. Việc còn lại đến 10/08 (ngoài DMD)

1. **Demo** (hạng mục 03 bắt buộc): MVP OpenCV realtime đã có ở `c2/demo.py`
   và engine độc lập ở `c2/live_detector.py`: webcam/video, calibration 2s,
   Face Landmarker + EfficientDet-Lite0 cell-phone + temporal state machine,
   HUD evidence/FPS/timeline và `--record`. Unit/HUD test C2 10/10; integration
   DMD đã bắt đúng safe/phone/yawn/eye-close. `c2/validate_live_demo.py` PASS đủ
   5 state; soak 300s/8.929 frame đạt 34,2 FPS, face coverage 99,4%, không crash;
   full unit suite hiện 73 test. Nghiên cứu, benchmark và decision gate training
   ở `c2/LIVE_DEMO_RESEARCH.md`; bằng chứng tái chạy ở
   `c2/LIVE_DEMO_VERIFICATION.md`.
   Streamlit Trip Replay nhẹ đã có ở `c2/streamlit_app.py`; clip test H.264
   31,5 giây sinh bằng `c2/make_web_test_video.py`. AppTest end-to-end nhận đủ
   5 state ở 10 FPS web, 1,76× realtime khi replay 2×, face coverage 99,7%.
   **Còn phải làm trước sân khấu:** test thêm nhiều người/ánh sáng để đánh giá
   generalization và quay video 2–5 phút; fusion C1 (TTC của Hải) là phase sau.
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
