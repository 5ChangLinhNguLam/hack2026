# Agent handoff — C2 realtime driver state

> Cập nhật: 28/07/2026. Đọc cùng `c2/HANDOFF.md` để có bối cảnh đầy đủ về
> pipeline CSV, DMD matching và các quyết định bake-off trước đó.

## 1. Mục tiêu hiện tại

C2 không còn chỉ sinh CSV. MVP hiện có hai bề mặt demo dùng chung một engine:

- OpenCV webcam/video HUD: `c2/demo.py`;
- Streamlit MP4 realtime replay: `c2/streamlit_app.py`.

Engine nhận diện năm trạng thái:

`alert`, `distracted`, `drowsy`, `yawning`, `microsleep`.

Đây là functional hackathon demo, không phải hệ thống y tế hoặc safety-certified.

## 2. Kiến trúc đã chốt

`c2/live_detector.py` chứa:

- MediaPipe Face Landmarker chạy mỗi frame;
- EfficientDet-Lite0 COCO cell-phone detector chạy theo interval;
- neutral calibration theo từng người;
- temporal state machine và arbitration giữa phone/head pose/yawn/eye closure;
- evidence, confidence, reason và event transition.

Không train multitask model cho C2 MVP vì dữ liệu có nhãn theo subject còn ít và
bake-off trước đó cho thấy model học fingerprint của subject. Xem
`c2/LIVE_DEMO_RESEARCH.md` trước khi đổi kiến trúc.

## 3. Các file cần biết

| File | Vai trò |
|---|---|
| `c2/live_detector.py` | Model adapters và temporal state machine |
| `c2/demo.py` | Webcam/video OpenCV HUD, headless smoke, record |
| `c2/streamlit_app.py` | Web MP4 realtime replay |
| `c2/validate_live_demo.py` | Integration gate trên bốn DMD segment |
| `c2/make_web_test_video.py` | Sinh MP4 local 31,5 giây để test web |
| `c2/requirements-live.txt` | Runtime dependencies |
| `c2/LIVE_DEMO_RESEARCH.md` | Research, benchmark và training decision gate |
| `c2/LIVE_DEMO_VERIFICATION.md` | Bằng chứng validation/soak/test |
| `c2/STREAMLIT_WEB_DEMO.md` | Hướng dẫn chạy web |
| `tests/test_c2_live_detector.py` | Unit và HUD tests |
| `run_c2_web.cmd` | Launcher một click trên Windows |

## 4. Chạy trên Windows

Repository local:

```text
C:\HackathonFPT\hack2026
```

Virtual environment đã được tạo và bị gitignore:

```powershell
cd C:\HackathonFPT\hack2026
.\.venv\Scripts\Activate.ps1
python -m streamlit run c2\streamlit_app.py
```

Không cần activate:

```powershell
.\.venv\Scripts\python.exe -m streamlit run c2\streamlit_app.py
```

Hoặc double-click `run_c2_web.cmd`.

## 5. Chạy trên GPU server

Team thông báo server AWS `g4dn.xlarge`, Tesla T4, disk 200 GB gp3, region
`ap-southeast-1`; thư mục làm việc bắt buộc là `/home/workspaces`.

Hostname, username và SSH config chưa được lưu trong repo. Không commit private
key, `.pem`, token, IP nội bộ hoặc credential.

Bootstrap dự kiến sau khi được cấp SSH:

```bash
cd /home/workspaces
git clone <repo-url> hack2026
cd hack2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r c2/requirements-dev.txt
python -m streamlit run c2/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Pipeline MediaPipe hiện chủ yếu chạy CPU; Tesla T4 chỉ thực sự cần khi agent mở
nhánh training PyTorch/CUDA sau khi training decision gate được thỏa mãn.

## 6. Model và dữ liệu local-only

Các file sau nằm trong `c2/cache/` và tuyệt đối không commit:

- `face_landmarker.task`;
- `efficientdet_lite0_int8.tflite`;
- `web_test_realtime.mp4`;
- mọi cache/ảnh/video/frame trích từ DMD.

`live_detector.ensure_models()` có thể tải model pretrained khi cache thiếu.

DMD là dữ liệu nghiên cứu CC BY-NC-ND. MP4 test được sinh local từ DMD chỉ dùng
để kiểm thử nội bộ, không deploy công khai và không đưa vào artifact GitHub.

## 7. Bằng chứng hiện tại

Kết quả gần nhất trên Windows/Python 3.12 CPU:

- full test suite: `73 passed`;
- DMD integration gate: đủ cả năm state;
- MP4 web: 938 raw frames, 31,52 giây, H.264, 640×360;
- Streamlit AppTest: 313 sampled frames, đủ năm state, face coverage 99,7%;
- replay 2× đạt 1,76× realtime trong AppTest;
- headless engine trên MP4: 35,1 FPS, face coverage 99,9%;
- soak 300 giây: 34,2 FPS, face coverage 99,4%, không crash.

Lệnh validation:

```powershell
python -m pytest -q
python c2/validate_live_demo.py --dmd-root C:\DMD\dmd
python c2/demo.py --video c2/cache/web_test_realtime.mp4 `
  --headless --max-seconds 32 --phone-every 5
```

## 8. Việc tiếp theo

Ưu tiên trước C2:

1. Test nhiều người, kính, ánh sáng và camera placement thực tế.
2. Quay backup demo video 2–5 phút không cắt ghép.
3. Tích hợp C1 TTC khi contract dữ liệu từ Hải ổn định.
4. Chỉ cân nhắc training model khi có split theo subject và validation đủ mạnh.

Không được:

- tuyên bố clinical PERCLOS hoặc safety certification;
- dùng leaderboard/test labels để tune;
- commit DMD material, model cache, `.venv` hoặc SSH credential;
- thay temporal engine bằng model học máy mà không chạy lại decision gate.

## 9. Git handoff

Branch làm việc: `C2`.

Trước khi commit/push:

```bash
git status -sb
git diff --check
python -m pytest -q
```

Chỉ stage source/docs/tests liên quan C2. Không dùng `git add -A` nếu worktree
có ảnh chat, data, predictions hoặc thay đổi của thành viên khác.
