# C2 realtime live-demo — research và quyết định kiến trúc

> Cập nhật: 28/07/2026. Phạm vi: MVP chạy được trước, không thay thế pipeline
> leaderboard trong `predict_final.py`.

## 1. Kết luận

**MVP không cần train model mới và chưa cần multi-task network.**

Kiến trúc phù hợp nhất với deadline và máy CPU-only:

```text
Webcam/video
  ├─ MediaPipe Face Landmarker
  │    └─ eye blink, jaw open, gaze proxy, head pose
  ├─ EfficientDet-Lite0 int8 (chạy thưa)
  │    └─ cell phone + bounding box
  └─ Temporal state machine
       └─ alert / drowsy / yawning / distracted / microsleep
```

Lý do:

1. Hai pretrained task đã đủ nhanh trên máy đích.
2. Dữ liệu DMD hiện có nhiều frame nhưng chỉ 14 người; frame liền nhau không phải
   306.000 quan sát độc lập.
3. Annotation DMD là các tín hiệu phụ (eye state, yawn, phone action...), không
   phải ground truth trực tiếp cho đủ năm lớp hackathon.
4. Bake-off hiện tại đã cho thấy classifier học từ ít subject bị subject leakage:
   MLP chỉ đạt 39,2 LOTO, trong khi semantic rules đạt 91,7.
5. Một model mới sẽ kéo theo remap nhãn, split theo subject, train, export TFLite/
   ONNX, quantization và kiểm thử domain webcam. Các việc này chưa cần để có một
   live demo đáng tin cậy.

## 2. Audit tài nguyên hiện tại

### Code và model

- `c2/features.py`: đã trích 478 face landmarks, 52 blendshapes và head pose,
  nhưng đang chạy theo batch trip.
- `c2/cache/face_landmarker.task`: pretrained model đã có.
- `c2/predict_final.py`: pipeline CSV/retrieval, không dùng được cho webcam mới.
- Camera 0 hoạt động ở 640×480, tốc độ capture quan sát được khoảng 21,7 FPS.
- Máy không có NVIDIA GPU; MediaPipe đang dùng CPU/XNNPACK.

### Dữ liệu DMD đã có trên máy

| Phần | Video face | Frame | Thời lượng |
|---|---:|---:|---:|
| s2 distraction | 16 | 218.342 | 2,04 giờ |
| s5 drowsiness | 16 | 88.222 | 0,82 giờ |
| Tổng | 32 | 306.564 | 2,86 giờ |

Thông số chung: 14 subject, RGB 1280×720, khoảng 29,76 FPS.

Nhãn s2 có trực tiếp `safe_drive`, `phonecall_left/right`,
`texting_left/right`, gaze on/off-road và object `cellphone`. Nhãn s5 có
`eyes_state/open/close/opening/closing`, blink và yawn.

Các khoảng annotation hiện có gồm khoảng:

- 116.318 frame phonecall/texting;
- 126.814 frame có object cellphone;
- 8.827 frame mắt đóng;
- 12.427 frame yawning.

Số trên có overlap và chỉ dùng để hiểu độ phủ annotation, không phải class
distribution năm lớp.

### Khoảng trống dữ liệu nếu muốn train

1. Không có nhãn `drowsy` trực tiếp theo đúng định nghĩa challenge. Trạng thái này
   phải suy ra theo thời gian từ eye closure/blink/gaze.
2. JSON DMD cho biết khoảng thời gian có điện thoại nhưng không có bounding box
   điện thoại. Fine-tune object detector sẽ cần annotate box thủ công.
3. Chỉ có 14 subject được phép tải hiện nay. Mọi validation phải split theo
   subject; random frame split sẽ làm điểm số ảo vì nền, khuôn mặt và góc camera
   của cùng một người xuất hiện ở cả train và test.
4. Chưa có test set quay bằng đúng webcam, ánh sáng và vị trí trình diễn.
5. DMD chỉ cho mục đích academic và dùng giấy phép CC BY-NC-ND 4.0. Không commit
   hoặc phân phối lại video/frame; khi dùng phải trích dẫn nguồn.

Nguồn chính thức: [DMD dataset](https://dmd.vicomtech.org/),
[DMD repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset),
[DMD paper](https://arxiv.org/abs/2008.12085).

## 3. Benchmark trên máy demo

### Face Landmarker

Đo 300 frame mỗi nguồn:

| Nguồn 640×360 | Mean | P95 | Face found |
|---|---:|---:|---:|
| Hackathon trip | 10,12 ms | 11,49 ms | 100% |
| DMD video | 9,78 ms | 11,20 ms | 100% |

MediaPipe hỗ trợ VIDEO/LIVE_STREAM, trả 478 landmarks và 52 blendshapes. VIDEO
và LIVE_STREAM dùng tracking để giảm latency. MVP dùng VIDEO mode đồng bộ vì
10 ms nhỏ hơn nhiều so với một chu kỳ camera 20 FPS và cho kết quả deterministic.
[Face Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).

### Cell-phone detector

EfficientDet-Lite0 int8, input 320×320:

| Đo trên 190 frame DMD cân bằng phone/no-phone | Kết quả |
|---|---:|
| Mean inference | 26,16 ms |
| P95 inference | 29,76 ms |
| Threshold 0,20 — precision | 87,8% |
| Threshold 0,20 — recall | 37,9% |
| Threshold 0,20 — false-positive rate | 5,3% |

Đây là sample audit, chưa phải benchmark học thuật toàn dataset. Recall từng
frame thấp nhưng video có nhiều cơ hội liên tiếp. State machine vì vậy yêu cầu
hai detection độc lập trong 1,5 giây và chỉ chạy detector mỗi năm frame.

MediaPipe khuyến nghị EfficientDet-Lite0 như lựa chọn cân bằng accuracy/latency;
model COCO có lớp `cell phone`.
[Object Detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python),
[model benchmark](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector),
[COCO labels](https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt).

### End-to-end

- DMD safe segment: 31,5 FPS headless, 100% face found, toàn bộ sau calibration
  là `alert`.
- DMD phone segment: 33,8 FPS headless; chuyển `alert → distracted` bằng hai
  phone detections, best score 0,48.
- DMD yawn segment: 43,5 FPS khi tắt phone detector; phát hiện `yawning`.
- DMD eye-close segment: phát hiện `microsleep`, sau đó `drowsy` theo fatigue
  trend.
- Integration gate `validate_live_demo.py` PASS đủ năm state trên bốn segment
  annotated, face coverage 100%, 41,9–56,4 FPS. Đây là functional regression
  test trên một subject, không phải claim cross-subject accuracy.
- Soak test toàn pipeline 300 giây source: 8.929 frame, 34,2 FPS headless,
  face coverage 99,4%, Face Landmarker 10,6 ms và phone detector 25,8 ms,
  không crash.

FPS headless không phải FPS camera thực tế; camera hiện giới hạn khoảng 20–22
FPS. Mục tiêu demo hợp lý là **≥15 FPS end-to-end**.

## 4. Temporal logic của MVP

MVP calibration hai giây với tư thế neutral, sau đó dùng head pose và eye-gaze
tương đối so với baseline từng người.

| State | Evidence chính |
|---|---|
| alert | Không có evidence nguy hiểm đủ lâu |
| yawning | Jaw open cao liên tục khoảng 0,65 giây |
| microsleep | Mắt đóng liên tục khoảng 1,2 giây, trừ eye-squeeze khi đang yawn |
| distracted | Hai phone hits/1,5 giây; hoặc head/gaze off-road khoảng 0,9 giây; hoặc mất mặt 1,2 giây |
| drowsy | Nhiều lần nhắm mắt dài hoặc tỷ lệ eye closure cao trong cửa sổ 10 giây |

Các giá trị trên là **demo thresholds**, không phải tiêu chuẩn y tế. PERCLOS
chuẩn xét tỷ lệ thời gian mắt đóng 80–100%, loại blink, thường trên cửa sổ một
phút; cửa sổ 10 giây ở đây chỉ được hiển thị là `fatigue trend`.
[NHTSA/FHWA PERCLOS brief](https://rosap.ntl.bts.gov/view/dot/113).

State machine có:

- calibration theo người;
- minimum dwell time;
- state hold 0,7 giây để tránh nhấp nháy;
- ưu tiên direct phone/eye/yawn evidence hơn fatigue trend;
- reason và evidence bars để giải thích dự đoán.

## 5. Multi-task architecture có cần không?

### Không cần cho MVP

Một multi-task network là hợp lý về nghiên cứu: backbone MobileNet dùng chung,
các head dự đoán eye closure, mouth opening và head direction, sau đó suy trạng
thái theo thời gian. Nghiên cứu MT-MobileNets cũng theo hướng dự đoán các facial
behaviors riêng rồi dùng duration/PERCLOS để suy driver status.
[Lightweight Multi-Task MobileNets](https://www.mdpi.com/1424-8220/19/14/3200).

Tuy nhiên, model như vậy không làm MVP nhanh hơn hoặc an toàn hơn model
pretrained đang chạy 10 ms/frame.

### Kiến trúc phase 2 nếu thực sự cần train

```text
MobileNetV3-Small / EfficientNet-Lite0 shared backbone
  ├─ eye-state head
  ├─ yawn/mouth head
  ├─ head/gaze head
  └─ distraction-action head
          ↓
  causal TCN hoặc GRU 1 chiều
          ↓
  five-state output
```

Điện thoại vẫn nên giữ một object detector riêng để có bounding box và bằng
chứng trực quan. Với video action model, DMD từng thử MobileNet+LSTM, Conv3D và
Conv2D-LSTM; temporal model có thể đạt tốt nhưng kết quả được đo trên GPU T4 và
task distraction riêng, không chứng minh rằng nó sẽ tổng quát ngay cho năm lớp
của challenge.
[DMD image/video comparison](https://www.scitepress.org/PublishedPapers/2021/102445/).

## 6. Decision gate trước khi train

Quay một validation set nhỏ nhưng đúng domain:

- ít nhất 5 người;
- mỗi người 1–2 phút;
- neutral, yawn, nhắm mắt >1,2 giây, nhìn trái/phải/xuống, texting và phonecall;
- thay đổi kính, khoảng cách và ánh sáng;
- split/report theo người.

Chỉ train khi một trong các điều sau xảy ra:

| Vấn đề sau live test | Hành động |
|---|---|
| Phone miss quá nhiều | Annotate 500–1.000 phone boxes, fine-tune detector nhẹ |
| Eye/yawn threshold lệch giữa người | Học auxiliary eye/yawn heads hoặc calibration tốt hơn |
| Head/gaze false alert | Thêm gaze-zone dataset/head-pose calibration |
| Năm state vẫn không ổn sau temporal tuning | Train causal TCN/GRU trên chuỗi auxiliary signals |

Không train direct five-class CNN trên random DMD frames.

## 7. Definition of Done cho MVP

- [x] Webcam và `--video` đều chạy.
- [x] Neutral calibration tối đa 2–3 giây.
- [x] HUD có state, confidence, reason, evidence, FPS và bounding box phone.
- [x] Integration gate quan sát đủ năm trạng thái.
- [x] End-to-end ≥15 FPS trên laptop.
- [x] Chạy 300 giây source/8.929 frame không crash.
- [x] Có thể quay HUD ra MP4 bằng `--record`.
- [x] Có video DMD dự phòng khi webcam/ánh sáng tại sân khấu gặp lỗi.
- [x] Không gọi retrieval hoặc đọc DMD annotation trong live runtime.

Việc test thêm 5–10 người vẫn cần trước khi claim khả năng tổng quát, nhưng không
phải blocker cho MVP demo.

## 8. Lệnh chạy

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
cd C:\HackathonFPT\hack2026

# Live webcam
& $py c2/demo.py

# Tắt phone detector để debug facial states
& $py c2/demo.py --no-phone

# Video dự phòng
& $py c2/demo.py --video C:\path\to\face_video.mp4

# Ghi video HUD
& $py c2/demo.py --record artifacts\c2_live_demo.mp4

# Regression gate đủ 5 state trên video annotated
& $py c2/validate_live_demo.py --dmd-root C:\DMD\dmd `
    --json-out c2\cache\live_validation.json
```
