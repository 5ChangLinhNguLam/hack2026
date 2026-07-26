"""C2 — trích đặc trưng khuôn mặt per-frame bằng MediaPipe FaceLandmarker (Tasks API).

Đầu ra mỗi trip: ``c2/cache/features_{trip}.npz``:
    features (n_frames, D) float32 — NaN nếu không bắt được mặt
    names    (D,) tên từng cột

Đặc trưng gồm 2 nhóm:
1. Hình học từ 478 landmark: ear_l, ear_r (eye aspect ratio), mar (mouth
   aspect ratio), yaw/pitch/roll (solvePnP), nose_dx/nose_dy (proxy lệch
   đầu so tâm bbox mặt), face_w/face_h, face_found.
2. 52 blendshape score học sẵn của model (eyeBlinkLeft, jawOpen,
   mouthFunnel...) — nghiên cứu 26/07 kết luận đặc trưng HỌC vượt ngưỡng
   hình học cứng (EAR 0.23 bị bác 0-3), nên nhóm này là tín hiệu chính,
   nhóm hình học làm bổ trợ/đối chứng.

Model: c2/cache/face_landmarker.task (tải từ storage.googleapis.com/mediapipe-models,
float16, 3.7MB — script tự tải lại nếu thiếu).

Chạy:  python c2/features.py                # toàn bộ 16 trip (bỏ qua trip đã có)
       python c2/features.py T01d T02d     # trip chỉ định
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = Path(__file__).resolve().parent / "cache"
MODEL = CACHE / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

GEO_NAMES = ["ear_l", "ear_r", "mar", "yaw", "pitch", "roll",
             "nose_dx", "nose_dy", "face_w", "face_h", "face_found"]
N_BLEND = 52
FRAME_MS = 50  # 20 FPS

# Landmark index chuẩn FaceMesh topology (dùng chung cho 468/478 điểm)
_EYE_L = [33, 160, 158, 133, 153, 144]
_EYE_R = [362, 385, 387, 263, 373, 380]
_MOUTH_V = [(13, 14), (81, 178), (311, 402)]
_MOUTH_H = (61, 291)
_PNP_IDS = [1, 152, 33, 263, 61, 291]
_PNP_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)


def _aspect(pts: np.ndarray, ids: list[int]) -> float:
    p = pts[ids]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = 2.0 * np.linalg.norm(p[0] - p[3])
    return float(v / h) if h > 1e-9 else np.nan


def _mar(pts: np.ndarray) -> float:
    v = sum(np.linalg.norm(pts[a] - pts[b]) for a, b in _MOUTH_V)
    h = np.linalg.norm(pts[_MOUTH_H[0]] - pts[_MOUTH_H[1]])
    return float(v / (3.0 * h)) if h > 1e-9 else np.nan


def _head_pose(pts_px: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_PNP_3D, pts_px[_PNP_IDS].astype(np.float64),
                               cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return (np.nan,) * 3
    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.hypot(rmat[0, 0], rmat[1, 0]))
    pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
    yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
    return yaw, pitch, roll


def _make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODEL.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp, vision.FaceLandmarker.create_from_options(opts)


def extract_trip(trip: str) -> tuple[np.ndarray, list[str]]:
    mp, landmarker = _make_landmarker()
    files = sorted((DATA / trip / "driver").glob("frame_*.jpg"))
    blend_names = [f"bs_{i}" for i in range(N_BLEND)]  # thay bằng tên thật khi gặp mặt đầu tiên
    named = False
    out = np.full((len(files), len(GEO_NAMES) + N_BLEND), np.nan, dtype=np.float32)
    out[:, len(GEO_NAMES) - 1] = 0.0  # face_found
    with landmarker:
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), i * FRAME_MS)
            if not res.face_landmarks:
                continue
            pts = np.array([(p.x, p.y) for p in res.face_landmarks[0]], dtype=np.float32)
            pts_px = pts * (w, h)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            fw, fh = x1 - x0, y1 - y0
            yaw, pitch, roll = _head_pose(pts_px, w, h)
            out[i, :len(GEO_NAMES)] = (
                _aspect(pts, _EYE_L), _aspect(pts, _EYE_R), _mar(pts),
                yaw, pitch, roll,
                (pts[1, 0] - cx) / fw if fw > 1e-6 else np.nan,
                (pts[1, 1] - cy) / fh if fh > 1e-6 else np.nan,
                fw, fh, 1.0,
            )
            if res.face_blendshapes:
                cats = res.face_blendshapes[0]
                if not named:
                    blend_names = [c.category_name for c in cats[:N_BLEND]]
                    named = True
                out[i, len(GEO_NAMES):len(GEO_NAMES) + len(cats)] = [
                    c.score for c in cats[:N_BLEND]]
    return out, GEO_NAMES + blend_names


def main() -> int:
    trips = sys.argv[1:] or (
        [f"T{i:02d}-Sample" for i in range(1, 7)] + [f"T{i:02d}d" for i in range(1, 11)]
    )
    CACHE.mkdir(exist_ok=True)
    for trip in trips:
        dest = CACHE / f"features_{trip}.npz"
        if dest.exists():
            print(f"{trip}: đã có, bỏ qua")
            continue
        feats, names = extract_trip(trip)
        np.savez(dest, features=feats, names=np.array(names))
        found = float((feats[:, GEO_NAMES.index("face_found")] == 1.0).mean())
        print(f"{trip}: {feats.shape} face_found={found:.1%}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
