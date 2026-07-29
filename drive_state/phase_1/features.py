"""Per-frame face features from MediaPipe FaceLandmarker.

Everything downstream (thresholds, temporal smoothing, evaluation) runs off the
:class:`FrameFeatures` rows this module produces, so extraction is done once per
trip and cached to CSV. That keeps the expensive part (landmarking 3,600 frames)
out of the threshold-tuning loop.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from drive_state.phase_1.phone import PhoneObservation

# Landmark indices, in the p1..p6 order the aspect-ratio formulas expect
# (outer corner, upper-outer, upper-inner, inner corner, lower-inner, lower-outer).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (61, 81, 13, 291, 14, 178)

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Blendshapes worth keeping: MediaPipe's own eye/jaw estimates, which are
# trained rather than geometric and survive head rotation far better than EAR.
BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthClose")


@dataclass(slots=True)
class FrameFeatures:
    """One row per frame. `face_found` false means every other field is 0."""

    frame_id: int
    timestamp: float
    face_found: int

    ear: float  # mean eye aspect ratio
    ear_left: float
    ear_right: float
    mar: float  # mouth aspect ratio

    # Head rotation in degrees, from the facial transformation matrix.
    # yaw > 0 turns to the driver's left, pitch > 0 tips the chin up.
    yaw: float
    pitch: float
    roll: float

    # Landmark-geometry fallbacks for head pose, scale-free so they survive the
    # driver sitting at different distances from the camera.
    nose_offset_x: float  # nose vs eye-midpoint, in inter-ocular widths
    nose_offset_y: float
    face_ratio: float  # face height / width; drops when the head tips down

    blink_left: float
    blink_right: float
    blink: float
    jaw_open: float
    mouth_close: float

    face_scale: float  # inter-ocular distance in px, for normalisation

    # Best `cell phone` detection in the cabin, and where it was. Zero when the
    # detector is disabled or found nothing. `phone_fresh` is 1 on frames the
    # detector actually ran and 0 on frames holding a previous result.
    phone_conf: float
    phone_x: float
    phone_y: float
    phone_w: float
    phone_h: float
    phone_fresh: int


FEATURE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FrameFeatures))


def _blank(frame_id: int, timestamp: float, phone: PhoneObservation | None = None) -> FrameFeatures:
    values = dict.fromkeys(FEATURE_FIELDS, 0.0)
    values["frame_id"] = frame_id
    values["timestamp"] = timestamp
    values["face_found"] = 0
    values.update(_phone_fields(phone))
    return FrameFeatures(**values)  # type: ignore[arg-type]


def _phone_fields(phone: PhoneObservation | None) -> dict[str, float | int]:
    if phone is None:
        return {
            "phone_conf": 0.0,
            "phone_x": 0.0,
            "phone_y": 0.0,
            "phone_w": 0.0,
            "phone_h": 0.0,
            "phone_fresh": 0,
        }
    x, y, w, h = phone.bbox
    return {
        "phone_conf": phone.confidence,
        "phone_x": float(x),
        "phone_y": float(y),
        "phone_w": float(w),
        "phone_h": float(h),
        "phone_fresh": int(phone.fresh),
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _aspect_ratio(points: list[tuple[float, float]]) -> float:
    """Generic |p2-p6| + |p3-p5| over 2*|p1-p4|; used for both eyes and mouth."""
    width = _distance(points[0], points[3])
    if width == 0:
        return 0.0
    return (_distance(points[1], points[5]) + _distance(points[2], points[4])) / (2.0 * width)


def _euler_from_matrix(matrix: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from MediaPipe's 4x4 head transform.

    The rotation block is applied in the canonical face frame, so a standard
    XYZ decomposition gives head angles directly. Sign convention is fixed up
    by the caller's threshold signs, not here.
    """
    r = matrix[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:  # gimbal lock; roll is undefined, fold it into yaw
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class FaceFeatureExtractor:
    """Wraps a MediaPipe FaceLandmarker in VIDEO mode.

    VIDEO mode (rather than IMAGE) is deliberate: it lets the landmarker track
    across frames, which stabilises the eye landmarks that EAR is most
    sensitive to. It requires monotonically increasing timestamps, so one
    extractor instance handles exactly one trip.
    """

    def __init__(self, model_path: str | Path = "models/face_landmarker.task") -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model missing: {model_path}. "
                "Run `python scripts/download_models.py --mediapipe-face`."
            )
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        # Last frame's raw landmarks, for the demo overlay. Kept off
        # FrameFeatures on purpose: 478 points per frame would bloat the cached
        # CSVs that the tuner reads thousands of times.
        self.last_points: list[tuple[float, float]] = []
        self.last_bbox: tuple[int, int, int, int] | None = None
        self._detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> FaceFeatureExtractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def extract(
        self,
        frame: npt.NDArray[np.uint8],
        frame_id: int,
        timestamp: float,
        phone: PhoneObservation | None = None,
    ) -> FrameFeatures:
        """Landmark one frame. `phone` comes from a separate detector so the
        expensive object model can run on its own schedule -- see `phone.py`."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe wants integer milliseconds and rejects a repeated stamp.
        result = self._detector.detect_for_video(image, int(round(timestamp * 1000)))
        if not result.face_landmarks:
            # A lost face does not invalidate the phone: the driver looking down
            # at a handset is exactly when tracking fails and the phone matters.
            self.last_points = []
            self.last_bbox = None
            return _blank(frame_id, timestamp, phone)

        height, width = frame.shape[:2]
        points = [(lm.x * width, lm.y * height) for lm in result.face_landmarks[0]]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.last_points = points
        self.last_bbox = (
            int(max(0, min(xs))),
            int(max(0, min(ys))),
            int(min(width, max(xs)) - max(0, min(xs))),
            int(min(height, max(ys)) - max(0, min(ys))),
        )

        left = [points[i] for i in LEFT_EYE]
        right = [points[i] for i in RIGHT_EYE]
        mouth = [points[i] for i in MOUTH]
        ear_left = _aspect_ratio(left)
        ear_right = _aspect_ratio(right)
        mar = _aspect_ratio(mouth)

        eye_l = points[LEFT_EYE_OUTER]
        eye_r = points[RIGHT_EYE_OUTER]
        face_scale = _distance(eye_l, eye_r) or 1.0
        eye_mid = ((eye_l[0] + eye_r[0]) / 2.0, (eye_l[1] + eye_r[1]) / 2.0)
        nose = points[NOSE_TIP]
        nose_offset_x = (nose[0] - eye_mid[0]) / face_scale
        nose_offset_y = (nose[1] - eye_mid[1]) / face_scale
        face_height = _distance(points[FOREHEAD], points[CHIN])
        face_width = _distance(points[LEFT_CHEEK], points[RIGHT_CHEEK]) or 1.0

        yaw = pitch = roll = 0.0
        if result.facial_transformation_matrixes:
            yaw, pitch, roll = _euler_from_matrix(
                np.asarray(result.facial_transformation_matrixes[0])
            )

        scores = {c.category_name: c.score for c in (result.face_blendshapes or [[]])[0]}
        blink_left = scores.get("eyeBlinkLeft", 0.0)
        blink_right = scores.get("eyeBlinkRight", 0.0)

        return FrameFeatures(
            frame_id=frame_id,
            timestamp=timestamp,
            face_found=1,
            ear=(ear_left + ear_right) / 2.0,
            ear_left=ear_left,
            ear_right=ear_right,
            mar=mar,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            nose_offset_x=nose_offset_x,
            nose_offset_y=nose_offset_y,
            face_ratio=face_height / face_width,
            blink_left=blink_left,
            blink_right=blink_right,
            blink=(blink_left + blink_right) / 2.0,
            jaw_open=scores.get("jawOpen", 0.0),
            mouth_close=scores.get("mouthClose", 0.0),
            face_scale=face_scale,
            **cast(Any, _phone_fields(phone)),
        )


def write_features_csv(path: str | Path, rows: Iterable[FrameFeatures]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _round(v) for k, v in asdict(row).items()})
    return path


def read_features_csv(path: str | Path) -> list[FrameFeatures]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [_row_from_csv(row) for row in csv.DictReader(handle)]


def iter_features_csv(path: str | Path) -> Iterator[FrameFeatures]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield _row_from_csv(row)


INT_FIELDS = frozenset({"frame_id", "face_found", "phone_fresh"})


def _row_from_csv(row: dict[str, str]) -> FrameFeatures:
    values: dict[str, object] = {}
    for name in FEATURE_FIELDS:
        # Phone columns were added after the first feature caches were written;
        # default them so an older CSV still loads (as if the detector was off).
        raw = row.get(name)
        if raw is None or raw == "":
            values[name] = 0
            continue
        values[name] = int(raw) if name in INT_FIELDS else float(raw)
    return FrameFeatures(**values)  # type: ignore[arg-type]


def _round(value: object) -> object:
    return round(value, 5) if isinstance(value, float) else value
