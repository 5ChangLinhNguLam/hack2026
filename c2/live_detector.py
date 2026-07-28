"""Realtime driver-state engine for the C2 live demo.

The leaderboard pipeline in :mod:`predict_final` uses retrieval against known
trips.  That is deliberately not used here: a webcam must work for a new
person.  This module fuses two lightweight, pretrained MediaPipe tasks with a
small temporal state machine:

* Face Landmarker -> eyes, mouth, gaze proxy and head pose.
* EfficientDet-Lite0 -> COCO ``cell phone`` detections.
* TemporalStateMachine -> stable C2 states and human-readable evidence.

No DMD frame, subject identity or leaderboard answer is required at runtime.
"""

from __future__ import annotations

import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import cv2
import numpy as np


C2_DIR = Path(__file__).resolve().parent
CACHE = C2_DIR / "cache"
FACE_MODEL = CACHE / "face_landmarker.task"
PHONE_MODEL = CACHE / "efficientdet_lite0_int8.tflite"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
PHONE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
)

# Landmark indices from the MediaPipe Face Mesh topology.
_MOUTH_V = ((13, 14), (81, 178), (311, 402))
_MOUTH_H = (61, 291)
_PNP_IDS = (1, 152, 33, 263, 61, 291)
_PNP_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

VALID_STATES = ("alert", "drowsy", "yawning", "distracted", "microsleep")


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...", flush=True)
    urllib.request.urlretrieve(url, path)


def ensure_models(*, phone: bool = True) -> None:
    """Download the small official MediaPipe models when not cached."""

    _download_if_missing(FACE_MODEL, FACE_MODEL_URL)
    if phone:
        _download_if_missing(PHONE_MODEL, PHONE_MODEL_URL)


def _mouth_aspect_ratio(points: np.ndarray) -> float:
    vertical = sum(np.linalg.norm(points[a] - points[b]) for a, b in _MOUTH_V)
    horizontal = np.linalg.norm(points[_MOUTH_H[0]] - points[_MOUTH_H[1]])
    return float(vertical / (3.0 * horizontal)) if horizontal > 1e-9 else 0.0


def _head_pose(
    points_px: np.ndarray, width: int, height: int
) -> tuple[float, float, float]:
    camera = np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    ok, rotation, _ = cv2.solvePnP(
        _PNP_3D,
        points_px[list(_PNP_IDS)].astype(np.float64),
        camera,
        np.zeros(4),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0
    matrix, _ = cv2.Rodrigues(rotation)
    sy = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    pitch = float(np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2])))
    yaw = float(np.degrees(np.arctan2(-matrix[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return yaw, pitch, roll


@dataclass(slots=True)
class FaceSignals:
    """Signals extracted from one frame."""

    face_found: bool = False
    jaw_open: float = 0.0
    eye_blink: float = 0.0
    look_down: float = 0.0
    mouth_aspect: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    inference_ms: float = 0.0
    face_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class PhoneDetection:
    score: float = 0.0
    box: tuple[int, int, int, int] | None = None
    inference_ms: float = 0.0


@dataclass(slots=True)
class LiveConfig:
    """Thresholds for a short, explainable hackathon demonstration.

    These are demo calibration values, not medical or regulatory thresholds.
    A two-second neutral calibration makes head-pose thresholds relative to
    the user's natural camera position.
    """

    calibration_seconds: float = 2.0
    calibration_min_frames: int = 15
    history_seconds: float = 12.0
    eye_closed_threshold: float = 0.52
    soft_eye_closed_threshold: float = 0.30
    microsleep_seconds: float = 1.20
    jaw_open_threshold: float = 0.32
    yawn_seconds: float = 0.65
    offroad_yaw_degrees: float = 22.0
    offroad_pitch_degrees: float = 18.0
    look_down_threshold: float = 0.20
    offroad_seconds: float = 0.90
    no_face_seconds: float = 1.20
    phone_score_threshold: float = 0.20
    phone_hits_required: int = 2
    phone_window_seconds: float = 1.50
    drowsy_window_seconds: float = 10.0
    drowsy_min_history_seconds: float = 5.0
    drowsy_perclos_threshold: float = 0.26
    drowsy_long_closure_seconds: float = 0.35
    drowsy_long_closures: int = 2
    state_hold_seconds: float = 0.70


@dataclass(slots=True)
class StateSnapshot:
    state: str
    confidence: float
    reason: str
    state_seconds: float
    calibrated: bool
    calibration_progress: float
    evidence: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _HistoryItem:
    timestamp: float
    face_found: bool
    jaw_open: float
    eye_blink: float
    look_down: float
    yaw_relative: float
    pitch_relative: float


class FaceSignalExtractor:
    """Synchronous Face Landmarker runner.

    VIDEO mode is intentional for the first MVP.  It uses MediaPipe tracking,
    processes frames deterministically, and benchmarks well below one webcam
    frame interval on the target laptop.  The surrounding UI is still
    realtime; LIVE_STREAM can replace this later without changing the state
    machine.
    """

    def __init__(self, model_path: Path = FACE_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        ensure_models(phone=False)
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceSignalExtractor":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def process(self, bgr: np.ndarray, timestamp_ms: int) -> FaceSignals:
        timestamp_ms = max(self._last_timestamp_ms + 1, int(timestamp_ms))
        self._last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not result.face_landmarks:
            return FaceSignals(inference_ms=elapsed_ms)

        height, width = bgr.shape[:2]
        points = np.array(
            [(point.x, point.y) for point in result.face_landmarks[0]],
            dtype=np.float32,
        )
        points_px = points * np.array((width, height), dtype=np.float32)
        x0, y0 = np.floor(points_px.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points_px.max(axis=0)).astype(int)
        yaw, pitch, roll = _head_pose(points_px, width, height)
        blend = (
            {item.category_name: float(item.score) for item in result.face_blendshapes[0]}
            if result.face_blendshapes
            else {}
        )
        return FaceSignals(
            face_found=True,
            jaw_open=blend.get("jawOpen", 0.0),
            eye_blink=(
                blend.get("eyeBlinkLeft", 0.0)
                + blend.get("eyeBlinkRight", 0.0)
            )
            / 2.0,
            look_down=(
                blend.get("eyeLookDownLeft", 0.0)
                + blend.get("eyeLookDownRight", 0.0)
            )
            / 2.0,
            mouth_aspect=_mouth_aspect_ratio(points),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            inference_ms=elapsed_ms,
            face_box=(
                max(0, x0),
                max(0, y0),
                min(width - 1, x1),
                min(height - 1, y1),
            ),
        )


class CellPhoneDetector:
    """COCO cell-phone detector backed by EfficientDet-Lite0 int8."""

    def __init__(
        self,
        model_path: Path = PHONE_MODEL,
        minimum_output_score: float = 0.01,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        ensure_models(phone=True)
        self._mp = mp
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            max_results=3,
            score_threshold=minimum_output_score,
            category_allowlist=["cell phone"],
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "CellPhoneDetector":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def process(self, bgr: np.ndarray) -> PhoneDetection:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        started = time.perf_counter()
        result = self._detector.detect(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        best_score = 0.0
        best_box = None
        for detection in result.detections:
            if not detection.categories:
                continue
            score = float(detection.categories[0].score)
            if score <= best_score:
                continue
            box = detection.bounding_box
            best_score = score
            best_box = (
                int(box.origin_x),
                int(box.origin_y),
                int(box.origin_x + box.width),
                int(box.origin_y + box.height),
            )
        return PhoneDetection(
            score=best_score,
            box=best_box,
            inference_ms=elapsed_ms,
        )


class TemporalStateMachine:
    """Convert noisy per-frame signals into stable, explainable C2 states."""

    def __init__(self, config: LiveConfig | None = None):
        self.config = config or LiveConfig()
        self._history: Deque[_HistoryItem] = deque()
        self._phone_history: Deque[tuple[float, float]] = deque()
        self._calibration: list[tuple[float, float, float, float, float]] = []
        self._calibration_started: float | None = None
        self._baseline_yaw = 0.0
        self._baseline_pitch = 0.0
        self._baseline_jaw = 0.0
        self._baseline_blink = 0.0
        self._baseline_look_down = 0.0
        self._calibrated = False
        self._last_face_time: float | None = None
        self._state = "alert"
        self._state_started = 0.0
        self._hold_until = 0.0

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def reset(self, timestamp: float = 0.0) -> None:
        self._history.clear()
        self._phone_history.clear()
        self._calibration.clear()
        self._calibration_started = None
        self._baseline_yaw = 0.0
        self._baseline_pitch = 0.0
        self._baseline_jaw = 0.0
        self._baseline_blink = 0.0
        self._baseline_look_down = 0.0
        self._calibrated = False
        self._last_face_time = None
        self._state = "alert"
        self._state_started = timestamp
        self._hold_until = timestamp

    def _calibration_progress(self, timestamp: float) -> float:
        if self._calibrated:
            return 1.0
        if self._calibration_started is None:
            return 0.0
        time_progress = (
            (timestamp - self._calibration_started)
            / max(self.config.calibration_seconds, 1e-6)
        )
        frame_progress = (
            len(self._calibration) / max(self.config.calibration_min_frames, 1)
        )
        return float(np.clip(min(time_progress, frame_progress), 0.0, 1.0))

    def _update_calibration(self, timestamp: float, face: FaceSignals) -> None:
        if self._calibrated or not face.face_found:
            return
        if self._calibration_started is None:
            self._calibration_started = timestamp
        self._calibration.append(
            (
                face.yaw,
                face.pitch,
                face.jaw_open,
                face.eye_blink,
                face.look_down,
            )
        )
        enough_time = (
            timestamp - self._calibration_started >= self.config.calibration_seconds
        )
        enough_frames = len(self._calibration) >= self.config.calibration_min_frames
        if enough_time and enough_frames:
            values = np.asarray(self._calibration, dtype=float)
            self._baseline_yaw = float(np.median(values[:, 0]))
            self._baseline_pitch = float(np.median(values[:, 1]))
            self._baseline_jaw = float(np.median(values[:, 2]))
            self._baseline_blink = float(np.median(values[:, 3]))
            self._baseline_look_down = float(np.median(values[:, 4]))
            self._calibrated = True

    def _trim(self, timestamp: float) -> None:
        oldest = timestamp - self.config.history_seconds
        while self._history and self._history[0].timestamp < oldest:
            self._history.popleft()
        phone_oldest = timestamp - self.config.phone_window_seconds
        while self._phone_history and self._phone_history[0][0] < phone_oldest:
            self._phone_history.popleft()

    def _consecutive_duration(self, predicate) -> float:
        if not self._history or not predicate(self._history[-1]):
            return 0.0
        end = self._history[-1].timestamp
        start = end
        for item in reversed(self._history):
            if not predicate(item):
                break
            start = item.timestamp
        return max(0.0, end - start)

    def _window_items(self, timestamp: float, seconds: float) -> list[_HistoryItem]:
        start = timestamp - seconds
        return [item for item in self._history if item.timestamp >= start]

    def _long_closure_count(
        self, items: list[_HistoryItem], jaw_threshold: float
    ) -> int:
        threshold = self.config.soft_eye_closed_threshold
        minimum = self.config.drowsy_long_closure_seconds
        count = 0
        start: float | None = None
        previous = 0.0
        for item in items:
            if (
                item.face_found
                and item.jaw_open < jaw_threshold
                and item.eye_blink >= threshold
            ):
                if start is None:
                    start = item.timestamp
                previous = item.timestamp
            elif start is not None:
                if previous - start >= minimum:
                    count += 1
                start = None
        if start is not None and previous - start >= minimum:
            count += 1
        return count

    def _set_state(self, candidate: str, timestamp: float) -> None:
        if candidate != "alert":
            self._hold_until = timestamp + self.config.state_hold_seconds
        elif timestamp < self._hold_until:
            candidate = self._state
        if candidate != self._state:
            self._state = candidate
            self._state_started = timestamp

    def update(
        self,
        timestamp: float,
        face: FaceSignals,
        phone_score: float | None = None,
    ) -> StateSnapshot:
        """Update the temporal model.

        ``phone_score`` is ``None`` on frames where the slower object detector
        did not run.  This distinction prevents cached results from counting as
        multiple independent phone hits.
        """

        if face.face_found:
            self._last_face_time = timestamp
        self._update_calibration(timestamp, face)
        if phone_score is not None:
            self._phone_history.append((timestamp, float(phone_score)))

        yaw_relative = face.yaw - self._baseline_yaw if self._calibrated else 0.0
        pitch_relative = (
            face.pitch - self._baseline_pitch if self._calibrated else 0.0
        )
        look_down_relative = (
            max(0.0, face.look_down - self._baseline_look_down)
            if self._calibrated
            else 0.0
        )
        self._history.append(
            _HistoryItem(
                timestamp=timestamp,
                face_found=face.face_found,
                jaw_open=face.jaw_open,
                eye_blink=face.eye_blink,
                look_down=look_down_relative,
                yaw_relative=yaw_relative,
                pitch_relative=pitch_relative,
            )
        )
        self._trim(timestamp)

        progress = self._calibration_progress(timestamp)
        if not self._calibrated:
            return StateSnapshot(
                state="alert",
                confidence=0.0,
                reason=(
                    "Keep a neutral face toward the camera"
                    if face.face_found
                    else "No face - move into the camera view"
                ),
                state_seconds=0.0,
                calibrated=False,
                calibration_progress=progress,
                evidence={
                    "eye": face.eye_blink,
                    "mouth": face.jaw_open,
                    "offroad": 0.0,
                    "phone": max((score for _, score in self._phone_history), default=0.0),
                    "fatigue": 0.0,
                },
            )

        eye_threshold = max(
            self.config.eye_closed_threshold, self._baseline_blink + 0.25
        )
        jaw_threshold = max(
            self.config.jaw_open_threshold, self._baseline_jaw + 0.20
        )
        closed_seconds = self._consecutive_duration(
            lambda item: item.face_found and item.eye_blink >= eye_threshold
        )
        yawn_seconds = self._consecutive_duration(
            lambda item: item.face_found and item.jaw_open >= jaw_threshold
        )
        offroad_seconds = self._consecutive_duration(
            lambda item: item.face_found
            and (
                abs(item.yaw_relative) >= self.config.offroad_yaw_degrees
                or abs(item.pitch_relative) >= self.config.offroad_pitch_degrees
                or item.look_down >= self.config.look_down_threshold
            )
        )
        no_face_seconds = (
            max(0.0, timestamp - self._last_face_time)
            if self._last_face_time is not None and not face.face_found
            else 0.0
        )

        phone_scores = [score for _, score in self._phone_history]
        phone_hits = sum(
            score >= self.config.phone_score_threshold for score in phone_scores
        )
        best_phone = max(phone_scores, default=0.0)

        drowsy_items = self._window_items(
            timestamp, self.config.drowsy_window_seconds
        )
        valid_eye = [
            item.eye_blink
            for item in drowsy_items
            if item.face_found and item.jaw_open < jaw_threshold
        ]
        perclos_proxy = (
            float(
                np.mean(
                    np.asarray(valid_eye) >= self.config.soft_eye_closed_threshold
                )
            )
            if valid_eye
            else 0.0
        )
        history_span = (
            drowsy_items[-1].timestamp - drowsy_items[0].timestamp
            if len(drowsy_items) >= 2
            else 0.0
        )
        long_closures = self._long_closure_count(drowsy_items, jaw_threshold)

        candidate = "alert"
        confidence = 1.0
        reason = "Face and attention signals are stable"

        # A strong yawn signal suppresses the common "eyes squeezed while
        # yawning" false microsleep.  Direct phone evidence remains higher
        # priority than head/gaze and fatigue trends.
        duration_epsilon = 1e-6
        yawn_active = (
            yawn_seconds + duration_epsilon >= self.config.yawn_seconds
        )
        microsleep_active = (
            closed_seconds + duration_epsilon >= self.config.microsleep_seconds
        )
        facial_event_visible = (
            face.eye_blink >= self.config.soft_eye_closed_threshold
            or face.jaw_open >= jaw_threshold
        )
        if no_face_seconds >= self.config.no_face_seconds:
            candidate = "distracted"
            confidence = min(
                1.0, no_face_seconds / self.config.no_face_seconds
            )
            reason = f"Face not visible for {no_face_seconds:.1f}s"
        elif microsleep_active and not yawn_active:
            candidate = "microsleep"
            confidence = min(
                1.0, closed_seconds / self.config.microsleep_seconds
            )
            reason = f"Eyes closed continuously for {closed_seconds:.1f}s"
        elif phone_hits >= self.config.phone_hits_required:
            candidate = "distracted"
            confidence = min(
                1.0,
                max(
                    best_phone / max(self.config.phone_score_threshold, 1e-6),
                    phone_hits / max(self.config.phone_hits_required, 1),
                )
                / 1.5,
            )
            reason = (
                f"Cell phone detected {phone_hits}x "
                f"(best score {best_phone:.2f})"
            )
        elif yawn_active:
            candidate = "yawning"
            confidence = min(1.0, yawn_seconds / self.config.yawn_seconds)
            reason = f"Mouth open continuously for {yawn_seconds:.1f}s"
        elif (
            offroad_seconds + duration_epsilon >= self.config.offroad_seconds
            and not facial_event_visible
        ):
            candidate = "distracted"
            confidence = min(
                1.0, offroad_seconds / self.config.offroad_seconds
            )
            reason = (
                f"Head/gaze away from road for {offroad_seconds:.1f}s "
                f"(yaw {yaw_relative:+.0f}°)"
            )
        elif (
            history_span >= self.config.drowsy_min_history_seconds
            and (
                perclos_proxy >= self.config.drowsy_perclos_threshold
                or long_closures >= self.config.drowsy_long_closures
            )
        ):
            candidate = "drowsy"
            confidence = min(
                1.0,
                max(
                    perclos_proxy
                    / max(self.config.drowsy_perclos_threshold, 1e-6),
                    long_closures / max(self.config.drowsy_long_closures, 1),
                )
                / 1.5,
            )
            reason = (
                f"Fatigue trend: closure={perclos_proxy:.0%}, "
                f"long closures={long_closures}"
            )
        else:
            strongest = max(
                closed_seconds / max(self.config.microsleep_seconds, 1e-6),
                yawn_seconds / max(self.config.yawn_seconds, 1e-6),
                offroad_seconds / max(self.config.offroad_seconds, 1e-6),
                perclos_proxy / max(self.config.drowsy_perclos_threshold, 1e-6),
                best_phone / max(self.config.phone_score_threshold, 1e-6),
            )
            confidence = float(np.clip(1.0 - 0.5 * strongest, 0.5, 1.0))

        self._set_state(candidate, timestamp)
        if self._state != candidate:
            reason = f"Holding {self._state} briefly to avoid flicker"
            confidence = max(confidence, 0.55)

        offroad_evidence = max(
            abs(yaw_relative) / max(self.config.offroad_yaw_degrees, 1e-6),
            abs(pitch_relative) / max(self.config.offroad_pitch_degrees, 1e-6),
            look_down_relative / max(self.config.look_down_threshold, 1e-6),
        )
        return StateSnapshot(
            state=self._state,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reason=reason,
            state_seconds=max(0.0, timestamp - self._state_started),
            calibrated=True,
            calibration_progress=1.0,
            evidence={
                "eye": float(np.clip(face.eye_blink, 0.0, 1.0)),
                "mouth": float(np.clip(face.jaw_open, 0.0, 1.0)),
                "offroad": float(np.clip(offroad_evidence, 0.0, 1.0)),
                "phone": float(np.clip(best_phone / 0.5, 0.0, 1.0)),
                "fatigue": float(
                    np.clip(
                        perclos_proxy
                        / max(self.config.drowsy_perclos_threshold, 1e-6),
                        0.0,
                        1.0,
                    )
                ),
            },
            raw={
                "eye_blink": face.eye_blink,
                "eye_threshold": eye_threshold,
                "closed_seconds": closed_seconds,
                "jaw_open": face.jaw_open,
                "jaw_threshold": jaw_threshold,
                "yawn_seconds": yawn_seconds,
                "yaw_relative": yaw_relative,
                "pitch_relative": pitch_relative,
                "look_down_relative": look_down_relative,
                "offroad_seconds": offroad_seconds,
                "phone_score": best_phone,
                "phone_hits": float(phone_hits),
                "perclos_proxy": perclos_proxy,
                "long_closures": float(long_closures),
            },
        )
