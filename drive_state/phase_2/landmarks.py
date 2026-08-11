"""MediaPipe Face Landmarker adapter and eye ROI projection."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from .data.evidence_regions import EvidenceRegionRecord
from .data.eye_crop_cache import EyeCropStore, save_eye_crop_cache
from .data.eye_rois import (
    EyeRoiRecord,
    box_from_normalized_landmarks,
    crop_eye_pair,
    load_eye_roi_cache,
    save_eye_roi_cache,
)
from .data.eye_sequences import EyeSequence
from .data.face_cache import FaceCropStore
from .data.mouth_cache import MouthCropStore, save_mouth_crop_cache
from .face_detection import FaceBox, FaceDetection, FaceDetector


# MediaPipe Face Mesh canonical eye contours.  Names refer to the driver's
# anatomical side, not screen position.
LEFT_EYE_INDICES = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYE_INDICES = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
MOUTH_INDICES = (13, 14, 61, 291, 78, 308)


class LandmarkLike(Protocol):
    x: float
    y: float


class FaceLandmarkDetector(Protocol):
    def detect(self, image: Image.Image) -> Sequence[LandmarkLike] | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class NormalizedLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class EyeCropCacheStats:
    total_frames: int
    both_visible: int
    one_visible: int
    missing_both: int


@dataclass(frozen=True)
class MouthCropCacheStats:
    total_frames: int
    visible: int
    missing: int


@dataclass(frozen=True)
class _NormalizedRegion:
    box: tuple[float, float, float, float]
    visibility: float


def _landmark_quality(landmarks: Sequence[LandmarkLike], indices: Sequence[int]) -> float:
    qualities: list[float] = []
    for index in indices:
        landmark = landmarks[index]
        values = [
            value
            for value in (
                getattr(landmark, "visibility", None),
                getattr(landmark, "presence", None),
            )
            if value is not None
        ]
        qualities.append(min(float(value) for value in values) if values else 1.0)
    return max(0.0, min(1.0, sum(qualities) / len(qualities)))


def _normalized_face_box(
    box: FaceBox,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    return (
        float(np.clip(box.x1 / image_width, 0.0, 1.0)),
        float(np.clip(box.y1 / image_height, 0.0, 1.0)),
        float(np.clip(box.x2 / image_width, 0.0, 1.0)),
        float(np.clip(box.y2 / image_height, 0.0, 1.0)),
    )


def _normalized_landmark_region(
    landmarks: Sequence[LandmarkLike] | None,
    indices: Sequence[int],
    *,
    padding: float,
) -> _NormalizedRegion:
    if landmarks is None:
        return _NormalizedRegion((0.0, 0.0, 0.0, 0.0), 0.0)
    if len(landmarks) <= max(indices):
        raise ValueError(
            f"face mesh has {len(landmarks)} landmarks; "
            f"need at least {max(indices) + 1}"
        )
    points = np.asarray(
        [
            (float(landmarks[index].x), float(landmarks[index].y))
            for index in indices
        ],
        dtype=np.float32,
    )
    if not np.isfinite(points).all():
        return _NormalizedRegion((0.0, 0.0, 0.0, 0.0), 0.0)
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = max(float(high[0] - low[0]), float(high[1] - low[1]))
    if span <= 0.0:
        return _NormalizedRegion((0.0, 0.0, 0.0, 0.0), 0.0)
    margin = span * padding
    box = (
        float(np.clip(low[0] - margin, 0.0, 1.0)),
        float(np.clip(low[1] - margin, 0.0, 1.0)),
        float(np.clip(high[0] + margin, 0.0, 1.0)),
        float(np.clip(high[1] + margin, 0.0, 1.0)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return _NormalizedRegion((0.0, 0.0, 0.0, 0.0), 0.0)
    return _NormalizedRegion(box, _landmark_quality(landmarks, indices))


def evidence_regions_from_landmarks(
    *,
    frame_id: int,
    face_box: FaceBox | None,
    landmarks: Sequence[LandmarkLike] | None,
    image_width: int,
    image_height: int,
    pitch: float,
    yaw: float,
    face_visibility: float = 1.0,
) -> EvidenceRegionRecord:
    """Return full-frame normalized face, eye, and mouth regions."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= face_visibility <= 1.0:
        raise ValueError("face visibility must be in [0, 1]")
    if face_box is None:
        return EvidenceRegionRecord.missing(frame_id)
    left = _normalized_landmark_region(
        landmarks,
        LEFT_EYE_INDICES,
        padding=0.20,
    )
    right = _normalized_landmark_region(
        landmarks,
        RIGHT_EYE_INDICES,
        padding=0.20,
    )
    mouth = _normalized_landmark_region(
        landmarks,
        MOUTH_INDICES,
        padding=0.35,
    )
    return EvidenceRegionRecord(
        frame_id=frame_id,
        boxes=np.asarray(
            (
                _normalized_face_box(face_box, image_width, image_height),
                left.box,
                right.box,
                mouth.box,
            ),
            dtype=np.float32,
        ),
        visibility=np.asarray(
            (
                face_visibility,
                left.visibility,
                right.visibility,
                mouth.visibility,
            ),
            dtype=np.float32,
        ),
        head_pose=np.asarray((pitch, yaw), dtype=np.float32),
    )


def eye_roi_record_from_landmarks(
    *,
    frame_id: int,
    landmarks: Sequence[LandmarkLike] | None,
    image_width: int,
    image_height: int,
) -> EyeRoiRecord:
    """Project a detected face mesh into two padded eye regions."""
    if landmarks is None:
        return EyeRoiRecord(frame_id=frame_id, left=None, right=None)
    required_index = max(max(LEFT_EYE_INDICES), max(RIGHT_EYE_INDICES))
    if len(landmarks) <= required_index:
        raise ValueError(f"face mesh has {len(landmarks)} landmarks; need at least {required_index + 1}")

    left_points = [(float(landmarks[index].x), float(landmarks[index].y)) for index in LEFT_EYE_INDICES]
    right_points = [(float(landmarks[index].x), float(landmarks[index].y)) for index in RIGHT_EYE_INDICES]
    left = box_from_normalized_landmarks(
        left_points, image_width=image_width, image_height=image_height
    )
    right = box_from_normalized_landmarks(
        right_points, image_width=image_width, image_height=image_height
    )
    return EyeRoiRecord(
        frame_id=frame_id,
        left=left,
        right=right,
        left_visibility=_landmark_quality(landmarks, LEFT_EYE_INDICES) if left else 0.0,
        right_visibility=_landmark_quality(landmarks, RIGHT_EYE_INDICES) if right else 0.0,
    )


def mouth_crop_from_landmarks(
    image: Image.Image,
    landmarks: Sequence[LandmarkLike] | None,
    *,
    output_size: tuple[int, int] = (96, 64),
    padding: float = 0.35,
) -> tuple[Image.Image, float]:
    """Project canonical lip landmarks into one fixed-size RGB mouth crop."""
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size values must be positive")
    if padding < 0.0:
        raise ValueError("padding cannot be negative")
    missing = Image.new("RGB", output_size)
    if landmarks is None:
        return missing, 0.0
    required_index = max(MOUTH_INDICES)
    if len(landmarks) <= required_index:
        raise ValueError(f"face mesh has {len(landmarks)} landmarks; need at least {required_index + 1}")

    rgb = image.convert("RGB")
    points = np.asarray(
        [
            (float(landmarks[index].x) * rgb.width, float(landmarks[index].y) * rgb.height)
            for index in MOUTH_INDICES
        ],
        dtype=np.float32,
    )
    if not np.isfinite(points).all():
        return missing, 0.0
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    span = max(float(x2 - x1), float(y2 - y1))
    if span <= 0.0:
        return missing, 0.0
    margin = span * padding
    left = max(0, int(np.floor(x1 - margin)))
    top = max(0, int(np.floor(y1 - margin)))
    right = min(rgb.width, int(np.ceil(x2 + margin)))
    bottom = min(rgb.height, int(np.ceil(y2 + margin)))
    if right <= left or bottom <= top:
        return missing, 0.0
    crop = rgb.crop((left, top, right, bottom)).resize(
        output_size, Image.Resampling.BILINEAR
    )
    return crop, _landmark_quality(landmarks, MOUTH_INDICES)


def precompute_mouth_crop_cache_from_face_store(
    sequence: EyeSequence,
    face_store: FaceCropStore,
    landmark_detector: FaceLandmarkDetector,
    cache_dir: Path | str,
    *,
    overwrite: bool = False,
) -> MouthCropCacheStats:
    """Create one dense 96x64 mouth crop per 20 FPS s5 source frame."""
    cache_dir = Path(cache_dir)
    if not overwrite:
        try:
            existing = MouthCropStore(cache_dir, sequence.session_name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            existing.rows_for(sequence.frame_ids)
            visible = int((existing.visibility > 0.0).sum())
            return MouthCropCacheStats(len(sequence), visible, len(sequence) - visible)

    if face_store.rate != "20fps":
        raise ValueError(f"mouth-cache face source must be 20fps, got {face_store.rate!r}")
    expected_ids = np.asarray(sequence.frame_ids, dtype=np.int64)
    if not np.array_equal(face_store.frame_ids, expected_ids):
        raise ValueError(f"incomplete/misaligned face cache for {sequence.session_name}")

    mouths = np.zeros((len(sequence), 64, 96, 3), dtype=np.uint8)
    visibility = np.zeros(len(sequence), dtype=np.float32)
    for row, frame_id in enumerate(sequence.frame_ids):
        sample = face_store.get(frame_id)
        landmarks = landmark_detector.detect(sample.image) if sample.visible else None
        crop, crop_visibility = mouth_crop_from_landmarks(sample.image, landmarks)
        mouths[row] = np.asarray(crop, dtype=np.uint8)
        visibility[row] = crop_visibility

    save_mouth_crop_cache(
        cache_dir,
        sequence.session_name,
        mouths=mouths,
        frame_ids=np.asarray(sequence.frame_ids, dtype=np.int32),
        visibility=visibility,
    )
    visible = int((visibility > 0.0).sum())
    return MouthCropCacheStats(len(sequence), visible, len(sequence) - visible)


def precompute_eye_roi_cache(
    sequence: EyeSequence,
    detector: FaceLandmarkDetector,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> tuple[EyeRoiRecord, ...]:
    """Detect and serialize eye boxes for one complete 20 FPS s5 session."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        cached = load_eye_roi_cache(output_path)
        missing_ids = set(sequence.frame_ids).difference(cached)
        if missing_ids:
            raise ValueError(
                f"incomplete eye cache {output_path}: missing {len(missing_ids)} frame ids"
            )
        return tuple(cached[frame_id] for frame_id in sequence.frame_ids)

    records: list[EyeRoiRecord] = []
    for frame_id, frame_path in zip(sequence.frame_ids, sequence.frame_paths, strict=True):
        if not frame_path.is_file():
            raise FileNotFoundError(f"DMD frame is missing: {frame_path}")
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
        landmarks = detector.detect(rgb)
        records.append(
            eye_roi_record_from_landmarks(
                frame_id=frame_id,
                landmarks=landmarks,
                image_width=rgb.width,
                image_height=rgb.height,
            )
        )
    save_eye_roi_cache(output_path, records)
    return tuple(records)


def precompute_eye_crop_cache(
    sequence: EyeSequence,
    detector: FaceLandmarkDetector,
    cache_dir: Path | str,
    *,
    output_size: tuple[int, int] = (96, 48),
    overwrite: bool = False,
) -> EyeCropCacheStats:
    """Create a dense, complete pair-of-eyes cache for one 20 FPS session.

    Detection misses are stored as black crops with visibility zero rather
    than dropping frames.  Consequently temporal labels and cache rows can
    never drift out of alignment.
    """
    cache_dir = Path(cache_dir)
    if not overwrite:
        try:
            existing = EyeCropStore(cache_dir, sequence.session_name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            existing.rows_for(sequence.frame_ids)
            visible_counts = (existing.visibility > 0.0).sum(axis=1)
            return EyeCropCacheStats(
                total_frames=len(sequence),
                both_visible=int((visible_counts == 2).sum()),
                one_visible=int((visible_counts == 1).sum()),
                missing_both=int((visible_counts == 0).sum()),
            )

    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size values must be positive")
    eyes = np.zeros((len(sequence), 2, height, width, 3), dtype=np.uint8)
    visibility = np.zeros((len(sequence), 2), dtype=np.float32)
    records: list[EyeRoiRecord] = []
    for row, (frame_id, frame_path) in enumerate(
        zip(sequence.frame_ids, sequence.frame_paths, strict=True)
    ):
        if not frame_path.is_file():
            raise FileNotFoundError(f"DMD frame is missing: {frame_path}")
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
        landmarks = detector.detect(image)
        record = eye_roi_record_from_landmarks(
            frame_id=frame_id,
            landmarks=landmarks,
            image_width=image.width,
            image_height=image.height,
        )
        left, right, eye_visibility = crop_eye_pair(
            image, record, output_size=output_size
        )
        eyes[row, 0] = np.asarray(left, dtype=np.uint8)
        eyes[row, 1] = np.asarray(right, dtype=np.uint8)
        visibility[row] = eye_visibility
        records.append(record)

    save_eye_crop_cache(
        cache_dir,
        sequence.session_name,
        eyes=eyes,
        frame_ids=np.asarray(sequence.frame_ids, dtype=np.int32),
        visibility=visibility,
    )
    save_eye_roi_cache(cache_dir / f"{sequence.session_name}.rois.csv", records)
    visible_counts = (visibility > 0.0).sum(axis=1)
    return EyeCropCacheStats(
        total_frames=len(sequence),
        both_visible=int((visible_counts == 2).sum()),
        one_visible=int((visible_counts == 1).sum()),
        missing_both=int((visible_counts == 0).sum()),
    )


def precompute_eye_crop_cache_from_face_store(
    sequence: EyeSequence,
    face_store: FaceCropStore,
    landmark_detector: FaceLandmarkDetector,
    cache_dir: Path | str,
    *,
    output_size: tuple[int, int] = (96, 48),
    overwrite: bool = False,
) -> EyeCropCacheStats:
    """Create dense eye crops from an already complete 20 FPS face cache."""
    cache_dir = Path(cache_dir)
    if not overwrite:
        try:
            existing = EyeCropStore(cache_dir, sequence.session_name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            existing.rows_for(sequence.frame_ids)
            visible_counts = (existing.visibility > 0.0).sum(axis=1)
            return EyeCropCacheStats(
                total_frames=len(sequence),
                both_visible=int((visible_counts == 2).sum()),
                one_visible=int((visible_counts == 1).sum()),
                missing_both=int((visible_counts == 0).sum()),
            )

    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size values must be positive")
    if face_store.rate != "20fps":
        raise ValueError(
            f"eye-cache face source must be 20fps, got {face_store.rate!r}"
        )
    expected_ids = np.asarray(sequence.frame_ids, dtype=np.int64)
    if not np.array_equal(face_store.frame_ids, expected_ids):
        raise ValueError(
            f"incomplete/misaligned face cache for {sequence.session_name}"
        )

    eyes = np.zeros((len(sequence), 2, height, width, 3), dtype=np.uint8)
    visibility = np.zeros((len(sequence), 2), dtype=np.float32)
    records: list[EyeRoiRecord] = []
    for row, frame_id in enumerate(sequence.frame_ids):
        sample = face_store.get(frame_id)
        image = sample.image
        landmarks = landmark_detector.detect(image) if sample.visible else None
        record = eye_roi_record_from_landmarks(
            frame_id=frame_id,
            landmarks=landmarks,
            image_width=image.width,
            image_height=image.height,
        )
        left, right, eye_visibility = crop_eye_pair(
            image, record, output_size=output_size
        )
        eyes[row, 0] = np.asarray(left, dtype=np.uint8)
        eyes[row, 1] = np.asarray(right, dtype=np.uint8)
        visibility[row] = eye_visibility
        records.append(record)

    save_eye_crop_cache(
        cache_dir,
        sequence.session_name,
        eyes=eyes,
        frame_ids=np.asarray(sequence.frame_ids, dtype=np.int32),
        visibility=visibility,
    )
    save_eye_roi_cache(cache_dir / f"{sequence.session_name}.rois.csv", records)
    visible_counts = (visibility > 0.0).sum(axis=1)
    return EyeCropCacheStats(
        total_frames=len(sequence),
        both_visible=int((visible_counts == 2).sum()),
        one_visible=int((visible_counts == 1).sum()),
        missing_both=int((visible_counts == 0).sum()),
    )


class FaceBoxLandmarkDetector:
    """Run landmarks inside an Ultra-Light face box and remap to full frame."""

    def __init__(
        self,
        face_detector: FaceDetector,
        landmark_detector: FaceLandmarkDetector,
        *,
        padding: float = 0.15,
    ) -> None:
        if padding < 0.0:
            raise ValueError("padding cannot be negative")
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.padding = padding
        self.latest_detection: FaceDetection | None = None

    def reset(self) -> None:
        """Clear detector state before a new independent camera session."""

        self.latest_detection = None
        for component in (self.face_detector, self.landmark_detector):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def detect(self, image: Image.Image) -> Sequence[NormalizedLandmark] | None:
        image = image.convert("RGB")
        detection = self.face_detector.detect(image)
        self.latest_detection = detection
        if detection is None:
            return None

        box = detection.box
        margin = round(max(box.x2 - box.x1, box.y2 - box.y1) * self.padding)
        x1 = max(0, box.x1 - margin)
        y1 = max(0, box.y1 - margin)
        x2 = min(image.width, box.x2 + margin)
        y2 = min(image.height, box.y2 + margin)
        crop = image.crop((x1, y1, x2, y2))
        landmarks = self.landmark_detector.detect(crop)
        if landmarks is None:
            return None

        crop_width = x2 - x1
        crop_height = y2 - y1
        remapped: list[NormalizedLandmark] = []
        for landmark in landmarks:
            source_visibility = getattr(landmark, "visibility", None)
            source_presence = getattr(landmark, "presence", None)
            remapped.append(
                NormalizedLandmark(
                    x=(x1 + float(landmark.x) * crop_width) / image.width,
                    y=(y1 + float(landmark.y) * crop_height) / image.height,
                    z=float(getattr(landmark, "z", 0.0)) * crop_width / image.width,
                    visibility=detection.confidence
                    * (1.0 if source_visibility is None else float(source_visibility)),
                    presence=detection.confidence
                    * (1.0 if source_presence is None else float(source_presence)),
                )
            )
        return remapped

    def close(self) -> None:
        close = getattr(self.landmark_detector, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "FaceBoxLandmarkDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncFaceLandmarkDetector:
    """Refresh landmarks on one worker and return only completed past results.

    The first call initializes synchronously through the worker. Later refreshes
    never block the inference thread: the current frame uses the most recent
    completed landmarks while a newer frame is processed in the background.
    This is causal because a result is published only after its source frame
    has already arrived.
    """

    def __init__(
        self, detector: FaceLandmarkDetector, *, interval_frames: int = 5
    ) -> None:
        if interval_frames <= 0:
            raise ValueError("interval_frames must be positive")
        self.detector = detector
        self.interval_frames = interval_frames
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="face-landmarks"
        )
        self._future: Future[Sequence[LandmarkLike] | None] | None = None
        self._latest: Sequence[LandmarkLike] | None = None
        self._initialized = False
        self._frame_index = 0
        self._closed = False

    def _submit(self, image: Image.Image) -> None:
        # The video/camera owner may reuse its input buffer after process()
        # returns, so the worker must own an immutable copy.
        self._future = self._executor.submit(self.detector.detect, image.copy())

    def _collect_if_ready(self) -> None:
        if self._future is not None and self._future.done():
            self._latest = self._future.result()
            self._future = None
            self._initialized = True

    def detect(self, image: Image.Image) -> Sequence[LandmarkLike] | None:
        if self._closed:
            raise RuntimeError("async landmark detector is closed")
        self._collect_if_ready()
        if not self._initialized:
            self._submit(image)
            assert self._future is not None
            self._latest = self._future.result()
            self._future = None
            self._initialized = True
            self._frame_index = 1
            return self._latest

        if (
            self._frame_index % self.interval_frames == 0
            and self._future is None
        ):
            self._submit(image)
        self._frame_index += 1
        return self._latest

    def close(self) -> None:
        if self._closed:
            return
        if self._future is not None:
            self._future.result()
            self._future = None
        self._executor.shutdown(wait=True)
        self._closed = True

    def __enter__(self) -> "AsyncFaceLandmarkDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class MediaPipeFaceLandmarker:
    """Small optional adapter around MediaPipe's Tasks Face Landmarker API."""

    def __init__(self, model_path: Path | str, *, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as error:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "MediaPipe is required for ROI preprocessing; install the 'landmarks' extra"
            ) from error

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model is missing: {model_path}. "
                "Pass an explicit downloaded face_landmarker.task file."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp: Any = mp
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: Image.Image) -> Sequence[LandmarkLike] | None:
        import numpy as np

        data = np.asarray(image.convert("RGB"))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=data)
        result = self._detector.detect(mp_image)
        return result.face_landmarks[0] if result.face_landmarks else None

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MediaPipeFaceLandmarker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
