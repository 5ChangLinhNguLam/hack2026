"""Ultra-Light face detection with deterministic NMS and short-gap tracking.

The ONNX preprocessing and output layout follow Linzaer's
Ultra-Light-Fast-Generic-Face-Detector-1MB reference implementation.  Model
assets are always explicit; this module never downloads weights at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"invalid face box: {self}")

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class FaceDetection:
    box: FaceBox
    confidence: float
    is_fallback: bool = False


class FaceDetector(Protocol):
    def detect(self, image: Image.Image) -> FaceDetection | None: ...


def _iou(box: FaceBox, others: list[FaceBox]) -> np.ndarray:
    if not others:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box.x1, np.array([other.x1 for other in others]))
    y1 = np.maximum(box.y1, np.array([other.y1 for other in others]))
    x2 = np.minimum(box.x2, np.array([other.x2 for other in others]))
    y2 = np.minimum(box.y2, np.array([other.y2 for other in others]))
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    other_areas = np.array([other.area for other in others])
    return intersection / np.maximum(box.area + other_areas - intersection, 1)


def decode_ultralight_outputs(
    confidences: np.ndarray,
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    score_threshold: float = 0.7,
    nms_iou_threshold: float = 0.3,
) -> tuple[FaceDetection, ...]:
    """Decode normalized Ultra-Light SSD outputs and perform hard NMS."""
    confidences = np.asarray(confidences)
    boxes = np.asarray(boxes)
    if confidences.ndim != 3 or confidences.shape[0] != 1 or confidences.shape[-1] < 2:
        raise ValueError(f"unexpected Ultra-Light confidence shape: {confidences.shape}")
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 4:
        raise ValueError(f"unexpected Ultra-Light box shape: {boxes.shape}")
    if boxes.shape[1] != confidences.shape[1]:
        raise ValueError("Ultra-Light confidence and box counts differ")

    candidates: list[FaceDetection] = []
    for normalized_box, score in zip(boxes[0], confidences[0, :, 1], strict=True):
        if float(score) < score_threshold:
            continue
        x1 = min(max(round(float(normalized_box[0]) * image_width), 0), image_width - 1)
        y1 = min(max(round(float(normalized_box[1]) * image_height), 0), image_height - 1)
        x2 = min(max(round(float(normalized_box[2]) * image_width), 1), image_width)
        y2 = min(max(round(float(normalized_box[3]) * image_height), 1), image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(FaceDetection(FaceBox(x1, y1, x2, y2), float(score)))

    candidates.sort(key=lambda detection: detection.confidence, reverse=True)
    kept: list[FaceDetection] = []
    for candidate in candidates:
        if kept and np.any(_iou(candidate.box, [detection.box for detection in kept]) > nms_iou_threshold):
            continue
        kept.append(candidate)
    return tuple(kept)


class UltraLightFaceDetector:
    """OpenCV-DNN inference for the 320x240 RFB or slim ONNX model."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        score_threshold: float = 0.7,
        nms_iou_threshold: float = 0.3,
    ) -> None:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("OpenCV is required for Ultra-Light ONNX inference") from error
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Ultra-Light ONNX model is missing: {model_path}. "
                "Use version-RFB-320.onnx from Linzaer's official repository."
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._output_names = self._net.getUnconnectedOutLayersNames()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def detect(self, image: Image.Image) -> FaceDetection | None:
        rgb = np.asarray(image.convert("RGB"))
        resized = self._cv2.resize(rgb, (320, 240))
        tensor = ((resized.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)[None]
        self._net.setInput(tensor)
        outputs = self._net.forward(self._output_names)
        confidences = next((output for output in outputs if output.shape[-1] == 2), None)
        boxes = next((output for output in outputs if output.shape[-1] == 4), None)
        if confidences is None or boxes is None:
            raise RuntimeError(
                f"unrecognized Ultra-Light outputs: {[output.shape for output in outputs]}"
            )
        detections = decode_ultralight_outputs(
            confidences,
            boxes,
            image_width=image.width,
            image_height=image.height,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
        )
        return detections[0] if detections else None


class PeriodicFaceDetector:
    """Run the expensive detector every N frames and hold its causal box."""

    def __init__(self, detector: FaceDetector, *, interval_frames: int = 5) -> None:
        if interval_frames <= 0:
            raise ValueError("interval_frames must be positive")
        self.detector = detector
        self.interval_frames = interval_frames
        self._frame_index = 0
        self._last: FaceDetection | None = None

    def reset(self) -> None:
        self._frame_index = 0
        self._last = None

    def detect(self, image: Image.Image) -> FaceDetection | None:
        # Before the first valid box, retry on every frame. Otherwise one miss
        # at startup would be expanded into an artificial interval-sized gap.
        should_detect = (
            self._last is None
            or self._frame_index % self.interval_frames == 0
        )
        self._frame_index += 1
        if should_detect:
            current = self.detector.detect(image)
            self._last = current
            if current is not None:
                return current
            return None
        if self._last is None:
            return None
        return replace(self._last, is_fallback=True)


class TrackedFaceDetector:
    """Smooth detected boxes and bridge a bounded number of missed frames."""

    def __init__(
        self,
        detector: FaceDetector,
        *,
        max_missed_frames: int = 10,
        confidence_decay: float = 0.85,
        smoothing: float = 0.65,
    ) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if not 0.0 <= confidence_decay <= 1.0 or not 0.0 <= smoothing <= 1.0:
            raise ValueError("confidence_decay and smoothing must be between 0 and 1")
        self.detector = detector
        self.max_missed_frames = max_missed_frames
        self.confidence_decay = confidence_decay
        self.smoothing = smoothing
        self._last: FaceDetection | None = None
        self._missed = 0

    def _smooth(self, current: FaceDetection) -> FaceDetection:
        if self._last is None:
            return current
        previous = self._last.box
        box = current.box
        alpha = self.smoothing
        coordinates = tuple(
            round(alpha * new + (1.0 - alpha) * old)
            for new, old in zip(
                (box.x1, box.y1, box.x2, box.y2),
                (previous.x1, previous.y1, previous.x2, previous.y2),
                strict=True,
            )
        )
        return replace(current, box=FaceBox(*coordinates))

    def detect(self, image: Image.Image) -> FaceDetection | None:
        current = self.detector.detect(image)
        if current is not None:
            current = self._smooth(current)
            self._last = current
            self._missed = 0
            return current
        self._missed += 1
        if self._last is None or self._missed > self.max_missed_frames:
            self._last = None
            return None
        fallback = FaceDetection(
            box=self._last.box,
            confidence=self._last.confidence * self.confidence_decay,
            is_fallback=True,
        )
        self._last = fallback
        return fallback
