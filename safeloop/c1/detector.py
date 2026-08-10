"""OpenCV-DNN adapter for the repository's YOLO11 ONNX detector."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Sequence

import cv2
import numpy as np

from .types import Detection


class ObjectDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[Detection]: ...


ROAD_USER_LABELS = frozenset({"person", "bicycle", "car", "motorcycle", "bus", "truck"})


class OpenCVDnnYoloDetector:
    """YOLO detector with letterbox preprocessing and no Ultralytics runtime.

    The checked-in ``models/yolo11s.onnx`` output is ``[1, 4+C, 8400]``:
    four ``cx, cy, w, h`` values followed by class confidences.  Keeping the
    adapter here makes the later detector swap (RTMDet/YOLO tiny) isolated from
    tracking and TTC logic.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        allowed_labels: Iterable[str] = ROAD_USER_LABELS,
        device: str = "auto",
    ) -> None:
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold phải nằm trong (0, 1)")
        if not 0.0 < nms_threshold < 1.0:
            raise ValueError("nms_threshold phải nằm trong (0, 1)")
        if input_size <= 0:
            raise ValueError("input_size phải > 0")

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy detector ONNX: {self.model_path}")
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy detector labels: {self.labels_path}")

        self.labels = tuple(
            line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        self.allowed_labels = frozenset(allowed_labels)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.device = self._configure_device(device)

    def _configure_device(self, requested: str) -> str:
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device phải là auto, cpu hoặc cuda")
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
        selected = "cuda" if requested == "auto" and cuda_count > 0 else requested
        if selected == "cuda":
            if cuda_count <= 0:
                raise RuntimeError(
                    "--device cuda được chọn nhưng OpenCV không thấy CUDA device; "
                    "dùng --device cpu hoặc cài OpenCV DNN có CUDA"
                )
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            return "cuda"
        # OpenCV defaults to its CPU backend.  Leaving the defaults untouched
        # also avoids a noisy graph-engine warning in OpenCV 5 builds.
        return "cpu"

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Ảnh detector phải có shape HxWx3, gặp {image_bgr.shape}")

        canvas, scale, pad_x, pad_y = self._letterbox(image_bgr)
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        raw = self.net.forward()
        rows = self._prediction_rows(raw)

        h, w = image_bgr.shape[:2]
        boxes_xywh: list[list[int]] = []
        candidates: list[Detection] = []
        scores: list[float] = []

        for row in rows:
            if row.size < 5:
                continue
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id >= len(self.labels) or confidence < self.confidence_threshold:
                continue
            label = self.labels[class_id]
            if label not in self.allowed_labels:
                continue

            cx, cy, bw, bh = (float(v) for v in row[:4])
            x1 = (cx - bw / 2.0 - pad_x) / scale
            y1 = (cy - bh / 2.0 - pad_y) / scale
            x2 = (cx + bw / 2.0 - pad_x) / scale
            y2 = (cy + bh / 2.0 - pad_y) / scale
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w - 1), x2), min(float(h - 1), y2)
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue

            boxes_xywh.append([
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            ])
            scores.append(confidence)
            candidates.append(Detection(class_id, label, confidence, (x1, y1, x2, y2)))

        if not candidates:
            return []
        kept = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            self.confidence_threshold,
            self.nms_threshold,
        )
        indices = np.asarray(kept).reshape(-1) if len(kept) else np.empty(0, dtype=int)
        return [candidates[int(i)] for i in indices]

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        resized_w = int(round(w * scale))
        resized_h = int(round(h * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized_w) // 2
        pad_y = (self.input_size - resized_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _prediction_rows(raw: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        if isinstance(raw, (tuple, list)):
            if len(raw) != 1:
                raise RuntimeError(f"Detector trả {len(raw)} output; baseline chỉ hỗ trợ 1")
            raw = raw[0]
        output = np.asarray(raw)
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise RuntimeError(f"YOLO output không hỗ trợ: shape={output.shape}")
        # YOLO11 ONNX: [4+C, N].  Also accept already-transposed [N, 4+C].
        if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
            output = output.T
        return output
