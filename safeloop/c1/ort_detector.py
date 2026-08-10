"""Optional ONNX Runtime backend for the unchanged C1 YOLO11s model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .detector import ROAD_USER_LABELS, OpenCVDnnYoloDetector
from .types import Detection


class OnnxRuntimeYoloDetector(OpenCVDnnYoloDetector):
    """YOLO detector using ONNX Runtime CPU or CUDA execution providers.

    Preprocessing, output decoding, road-user filtering, and NMS are inherited
    from :class:`OpenCVDnnYoloDetector`.  This keeps the detector model and its
    observable postprocessing contract identical while changing only the ONNX
    graph execution backend.  ``onnxruntime`` remains an optional dependency:
    importing :mod:`safeloop.c1` does not require it.
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
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device phải là auto, cpu hoặc cuda")

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy detector ONNX: {self.model_path}")
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy detector labels: {self.labels_path}")

        self.labels = tuple(
            line.strip()
            for line in self.labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        self.allowed_labels = frozenset(allowed_labels)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        ort = self._import_onnxruntime()
        available = frozenset(ort.get_available_providers())
        selected = "cuda" if device == "auto" and "CUDAExecutionProvider" in available else device
        if selected == "auto":
            selected = "cpu"
        provider = (
            "CUDAExecutionProvider" if selected == "cuda" else "CPUExecutionProvider"
        )
        if provider not in available:
            raise RuntimeError(
                f"ONNX Runtime không có {provider}; available={sorted(available)}"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if selected == "cuda"
            else ["CPUExecutionProvider"]
        )
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:  # pragma: no cover - exact ORT exception varies by build
            raise RuntimeError(
                f"Không khởi tạo được ONNX Runtime {selected}: {exc}"
            ) from exc

        active = tuple(self.session.get_providers())
        if not active or active[0] != provider:
            raise RuntimeError(
                f"ONNX Runtime không kích hoạt {provider}; active={list(active)}"
            )
        inputs = tuple(self.session.get_inputs())
        outputs = tuple(self.session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                "YOLO baseline yêu cầu đúng 1 input và 1 output; "
                f"gặp {len(inputs)} input/{len(outputs)} output"
            )
        self._validate_input(inputs[0])
        self.input_name = str(inputs[0].name)
        self.output_name = str(outputs[0].name)
        self.execution_provider = provider
        self.device = selected

    @staticmethod
    def _import_onnxruntime() -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu onnxruntime; cài onnxruntime-gpu trong môi trường GPU "
                "hoặc onnxruntime cho CPU"
            ) from exc
        return ort

    def _validate_input(self, input_meta: Any) -> None:
        if getattr(input_meta, "type", None) != "tensor(float)":
            raise RuntimeError(
                f"YOLO input phải là tensor(float), gặp {getattr(input_meta, 'type', None)!r}"
            )
        shape = tuple(getattr(input_meta, "shape", ()))
        if len(shape) != 4:
            raise RuntimeError(f"YOLO input phải là NCHW 4-D, gặp shape={shape}")
        expected = (1, 3, self.input_size, self.input_size)
        for actual, wanted in zip(shape, expected):
            if isinstance(actual, int) and actual != wanted:
                raise RuntimeError(
                    f"YOLO input shape không khớp: model={shape}, expected={expected}"
                )

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        blob, scale, pad_x, pad_y = self._prepare_input(image_bgr)
        try:
            raw = self.session.run(
                [self.output_name],
                {self.input_name: np.ascontiguousarray(blob, dtype=np.float32)},
            )
        except Exception as exc:  # pragma: no cover - exact ORT exception varies by build
            raise RuntimeError(f"ONNX Runtime detector inference thất bại: {exc}") from exc
        return self._postprocess(
            raw,
            image_shape=image_bgr.shape,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
