from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from safeloop.c1.detector import OpenCVDnnYoloDetector
from safeloop.c1.ort_detector import OnnxRuntimeYoloDetector


@dataclass(frozen=True)
class _TensorMeta:
    name: str
    shape: tuple[int, ...]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(self, raw: np.ndarray, providers: list[str]) -> None:
        self.raw = raw
        self.providers = tuple(providers)
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def get_inputs(self) -> list[_TensorMeta]:
        return [_TensorMeta("images", (1, 3, 640, 640))]

    def get_outputs(self) -> list[_TensorMeta]:
        return [_TensorMeta("output0", (1, 6, 10))]

    def run(
        self,
        output_names: list[str],
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names == ["output0"]
        self.feeds.append(feeds)
        return [self.raw]


class _FakeNet:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw
        self.blob: np.ndarray | None = None

    def setInput(self, blob: np.ndarray) -> None:
        self.blob = blob.copy()

    def forward(self) -> np.ndarray:
        return self.raw


def _raw_predictions() -> np.ndarray:
    raw = np.zeros((1, 6, 10), dtype=np.float32)
    raw[0, :, 0] = (320.0, 320.0, 100.0, 200.0, 0.90, 0.05)
    raw[0, :, 1] = (322.0, 320.0, 100.0, 200.0, 0.70, 0.05)
    raw[0, :, 2] = (100.0, 200.0, 40.0, 80.0, 0.05, 0.80)
    return raw


def _files(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "detector.onnx"
    labels = tmp_path / "labels.txt"
    model.write_bytes(b"test model placeholder")
    labels.write_text("person\ncar\n", encoding="utf-8")
    return model, labels


def _fake_ort(
    raw: np.ndarray,
    *,
    available: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    active: tuple[str, ...] | None = None,
) -> tuple[SimpleNamespace, list[_FakeSession]]:
    sessions: list[_FakeSession] = []

    def factory(_path: str, *, providers: list[str]) -> _FakeSession:
        session = _FakeSession(raw, list(active) if active is not None else providers)
        sessions.append(session)
        return session

    module = SimpleNamespace(
        get_available_providers=lambda: list(available),
        InferenceSession=factory,
    )
    return module, sessions


def test_ort_cuda_uses_same_preprocess_decode_and_nms_as_opencv(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, labels = _files(tmp_path)
    raw = _raw_predictions()
    fake_net = _FakeNet(raw)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _path: fake_net)
    opencv = OpenCVDnnYoloDetector(model, labels, device="cpu")

    fake_ort, sessions = _fake_ort(raw)
    monkeypatch.setattr(
        OnnxRuntimeYoloDetector,
        "_import_onnxruntime",
        staticmethod(lambda: fake_ort),
    )
    ort = OnnxRuntimeYoloDetector(model, labels, device="cuda")
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    assert ort.detect(image) == opencv.detect(image)
    assert ort.device == "cuda"
    assert ort.execution_provider == "CUDAExecutionProvider"
    assert tuple(sessions[0].feeds[0]) == ("images",)
    ort_blob = sessions[0].feeds[0]["images"]
    assert ort_blob.shape == (1, 3, 640, 640)
    assert ort_blob.dtype == np.float32
    assert ort_blob.flags.c_contiguous
    np.testing.assert_array_equal(ort_blob, fake_net.blob)


def test_ort_auto_falls_back_to_cpu_when_cuda_provider_is_absent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, labels = _files(tmp_path)
    fake_ort, sessions = _fake_ort(
        _raw_predictions(), available=("CPUExecutionProvider",)
    )
    monkeypatch.setattr(
        OnnxRuntimeYoloDetector,
        "_import_onnxruntime",
        staticmethod(lambda: fake_ort),
    )

    detector = OnnxRuntimeYoloDetector(model, labels, device="auto")

    assert detector.device == "cpu"
    assert detector.execution_provider == "CPUExecutionProvider"
    assert sessions[0].providers == ("CPUExecutionProvider",)


def test_ort_explicit_cuda_fails_when_provider_is_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, labels = _files(tmp_path)
    fake_ort, sessions = _fake_ort(
        _raw_predictions(), available=("CPUExecutionProvider",)
    )
    monkeypatch.setattr(
        OnnxRuntimeYoloDetector,
        "_import_onnxruntime",
        staticmethod(lambda: fake_ort),
    )

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        OnnxRuntimeYoloDetector(model, labels, device="cuda")
    assert sessions == []


def test_ort_rejects_silent_cuda_fallback_to_cpu(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, labels = _files(tmp_path)
    fake_ort, _sessions = _fake_ort(
        _raw_predictions(), active=("CPUExecutionProvider",)
    )
    monkeypatch.setattr(
        OnnxRuntimeYoloDetector,
        "_import_onnxruntime",
        staticmethod(lambda: fake_ort),
    )

    with pytest.raises(RuntimeError, match="không kích hoạt CUDAExecutionProvider"):
        OnnxRuntimeYoloDetector(model, labels, device="cuda")
