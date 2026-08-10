from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

import C1.runtime as runtime_module
from C1.runtime import StudentTTCRuntime


class FakeModel:
    def __init__(self) -> None:
        self.features: list[np.ndarray] = []
        self.outputs = iter(((0.9, 1.0), (0.2, float("inf"))))
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict(self, _frame: object, scalar: object) -> tuple[float, float]:
        self.features.append(scalar.detach().cpu().numpy()[0])
        return next(self.outputs)


class FakeBundle:
    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.timestamp = frame_id / 20.0
        self.ego = {"speed_kmh": frame_id * 1.8, "lateral_accel": 0.0}
        self.targets = []
        self.left_reads = 0

    def left(self) -> np.ndarray:
        self.left_reads += 1
        return np.zeros((16, 16, 3), dtype=np.uint8)


def test_runtime_updates_at_10hz_and_forward_fills(monkeypatch, tmp_path) -> None:
    fake = FakeModel()
    checkpoint = tmp_path / "student.pth"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(
        runtime_module,
        "load_student_ttc",
        lambda *_args, **_kwargs: (fake, {"feat_use": list(runtime_module.FEATURES_USED)}),
    )
    model = StudentTTCRuntime(
        checkpoint,
        metadata={"speed_limit_kmh": 60, "weather": {}},
        source_fps=20.0,
        stride=2,
        device="cpu",
    )
    bundles = [FakeBundle(index) for index in range(4)]

    predictions = [model.process_bundle(bundle) for bundle in bundles]

    assert [prediction.model_updated for prediction in predictions] == [True, False, True, False]
    assert [bundle.left_reads for bundle in bundles] == [1, 0, 1, 0]
    assert predictions[1].predicted_ttc_s == predictions[0].predicted_ttc_s
    assert predictions[1].model_frame_id == 0
    assert predictions[3].model_frame_id == 2
    assert len(fake.features) == 2
