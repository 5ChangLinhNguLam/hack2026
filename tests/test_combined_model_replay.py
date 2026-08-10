from dataclasses import dataclass

import pytest

from safeloop.combined_replay import CombinedModelReplay
from safeloop.replay_models import build_parser


@dataclass
class FakePrediction:
    frame_id: int
    timestamp: float
    value: object
    challenge: str

    def submission_row(self) -> dict[str, object]:
        if self.challenge == "c1":
            return {
                "frame_id": self.frame_id,
                "timestamp": self.timestamp,
                "predicted_ttc": self.value,
            }
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "predicted_driver_state": self.value,
        }


class FakeBundle:
    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.timestamp = frame_id / 20.0
        self.driver_reads = 0

    def driver(self) -> str:
        self.driver_reads += 1
        return f"driver-{self.frame_id}"


class FakeC1:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def process_bundle(self, bundle: FakeBundle) -> FakePrediction:
        self.frames.append(bundle.frame_id)
        return FakePrediction(bundle.frame_id, bundle.timestamp, 1.5, "c1")


class FakeC2:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []

    def process_bgr(
        self, image: str, *, frame_id: int, timestamp: float
    ) -> FakePrediction:
        self.calls.append((image, frame_id, timestamp))
        return FakePrediction(frame_id, timestamp, "alert", "c2")


def test_combined_replay_consumes_one_ordered_stream() -> None:
    bundles = [FakeBundle(index) for index in range(5)]
    c1 = FakeC1()
    c2 = FakeC2()

    predictions = list(CombinedModelReplay(iter(bundles), c1=c1, c2=c2))

    assert c1.frames == [0, 1, 2, 3, 4]
    assert [call[1] for call in c2.calls] == [0, 1, 2, 3, 4]
    assert [bundle.driver_reads for bundle in bundles] == [1, 1, 1, 1, 1]
    assert predictions[-1].submission_row() == {
        "frame_id": 4,
        "timestamp": 0.2,
        "predicted_ttc": 1.5,
        "predicted_driver_state": "alert",
    }


def test_combined_prediction_rejects_frame_mismatch() -> None:
    bundle = FakeBundle(2)

    class WrongC2(FakeC2):
        def process_bgr(self, image: str, *, frame_id: int, timestamp: float) -> FakePrediction:
            return FakePrediction(frame_id + 1, timestamp, "alert", "c2")

    with pytest.raises(ValueError, match="C2 frame mismatch"):
        list(CombinedModelReplay([bundle], c1=FakeC1(), c2=WrongC2()))


def test_unified_cli_defaults_to_both_repository_models() -> None:
    args = build_parser().parse_args(
        ["--dataset", "data", "--trip", "T01-Sample", "--write-video"]
    )
    assert str(args.c1_checkpoint) == "C1/student_ttc.pth"
    assert str(args.c2_bundle) == "models/driver_state_phase_2_v13"
    assert args.c1_stride == 2
    assert args.write_video is True
