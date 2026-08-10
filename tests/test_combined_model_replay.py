from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from safeloop.combined_replay import CombinedModelReplay
from safeloop.drive_quality import DriveQualityAccumulator
from safeloop.replay_models import (
    DRIVE_QUALITY_DIAGNOSTIC_FIELDS,
    SUBMISSION_FIELDS,
    _archive_existing_outputs,
    build_parser,
)


@dataclass
class FakePrediction:
    frame_id: int
    timestamp: float
    value: object
    challenge: str

    @property
    def predicted_ttc_s(self) -> float:
        if self.challenge != "c1":
            raise AttributeError("only C1 predictions carry TTC")
        return float(self.value)

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


class FakeC3:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def update(self, bundle, *, predicted_ttc_s):
        self.frames.append(bundle.frame_id)
        return FakePrediction(bundle.frame_id, bundle.timestamp, 95.0, "c3")


class FakeDriveQuality:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def update(self, c3_frame):
        self.frames.append(c3_frame.frame_id)
        return FakePrediction(
            c3_frame.frame_id,
            c3_frame.timestamp,
            91.0,
            "drive_quality",
        )


class FakeRisk:
    score_pct = 12.5


class FakeRiskPolicy:
    def evaluate(self, c1, c2):
        return FakeRisk()


def test_combined_replay_consumes_one_ordered_stream() -> None:
    bundles = [FakeBundle(index) for index in range(5)]
    c1 = FakeC1()
    c2 = FakeC2()
    c3 = FakeC3()
    drive_quality = FakeDriveQuality()

    predictions = list(
        CombinedModelReplay(
            iter(bundles),
            c1=c1,
            c2=c2,
            c3=c3,
            drive_quality=drive_quality,
            contextual_risk=FakeRiskPolicy(),
        )
    )

    assert c1.frames == [0, 1, 2, 3, 4]
    assert [call[1] for call in c2.calls] == [0, 1, 2, 3, 4]
    assert c3.frames == [0, 1, 2, 3, 4]
    assert drive_quality.frames == [0, 1, 2, 3, 4]
    assert [bundle.driver_reads for bundle in bundles] == [1, 1, 1, 1, 1]
    assert predictions[-1].submission_row() == {
        "frame_id": 4,
        "timestamp": 0.2,
        "predicted_ttc": 1.5,
        "predicted_driver_state": "alert",
        "predicted_risk_score": 12.5,
    }


def test_combined_prediction_rejects_frame_mismatch() -> None:
    bundle = FakeBundle(2)

    class WrongC2(FakeC2):
        def process_bgr(self, image: str, *, frame_id: int, timestamp: float) -> FakePrediction:
            return FakePrediction(frame_id + 1, timestamp, "alert", "c2")

    with pytest.raises(ValueError, match="C2 frame mismatch"):
        list(
            CombinedModelReplay(
                [bundle],
                c1=FakeC1(),
                c2=WrongC2(),
                c3=FakeC3(),
                drive_quality=FakeDriveQuality(),
                contextual_risk=FakeRiskPolicy(),
            )
        )


def test_combined_prediction_rejects_drive_quality_frame_mismatch() -> None:
    class WrongDriveQuality(FakeDriveQuality):
        def update(self, c3_frame):
            return FakePrediction(
                c3_frame.frame_id + 1,
                c3_frame.timestamp,
                91.0,
                "drive_quality",
            )

    with pytest.raises(ValueError, match="Drive Quality frame mismatch"):
        list(
            CombinedModelReplay(
                [FakeBundle(0)],
                c1=FakeC1(),
                c2=FakeC2(),
                c3=FakeC3(),
                drive_quality=WrongDriveQuality(),
                contextual_risk=FakeRiskPolicy(),
            )
        )


def test_combined_prediction_rejects_nonfinite_submission_risk() -> None:
    class InvalidRiskPolicy:
        def evaluate(self, c1, c2):
            return type("InvalidRisk", (), {"score_pct": float("nan")})()

    with pytest.raises(ValueError, match="contextual risk score"):
        list(
            CombinedModelReplay(
                [FakeBundle(0)],
                c1=FakeC1(),
                c2=FakeC2(),
                c3=FakeC3(),
                drive_quality=FakeDriveQuality(),
                contextual_risk=InvalidRiskPolicy(),
            )
        )


def test_unified_cli_defaults_to_both_repository_models() -> None:
    args = build_parser().parse_args(
        ["--dataset", "data", "--trip", "T01-Sample", "--write-video"]
    )
    assert str(args.c1_checkpoint) == "C1/student_ttc.pth"
    assert str(args.c2_bundle) == "models/driver_state_phase_2_v13"
    assert args.c1_stride == 2
    assert args.write_video is True


def test_drive_quality_diagnostics_are_not_added_to_submission_schema() -> None:
    estimate = DriveQualityAccumulator().update(
        SimpleNamespace(
            frame_id=0,
            timestamp=0.0,
            is_near_miss=False,
            is_harsh_brake=False,
            is_harsh_accel=False,
            is_harsh_corner=False,
            is_speeding=False,
            trip_complete=False,
        )
    )

    assert tuple(estimate.diagnostic_row()) == DRIVE_QUALITY_DIAGNOSTIC_FIELDS
    assert SUBMISSION_FIELDS == (
        "frame_id",
        "timestamp",
        "predicted_ttc",
        "predicted_driver_state",
        "predicted_risk_score",
    )


def test_new_run_archives_stale_full_and_partial_outputs(tmp_path) -> None:
    final = tmp_path / "T01-Sample.csv"
    partial = tmp_path / "partial" / "T01-Sample.csv"
    final.write_text("old-full", encoding="utf-8")
    partial.parent.mkdir()
    partial.write_text("old-partial", encoding="utf-8")

    archived = _archive_existing_outputs(
        tmp_path,
        archive_id="run-2",
        paths=(final, partial),
    )

    assert not final.exists()
    assert not partial.exists()
    assert archived == [
        tmp_path / "archive" / "run-2" / "T01-Sample.csv",
        tmp_path / "archive" / "run-2" / "partial" / "T01-Sample.csv",
    ]
    assert archived[0].read_text(encoding="utf-8") == "old-full"
    assert archived[1].read_text(encoding="utf-8") == "old-partial"
