from pathlib import Path
from types import SimpleNamespace

import pytest

from drive_state.phase_2.replay import DMSBundle, frame_prediction_from_runtime


def runtime_prediction(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "state_id": 4,
        "probabilities": (0.1, 0.05, 0.0, 0.05, 0.8),
        "closed_probability": 0.1,
        "ocular_reliability": 0.9,
        "closure_duration_seconds": 0.0,
        "perclos": (0.1, 0.1, 0.1),
        "slow_perclos": (0.0, 0.0, 0.0),
        "perclos_reliable": (True, False, False),
        "nod_probability": 0.0,
        "microsleep_active": False,
        "drowsy_episode_active": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_distraction_maps_to_submission_vocabulary_and_vss() -> None:
    frame = frame_prediction_from_runtime(
        frame_id=7,
        timestamp=0.35,
        prediction=runtime_prediction(),
        latency_ms=10.0,
    )

    assert frame.state == "distracted"
    assert frame.submission_row() == {
        "frame_id": 7,
        "timestamp": 0.35,
        "predicted_driver_state": "distracted",
    }
    assert frame.vss_signals().as_vss_dict() == {
        "Vehicle.Driver.AttentiveProbability": 10.0,
        "Vehicle.Driver.DistractionLevel": 80.0,
        "Vehicle.Driver.FatigueLevel": 5.0,
        "Vehicle.Driver.IsEyesOnRoad": False,
        "Vehicle.ADAS.DMS.IsWarning": True,
    }


def test_microsleep_is_a_fatigue_warning() -> None:
    frame = frame_prediction_from_runtime(
        frame_id=40,
        timestamp=2.0,
        prediction=runtime_prediction(
            state_id=2,
            probabilities=(0.0, 0.0, 1.0, 0.0, 0.0),
            microsleep_active=True,
        ),
        latency_ms=8.0,
    )

    signals = frame.vss_signals()
    assert signals.fatigue_level == 100.0
    assert signals.is_warning is True
    assert signals.is_eyes_on_road is True


def test_probability_contract_rejects_invalid_distribution() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        frame_prediction_from_runtime(
            frame_id=0,
            timestamp=0.0,
            prediction=runtime_prediction(probabilities=(0.1,) * 5),
            latency_ms=1.0,
        )


def test_bundle_validation_lists_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="visual/best.pt"):
        DMSBundle.at(tmp_path).validate()
