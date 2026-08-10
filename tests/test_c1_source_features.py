import numpy as np
import pytest

from C1.features import FEATURES_USED, ScalarFeatureStream, build_scalars


def metadata() -> dict[str, object]:
    return {
        "speed_limit_kmh": 60.0,
        "weather": {
            "cloudiness": 20.0,
            "precipitation": 10.0,
            "wetness": 30.0,
            "precipitation_deposits": 40.0,
            "fog_density": 5.0,
            "sun_altitude_angle": 75.0,
        },
    }


def test_feature_contract_order_is_checkpoint_order() -> None:
    assert FEATURES_USED == (
        "speed",
        "accel",
        "jerk",
        "lat_accel",
        "speed_ratio",
        "has_limit",
        "cloud",
        "rain",
        "wet",
        "fog",
        "sun",
    )


def test_scalar_stream_derives_acceleration_and_jerk_at_sample_rate() -> None:
    stream = ScalarFeatureStream(metadata(), sample_hz=10.0)
    first = stream.step(
        {"speed_kmh": 0.0, "longitudinal_accel": 999.0, "lateral_accel": -1.0},
        target_count=3,
    )
    second = stream.step(
        {"speed_kmh": 3.6, "longitudinal_accel": -999.0, "lateral_accel": -1.0},
        target_count=4,
    )
    third = stream.step(
        {"speed_kmh": 7.2, "longitudinal_accel": 0.0, "lateral_accel": -1.0},
        target_count=5,
    )

    accel_index = FEATURES_USED.index("accel")
    jerk_index = FEATURES_USED.index("jerk")
    lateral_index = FEATURES_USED.index("lat_accel")
    assert first[accel_index] == 0.0
    assert first[jerk_index] == 0.0
    assert second[accel_index] == pytest.approx(np.tanh(10.0 / 3.0))
    assert second[jerk_index] == pytest.approx(1.0, abs=1e-6)
    assert third[accel_index] == pytest.approx(second[accel_index])
    assert third[jerk_index] == pytest.approx(0.0, abs=1e-6)
    assert first[lateral_index] == pytest.approx(np.tanh(0.5))


def test_build_scalars_rejects_unknown_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="unknown C1 scalar"):
        build_scalars([0], [0], [0], [0], [0], feature_order=("missing",))
    with pytest.raises(ValueError, match="equal length"):
        build_scalars([0, 1], [0], [0], [0], [0])
