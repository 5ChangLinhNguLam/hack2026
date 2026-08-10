from __future__ import annotations

import pytest

from safeloop.c1.corridor import CorridorConfig, CorridorEstimator
from safeloop.c1.lane import LaneEstimate


IMAGE_SHAPE = (360, 640)


def _lane(*, valid: bool = True, confidence: float = 0.9) -> LaneEstimate:
    return LaneEstimate(
        valid=valid,
        confidence=confidence,
        left_points=((280, 180), (240, 270), (190, 359)),
        right_points=((360, 180), (400, 270), (450, 359)),
        lane_center_offset=0.0,
        heading_error_deg=0.0,
        lane_width_px=250.0,
        departure_warning=False,
        inferred_side=None,
    )


def test_reliable_lane_builds_corridor_at_bbox_footpoint() -> None:
    estimator = CorridorEstimator()
    corridor = estimator.estimate(_lane(), IMAGE_SHAPE)
    evidence = estimator.assess(corridor, (170.0, 120.0, 230.0, 350.0))

    assert corridor.source == "lane"
    assert evidence.footpoint == (200.0, 350.0)
    # The same x would be outside the narrower corridor at the bbox centre.
    center_bounds = corridor.bounds_at((120.0 + 350.0) / 2.0)
    assert evidence.current_bounds[0] < evidence.footpoint[0]
    assert evidence.footpoint[0] < evidence.current_bounds[1]
    assert evidence.footpoint[0] < center_bounds[0]
    assert evidence.in_path_probability > 0.5


@pytest.mark.parametrize(
    "lane",
    (
        None,
        _lane(valid=False),
        _lane(confidence=0.20),
        _lane(confidence=0.56),
    ),
)
def test_missing_or_unreliable_lane_uses_fixed_camera_trapezoid(lane) -> None:
    estimator = CorridorEstimator(principal_x_px=315.0)
    corridor = estimator.estimate(lane, IMAGE_SHAPE)
    top = corridor.bounds_at(0.35 * IMAGE_SHAPE[0])
    bottom = corridor.bounds_at(IMAGE_SHAPE[0] - 1)

    assert corridor.source == "fixed"
    assert sum(top) / 2.0 == pytest.approx(315.0)
    assert sum(bottom) / 2.0 == pytest.approx(315.0)
    assert bottom[1] - bottom[0] > top[1] - top[0]


def test_short_horizon_footpoint_motion_identifies_cut_in() -> None:
    config = CorridorConfig(prediction_horizon_s=0.75)
    estimator = CorridorEstimator(config)
    corridor = estimator.estimate(None, IMAGE_SHAPE)
    bbox = (460.0, 220.0, 520.0, 320.0)

    entering = estimator.assess(
        corridor,
        bbox,
        velocity_px_s=(-180.0, 0.0),
        track_confidence=0.8,
    )
    leaving = estimator.assess(
        corridor,
        bbox,
        velocity_px_s=(180.0, 0.0),
        track_confidence=0.8,
    )

    assert entering.predicted_footpoint[0] < entering.footpoint[0]
    assert entering.predicted_corridor_overlap > entering.corridor_overlap
    assert entering.cut_in_probability > 0.25
    assert leaving.cut_in_probability == 0.0
    assert entering.track_confidence == pytest.approx(0.8)


def test_coast_and_motion_error_raise_lateral_uncertainty() -> None:
    estimator = CorridorEstimator()
    corridor = estimator.estimate(_lane(), IMAGE_SHAPE)
    bbox = (280.0, 180.0, 360.0, 330.0)
    fresh = estimator.assess(corridor, bbox)
    coasting = estimator.assess(
        corridor,
        bbox,
        velocity_px_s=(20.0, 0.0),
        velocity_uncertainty_px_s=25.0,
        coast_age_s=0.4,
    )

    assert coasting.lateral_uncertainty_px > fresh.lateral_uncertainty_px
    assert coasting.footpoint[0] == pytest.approx(fresh.footpoint[0] + 8.0)
    assert coasting.predicted_footpoint[0] > coasting.footpoint[0]
    assert coasting.corridor_source == "lane"


def test_invalid_geometry_is_rejected() -> None:
    estimator = CorridorEstimator()
    with pytest.raises(ValueError, match="image_shape"):
        estimator.estimate(None, (1, 640))

    corridor = estimator.estimate(None, IMAGE_SHAPE)
    with pytest.raises(ValueError, match="positive width"):
        estimator.assess(corridor, (10.0, 20.0, 10.0, 40.0))
    with pytest.raises(ValueError, match="non-negative"):
        estimator.assess(corridor, (10.0, 20.0, 30.0, 40.0), coast_age_s=-0.1)
