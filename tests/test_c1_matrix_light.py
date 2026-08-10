from __future__ import annotations

import numpy as np

from safeloop.c1.matrix_light import MatrixLightController, MatrixLightOfflineEvaluator
from safeloop.c1.types import C1FramePrediction, TrackRisk


def _risk(ttc: float = 1.2) -> TrackRisk:
    return TrackRisk(
        track_id=7,
        label="car",
        confidence=0.9,
        bbox=(270.0, 150.0, 370.0, 290.0),
        collision_relevant=True,
        predicted_ttc_s=ttc,
        scale_ttc_s=ttc,
        range_ttc_s=ttc,
        range_closing_speed_mps=8.0,
        range_trend_confidence=0.9,
        lateral_ttc_s=float("inf"),
        ego_fallback_ttc_s=float("inf"),
        estimated_distance_m=12.0,
    )


def _prediction(risks=(_risk(),)) -> C1FramePrediction:
    return C1FramePrediction(1, 0.05, 1.2, True, risks, 5.0)


def test_matrix_light_covers_selected_bbox_and_is_simulation_only() -> None:
    controller = MatrixLightController()
    frame = controller.compute(np.full((360, 640, 3), 180, np.uint8), _prediction())

    assert frame.simulation_only is True
    assert len(frame.targets) == 1
    assert frame.targets[0].geometric_coverage >= 0.95
    assert frame.grid_mask.any()


def test_matrix_light_detects_night_and_rejects_non_danger() -> None:
    controller = MatrixLightController()
    safe = _risk(ttc=5.0)
    frame = controller.compute(
        np.full((360, 640, 3), 15, np.uint8), _prediction((safe,))
    )

    assert frame.mode == "NIGHT"
    assert frame.targets == ()
    assert not frame.grid_mask.any()


def test_offline_evaluator_uses_independent_gt_bbox() -> None:
    controller = MatrixLightController()
    frame = controller.compute(np.full((360, 640, 3), 180, np.uint8), _prediction())
    evaluator = MatrixLightOfflineEvaluator((360, 640), frame.grid_mask.shape)
    evaluator.add(frame, [(275.0, 155.0, 365.0, 285.0)])
    report = evaluator.report()

    assert report.frame_precision == 1.0
    assert report.frame_recall == 1.0
    assert report.target_precision == 1.0
    assert report.gt_coverage_95_recall == 1.0
