from __future__ import annotations

import math

from safeloop.c1.pseudo_label import PSEUDO_SOURCE, PseudoLabelEnsemble
from safeloop.c1.tracker import MonocularTTCTracker, TrackerConfig
from safeloop.c1.types import Detection


def _car(distance_m: float) -> Detection:
    height = 320.0 * 2.84 / distance_m
    return Detection(2, "car", 0.9, (290.0, 230.0 - height, 350.0, 230.0))


def test_pseudo_label_requires_consensus_and_names_its_source() -> None:
    trackers = [
        MonocularTTCTracker(TrackerConfig(min_history=2))
        for _ in range(3)
    ]
    teacher = PseudoLabelEnsemble(trackers)
    label = None
    for frame_id, distance in enumerate((20.0, 18.5, 17.0, 15.5)):
        label = teacher.update(
            [_car(distance)],
            frame_id=frame_id,
            timestamp=frame_id * 0.1,
            image_shape=(360, 640),
            ego_speed_kmh=30.0,
        )

    assert label is not None
    assert math.isfinite(label.pseudo_ttc_s)
    assert label.support_count == 3
    assert label.agreement > 0.99
    assert label.source == PSEUDO_SOURCE
    assert "pseudo_ttc" in label.row()
    assert "ground_truth" not in label.row()


def test_pseudo_normal_is_lower_confidence_when_geometry_is_ambiguous() -> None:
    trackers = [MonocularTTCTracker() for _ in range(2)]
    teacher = PseudoLabelEnsemble(trackers)
    empty = teacher.update(
        [], frame_id=0, timestamp=0.0, image_shape=(360, 640), ego_speed_kmh=20.0
    )
    static = teacher.update(
        [_car(20.0)],
        frame_id=1,
        timestamp=0.1,
        image_shape=(360, 640),
        ego_speed_kmh=20.0,
    )

    assert math.isinf(empty.pseudo_ttc_s)
    assert math.isinf(static.pseudo_ttc_s)
    assert empty.confidence > static.confidence
