from __future__ import annotations

import pytest

from safeloop.c3 import Challenge3FrameEstimate
from safeloop.drive_quality import (
    DriveQualityAccumulator,
    FULL_TRIP,
    NO_DATA,
    PREFIX,
    ROLLING_60S,
)


def _c3(
    frame_id: int,
    timestamp: float,
    *,
    near: bool = False,
    brake: bool = False,
    accel: bool = False,
    corner: bool = False,
    speeding: bool = False,
    complete: bool = False,
) -> Challenge3FrameEstimate:
    return Challenge3FrameEstimate(
        frame_id=frame_id,
        timestamp=timestamp,
        safe_score_estimate=100.0,
        risk_score_pct=0.0,
        grade="A",
        total_penalty=0.0,
        processed_frames=frame_id + 1,
        near_miss_frames=int(near),
        harsh_brake_frames=int(brake),
        harsh_accel_frames=int(accel),
        harsh_corner_frames=int(corner),
        speeding_frames=int(speeding),
        speeding_pct_time=100.0 if speeding else 0.0,
        is_near_miss=near,
        is_harsh_brake=brake,
        is_harsh_accel=accel,
        is_harsh_corner=corner,
        is_speeding=speeding,
        trip_complete=complete,
    )


def test_summary_is_explicitly_unavailable_before_first_frame() -> None:
    result = DriveQualityAccumulator().summary()

    assert result["scope"] == NO_DATA
    assert result["score_available"] is False
    assert result["score_pct"] is None
    assert result["grade"] == "N/A"


def test_first_frame_produces_a_prefix_score() -> None:
    result = DriveQualityAccumulator().update(_c3(0, 0.0))

    assert result.score_available
    assert result.score_pct == 100.0
    assert result.total_penalty == 0.0
    assert result.grade == "A"
    assert result.scope == PREFIX
    assert result.observed_seconds == 0.0
    assert not result.window_ready


def test_nearby_positive_frames_are_one_event() -> None:
    accumulator = DriveQualityAccumulator(merge_gap_seconds=0.5)
    inputs = (
        _c3(0, 0.0, brake=True),
        _c3(1, 0.1),
        _c3(2, 0.5, brake=True),
        _c3(3, 0.6),
        _c3(4, 1.11, brake=True),
    )

    results = [accumulator.update(frame) for frame in inputs]

    assert results[0].new_events["harsh_brake"]
    assert not results[2].new_events["harsh_brake"]
    assert results[4].new_events["harsh_brake"]
    assert results[-1].event_counts_window["harsh_brake"] == 2
    assert results[-1].score_pct == 94.0


def test_exact_half_second_inactive_gap_is_merged() -> None:
    accumulator = DriveQualityAccumulator(merge_gap_seconds=0.5)
    accumulator.update(_c3(0, 0.0, brake=True))
    accumulator.update(_c3(1, 0.1))
    merged = accumulator.update(_c3(2, 0.6, brake=True))
    accumulator.update(_c3(3, 0.7))
    separate = accumulator.update(_c3(4, 1.21, brake=True))

    assert not merged.new_events["harsh_brake"]
    assert separate.new_events["harsh_brake"]
    assert separate.event_counts_window["harsh_brake"] == 2


def test_formula_counts_events_but_speeding_remains_percentage() -> None:
    accumulator = DriveQualityAccumulator()
    result = None
    for index in range(4):
        result = accumulator.update(
            _c3(
                index,
                index * 0.1,
                near=index < 2,
                brake=index == 0,
                accel=index == 0,
                corner=index == 0,
                speeding=index < 2,
            )
        )

    assert result is not None
    assert dict(result.event_counts_window) == {
        "near_miss": 1,
        "harsh_brake": 1,
        "harsh_accel": 1,
        "harsh_corner": 1,
    }
    assert result.speeding_pct_window == 50.0
    assert result.total_penalty == pytest.approx(
        5.0 + 3.0 + 2.0 + 2.0 + 50.0 * 0.15
    )
    assert result.score_pct == pytest.approx(80.5)
    assert result.grade == "B"


def test_trip_completion_changes_prefix_to_full_trip_and_locks_input() -> None:
    accumulator = DriveQualityAccumulator()
    accumulator.update(_c3(0, 0.0))
    result = accumulator.update(_c3(1, 0.05, complete=True))

    assert result.scope == FULL_TRIP
    assert result.trip_complete
    assert accumulator.summary()["scope"] == FULL_TRIP
    with pytest.raises(ValueError, match="already complete"):
        accumulator.update(_c3(2, 0.1))


def test_rolling_window_discards_old_events_and_speed_samples() -> None:
    accumulator = DriveQualityAccumulator(window_seconds=60.0)
    accumulator.update(_c3(0, 0.0, near=True, speeding=True))
    accumulator.update(_c3(1, 0.6))
    before = accumulator.update(_c3(2, 59.9))
    after = accumulator.update(_c3(3, 60.1))

    assert before.scope == PREFIX
    assert before.event_counts_window["near_miss"] == 1
    assert before.speeding_pct_window == pytest.approx(100.0 / 3.0)
    assert after.scope == ROLLING_60S
    assert after.window_ready
    assert after.observed_seconds == 60.0
    assert after.event_counts_window["near_miss"] == 0
    assert after.speeding_pct_window == 0.0


def test_sustained_event_remains_after_its_onset_leaves_window() -> None:
    accumulator = DriveQualityAccumulator()
    first = accumulator.update(_c3(0, 0.0, near=True))
    accumulator.update(_c3(1, 30.0, near=True))
    accumulator.update(_c3(2, 60.0, near=True))
    latest = accumulator.update(_c3(3, 61.0, near=True))

    assert first.new_events["near_miss"]
    assert not latest.new_events["near_miss"]
    assert latest.scope == ROLLING_60S
    assert latest.event_counts_window["near_miss"] == 1
    assert latest.score_pct == 95.0


def test_window_shorter_than_merge_gap_handles_carry_in_episode() -> None:
    accumulator = DriveQualityAccumulator(
        window_seconds=0.1, merge_gap_seconds=0.5
    )
    accumulator.update(_c3(0, 0.0, near=True))
    accumulator.update(_c3(1, 0.1))
    accumulator.update(_c3(2, 0.2))
    result = accumulator.update(_c3(3, 0.3, near=True))

    assert not result.new_events["near_miss"]
    assert result.event_counts_window["near_miss"] == 1


def test_long_completed_trip_remains_a_rolling_window_score() -> None:
    accumulator = DriveQualityAccumulator(window_seconds=60.0)
    accumulator.update(_c3(0, 0.0))
    result = accumulator.update(_c3(1, 61.0, complete=True))

    assert result.scope == ROLLING_60S
    assert result.trip_complete


@pytest.mark.parametrize(
    ("first,second,message"),
    (
        (_c3(1, 0.0), None, "start at frame 0"),
        (_c3(0, 0.0), _c3(2, 0.1), "contiguous and ordered"),
        (_c3(0, 0.0), _c3(1, 0.0), "strictly increasing"),
    ),
)
def test_invalid_stream_position_is_rejected(
    first: Challenge3FrameEstimate,
    second: Challenge3FrameEstimate | None,
    message: str,
) -> None:
    accumulator = DriveQualityAccumulator()
    if second is None:
        with pytest.raises(ValueError, match=message):
            accumulator.update(first)
        return
    accumulator.update(first)
    with pytest.raises(ValueError, match=message):
        accumulator.update(second)


def test_reset_starts_a_fresh_trip() -> None:
    accumulator = DriveQualityAccumulator()
    accumulator.update(_c3(0, 0.0, brake=True, complete=True))

    accumulator.reset()
    result = accumulator.update(_c3(0, 0.0))

    assert result.score_pct == 100.0
    assert result.event_counts_window["harsh_brake"] == 0
    assert not result.trip_complete


def test_diagnostic_row_uses_drive_quality_prefix() -> None:
    result = DriveQualityAccumulator().update(
        _c3(0, 0.0, near=True, speeding=True)
    )

    row = result.diagnostic_row()
    assert row["drive_quality_score_pct"] == 80.0
    assert row["drive_quality_near_miss_events_window"] == 1
    assert row["drive_quality_speeding_pct_window"] == 100.0
    assert row["drive_quality_new_near_miss_event"] is True
    assert all(key.startswith("drive_quality_") for key in row)


@pytest.mark.parametrize(
    ("kwargs,message"),
    (
        ({"window_seconds": 0.0}, "window_seconds"),
        ({"window_seconds": float("nan")}, "window_seconds"),
        ({"merge_gap_seconds": -0.1}, "merge_gap_seconds"),
    ),
)
def test_configuration_is_validated(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DriveQualityAccumulator(**kwargs)
