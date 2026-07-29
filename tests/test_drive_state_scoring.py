from __future__ import annotations

import pytest

from drive_state.phase_1.scoring import overall_composite, score_trip
from drive_state.phase_1.states import DRIVER_STATE_CLASSES, state_from_signals


def test_perfect_prediction_scores_100() -> None:
    truth = {0: "alert", 1: "alert", 2: "drowsy"}
    score = score_trip("T", dict(truth), truth)
    assert score.accuracy == 1.0
    assert score.macro_f1 == 1.0
    assert score.composite == pytest.approx(100.0)


def test_macro_f1_ignores_classes_absent_from_ground_truth() -> None:
    """The organiser averages F1 only over classes present in the trip's truth.

    Here `microsleep` is predicted but never true. It must not contribute a
    zero to the macro average -- it may only cost accuracy and alert's recall.
    """
    truth = {0: "alert", 1: "alert", 2: "alert", 3: "alert"}
    predicted = {0: "alert", 1: "alert", 2: "alert", 3: "microsleep"}
    score = score_trip("T", predicted, truth)

    assert score.present_classes == ("alert",)
    assert score.accuracy == pytest.approx(0.75)
    # alert: precision 3/3, recall 3/4 -> F1 = 6/7. Macro over {alert} only.
    assert score.macro_f1 == pytest.approx(6 / 7)
    assert score.composite == pytest.approx(100.0 * (0.5 * 0.75 + 0.5 * 6 / 7))


def test_absent_class_costs_less_than_a_wrong_present_class() -> None:
    """The asymmetry the threshold choices lean on.

    Both predictions get one frame wrong out of four, so accuracy is equal.
    Confusing two classes that are both present drags two F1 scores down;
    inventing an absent class drags only one.
    """
    truth = {0: "alert", 1: "alert", 2: "alert", 3: "drowsy"}
    invented = score_trip("T", {0: "alert", 1: "alert", 2: "alert", 3: "microsleep"}, truth)
    confused = score_trip("T", {0: "alert", 1: "alert", 2: "alert", 3: "alert"}, truth)

    assert invented.accuracy == confused.accuracy
    assert invented.composite > confused.composite


def test_score_trip_uses_only_overlapping_frames() -> None:
    truth = {0: "alert", 1: "drowsy"}
    score = score_trip("T", {0: "alert", 1: "drowsy", 99: "yawning"}, truth)
    assert score.n_frames == 2


def test_score_trip_rejects_disjoint_frame_ids() -> None:
    with pytest.raises(ValueError):
        score_trip("T", {5: "alert"}, {0: "alert"})


def test_overall_composite_is_unweighted_mean_over_trips() -> None:
    truth_a = dict.fromkeys(range(10), "alert")
    truth_b = dict.fromkeys(range(1000), "drowsy")
    scores = [
        score_trip("A", dict(truth_a), truth_a),
        score_trip("B", dict.fromkeys(truth_b, "alert"), truth_b),
    ]
    # A is perfect, B is entirely wrong; the 100x size difference must not tilt it.
    assert overall_composite(scores) == pytest.approx(50.0)


def test_signal_table_covers_the_five_labelled_combinations() -> None:
    assert state_from_signals("open", "normal", "normal") == "alert"
    assert state_from_signals("open", "side", "normal") == "distracted"
    assert state_from_signals("partial", "down", "normal") == "drowsy"
    assert state_from_signals("partial", "normal", "yawning") == "yawning"
    assert state_from_signals("closed", "down", "normal") == "microsleep"


def test_unlabelled_signal_combinations_fall_back_by_risk() -> None:
    # Closed eyes with a head that is not down never appears in the labels, but
    # still has to resolve to something -- and to the riskier reading.
    assert state_from_signals("closed", "normal", "normal") == "microsleep"
    assert state_from_signals("open", "down", "yawning") == "yawning"
    assert state_from_signals("open", "down", "normal") == "distracted"


def test_fallback_always_returns_a_valid_class() -> None:
    for eye in ("open", "partial", "closed"):
        for head in ("normal", "side", "down"):
            for mouth in ("normal", "yawning"):
                assert state_from_signals(eye, head, mouth) in DRIVER_STATE_CLASSES
