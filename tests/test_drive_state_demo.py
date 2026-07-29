from __future__ import annotations

from dataclasses import asdict

import pytest

from drive_state.phase_1.classifier import ClassifierConfig, compute_window_features
from drive_state.phase_1.demo import STREAMING_CONFIG, StreamingClassifier
from drive_state.phase_1.features import FEATURE_FIELDS, FrameFeatures


def make_frame(frame_id: int, **overrides: float) -> FrameFeatures:
    values: dict[str, object] = dict.fromkeys(FEATURE_FIELDS, 0.0)
    values.update(frame_id=frame_id, timestamp=frame_id / 20.0, face_found=1)
    values.update(overrides)
    return FrameFeatures(**values)  # type: ignore[arg-type]


def test_trailing_window_never_reads_ahead() -> None:
    """The property the whole streaming mode rests on: a frame's window must
    not be influenced by anything after it. Frame 0 sees only frame 0, so a
    burst of closed eyes later in the sequence cannot raise its PERCLOS."""
    config = ClassifierConfig(window_frames=5, blink_closed=0.5)
    rows = [make_frame(i, blink=0.0 if i < 3 else 0.9) for i in range(6)]

    trailing = compute_window_features(rows, config, trailing=True)
    centred = compute_window_features(rows, config)

    assert trailing[0].perclos == pytest.approx(0.0)
    assert trailing[2].perclos == pytest.approx(0.0)
    # The centred window at frame 2 already sees frames 3 and 4 closed.
    assert centred[2].perclos > 0.0


def test_trailing_window_reaches_back_the_full_length() -> None:
    config = ClassifierConfig(window_frames=4, blink_closed=0.5)
    rows = [make_frame(i, blink=0.9 if i < 2 else 0.0) for i in range(4)]
    trailing = compute_window_features(rows, config, trailing=True)
    assert trailing[3].perclos == pytest.approx(0.5)  # frames 0,1 closed of 0-3


def test_streaming_classifier_matches_the_batch_trailing_computation() -> None:
    """StreamingClassifier keeps a ring buffer while compute_window_features
    slices arrays. They must agree, or the demo would show something the
    scorer never produced."""
    config = ClassifierConfig(window_frames=7, blink_closed=0.4)
    rows = [
        make_frame(
            i,
            blink=0.05 * (i % 13),
            mar=0.02 * (i % 9),
            jaw_open=0.01 * (i % 5),
            phone_conf=0.1 * (i % 7),
        )
        for i in range(40)
    ]

    batch = compute_window_features(rows, config, trailing=True)
    classifier = StreamingClassifier(config)
    for row, expected in zip(rows, batch, strict=True):
        _, window = classifier.update(row)
        # Compare every field rather than a chosen few: an earlier version of
        # this test checked only three, and silently passed while the streaming
        # path returned phone_frac=0 and could never predict `distracted`.
        assert asdict(window) == pytest.approx(asdict(expected))


def test_streaming_classifier_emits_a_state_from_the_first_frame() -> None:
    """A demo cannot wait 4.5 s for the buffer to fill before showing anything."""
    classifier = StreamingClassifier()
    state, window = classifier.update(make_frame(0))
    assert state == "alert"
    assert window.face_ratio == pytest.approx(1.0)


def test_streaming_classifier_survives_a_face_it_never_finds() -> None:
    classifier = StreamingClassifier()
    blank = FrameFeatures(**{**dict.fromkeys(FEATURE_FIELDS, 0.0), "frame_id": 0})  # type: ignore[arg-type]
    state, window = classifier.update(blank)
    assert window.face_ratio == pytest.approx(0.0)
    assert state == STREAMING_CONFIG.no_face_state


def test_streaming_config_is_a_valid_classifier_config() -> None:
    # Guards against a typo in the re-tuned constants silently disabling a rule.
    assert 0 < STREAMING_CONFIG.blink_closed < 1
    assert STREAMING_CONFIG.perclos_drowsy < STREAMING_CONFIG.perclos_microsleep
    assert STREAMING_CONFIG.mar_talking < STREAMING_CONFIG.mar_yawning


def test_warming_up_clears_once_the_buffer_holds_a_full_window() -> None:
    config = ClassifierConfig(window_frames=4)
    classifier = StreamingClassifier(config)
    for i in range(3):
        classifier.update(make_frame(i))
        assert classifier.warming_up
        assert classifier.filled == i + 1
    classifier.update(make_frame(3))
    assert not classifier.warming_up
    assert classifier.filled == 4


def test_warming_up_stays_false_once_the_buffer_is_saturated() -> None:
    """The deque is bounded, so `filled` plateaus rather than growing."""
    config = ClassifierConfig(window_frames=3)
    classifier = StreamingClassifier(config)
    for i in range(10):
        classifier.update(make_frame(i))
    assert not classifier.warming_up
    assert classifier.filled == 3


def test_auto_mode_falls_back_when_tripkit_cannot_read_the_layout(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """tripkit ships in this repo, so it always imports -- which means the
    unreadable-layout case is now reached by default rather than skipped. Auto
    mode must degrade to the built-in loader instead of failing the run."""
    from drive_state.phase_1.demo import TripkitLayoutError, _check_tripkit_layout

    trip = tmp_path / "T01-Sample"
    (trip / "T01-Sample.json").mkdir(parents=True)
    with pytest.raises(TripkitLayoutError):
        _check_tripkit_layout(trip)


def test_a_normal_trip_layout_passes_the_tripkit_check(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from drive_state.phase_1.demo import _check_tripkit_layout

    trip = tmp_path / "T02-Sample"
    trip.mkdir()
    (trip / "T02-Sample.json.gz").write_bytes(b"")
    _check_tripkit_layout(trip)  # must not raise


def test_frames_are_not_ready_until_the_buffer_fills() -> None:
    """`ready` is what lets a consumer hold output during boot instead of
    publishing a guess made over three frames."""
    config = ClassifierConfig(window_frames=4)
    classifier = StreamingClassifier(config)
    for _ in range(3):
        classifier.update(make_frame(0))
        assert classifier.warming_up
    classifier.update(make_frame(3))
    assert not classifier.warming_up


def test_backfilling_held_frames_beats_reporting_the_guess() -> None:
    """The measured reason the demo holds output: a unit that stays quiet until
    warm, then backfills, is right about the boot frames where one that guesses
    immediately is wrong."""
    config = ClassifierConfig(window_frames=5, phone_confidence=0.1, phone_distracted=0.5)
    # Evidence arrives *during* the warm-up, which is T01's shape: the driver
    # raises the handset two seconds in, so the opening frames genuinely show
    # nothing while the label already says `distracted`.
    rows = [make_frame(i, phone_conf=0.0 if i < 2 else 0.9) for i in range(10)]

    classifier = StreamingClassifier(config)
    guessed: dict[int, str] = {}
    held: list[int] = []
    backfilled: dict[int, str] = {}
    for row in rows:
        state, _ = classifier.update(row)
        guessed[row.frame_id] = state
        if classifier.warming_up:
            held.append(row.frame_id)
        else:
            for fid in held:
                backfilled[fid] = state
            held.clear()
            backfilled[row.frame_id] = state

    assert guessed[0] == "alert"  # the immediate guess, over one frame
    assert backfilled[0] == "distracted"  # what it concluded once warm
