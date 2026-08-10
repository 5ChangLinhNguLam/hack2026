from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safeloop.c1.temporal_features import FEATURE_DIM, TemporalTripFeatures
from safeloop.c1.temporal_model import (
    CausalTemporalTTC,
    FeatureNormalizer,
    TemporalModelConfig,
)
from safeloop.c1.train_fusion_fp32 import (
    EXPECTED_EVALUATOR_SHA256,
    FP32_SEEDS,
    FusionCandidateResult,
    _assert_fp32_finite,
    _evaluator_sha256,
    aggregate_seed_reports,
    build_parser,
    evaluate_fusion_candidate_inner,
    select_fusion_candidate,
)
from safeloop.c1.train_temporal import (
    InnerFold,
    OuterFold,
    RawModelPredictions,
    SupervisedTrip,
    TTCSupervision,
    ThresholdChoice,
    TrainSettings,
)


def _supervised_trip(trip_id: str, n_frames: int = 8) -> SupervisedTrip:
    frame_ids = np.arange(n_frames, dtype=np.int64)
    timestamps = frame_ids.astype(np.float64) * 0.05
    features = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
    features[:, 0] = 1.0
    runtime = TemporalTripFeatures(
        trip_id=trip_id,
        frame_ids=frame_ids,
        timestamps=timestamps,
        features=features,
        physics_ttc_s=np.full(n_frames, 3.0, dtype=np.float32),
    )
    labels = TTCSupervision(
        trip_id=trip_id,
        frame_ids=frame_ids,
        ttc_s=np.full(n_frames, 2.5, dtype=np.float32),
    )
    return SupervisedTrip(runtime, labels)


def _choice(score: float) -> ThresholdChoice:
    return ThresholdChoice(
        threshold=0.5,
        macro_composite=score,
        macro_f1=0.4,
        macro_mae_critical=1.2,
    )


def _seed_report(
    composite: float,
    *,
    mae: float = 1.0,
    f1: float = 0.4,
    inverse_mae: float = 0.2,
    frames: int = 3600,
) -> dict[str, object]:
    return {
        "frames": frames,
        "fusion": {
            "macro": {
                "mae_critical": mae,
                "f1": f1,
                "inv_ttc_mae": inverse_mae,
                "composite_score": composite,
            }
        },
    }


def test_fp32_protocol_locks_three_seeds_and_has_no_seed_or_amp_cli() -> None:
    assert FP32_SEEDS == (2026, 2027, 2028)
    assert _evaluator_sha256() == EXPECTED_EVALUATOR_SHA256
    args = build_parser().parse_args(
        ["--data-dir", "/data", "--detection-cache-dir", "/cache"]
    )
    assert not hasattr(args, "seed")
    assert not hasattr(args, "amp")
    assert not hasattr(args, "candidates")
    assert not hasattr(args, "thresholds")
    assert not hasattr(args, "expected_physics_score")
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--data-dir",
                "/data",
                "--detection-cache-dir",
                "/cache",
                "--seed",
                "999",
            ]
        )


def test_inner_runner_trains_fusion_only_and_excludes_outer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import safeloop.c1.train_fusion_fp32 as runner

    supervised = {
        trip_id: _supervised_trip(trip_id)
        for trip_id in ("T01-Sample", "T02-Sample", "T06-Sample")
    }
    outer = OuterFold(
        train_trip_ids=("T01-Sample", "T02-Sample"),
        test_trip_id="T06-Sample",
    )
    inner = (
        InnerFold(
            train_trip_ids=("T01-Sample",),
            validation_trip_id="T02-Sample",
        ),
    )
    config = TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1)
    variants: list[str] = []
    train_trip_sets: list[set[str]] = []

    def fake_train(
        train_trips,
        validation_trips,
        model_config,
        settings,
        *,
        device,
        log_dir,
        seed,
        variant,
    ):
        del validation_trips, settings, log_dir, seed
        variants.append(variant)
        train_trip_sets.append({trip.trip_id for trip in train_trips})
        model = CausalTemporalTTC(model_config).to(device).eval()
        normalizer = FeatureNormalizer.fit(
            trip.runtime.features for trip in train_trips
        )
        return model, normalizer, 3

    def fake_predict(model, normalizer, runtime, *, device):
        del model, normalizer, device
        n_frames = runtime.n_frames
        zeros = np.zeros(n_frames, dtype=np.float32)
        return RawModelPredictions(zeros, zeros, zeros, zeros)

    threshold_variants: list[str] = []

    def fake_select(raw, labels, *, variant, thresholds):
        assert set(raw) == {"T02-Sample"}
        assert set(labels) == {"T02-Sample"}
        assert tuple(thresholds) == (0.5,)
        threshold_variants.append(variant)
        return _choice(51.0)

    monkeypatch.setattr(runner, "train_inner_model", fake_train)
    monkeypatch.setattr(runner, "predict_raw", fake_predict)
    monkeypatch.setattr(runner, "select_threshold", fake_select)
    result = evaluate_fusion_candidate_inner(
        supervised,
        outer,
        inner,
        config,
        TrainSettings(max_epochs=1, patience=1, amp=False, seed=2026),
        thresholds=(0.5,),
        device=torch.device("cpu"),
        log_root=tmp_path,
        seed=2026,
    )
    assert variants == ["fusion"]
    assert threshold_variants == ["fusion"]
    assert train_trip_sets == [{"T01-Sample"}]
    assert all("T06-Sample" not in trip_ids for trip_ids in train_trip_sets)
    assert result.validation_trip_ids == ("T02-Sample",)
    assert result.best_epochs == (3,)


def test_inner_runner_rejects_amp_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import safeloop.c1.train_fusion_fp32 as runner

    monkeypatch.setattr(
        runner,
        "train_inner_model",
        lambda *args, **kwargs: pytest.fail("AMP gate must run before training"),
    )
    outer = OuterFold(("T01-Sample", "T02-Sample"), "T06-Sample")
    with pytest.raises(ValueError, match="forbids AMP"):
        evaluate_fusion_candidate_inner(
            {
                trip_id: _supervised_trip(trip_id)
                for trip_id in ("T01-Sample", "T02-Sample", "T06-Sample")
            },
            outer,
            (InnerFold(("T01-Sample",), "T02-Sample"),),
            TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1),
            TrainSettings(max_epochs=1, patience=1, amp=True),
            thresholds=(0.5,),
            device=torch.device("cpu"),
            log_root=tmp_path,
            seed=2026,
        )


def test_fusion_candidate_selection_uses_inner_score_and_rejects_leakage() -> None:
    small = TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1)
    large = TemporalModelConfig(sequence_length=16, hidden_size=64, gru_layers=1)
    train_ids = ("T01-Sample", "T02-Sample")
    results = (
        FusionCandidateResult(small, _choice(49.0), (2,), train_ids),
        FusionCandidateResult(large, _choice(52.0), (4,), train_ids),
    )
    selected = select_fusion_candidate(
        results,
        outer_train_trip_ids=train_ids,
        outer_test_trip_id="T06-Sample",
    )
    assert selected.config == large

    leaked = FusionCandidateResult(
        small,
        _choice(99.0),
        (2,),
        ("T06-Sample",),
    )
    with pytest.raises(ValueError, match="outer-test"):
        select_fusion_candidate(
            (leaked,),
            outer_train_trip_ids=train_ids,
            outer_test_trip_id="T06-Sample",
        )


def test_three_seed_aggregate_reports_sample_std_without_best_seed() -> None:
    reports = {
        2026: _seed_report(54.0, mae=4.0, f1=0.30),
        2027: _seed_report(55.0, mae=5.0, f1=0.40),
        2028: _seed_report(53.0, mae=6.0, f1=0.50),
    }
    summary = aggregate_seed_reports(reports, physics_score=55.2)
    composite = summary["macro_mean_std"]["composite_score"]
    assert composite["values_in_seed_order"] == [54.0, 55.0, 53.0]
    assert composite["mean"] == 54.0
    assert composite["sample_std_ddof_1"] == statistics.stdev([54.0, 55.0, 53.0])
    assert summary["seed_selection"] is False
    assert "best_seed" not in summary
    assert summary["p1_closed"] is True
    assert summary["decision"] == "close_p1_keep_physics_default"


def test_three_seed_aggregate_requires_every_seed_and_exact_3600() -> None:
    reports = {
        2026: _seed_report(54.0),
        2027: _seed_report(55.0),
    }
    with pytest.raises(ValueError, match="exactly the declared seeds"):
        aggregate_seed_reports(reports, physics_score=55.2)

    reports[2028] = _seed_report(53.0, frames=3599)
    with pytest.raises(ValueError, match="3,600"):
        aggregate_seed_reports(reports, physics_score=55.2)


def test_fp32_guard_rejects_nonfinite_or_half_precision_weights() -> None:
    config = TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1)
    model = CausalTemporalTTC(config)
    _assert_fp32_finite(model)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        _assert_fp32_finite(model)

    half_model = CausalTemporalTTC(config).half()
    with pytest.raises(RuntimeError, match="dtype"):
        _assert_fp32_finite(half_model)
