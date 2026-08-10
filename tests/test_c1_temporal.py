from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safeloop.c1.prediction_preflight import preflight_prediction_csv
from safeloop.c1.temporal_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    TemporalTripFeatures,
    inverse_ttc,
)
from safeloop.c1.temporal_model import (
    PARAMETER_BUDGET,
    CausalTemporalTTC,
    FeatureNormalizer,
    TemporalModelConfig,
    build_causal_windows,
    calibrated_ttc,
    export_temporal_onnx,
    fused_inverse_ttc,
)
from safeloop.c1.train_temporal import (
    CandidateResult,
    OuterFold,
    SupervisedTrip,
    TTCSupervision,
    ThresholdChoice,
    TrainSettings,
    _ablation_report,
    _loss,
    _require_exact_oof_csv_set,
    _write_prediction_csv,
    nested_lopo_splits,
    outer_training_trips,
    select_candidate,
    train_inner_model,
)
from team_kit.evaluation import compute_trip_metrics


def _choice(score: float = 50.0) -> ThresholdChoice:
    return ThresholdChoice(
        threshold=0.5,
        macro_composite=score,
        macro_f1=0.5,
        macro_mae_critical=1.0,
    )


def _supervised_trip(
    trip_id: str,
    *,
    n_frames: int = 8,
    feature_value: float = 0.0,
) -> SupervisedTrip:
    frame_ids = np.arange(n_frames, dtype=np.int64)
    timestamps = frame_ids.astype(np.float64) * 0.05
    features = np.full((n_frames, FEATURE_DIM), feature_value, dtype=np.float32)
    features[:, 0] = 1.0
    ttc = np.full(n_frames, np.inf, dtype=np.float32)
    ttc[::2] = 1.5
    runtime = TemporalTripFeatures(
        trip_id=trip_id,
        frame_ids=frame_ids,
        timestamps=timestamps,
        features=features,
        physics_ttc_s=np.full(n_frames, np.inf, dtype=np.float32),
    )
    labels = TTCSupervision(trip_id=trip_id, frame_ids=frame_ids, ttc_s=ttc)
    return SupervisedTrip(runtime, labels)


def test_all_supported_grus_are_below_parameter_budget() -> None:
    largest = 0
    for sequence_length in (8, 16, 24):
        for hidden_size in (32, 64):
            for layers in (1, 2):
                model = CausalTemporalTTC(
                    TemporalModelConfig(
                        sequence_length=sequence_length,
                        hidden_size=hidden_size,
                        gru_layers=layers,
                    )
                )
                largest = max(largest, model.parameter_count)
                assert model.parameter_count < PARAMETER_BUDGET
    # Guard against a vacuous tiny stub while keeping a wide edge budget.
    assert 10_000 < largest < 100_000


def test_temporal_forward_shape_and_strict_causality() -> None:
    torch.manual_seed(7)
    model = CausalTemporalTTC(
        TemporalModelConfig(sequence_length=16, hidden_size=32, gru_layers=2)
    ).eval()
    features = torch.randn(2, 16, FEATURE_DIM)
    with torch.inference_mode():
        full = model(features)
        prefix = model(features[:, :9])
        changed_future = features.clone()
        changed_future[:, 9:] = torch.randn_like(changed_future[:, 9:]) * 100.0
        changed = model(changed_future)
    assert full.shape == (2, 16, 4)
    torch.testing.assert_close(full[:, :9], prefix, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full[:, :9], changed[:, :9], rtol=1e-5, atol=1e-6)


def test_causal_windows_never_include_future_frames() -> None:
    raw = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    raw[:, 0] = np.arange(30)
    windows, lengths = build_causal_windows(raw, 8)
    assert lengths.tolist()[:9] == [1, 2, 3, 4, 5, 6, 7, 8, 8]
    assert windows[4, :5, 0].tolist() == [0, 1, 2, 3, 4]
    assert np.all(windows[4, 5:] == 0.0)
    assert windows[20, :, 0].tolist() == list(range(13, 21))


def test_temporal_schema_excludes_fitted_physics_fields() -> None:
    joined = " ".join(FEATURE_NAMES)
    assert FEATURE_DIM == 48
    for forbidden in (
        "ttc",
        "distance",
        "closing_speed",
        "trend_confidence",
        "depth",
        "target",
        "event",
    ):
        assert forbidden not in joined
    assert any(name.endswith("bbox_velocity_x") for name in FEATURE_NAMES)
    assert any(name.endswith("association_age") for name in FEATURE_NAMES)


def test_fusion_residual_and_danger_calibration() -> None:
    physics = torch.tensor([0.0, 0.5, 1.0])
    unchanged = fused_inverse_ttc(physics, torch.zeros(3))
    torch.testing.assert_close(unchanged, physics)
    adjusted = fused_inverse_ttc(
        physics, torch.tensor([2.0, -2.0, 0.5]), residual_limit=0.25
    )
    assert adjusted[0] > physics[0]
    assert adjusted[1] < physics[1]
    assert adjusted[2] > physics[2]

    decoded = calibrated_ttc(
        np.asarray([1.0, 1.0, 0.01]),
        np.asarray([-10.0, 10.0, 10.0]),
        danger_threshold=0.5,
    )
    assert decoded[0] == 2.0  # non-danger is never < official 2 s threshold
    assert decoded[1] == pytest.approx(1.0)
    assert decoded[2] == pytest.approx(1.999)


def test_normalizer_and_inverse_ttc_are_finite_and_stable() -> None:
    values = np.zeros((4, FEATURE_DIM), dtype=np.float32)
    values[:, 2] = [0.0, 1.0, 2.0, 3.0]
    normalizer = FeatureNormalizer.fit([values])
    transformed = normalizer.transform(values)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(inverse_ttc(np.asarray([np.inf, 2.0, 0.05])), [0, 0.5, 10])


def test_nested_lopo_has_no_outer_or_inner_trip_leakage() -> None:
    trip_ids = tuple(f"T{index:02d}-Sample" for index in range(1, 7))
    plan = nested_lopo_splits(trip_ids)
    assert len(plan) == 6
    assert {outer.test_trip_id for outer, _ in plan} == set(trip_ids)
    for outer, inner_folds in plan:
        assert len(outer.train_trip_ids) == 5
        assert outer.test_trip_id not in outer.train_trip_ids
        assert len(inner_folds) == 5
        assert {fold.validation_trip_id for fold in inner_folds} == set(
            outer.train_trip_ids
        )
        for inner in inner_folds:
            assert len(inner.train_trip_ids) == 4
            assert inner.validation_trip_id not in inner.train_trip_ids
            assert outer.test_trip_id not in inner.train_trip_ids
            assert outer.test_trip_id != inner.validation_trip_id


def test_outer_final_fit_resolver_excludes_test_supervision() -> None:
    trips = {
        f"T{index:02d}-Sample": _supervised_trip(f"T{index:02d}-Sample")
        for index in range(1, 7)
    }
    outer = OuterFold(
        train_trip_ids=tuple(f"T{index:02d}-Sample" for index in range(1, 6)),
        test_trip_id="T06-Sample",
    )
    selected = outer_training_trips(trips, outer)
    assert {trip.trip_id for trip in selected} == set(outer.train_trip_ids)
    assert all(trip.labels.trip_id != outer.test_trip_id for trip in selected)


def test_inner_normalizer_is_fitted_only_on_training_trip(tmp_path: Path) -> None:
    training = _supervised_trip("T01-Sample", feature_value=3.0)
    validation = _supervised_trip("T02-Sample", feature_value=99.0)
    _, normalizer, _ = train_inner_model(
        [training],
        [validation],
        TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1),
        TrainSettings(
            max_epochs=1,
            patience=1,
            batch_size=4,
            amp=False,
            seed=17,
        ),
        device=torch.device("cpu"),
        log_dir=tmp_path / "tb",
        seed=17,
        variant="temporal",
    )
    expected = training.runtime.features.mean(axis=0)
    np.testing.assert_allclose(normalizer.mean, expected)
    assert not np.allclose(normalizer.mean, validation.runtime.features.mean(axis=0))


def test_candidate_selection_rejects_outer_test_provenance() -> None:
    config = TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1)
    leaked = CandidateResult(
        config=config,
        temporal_choice=_choice(),
        fusion_choice=_choice(),
        temporal_best_epochs=(3,),
        fusion_best_epochs=(4,),
        validation_trip_ids=("T06-Sample",),
    )
    with pytest.raises(ValueError, match="outer-test"):
        select_candidate(
            [leaked],
            outer_train_trip_ids=(
                "T01-Sample",
                "T02-Sample",
                "T03-Sample",
                "T04-Sample",
                "T05-Sample",
            ),
            outer_test_trip_id="T06-Sample",
            variant="fusion",
        )


def test_temporal_and_fusion_losses_do_not_cross_train_heads() -> None:
    config = TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1)
    temporal_model = CausalTemporalTTC(config)
    features = torch.randn(4, 8, FEATURE_DIM)
    current = temporal_model(features)[:, -1]
    loss = _loss(
        current,
        torch.tensor([0.0, 0.2, 0.5, 1.0]),
        torch.tensor([0.0, 0.4, 0.6, 1.2]),
        torch.tensor([float("inf"), 2.5, 1.5, 0.8]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        config=config,
        positive_weight=torch.tensor(1.0),
        variant="temporal",
    )
    loss.backward()
    assert temporal_model.temporal_inverse_head.weight.grad is not None
    assert temporal_model.temporal_danger_head.weight.grad is not None
    assert torch.count_nonzero(temporal_model.residual_head.weight.grad) == 0
    assert torch.count_nonzero(temporal_model.fusion_danger_head.weight.grad) == 0

    fusion_model = CausalTemporalTTC(config)
    fusion_current = fusion_model(features)[:, -1]
    fusion_loss = _loss(
        fusion_current,
        torch.tensor([0.0, 0.2, 0.5, 1.0]),
        torch.tensor([0.0, 0.4, 0.6, 1.2]),
        torch.tensor([float("inf"), 2.5, 1.5, 0.8]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        config=config,
        positive_weight=torch.tensor(1.0),
        variant="fusion",
    )
    fusion_loss.backward()
    assert fusion_model.residual_head.weight.grad is not None
    assert fusion_model.fusion_danger_head.weight.grad is not None
    assert torch.count_nonzero(fusion_model.temporal_inverse_head.weight.grad) == 0
    assert torch.count_nonzero(fusion_model.temporal_danger_head.weight.grad) == 0


def test_six_trip_oof_writer_produces_exactly_3600_unique_frames(
    tmp_path: Path,
) -> None:
    rows = 0
    for index in range(1, 7):
        trip_id = f"T{index:02d}-Sample"
        trip = _supervised_trip(trip_id, n_frames=600)
        path = tmp_path / f"{trip_id}.csv"
        predictions = np.full(600, np.inf, dtype=np.float64)
        predictions[10:20] = 1.5
        _write_prediction_csv(path, trip.runtime, predictions)
        report = preflight_prediction_csv(path, expected_frames=600)
        rows += report.rows
    assert rows == 3600
    assert len(_require_exact_oof_csv_set(tmp_path)) == 6
    (tmp_path / "stale.csv").write_text(
        "frame_id,timestamp,predicted_ttc\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected"):
        _require_exact_oof_csv_set(tmp_path)


def test_ablation_report_uses_official_trip_metric_implementation() -> None:
    labels = {}
    predictions = {}
    direct_scores = []
    for index in range(1, 7):
        trip_id = f"T{index:02d}-Sample"
        trip = _supervised_trip(trip_id, n_frames=600)
        labels[trip_id] = trip.labels
        prediction = np.full(600, np.inf, dtype=np.float64)
        prediction[::2] = 1.6
        predictions[trip_id] = prediction
        pairs = {
            frame_id: (float(prediction[frame_id]), float(trip.labels.ttc_s[frame_id]))
            for frame_id in range(600)
        }
        direct_scores.append(compute_trip_metrics(trip_id, pairs))
    report = _ablation_report(predictions, labels)
    assert report["frames"] == 3600
    assert report["macro"]["composite_score"] == round(
        float(np.mean([metric.composite_score for metric in direct_scores])), 1
    )
    assert report["per_trip"][0]["mae_critical"] == direct_scores[0].mae_critical


def test_onnx_export_is_fixed_shape_dynamo_opset20_and_runtime_equivalent(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(11)
    config = TemporalModelConfig(sequence_length=8, hidden_size=32, gru_layers=1)
    model = CausalTemporalTTC(config).eval()
    path = export_temporal_onnx(model, tmp_path / "temporal.onnx")
    assert path.is_file()
    document = onnx.load(path)
    onnx.checker.check_model(document)
    assert document.ir_version <= 10
    assert document.opset_import[0].version == 20
    dimensions = document.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in dimensions] == [1, 8, FEATURE_DIM]

    rng = np.random.default_rng(11)
    sample = rng.normal(size=(1, 8, FEATURE_DIM)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(sample)).numpy()
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(path), providers=providers)
    actual = session.run(None, {"features": sample})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
