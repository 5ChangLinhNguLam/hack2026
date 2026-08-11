"""Explicit, causal P2-C runtime variants layered over the accepted P2-B policy."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from .lane import LaneDetector, LaneEstimate
from .p2b_runtime import P2DeployableRuntime, P2FrameResult, P2Variant
from .p2b_tracker import P2CausalTracker, P2TrackSnapshot
from .p2c_range import (
    CausalScaleExpansionEstimator,
    CausalTTCFusion,
    ClassPriorRangeResult,
    ClassSizePriorConfig,
    GroundPlaneConfig,
    GroundPlaneHorizon,
    GroundPlaneRangeResult,
    P2CReason,
    ScaleExpansionConfig,
    ScaleExpansionResult,
    TTCFusionConfig,
    TTCFusionInput,
    TTCFusionResult,
    estimate_class_prior_range,
    estimate_ground_plane_range,
    project_scale_expansion_result,
    resolve_ground_plane_horizon,
)
from .range_ttc import (
    CausalRangeTTCEstimator,
    RangeTTCConfig,
    RangeTTCReason,
    RangeTTCResult,
    project_range_ttc_result,
)
from .target_selector import TargetAssessment
from .temporal_features import CameraGeometry
from .types import BBox, Detection, TrackRisk


class P2CVariant(str, Enum):
    PHYSICS = "physics"
    P2B_FULL = "p2b_full"
    GROUND_PLANE = "p2b_ground_plane"
    SCALE_EXPANSION = "p2b_scale_expansion"
    CLASS_PRIOR = "p2b_class_prior"
    FULL_FUSION = "p2c_full_fusion"


@dataclass(frozen=True, slots=True)
class P2CVariantConfig:
    variant: P2CVariant
    delegated_p2_variant: P2Variant
    use_ground_plane: bool
    use_scale_expansion: bool
    use_class_prior: bool
    use_safety_veto: bool

    @property
    def uses_p2c_estimators(self) -> bool:
        return self.use_ground_plane or self.use_scale_expansion or self.use_class_prior


LOCKED_P2C_VARIANTS: Mapping[P2CVariant, P2CVariantConfig] = {
    P2CVariant.PHYSICS: P2CVariantConfig(
        P2CVariant.PHYSICS, P2Variant.PHYSICS_CURRENT, False, False, False, False
    ),
    P2CVariant.P2B_FULL: P2CVariantConfig(
        P2CVariant.P2B_FULL, P2Variant.FULL, False, False, False, False
    ),
    P2CVariant.GROUND_PLANE: P2CVariantConfig(
        P2CVariant.GROUND_PLANE, P2Variant.FULL, True, False, False, False
    ),
    P2CVariant.SCALE_EXPANSION: P2CVariantConfig(
        P2CVariant.SCALE_EXPANSION, P2Variant.FULL, False, True, False, False
    ),
    P2CVariant.CLASS_PRIOR: P2CVariantConfig(
        P2CVariant.CLASS_PRIOR, P2Variant.FULL, False, False, True, False
    ),
    P2CVariant.FULL_FUSION: P2CVariantConfig(
        P2CVariant.FULL_FUSION, P2Variant.FULL, True, True, True, True
    ),
}


def _range_ttc_defaults() -> RangeTTCConfig:
    return RangeTTCConfig(
        history_size=12,
        min_samples=4,
        min_time_span_s=0.15,
        min_closing_speed_mps=0.30,
        min_closing_fraction=0.60,
        max_relative_closing_uncertainty=0.90,
        max_range_uncertainty_m=5.0,
        max_extrapolation_s=0.45,
        # Theil--Sen + MAD provide the robust initialization/gate; two bounded
        # IRLS refinements retain Huber weighting within the edge budget.
        huber_iterations=2,
    )


def _scale_expansion_defaults() -> ScaleExpansionConfig:
    return ScaleExpansionConfig(huber_iterations=2)


def _safety_buffers() -> Mapping[str, float]:
    return {
        "person": 2.5,
        "bicycle": 3.5,
        "motorcycle": 3.5,
        "car": 4.5,
        "truck": 5.5,
        "bus": 5.5,
    }


@dataclass(frozen=True)
class P2CRuntimeConfig:
    principal_y_px: float | None = None
    ground_plane: GroundPlaneConfig = field(default_factory=GroundPlaneConfig)
    class_prior: ClassSizePriorConfig = field(default_factory=ClassSizePriorConfig)
    scale_expansion: ScaleExpansionConfig = field(
        default_factory=_scale_expansion_defaults
    )
    range_ttc: RangeTTCConfig = field(default_factory=_range_ttc_defaults)
    fusion: TTCFusionConfig = field(default_factory=TTCFusionConfig)
    safety_buffer_by_class: Mapping[str, float] = field(default_factory=_safety_buffers)
    default_safety_buffer_m: float = 4.0
    minimum_component_uncertainty_s: float = 0.05
    warning_ttc_s: float = 2.0
    danger_ttc_s: float = 3.0
    minimum_warning_path_score: float = 0.35
    maximum_warning_assessment_uncertainty: float = 0.75
    maximum_new_warning_estimator_uncertainty_s: float = 1.50
    maximum_warning_coast_updates: int = 3
    minimum_fresh_physics_uncertainty_s: float = 0.35
    maximum_scale_safety_age_s: float = 0.30
    maximum_scale_fit_disagreement_sigma: float = 2.50

    def __post_init__(self) -> None:
        if self.principal_y_px is not None and not math.isfinite(self.principal_y_px):
            raise ValueError("principal_y_px must be finite")
        positive = (
            self.default_safety_buffer_m,
            self.minimum_component_uncertainty_s,
            self.warning_ttc_s,
            self.danger_ttc_s,
            self.minimum_warning_path_score,
            self.maximum_warning_assessment_uncertainty,
            self.maximum_new_warning_estimator_uncertainty_s,
            self.minimum_fresh_physics_uncertainty_s,
            self.maximum_scale_safety_age_s,
            self.maximum_scale_fit_disagreement_sigma,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("P2-C runtime thresholds must be finite and positive")
        if self.warning_ttc_s >= self.danger_ttc_s:
            raise ValueError("warning TTC must be below danger TTC")
        if self.maximum_warning_coast_updates < 0:
            raise ValueError("maximum_warning_coast_updates must be non-negative")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.safety_buffer_by_class.values()
        ):
            raise ValueError("safety buffers must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class P2CComponentDiagnostic:
    source: str
    ttc_s: float
    ttc_uncertainty_s: float
    range_m: float
    range_uncertainty_m: float
    reason: str
    measurement_reason: str
    observed_this_frame: bool
    sample_count: int
    inlier_count: int


@dataclass(frozen=True, slots=True)
class P2CFrameResult:
    # P2FrameResult-compatible fields.
    timestamp: float
    predicted_ttc_s: float
    primary_track_id: int | None
    dangerous_track_ids: tuple[int, ...]
    target_switched: bool
    held_by_hysteresis: bool
    warning: bool
    invalid_reason: str
    ttc_source: str
    lane_source: str
    lane_confidence: float
    candidate_count: int
    assessments: tuple[TargetAssessment, ...]
    risks: tuple[TrackRisk, ...]
    downstream_latency_ms: float
    # P2-C audit fields.
    primary_bbox: BBox | None
    estimator_source: str
    estimator_reason: str
    estimator_uncertainty_s: float
    estimator_sources: tuple[str, ...]
    component_diagnostics: tuple[P2CComponentDiagnostic, ...]
    base_predicted_ttc_s: float
    base_ttc_source: str


class _CapturingLaneDetector:
    """Capture the exact lane estimate consumed once by the P2-B runtime."""

    def __init__(self, detector: LaneDetector) -> None:
        self._detector = detector
        self.last_estimate: LaneEstimate | None = None

    def reset(self) -> None:
        self._detector.reset()
        self.last_estimate = None

    def detect(self, image_bgr: np.ndarray) -> LaneEstimate:
        estimate = self._detector.detect(image_bgr)
        self.last_estimate = estimate
        return estimate


@dataclass
class _TrackEstimatorState:
    ground_range: CausalRangeTTCEstimator | None
    class_range: CausalRangeTTCEstimator | None
    scale: CausalScaleExpansionEstimator | None
    fusion: CausalTTCFusion
    ground_measurement: GroundPlaneRangeResult | None = None
    class_measurement: ClassPriorRangeResult | None = None
    ground_trend: RangeTTCResult | None = None
    class_trend: RangeTTCResult | None = None
    scale_result: ScaleExpansionResult | None = None
    ground_fit_anchor: RangeTTCResult | None = None
    class_fit_anchor: RangeTTCResult | None = None
    scale_fit_anchor: ScaleExpansionResult | None = None
    ground_fit_tip_s: float | None = None
    class_fit_tip_s: float | None = None
    scale_fit_tip_s: float | None = None
    ground_anchor_uncertainty_m: float = float("inf")
    class_anchor_uncertainty_m: float = float("inf")


def _assessment_for(
    result: P2FrameResult, track_id: int | None
) -> TargetAssessment | None:
    if track_id is None:
        return None
    return next(
        (item for item in result.assessments if item.track_id == track_id), None
    )


class P2CDeployableRuntime:
    """Runtime-only P2-C layer enabled solely through an explicit variant."""

    def __init__(
        self,
        geometry: CameraGeometry,
        variant: P2CVariant,
        *,
        config: P2CRuntimeConfig | None = None,
    ) -> None:
        if variant not in LOCKED_P2C_VARIANTS:
            raise ValueError(f"Unknown P2-C variant: {variant}")
        self.geometry = geometry
        self.variant = variant
        self.variant_config = LOCKED_P2C_VARIANTS[variant]
        self.config = config or P2CRuntimeConfig()
        self.principal_y_px = (
            float(self.config.principal_y_px)
            if self.config.principal_y_px is not None
            else 0.5 * geometry.height
        )
        self.base_runtime = P2DeployableRuntime(
            geometry, self.variant_config.delegated_p2_variant
        )
        self._lane_capture: _CapturingLaneDetector | None = None
        if self.variant_config.uses_p2c_estimators:
            if self.base_runtime.lane_detector is None:
                raise AssertionError("P2-C estimators require the P2-B lane runtime")
            self._lane_capture = _CapturingLaneDetector(
                self.base_runtime.lane_detector
            )
            # P2-B continues to call the same detector exactly once; this proxy
            # only retains the returned causal estimate for ground-plane range.
            self.base_runtime.lane_detector = self._lane_capture  # type: ignore[assignment]
        self.reset()

    @property
    def active_estimator_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._track_states))

    def reset(self) -> None:
        self.base_runtime.reset()
        self._track_states: dict[int, _TrackEstimatorState] = {}
        self._warning_latched_tracks: set[int] = set()
        self._prewarning_latched_tracks: set[int] = set()

    def _new_track_state(self) -> _TrackEstimatorState:
        cfg = self.variant_config
        return _TrackEstimatorState(
            ground_range=(
                CausalRangeTTCEstimator(self.config.range_ttc)
                if cfg.use_ground_plane
                else None
            ),
            class_range=(
                CausalRangeTTCEstimator(self.config.range_ttc)
                if cfg.use_class_prior
                else None
            ),
            scale=(
                CausalScaleExpansionEstimator(self.config.scale_expansion)
                if cfg.use_scale_expansion
                else None
            ),
            fusion=CausalTTCFusion(self.config.fusion),
        )

    def _safety_buffer(self, posterior: Mapping[str, float]) -> float:
        usable = [
            (label, max(0.0, float(probability)))
            for label, probability in posterior.items()
            if math.isfinite(float(probability)) and float(probability) > 0.0
        ]
        total = sum(probability for _, probability in usable)
        if total <= 0.0:
            return self.config.default_safety_buffer_m
        return sum(
            probability
            * self.config.safety_buffer_by_class.get(
                label, self.config.default_safety_buffer_m
            )
            for label, probability in usable
        ) / total

    def _update_ground(
        self,
        state: _TrackEstimatorState,
        snapshot: P2TrackSnapshot,
        horizon: GroundPlaneHorizon | None,
        *,
        timestamp: float,
    ) -> None:
        estimator = state.ground_range
        if estimator is None:
            return
        if snapshot.observed_this_call:
            measurement = estimate_ground_plane_range(
                snapshot.bbox,
                timestamp_s=timestamp,
                image_shape=(self.geometry.height, self.geometry.width),
                focal_y_px=self.geometry.focal_y_px,
                principal_y_px=self.principal_y_px,
                resolved_horizon=horizon,
                config=self.config.ground_plane,
            )
            state.ground_measurement = measurement
            if measurement.reliable:
                state.ground_anchor_uncertainty_m = measurement.range_uncertainty_m
                estimator.observe(
                    timestamp_s=timestamp,
                    range_m=measurement.range_m,
                )

    def _update_class_prior(
        self,
        state: _TrackEstimatorState,
        snapshot: P2TrackSnapshot,
        *,
        timestamp: float,
    ) -> None:
        estimator = state.class_range
        if estimator is None:
            return
        if snapshot.observed_this_call:
            measurement = estimate_class_prior_range(
                snapshot.bbox,
                snapshot.class_posterior,
                timestamp_s=timestamp,
                focal_y_px=self.geometry.focal_y_px,
                config=self.config.class_prior,
            )
            state.class_measurement = measurement
            if measurement.reliable:
                state.class_anchor_uncertainty_m = measurement.range_uncertainty_m
                estimator.observe(
                    timestamp_s=timestamp,
                    range_m=measurement.range_m,
                )

    def _update_scale(
        self,
        state: _TrackEstimatorState,
        snapshot: P2TrackSnapshot,
        *,
        timestamp: float,
    ) -> None:
        if state.scale is None:
            return
        if snapshot.observed_this_call:
            state.scale.observe(
                timestamp_s=timestamp,
                bbox=snapshot.bbox,
                confidence=snapshot.confidence,
            )

    @staticmethod
    def _history_tip(estimator: object) -> float | None:
        history = getattr(estimator, "history", ())
        return float(history[-1].timestamp_s) if history else None

    def _evaluate_primary_state(
        self,
        state: _TrackEstimatorState,
        *,
        timestamp: float,
        safety_buffer_m: float,
    ) -> None:
        """Fit only dirty primary history; project cached fits on coast frames."""

        if state.ground_range is not None:
            tip = self._history_tip(state.ground_range)
            if state.ground_fit_anchor is None or tip != state.ground_fit_tip_s:
                state.ground_fit_anchor = state.ground_range.predict(
                    timestamp_s=timestamp,
                    safety_buffer_m=safety_buffer_m,
                )
                state.ground_fit_tip_s = tip
            state.ground_trend = project_range_ttc_result(
                state.ground_fit_anchor,
                evaluation_timestamp_s=timestamp,
                safety_buffer_m=safety_buffer_m,
                config=self.config.range_ttc,
            )

        if state.class_range is not None:
            tip = self._history_tip(state.class_range)
            if state.class_fit_anchor is None or tip != state.class_fit_tip_s:
                state.class_fit_anchor = state.class_range.predict(
                    timestamp_s=timestamp,
                    safety_buffer_m=safety_buffer_m,
                )
                state.class_fit_tip_s = tip
            state.class_trend = project_range_ttc_result(
                state.class_fit_anchor,
                evaluation_timestamp_s=timestamp,
                safety_buffer_m=safety_buffer_m,
                config=self.config.range_ttc,
            )

        if state.scale is not None:
            tip = self._history_tip(state.scale)
            if state.scale_fit_anchor is None or tip != state.scale_fit_tip_s:
                state.scale_fit_anchor = state.scale.predict(timestamp_s=timestamp)
                state.scale_fit_tip_s = tip
            state.scale_result = project_scale_expansion_result(
                state.scale_fit_anchor,
                evaluation_timestamp_s=timestamp,
                config=self.config.scale_expansion,
            )

    def _update_track_states(
        self,
        snapshots: Sequence[P2TrackSnapshot],
        *,
        lane: LaneEstimate | None,
        timestamp: float,
        primary_track_id: int | None,
    ) -> None:
        live_ids = {snapshot.track_id for snapshot in snapshots}
        self._track_states = {
            track_id: state
            for track_id, state in self._track_states.items()
            if track_id in live_ids
        }
        self._warning_latched_tracks.intersection_update(live_ids)
        self._prewarning_latched_tracks.intersection_update(live_ids)
        horizon = (
            resolve_ground_plane_horizon(
                lane,
                image_height=self.geometry.height,
                focal_y_px=self.geometry.focal_y_px,
                principal_y_px=self.principal_y_px,
                config=self.config.ground_plane,
            )
            if self.variant_config.use_ground_plane
            else None
        )
        for snapshot in snapshots:
            state = self._track_states.setdefault(
                snapshot.track_id, self._new_track_state()
            )
            self._update_ground(
                state,
                snapshot,
                horizon,
                timestamp=timestamp,
            )
            self._update_class_prior(
                state,
                snapshot,
                timestamp=timestamp,
            )
            self._update_scale(state, snapshot, timestamp=timestamp)
        primary = next(
            (item for item in snapshots if item.track_id == primary_track_id), None
        )
        if primary is not None:
            state = self._track_states[primary.track_id]
            self._evaluate_primary_state(
                state,
                timestamp=timestamp,
                safety_buffer_m=self._safety_buffer(primary.class_posterior),
            )

    def _range_component(
        self,
        *,
        source: str,
        trend: RangeTTCResult | None,
        measurement_reason: str,
        anchor_uncertainty_m: float,
        observed: bool,
    ) -> tuple[P2CComponentDiagnostic, TTCFusionInput | None]:
        if trend is None:
            diagnostic = P2CComponentDiagnostic(
                source,
                float("inf"),
                float("inf"),
                float("nan"),
                float("inf"),
                RangeTTCReason.INSUFFICIENT_HISTORY,
                measurement_reason,
                observed,
                0,
                0,
            )
            return diagnostic, None
        reliable = trend.reason_code in {
            RangeTTCReason.OK,
            RangeTTCReason.WITHIN_SAFETY_BUFFER,
        } and math.isfinite(trend.ttc_s)
        systematic_ttc_uncertainty = (
            anchor_uncertainty_m / max(trend.closing_speed_mps, 0.30)
            if math.isfinite(anchor_uncertainty_m)
            else float("inf")
        )
        uncertainty = math.hypot(
            trend.ttc_uncertainty_s, systematic_ttc_uncertainty
        )
        uncertainty = max(self.config.minimum_component_uncertainty_s, uncertainty)
        diagnostic = P2CComponentDiagnostic(
            source=source,
            ttc_s=float(trend.ttc_s) if reliable else float("inf"),
            ttc_uncertainty_s=uncertainty if reliable else float("inf"),
            range_m=trend.range_m,
            range_uncertainty_m=trend.range_uncertainty_m,
            reason=trend.reason_code,
            measurement_reason=measurement_reason,
            observed_this_frame=observed,
            sample_count=trend.sample_count,
            inlier_count=trend.inlier_count,
        )
        signal = (
            TTCFusionInput(source, trend.ttc_s, uncertainty, trend.reason_code)
            if reliable and math.isfinite(uncertainty)
            else None
        )
        return diagnostic, signal

    def _scale_component(
        self,
        result: ScaleExpansionResult | None,
        *,
        observed: bool,
    ) -> tuple[P2CComponentDiagnostic, TTCFusionInput | None]:
        if result is None:
            diagnostic = P2CComponentDiagnostic(
                "scale_expansion",
                float("inf"),
                float("inf"),
                float("nan"),
                float("inf"),
                P2CReason.INSUFFICIENT_HISTORY,
                P2CReason.INSUFFICIENT_HISTORY,
                observed,
                0,
                0,
            )
            return diagnostic, None
        sample_count = max(result.height_fit.sample_count, result.area_fit.sample_count)
        inlier_count = min(result.height_fit.inlier_count, result.area_fit.inlier_count)
        uncertainty = max(
            self.config.minimum_component_uncertainty_s,
            result.ttc_uncertainty_s,
        )
        diagnostic = P2CComponentDiagnostic(
            "scale_expansion",
            result.ttc_s,
            uncertainty if result.reliable else float("inf"),
            float("nan"),
            float("inf"),
            result.reason_code,
            result.source,
            observed,
            sample_count,
            inlier_count,
        )
        signal = (
            TTCFusionInput(
                "scale_expansion", result.ttc_s, uncertainty, result.reason_code
            )
            if result.reliable and math.isfinite(uncertainty)
            else None
        )
        return diagnostic, signal

    def _components(
        self,
        state: _TrackEstimatorState,
        snapshot: P2TrackSnapshot,
    ) -> tuple[tuple[P2CComponentDiagnostic, ...], tuple[TTCFusionInput, ...]]:
        diagnostics: list[P2CComponentDiagnostic] = []
        signals: list[TTCFusionInput] = []
        if self.variant_config.use_ground_plane:
            diagnostic, signal = self._range_component(
                source="ground_plane",
                trend=state.ground_trend,
                measurement_reason=(
                    state.ground_measurement.reason_code
                    if state.ground_measurement is not None
                    else P2CReason.INSUFFICIENT_HISTORY
                ),
                anchor_uncertainty_m=state.ground_anchor_uncertainty_m,
                observed=snapshot.observed_this_call,
            )
            diagnostics.append(diagnostic)
            if signal is not None:
                signals.append(signal)
        if self.variant_config.use_scale_expansion:
            diagnostic, signal = self._scale_component(
                state.scale_result, observed=snapshot.observed_this_call
            )
            diagnostics.append(diagnostic)
            if signal is not None:
                signals.append(signal)
        if self.variant_config.use_class_prior:
            diagnostic, signal = self._range_component(
                source="class_prior",
                trend=state.class_trend,
                measurement_reason=(
                    state.class_measurement.reason_code
                    if state.class_measurement is not None
                    else P2CReason.INSUFFICIENT_HISTORY
                ),
                anchor_uncertainty_m=state.class_anchor_uncertainty_m,
                observed=snapshot.observed_this_call,
            )
            diagnostics.append(diagnostic)
            if signal is not None:
                signals.append(signal)
        return tuple(diagnostics), tuple(signals)

    def _assessment_quality(self, assessment: TargetAssessment | None) -> bool:
        return bool(
            assessment is not None
            and assessment.confirmed
            and assessment.path_score >= self.config.minimum_warning_path_score
            and assessment.uncertainty
            <= self.config.maximum_warning_assessment_uncertainty
        )

    def _fresh_physics_signal(
        self,
        snapshot: P2TrackSnapshot,
        assessment: TargetAssessment | None,
    ) -> TTCFusionInput | None:
        """Expose only a confirmed physics value from the current detection.

        Raw tracker physics and the P2-C image estimators share geometry, so
        this is deliberately assigned a conservative uncertainty rather than
        being treated as a perfectly known anchor.  Coast/predict callbacks
        cannot manufacture a new physics signal.
        """

        raw_ttc = float(snapshot.raw_physics_ttc_s)
        if (
            assessment is None
            or not snapshot.observed_this_call
            or not self._assessment_quality(assessment)
            or not math.isfinite(raw_ttc)
            or raw_ttc < 0.0
        ):
            return None
        uncertainty = max(
            self.config.minimum_fresh_physics_uncertainty_s,
            float(assessment.uncertainty) * max(raw_ttc, 1.0),
        )
        return TTCFusionInput(
            "fresh_raw_physics",
            raw_ttc,
            uncertainty,
            "valid_legacy_physics",
        )

    @staticmethod
    def _inflate_correlated_signals(
        signals: Sequence[TTCFusionInput],
    ) -> tuple[TTCFusionInput, ...]:
        """Prevent shared-bbox inputs from creating false confidence gains.

        With ``n`` fully correlated signals, multiplying every marginal
        uncertainty by ``sqrt(n)`` prevents inverse-variance fusion from
        reporting a smaller within-signal variance merely because the same
        detector/association error appears in several estimators.  Relative
        weighting and the explicit disagreement variance remain unchanged.
        """

        if len(signals) <= 1:
            return tuple(signals)
        inflation = math.sqrt(len(signals))
        return tuple(
            TTCFusionInput(
                signal.source,
                signal.ttc_s,
                signal.uncertainty_s * inflation,
                signal.reason_code,
            )
            for signal in signals
        )

    def _strict_scale_safety_evidence(
        self,
        state: _TrackEstimatorState,
        assessment: TargetAssessment | None,
    ) -> bool:
        """Require a one-sided safe bound from both height and area fits."""

        result = state.scale_result
        if (
            result is None
            or not self._assessment_quality(assessment)
            or result.last_observation_age_s
            > self.config.maximum_scale_safety_age_s
        ):
            return False
        fits = (result.height_fit, result.area_fit)
        # The one-sided bound answers the alert decision (<2 s).  The
        # published point estimate still governs the wider <3 s danger set.
        safe_inverse_limit = 1.0 / (
            self.config.warning_ttc_s + result.last_observation_age_s
        )
        if not all(
            fit.reason_code in {P2CReason.OK, P2CReason.NON_EXPANDING}
            and fit.sample_count >= self.config.scale_expansion.min_samples
            and fit.inlier_count >= self.config.scale_expansion.min_samples
            and math.isfinite(fit.inverse_ttc_s)
            and math.isfinite(fit.inverse_ttc_uncertainty_s)
            and fit.inverse_ttc_s
            + self.config.maximum_scale_fit_disagreement_sigma
            * fit.inverse_ttc_uncertainty_s
            <= safe_inverse_limit
            for fit in fits
        ):
            return False
        disagreement = abs(
            fits[0].inverse_ttc_s - fits[1].inverse_ttc_s
        )
        disagreement_sigma = math.hypot(
            fits[0].inverse_ttc_uncertainty_s,
            fits[1].inverse_ttc_uncertainty_s,
        )
        return disagreement <= max(
            self.config.scale_expansion.inverse_uncertainty_floor_s,
            self.config.maximum_scale_fit_disagreement_sigma
            * disagreement_sigma,
        )

    def _strong_safe_evidence(
        self,
        state: _TrackEstimatorState,
        snapshot: P2TrackSnapshot,
        assessment: TargetAssessment | None,
        signals: Sequence[TTCFusionInput],
    ) -> bool:
        physics = self._fresh_physics_signal(snapshot, assessment)
        if physics is not None and physics.ttc_s < self.config.danger_ttc_s:
            return False
        fresh_physics_safe = bool(
            physics is not None and physics.ttc_s >= self.config.danger_ttc_s
        )
        return fresh_physics_safe or self._strict_scale_safety_evidence(
            state, assessment
        )

    def _fuse(
        self,
        state: _TrackEstimatorState,
        snapshot: P2TrackSnapshot,
        base: P2FrameResult,
        signals: Sequence[TTCFusionInput],
        *,
        timestamp: float,
        assessment: TargetAssessment | None = None,
    ) -> TTCFusionResult:
        physics = self._fresh_physics_signal(snapshot, assessment)
        runtime_signals = tuple(signals) + (
            (physics,) if physics is not None else ()
        )
        runtime_signals = self._inflate_correlated_signals(runtime_signals)
        fallback = (
            TTCFusionInput(
                "p2b_full",
                base.predicted_ttc_s,
                float("inf"),
                base.ttc_source,
            )
            if math.isfinite(base.predicted_ttc_s)
            else None
        )
        base_danger = (
            math.isfinite(base.predicted_ttc_s)
            and base.predicted_ttc_s < self.config.danger_ttc_s
        )
        fallback_like = base.ttc_source in {
            "robust_range",
            "last_good",
            "physics_hold",
            "uncertain_physics",
        }
        fresh_physics_danger = bool(
            physics is not None and physics.ttc_s < self.config.danger_ttc_s
        )
        strict_scale_safe = self._strict_scale_safety_evidence(
            state, assessment
        )
        can_clear_fallback_danger = bool(
            base_danger
            and self.variant_config.use_safety_veto
            and fallback_like
            and not fresh_physics_danger
            and strict_scale_safe
        )
        if can_clear_fallback_danger:
            # A prior danger anchor would otherwise upward-limit a genuinely
            # safe result.  Reset only after the strict dual-fit gate above.
            state.fusion.reset()
            scale = state.scale_result
            if scale is not None and scale.reliable:
                return TTCFusionResult(
                    timestamp_s=timestamp,
                    ttc_s=scale.ttc_s,
                    uncertainty_s=scale.ttc_uncertainty_s,
                    reason_code=P2CReason.SCALE_SAFETY_VETO,
                    source="scale_safe_upper_bound",
                    contributing_sources=("scale_expansion",),
                    fallback_used=False,
                )
            return TTCFusionResult(
                timestamp_s=timestamp,
                ttc_s=float("inf"),
                uncertainty_s=float("inf"),
                reason_code=P2CReason.SCALE_SAFETY_VETO,
                source="scale_nonexpanding_upper_bound",
                contributing_sources=("scale_expansion",),
                fallback_used=False,
            )
        result = state.fusion.update(
            runtime_signals, timestamp_s=timestamp, fallback=fallback
        )
        erases_base_warning = bool(
            base.warning
            and (
                not math.isfinite(result.ttc_s)
                or result.ttc_s >= self.config.warning_ttc_s
            )
        )
        erases_base_danger = bool(
            base_danger
            and (
                not math.isfinite(result.ttc_s)
                or result.ttc_s >= self.config.danger_ttc_s
            )
        )
        if (erases_base_warning or erases_base_danger) and not can_clear_fallback_danger:
            # P2-C may add a more urgent causal estimate, but a single
            # monocular component cannot erase an accepted P2-B danger.
            state.fusion.reset()
            state.fusion.update((), timestamp_s=timestamp, fallback=fallback)
            return TTCFusionResult(
                timestamp_s=timestamp,
                ttc_s=base.predicted_ttc_s,
                uncertainty_s=float("inf"),
                reason_code=P2CReason.FALLBACK,
                source=f"safety_envelope:{base.ttc_source}",
                contributing_sources=tuple(
                    dict.fromkeys(
                        (*result.contributing_sources, "p2b_full")
                    )
                ),
                fallback_used=True,
            )
        return result

    def _warning(
        self,
        base: P2FrameResult,
        assessment: TargetAssessment | None,
        fusion: TTCFusionResult,
        *,
        strong_safe_evidence: bool = False,
    ) -> tuple[float, bool]:
        track_id = base.primary_track_id
        predicted = fusion.ttc_s
        if track_id is None or assessment is None:
            return predicted, False
        quality = self._assessment_quality(assessment)
        estimator_can_arm = (
            not fusion.fallback_used
            and math.isfinite(fusion.uncertainty_s)
            and fusion.uncertainty_s
            <= self.config.maximum_new_warning_estimator_uncertainty_s
        )
        if assessment.observed_this_update:
            if (
                strong_safe_evidence
                and (
                    not math.isfinite(predicted)
                    or predicted >= self.config.danger_ttc_s
                )
            ):
                self._warning_latched_tracks.discard(track_id)
                self._prewarning_latched_tracks.discard(track_id)
            elif (
                quality
                and predicted < self.config.warning_ttc_s
                and estimator_can_arm
            ):
                self._warning_latched_tracks.add(track_id)
                self._prewarning_latched_tracks.discard(track_id)
            elif quality and estimator_can_arm:
                self._prewarning_latched_tracks.add(track_id)
        elif assessment.missed_updates == 0 and quality and estimator_can_arm:
            if (
                predicted < self.config.warning_ttc_s
                and track_id in self._prewarning_latched_tracks
            ):
                self._warning_latched_tracks.add(track_id)
                self._prewarning_latched_tracks.discard(track_id)
        elif assessment.missed_updates > self.config.maximum_warning_coast_updates:
            self._warning_latched_tracks.discard(track_id)
            self._prewarning_latched_tracks.discard(track_id)

        warning = (
            math.isfinite(predicted)
            and predicted < self.config.warning_ttc_s
            and quality
            and (
                (base.warning and assessment.confirmed)
                or track_id in self._warning_latched_tracks
            )
        )
        if math.isfinite(predicted) and predicted < self.config.warning_ttc_s and not warning:
            predicted = self.config.warning_ttc_s
        return predicted, warning

    def _primary_bbox(self, track_id: int | None) -> BBox | None:
        if track_id is None or not isinstance(self.base_runtime.tracker, P2CausalTracker):
            return None
        return next(
            (
                snapshot.bbox
                for snapshot in self.base_runtime.tracker.candidate_states()
                if snapshot.track_id == track_id
            ),
            None,
        )

    def _sanitized_base_dangerous_ids(
        self, base: P2FrameResult
    ) -> tuple[int, ...]:
        """A primary may not remain dangerous after its TTC becomes safe/invalid."""

        primary = base.primary_track_id
        if primary is None or (
            math.isfinite(base.predicted_ttc_s)
            and base.predicted_ttc_s < self.config.danger_ttc_s
        ):
            return base.dangerous_track_ids
        return tuple(
            track_id
            for track_id in base.dangerous_track_ids
            if track_id != primary
        )

    def _published_dangerous_ids(
        self,
        base: P2FrameResult,
        assessment: TargetAssessment | None,
        predicted_ttc_s: float,
    ) -> tuple[int, ...]:
        """Reconcile the primary danger bit after P2-C publication."""

        dangerous_ids = [
            track_id
            for track_id in base.dangerous_track_ids
            if track_id != base.primary_track_id
        ]
        if (
            base.primary_track_id is not None
            and assessment is not None
            and assessment.confirmed
            and assessment.path_score >= self.config.minimum_warning_path_score
            and math.isfinite(predicted_ttc_s)
            and predicted_ttc_s < self.config.danger_ttc_s
            and base.primary_track_id not in dangerous_ids
        ):
            dangerous_ids.append(base.primary_track_id)
        return tuple(dangerous_ids)

    def _from_base(self, base: P2FrameResult) -> P2CFrameResult:
        sources = (base.ttc_source,) if math.isfinite(base.predicted_ttc_s) else ()
        return P2CFrameResult(
            timestamp=base.timestamp,
            predicted_ttc_s=base.predicted_ttc_s,
            primary_track_id=base.primary_track_id,
            dangerous_track_ids=self._sanitized_base_dangerous_ids(base),
            target_switched=base.target_switched,
            held_by_hysteresis=base.held_by_hysteresis,
            warning=base.warning,
            invalid_reason=base.invalid_reason,
            ttc_source=base.ttc_source,
            lane_source=base.lane_source,
            lane_confidence=base.lane_confidence,
            candidate_count=base.candidate_count,
            assessments=base.assessments,
            risks=base.risks,
            downstream_latency_ms=base.downstream_latency_ms,
            primary_bbox=self._primary_bbox(base.primary_track_id),
            estimator_source=base.ttc_source,
            estimator_reason=base.invalid_reason,
            estimator_uncertainty_s=float("inf"),
            estimator_sources=sources,
            component_diagnostics=(),
            base_predicted_ttc_s=base.predicted_ttc_s,
            base_ttc_source=base.ttc_source,
        )

    def step(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        detector_update: bool,
        timestamp: float,
        ego_speed_kmh: float,
    ) -> P2CFrameResult:
        started = time.perf_counter()
        base = self.base_runtime.step(
            image_bgr,
            detections,
            detector_update=detector_update,
            timestamp=timestamp,
            ego_speed_kmh=ego_speed_kmh,
        )
        if not self.variant_config.uses_p2c_estimators:
            return self._from_base(base)
        if not isinstance(self.base_runtime.tracker, P2CausalTracker):
            raise AssertionError("P2-C estimators require P2CausalTracker")
        snapshots = self.base_runtime.tracker.candidate_states()
        lane = self._lane_capture.last_estimate if self._lane_capture is not None else None
        self._update_track_states(
            snapshots,
            lane=lane,
            timestamp=timestamp,
            primary_track_id=base.primary_track_id,
        )
        snapshot_by_id = {snapshot.track_id: snapshot for snapshot in snapshots}
        snapshot = snapshot_by_id.get(base.primary_track_id)
        assessment = _assessment_for(base, base.primary_track_id)
        if snapshot is None:
            converted = self._from_base(base)
            return P2CFrameResult(
                **{
                    name: getattr(converted, name)
                    for name in converted.__dataclass_fields__
                    if name != "downstream_latency_ms"
                },
                downstream_latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        state = self._track_states[snapshot.track_id]
        diagnostics, signals = self._components(state, snapshot)
        fusion = self._fuse(
            state,
            snapshot,
            base,
            signals,
            timestamp=timestamp,
            assessment=assessment,
        )
        if assessment is None or not assessment.confirmed:
            # An unconfirmed cold-start output is diagnostic-only in P2-C and
            # must not become the jump/hold anchor that biases the first
            # genuinely confirmed observation.
            state.fusion.reset()
        strong_safe_evidence = self._strong_safe_evidence(
            state, snapshot, assessment, signals
        )
        predicted, warning = self._warning(
            base,
            assessment,
            fusion,
            strong_safe_evidence=strong_safe_evidence,
        )
        dangerous_ids = self._published_dangerous_ids(
            base, assessment, predicted
        )
        sources = fusion.contributing_sources
        if fusion.fallback_used and not sources:
            sources = ("p2b_full",)
        return P2CFrameResult(
            timestamp=base.timestamp,
            predicted_ttc_s=predicted,
            primary_track_id=base.primary_track_id,
            dangerous_track_ids=dangerous_ids,
            target_switched=base.target_switched,
            held_by_hysteresis=base.held_by_hysteresis,
            warning=warning,
            invalid_reason=("valid" if math.isfinite(predicted) else fusion.reason_code),
            ttc_source=fusion.source,
            lane_source=base.lane_source,
            lane_confidence=base.lane_confidence,
            candidate_count=base.candidate_count,
            assessments=base.assessments,
            risks=base.risks,
            downstream_latency_ms=(time.perf_counter() - started) * 1000.0,
            primary_bbox=snapshot.bbox,
            estimator_source=fusion.source,
            estimator_reason=fusion.reason_code,
            estimator_uncertainty_s=fusion.uncertainty_s,
            estimator_sources=tuple(sources),
            component_diagnostics=diagnostics,
            base_predicted_ttc_s=base.predicted_ttc_s,
            base_ttc_source=base.ttc_source,
        )
