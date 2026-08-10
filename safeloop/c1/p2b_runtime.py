"""Deployable, causal runtime policy and locked P2-B ablation variants."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from .corridor import CorridorEstimator, EgoCorridor
from .lane import LaneDetector, LaneEstimate
from .p2b_tracker import (
    P2AssociationConfig,
    P2CausalTracker,
    P2RangePolicyConfig,
    P2TrackSnapshot,
)
from .target_selector import (
    DeployableTargetSelector,
    RuntimeTargetCandidate,
    TargetAssessment,
    TargetSelection,
    TargetSelectorConfig,
)
from .temporal_features import CameraGeometry, baseline_tracker
from .tracker import MonocularTTCTracker
from .types import Detection, TrackRisk


class P2Variant(str, Enum):
    PHYSICS_CURRENT = "physics_current"
    CORRIDOR_SELECTOR = "corridor_selector"
    SELECTOR_CLASS_HISTORY = "selector_class_history"
    SELECTOR_ROBUST_RANGE = "selector_robust_range"
    FULL = "selector_association_robust_range_fallback_hysteresis"


@dataclass(frozen=True)
class P2VariantConfig:
    variant: P2Variant
    use_corridor_selector: bool
    use_class_history_association: bool
    use_robust_range: bool
    enable_last_good_fallback: bool
    enable_hysteresis: bool
    enable_cold_start_ttc: bool


LOCKED_P2_VARIANTS: Mapping[P2Variant, P2VariantConfig] = {
    P2Variant.PHYSICS_CURRENT: P2VariantConfig(
        P2Variant.PHYSICS_CURRENT, False, False, False, False, False, False
    ),
    P2Variant.CORRIDOR_SELECTOR: P2VariantConfig(
        P2Variant.CORRIDOR_SELECTOR, True, False, False, False, False, False
    ),
    P2Variant.SELECTOR_CLASS_HISTORY: P2VariantConfig(
        P2Variant.SELECTOR_CLASS_HISTORY, True, True, False, False, False, False
    ),
    P2Variant.SELECTOR_ROBUST_RANGE: P2VariantConfig(
        P2Variant.SELECTOR_ROBUST_RANGE, True, False, True, False, False, False
    ),
    P2Variant.FULL: P2VariantConfig(
        P2Variant.FULL, True, True, True, True, True, True
    ),
}


@dataclass(frozen=True)
class P2FrameResult:
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


@dataclass(frozen=True)
class _PublishedTTCState:
    timestamp: float
    ttc_s: float
    warning: bool
    source: str


# Publishing a fresh physics estimate and briefly holding an already stable
# one are deliberately separate gates.  The wider hold gate cannot create a
# warning: it only counts down a same-target physics value that was previously
# accepted with lower uncertainty.
_MAX_FRESH_PHYSICS_UNCERTAINTY = 0.70
_MAX_HELD_PHYSICS_UNCERTAINTY = 0.85


def _minimum_risk_ttc(risks: Sequence[TrackRisk]) -> float:
    return min(
        (
            risk.predicted_ttc_s
            for risk in risks
            if math.isfinite(risk.predicted_ttc_s)
        ),
        default=float("inf"),
    )


def _posterior_uncertainty(snapshot: P2TrackSnapshot) -> float:
    probabilities = np.asarray(tuple(snapshot.class_posterior.values()), dtype=float)
    probabilities = probabilities[probabilities > 0.0]
    if probabilities.size <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / math.log(len(probabilities)), 0.0, 1.0))


class P2DeployableRuntime:
    """One-camera detector-cache/live-detector compatible P2-B runtime."""

    def __init__(
        self,
        geometry: CameraGeometry,
        variant: P2Variant,
        *,
        selector_config: TargetSelectorConfig | None = None,
    ) -> None:
        if variant not in LOCKED_P2_VARIANTS:
            raise ValueError(f"Unknown P2 variant: {variant}")
        self.geometry = geometry
        self.variant = variant
        self.variant_config = LOCKED_P2_VARIANTS[variant]
        frozen = baseline_tracker(geometry)
        if variant == P2Variant.PHYSICS_CURRENT:
            self.tracker: MonocularTTCTracker | P2CausalTracker = frozen
            self.lane_detector = None
            self.corridor_estimator = None
            self.selector = None
        else:
            self.tracker = P2CausalTracker(
                frozen.config,
                association=P2AssociationConfig(
                    enabled=self.variant_config.use_class_history_association
                ),
                range_policy=P2RangePolicyConfig(
                    enabled=self.variant_config.use_robust_range,
                    enable_last_good_fallback=self.variant_config.enable_last_good_fallback,
                ),
                focal_y_px=geometry.focal_y_px,
                focal_x_px=geometry.focal_x_px,
                principal_x_px=geometry.principal_x_px,
            )
            self.lane_detector = LaneDetector()
            self.corridor_estimator = CorridorEstimator(
                principal_x_px=geometry.principal_x_px
            )
            base_selector = selector_config or TargetSelectorConfig()
            if selector_config is not None and (
                selector_config.use_robust_ttc
                != self.variant_config.use_robust_range
                or selector_config.enable_hysteresis
                != self.variant_config.enable_hysteresis
                or selector_config.enable_last_good_fallback
                != self.variant_config.enable_last_good_fallback
                or selector_config.enable_cold_start_ttc
                != self.variant_config.enable_cold_start_ttc
            ):
                raise ValueError("selector feature toggles must match the locked variant")
            if selector_config is None:
                base_selector = TargetSelectorConfig(
                    use_robust_ttc=self.variant_config.use_robust_range,
                    enable_hysteresis=self.variant_config.enable_hysteresis,
                    enable_last_good_fallback=self.variant_config.enable_last_good_fallback,
                    enable_cold_start_ttc=self.variant_config.enable_cold_start_ttc,
                )
            self.selector = DeployableTargetSelector(base_selector)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        if self.lane_detector is not None:
            self.lane_detector.reset()
        if self.selector is not None:
            self.selector.reset()
        self._published_ttc_by_track: dict[int, _PublishedTTCState] = {}

    def _tracker_step(
        self,
        detections: Sequence[Detection],
        *,
        detector_update: bool,
        timestamp: float,
        ego_speed_kmh: float,
    ) -> tuple[TrackRisk, ...]:
        arguments = {
            "timestamp": timestamp,
            "image_shape": (self.geometry.height, self.geometry.width),
            "ego_speed_kmh": ego_speed_kmh,
        }
        if detector_update:
            return tuple(self.tracker.update(detections, **arguments))
        return tuple(self.tracker.predict(**arguments))

    def _candidate(
        self,
        snapshot: P2TrackSnapshot,
        corridor: EgoCorridor,
    ) -> RuntimeTargetCandidate:
        if self.corridor_estimator is None:
            raise AssertionError("corridor estimator is required")
        velocity_uncertainty = snapshot.foot_velocity_uncertainty_px_s
        if velocity_uncertainty is None:
            # Association-disabled ablation rungs retain their original
            # history-count heuristic exactly.  Kalman-enabled rungs instead
            # expose the filter's causal lateral-velocity covariance.
            velocity_uncertainty = 36.0 / math.sqrt(max(1, snapshot.hits))
        motion_state_confidence = (
            snapshot.motion_state_confidence
            if snapshot.motion_state_confidence is not None
            else 1.0
        )
        covariance_aware_association = (
            snapshot.association_confidence * motion_state_confidence
        )
        evidence = self.corridor_estimator.assess(
            corridor,
            snapshot.bbox,
            velocity_px_s=(
                snapshot.foot_velocity_x_px_s,
                snapshot.foot_velocity_y_px_s,
            ),
            velocity_uncertainty_px_s=velocity_uncertainty,
            coast_age_s=snapshot.last_seen_age_s,
            # Detection/association confidence is a distinct observation
            # signal.  KF covariance already reaches the corridor through
            # velocity_uncertainty_px_s and reaches the selector's generic
            # uncertainty below; multiplying it here would count the same
            # coast uncertainty again in cold-start confidence gates.
            track_confidence=(
                snapshot.confidence * snapshot.association_confidence
            ),
        )
        range_uncertainty = snapshot.range_uncertainty_m
        if (
            not self.variant_config.use_robust_range
            and not math.isfinite(range_uncertainty)
            and math.isfinite(
            snapshot.estimated_range_m
            )
        ):
            range_uncertainty = snapshot.estimated_range_m * (
                1.0 - float(np.clip(snapshot.range_stability, 0.0, 1.0))
            )
        uncertainty = float(
            np.clip(
                0.42 * (
                    _posterior_uncertainty(snapshot)
                    if self.variant_config.use_class_history_association
                    else 0.0
                )
                + 0.38 * (1.0 - covariance_aware_association)
                + 0.20 * min(1.0, snapshot.last_seen_age_s / 0.60),
                0.0,
                1.0,
            )
        )
        candidate = RuntimeTargetCandidate.from_snapshot(
            snapshot, evidence, uncertainty=uncertainty
        )
        if range_uncertainty == candidate.range_uncertainty_m:
            return candidate
        return RuntimeTargetCandidate(
            **{
                field_name: getattr(candidate, field_name)
                for field_name in candidate.__dataclass_fields__
                if field_name != "range_uncertainty_m"
            },
            range_uncertainty_m=range_uncertainty,
        )

    @staticmethod
    def _selection_source(selection: TargetSelection) -> str:
        if selection.primary_track_id is None:
            return "invalid"
        assessment = next(
            (
                item
                for item in selection.assessments
                if item.track_id == selection.primary_track_id
            ),
            None,
        )
        return assessment.ttc_source if assessment is not None else "invalid"

    def _stabilize_full_output(
        self,
        selection: TargetSelection,
        candidates: Sequence[RuntimeTargetCandidate],
        *,
        timestamp: float,
        source: str,
    ) -> tuple[float, str, bool]:
        """Bound danger-disappearing jumps and hold only stable physics.

        Downward TTC changes are never delayed, preserving cut-in response.
        A missing/high-uncertainty estimate may count down one recent physics
        value only while the same primary still has strong causal path/closing
        evidence.  Such a hold cannot create a warning unless that target was
        already warning.
        """

        predicted = selection.primary_ttc_s
        warning = selection.warning
        track_id = selection.primary_track_id
        live_ids = {candidate.track_id for candidate in candidates}
        self._published_ttc_by_track = {
            item_id: state
            for item_id, state in self._published_ttc_by_track.items()
            if item_id in live_ids
        }
        if track_id is None:
            return predicted, source, warning

        assessment = next(
            (item for item in selection.assessments if item.track_id == track_id),
            None,
        )
        candidate = next(
            (item for item in candidates if item.track_id == track_id),
            None,
        )
        previous = self._published_ttc_by_track.get(track_id)
        if assessment is None or candidate is None:
            return predicted, source, warning
        strong_path_closing_evidence = (
            assessment.path_score >= 0.40
            and assessment.closing_score >= 0.45
            and candidate.missed_updates <= 3
        )
        strong_evidence = (
            strong_path_closing_evidence
            and assessment.uncertainty <= _MAX_FRESH_PHYSICS_UNCERTAINTY
        )
        bounded_hold_evidence = (
            strong_path_closing_evidence
            and assessment.uncertainty <= _MAX_HELD_PHYSICS_UNCERTAINTY
        )

        # A noisy finite physics value is not automatically publishable.  It
        # may fall through to the bounded same-primary hold below; without a
        # previously accepted physics anchor it stays invalid and therefore
        # cannot create a warning merely because evidence is missing/noisy.
        if (
            math.isfinite(predicted)
            and source == "physics"
            and assessment.uncertainty > _MAX_FRESH_PHYSICS_UNCERTAINTY
        ):
            predicted = float("inf")
            source = "uncertain_physics"
            warning = False

        if previous is not None:
            elapsed = max(0.0, timestamp - previous.timestamp)
            expected = max(0.1, previous.ttc_s - elapsed)
            physics_origins = {"physics", "jump_limited_physics"}
            if (
                math.isfinite(predicted)
                and source == "physics"
                and previous.source in physics_origins
            ):
                # Limit only upward residuals: a sudden lower TTC may be a
                # genuine cut-in and must remain immediate.
                maximum = expected + 1.0
                if predicted > maximum:
                    predicted = maximum
                    source = "jump_limited_physics"
                    warning = warning or (
                        previous.warning and strong_evidence and predicted < 2.0
                    )
            elif (
                not math.isfinite(predicted)
                and elapsed <= 0.45
                and previous.source in physics_origins
                and bounded_hold_evidence
            ):
                predicted = expected
                source = "physics_hold"
                warning = previous.warning and predicted < 2.0

        return predicted, source, warning

    def _remember_full_output(
        self,
        track_id: int | None,
        *,
        timestamp: float,
        predicted_ttc_s: float,
        warning: bool,
        source: str,
    ) -> None:
        if (
            track_id is None
            or not math.isfinite(predicted_ttc_s)
            or source == "physics_hold"
        ):
            return
        self._published_ttc_by_track[track_id] = _PublishedTTCState(
            timestamp=float(timestamp),
            ttc_s=float(predicted_ttc_s),
            warning=bool(warning),
            source=source,
        )

    @staticmethod
    def _invalid_reason(
        selection: TargetSelection,
        candidates: Sequence[RuntimeTargetCandidate],
    ) -> str:
        if selection.primary_track_id is not None and math.isfinite(
            selection.primary_ttc_s
        ):
            return "valid"
        if not candidates:
            return "no_tracks"
        assessment_by_id = {
            assessment.track_id: assessment
            for assessment in selection.assessments
        }
        path_candidates = sorted(
            (
                (candidate, assessment_by_id[candidate.track_id])
                for candidate in candidates
            ),
            key=lambda item: (
                item[1].path_score,
                item[1].evidence_score,
                -item[0].track_id,
            ),
            reverse=True,
        )
        candidate, assessment = path_candidates[0]
        if assessment.path_score < 0.20:
            return "no_in_path_target"
        if not math.isfinite(assessment.selected_ttc_s):
            return candidate.robust_reason_code
        return "low_target_evidence"

    def step(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        detector_update: bool,
        timestamp: float,
        ego_speed_kmh: float,
    ) -> P2FrameResult:
        started = time.perf_counter()
        if image_bgr.shape[:2] != (self.geometry.height, self.geometry.width):
            raise ValueError(
                f"P2 image shape {image_bgr.shape[:2]} does not match camera geometry"
            )
        risks = self._tracker_step(
            detections,
            detector_update=detector_update,
            timestamp=timestamp,
            ego_speed_kmh=ego_speed_kmh,
        )
        if self.variant == P2Variant.PHYSICS_CURRENT:
            predicted = _minimum_risk_ttc(risks)
            latency = (time.perf_counter() - started) * 1000.0
            return P2FrameResult(
                timestamp,
                predicted,
                None,
                (),
                False,
                False,
                predicted < 2.0,
                "valid" if math.isfinite(predicted) else "no_finite_physics_ttc",
                "physics",
                "legacy",
                0.0,
                len(risks),
                (),
                risks,
                latency,
            )

        if not isinstance(self.tracker, P2CausalTracker):
            raise AssertionError("P2 selector requires P2CausalTracker")
        if self.lane_detector is None or self.corridor_estimator is None or self.selector is None:
            raise AssertionError("P2 selector components are not initialized")
        lane: LaneEstimate = self.lane_detector.detect(image_bgr)
        corridor = self.corridor_estimator.estimate(
            lane, (self.geometry.height, self.geometry.width)
        )
        candidates = tuple(
            self._candidate(snapshot, corridor)
            for snapshot in self.tracker.candidate_states()
        )
        selection = self.selector.select(candidates)
        predicted = selection.primary_ttc_s
        ttc_source = self._selection_source(selection)
        effective_warning = selection.warning
        if self.variant == P2Variant.FULL:
            predicted, ttc_source, effective_warning = self._stabilize_full_output(
                selection,
                candidates,
                timestamp=timestamp,
                source=ttc_source,
            )
        # The selector is the warning-confidence gate for every deployable
        # rung.  Exact 2.0 keeps a finite TTC diagnostic without publishing a
        # fresh strict-<2 s warning from uncertain/missing evidence.
        if (
            math.isfinite(predicted)
            and predicted < 2.0
            and not effective_warning
        ):
            predicted = 2.0
        effective_warning = predicted < 2.0 and effective_warning
        if self.variant == P2Variant.FULL:
            self._remember_full_output(
                selection.primary_track_id,
                timestamp=timestamp,
                predicted_ttc_s=predicted,
                warning=effective_warning,
                source=ttc_source,
            )
        latency = (time.perf_counter() - started) * 1000.0
        return P2FrameResult(
            timestamp=timestamp,
            predicted_ttc_s=predicted,
            primary_track_id=selection.primary_track_id,
            dangerous_track_ids=selection.dangerous_track_ids,
            target_switched=selection.switched,
            held_by_hysteresis=selection.held_by_hysteresis,
            warning=effective_warning,
            invalid_reason=(
                "valid"
                if math.isfinite(predicted)
                else "high_physics_uncertainty"
                if ttc_source == "uncertain_physics"
                else self._invalid_reason(selection, candidates)
            ),
            ttc_source=ttc_source,
            lane_source=corridor.source,
            lane_confidence=corridor.confidence,
            candidate_count=len(candidates),
            assessments=selection.assessments,
            risks=risks,
            downstream_latency_ms=latency,
        )
