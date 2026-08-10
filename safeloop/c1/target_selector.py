"""Causal, deployable target ranking for the C1 collision pipeline.

The selector consumes tracker, monocular TTC and image-space corridor evidence
that already exists at runtime.  It does not inspect challenge annotations,
depth, event metadata, trip names or frame identifiers.  Keeping the state
machine here independent of the tracker also makes selector-only and
hysteresis ablations explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

from .corridor import CorridorEvidence
from .types import BBox


TTCSource = Literal[
    "robust_range",
    "physics",
    "ego_motion_cold_start",
    "last_good",
    "invalid",
]


def _probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


class RuntimeTrackSnapshot(Protocol):
    """Structural tracker contract used by :meth:`from_snapshot`."""

    track_id: int
    label: str
    class_posterior: Mapping[str, float]
    bbox: BBox
    confidence: float
    foot_velocity_x_px_s: float
    foot_velocity_y_px_s: float
    raw_physics_ttc_s: float
    ego_motion_ttc_s: float
    scale_ttc_s: float
    legacy_range_ttc_s: float
    robust_range_ttc_s: float
    closing_speed_mps: float
    estimated_range_m: float
    range_stability: float
    range_uncertainty_m: float
    hits: int
    missed_updates: int
    association_confidence: float
    observed_this_call: bool
    last_good_ttc_s: float
    reason_code: str


@dataclass(frozen=True, slots=True)
class RuntimeTargetCandidate:
    """Runtime-only features for one active track.

    ``observed_this_update`` is deliberately distinct from ``missed_updates``.
    On detector-stride frames a tracker may predict a box without incrementing
    its missed counter.  Such a prediction must not count as a new detector
    observation when confirming a primary-target switch.
    """

    track_id: int
    bbox: BBox
    corridor: CorridorEvidence
    label: str = "unknown"
    confidence: float = 0.0
    foot_velocity_x_px_s: float = 0.0
    foot_velocity_y_px_s: float = 0.0
    raw_physics_ttc_s: float = float("inf")
    ego_motion_ttc_s: float = float("inf")
    scale_ttc_s: float = float("inf")
    legacy_range_ttc_s: float = float("inf")
    robust_ttc_s: float = float("inf")
    closing_speed_mps: float = 0.0
    range_m: float = float("inf")
    range_stability: float = 0.0
    range_uncertainty_m: float = float("inf")
    hits: int = 1
    missed_updates: int = 0
    association_confidence: float = 0.0
    observed_this_update: bool = True
    last_good_ttc_s: float = float("inf")
    robust_reason_code: str = "insufficient_history"
    uncertainty: float = 0.0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: RuntimeTrackSnapshot,
        corridor: CorridorEvidence,
        *,
        uncertainty: float = 0.0,
    ) -> RuntimeTargetCandidate:
        """Adapt the tracker snapshot without coupling the two modules."""

        return cls(
            track_id=snapshot.track_id,
            bbox=snapshot.bbox,
            corridor=corridor,
            label=snapshot.label,
            confidence=snapshot.confidence,
            foot_velocity_x_px_s=snapshot.foot_velocity_x_px_s,
            foot_velocity_y_px_s=snapshot.foot_velocity_y_px_s,
            raw_physics_ttc_s=snapshot.raw_physics_ttc_s,
            # ``getattr`` keeps the adapter compatible while the tracker and
            # selector land as separate reviewed changes.  A tracker without
            # the new causal signal simply cannot enter the cold-start path.
            ego_motion_ttc_s=float(
                getattr(snapshot, "ego_motion_ttc_s", float("inf"))
            ),
            scale_ttc_s=snapshot.scale_ttc_s,
            legacy_range_ttc_s=snapshot.legacy_range_ttc_s,
            robust_ttc_s=snapshot.robust_range_ttc_s,
            closing_speed_mps=snapshot.closing_speed_mps,
            range_m=snapshot.estimated_range_m,
            range_stability=snapshot.range_stability,
            range_uncertainty_m=snapshot.range_uncertainty_m,
            hits=snapshot.hits,
            missed_updates=snapshot.missed_updates,
            association_confidence=snapshot.association_confidence,
            observed_this_update=snapshot.observed_this_call,
            last_good_ttc_s=snapshot.last_good_ttc_s,
            robust_reason_code=snapshot.reason_code,
            uncertainty=uncertainty,
        )

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        x1, y1, x2, y2 = self.bbox
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        if self.hits < 0 or self.missed_updates < 0:
            raise ValueError("hits and missed_updates must be non-negative")
        if not math.isfinite(self.foot_velocity_x_px_s) or not math.isfinite(
            self.foot_velocity_y_px_s
        ):
            raise ValueError("footpoint velocity must be finite")


@dataclass(frozen=True, slots=True)
class TargetSelectorConfig:
    """Fixed, trip-independent selector controls.

    Feature toggles are intentionally explicit so an ablation cannot silently
    turn hysteresis, cold start, a remembered TTC, or robust range on as a side
    effect of another option.  Normal TTC sources also require an explicitly
    confirmed track; ego-motion cold start is the sole unconfirmed exception.
    """

    use_robust_ttc: bool = False
    enable_hysteresis: bool = False
    enable_last_good_fallback: bool = False
    enable_cold_start_ttc: bool = True
    warning_ttc_s: float = 2.0
    danger_ttc_s: float = 3.0
    monitor_ttc_s: float = 10.0
    min_in_path_probability: float = 0.28
    min_cut_in_probability: float = 0.20
    min_corridor_overlap: float = 0.08
    min_evidence_score: float = 0.26
    min_danger_score: float = 0.34
    closing_speed_scale_mps: float = 6.0
    near_range_m: float = 5.0
    far_range_m: float = 60.0
    min_confirmed_hits: int = 2
    mature_track_hits: int = 6
    robust_min_range_stability: float = 0.40
    robust_max_relative_range_uncertainty: float = 0.35
    cold_start_max_hits: int = 2
    cold_start_min_detection_confidence: float = 0.40
    cold_start_min_track_confidence: float = 0.15
    cold_start_min_association_confidence: float = 0.40
    cold_start_min_in_path_probability: float = 0.12
    cold_start_min_cut_in_probability: float = 0.20
    cold_start_min_corridor_overlap: float = 0.04
    cold_start_max_uncertainty: float = 0.60
    cold_start_max_lateral_uncertainty_widths: float = 0.90
    cold_start_near_uncertainty_scale: float = 1.60
    cold_start_min_evidence_score: float = 0.18
    cold_start_max_missed_updates: int = 1
    switch_score_margin: float = 0.08
    switch_confirmation_observations: int = 2
    primary_hold_updates: int = 2
    fallback_hold_updates: int = 2
    max_fallback_uncertainty: float = 0.60
    max_coast_warning_updates: int = 3
    max_coast_warning_uncertainty: float = 0.70
    min_coast_in_path_probability: float = 0.40
    min_coast_closing_speed_mps: float = 0.50
    min_coast_physics_closing_score: float = 0.45
    min_coast_range_stability: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.warning_ttc_s < self.danger_ttc_s <= self.monitor_ttc_s:
            raise ValueError("TTC thresholds must satisfy warning < danger <= monitor")
        probability_fields = (
            self.min_in_path_probability,
            self.min_cut_in_probability,
            self.min_corridor_overlap,
            self.min_evidence_score,
            self.min_danger_score,
            self.robust_min_range_stability,
            self.robust_max_relative_range_uncertainty,
            self.cold_start_min_detection_confidence,
            self.cold_start_min_track_confidence,
            self.cold_start_min_association_confidence,
            self.cold_start_min_in_path_probability,
            self.cold_start_min_cut_in_probability,
            self.cold_start_min_corridor_overlap,
            self.cold_start_max_uncertainty,
            self.cold_start_min_evidence_score,
            self.max_fallback_uncertainty,
            self.max_coast_warning_uncertainty,
            self.min_coast_in_path_probability,
            self.min_coast_physics_closing_score,
            self.min_coast_range_stability,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in probability_fields
        ):
            raise ValueError("selector probability/uncertainty thresholds must be in [0, 1]")
        if self.closing_speed_scale_mps <= 0.0:
            raise ValueError("closing_speed_scale_mps must be positive")
        if not 0.0 <= self.near_range_m < self.far_range_m:
            raise ValueError("range bounds must satisfy 0 <= near < far")
        if self.min_confirmed_hits < 1 or self.mature_track_hits < 1:
            raise ValueError("track hit thresholds must be positive")
        if self.cold_start_max_hits < 1:
            raise ValueError("cold_start_max_hits must be positive")
        if self.cold_start_max_missed_updates < 0:
            raise ValueError("cold_start_max_missed_updates must be non-negative")
        if (
            not math.isfinite(self.cold_start_max_lateral_uncertainty_widths)
            or self.cold_start_max_lateral_uncertainty_widths <= 0.0
            or not math.isfinite(self.cold_start_near_uncertainty_scale)
            or self.cold_start_near_uncertainty_scale < 0.0
        ):
            raise ValueError("cold-start uncertainty scales must be finite and valid")
        if self.switch_score_margin < 0.0:
            raise ValueError("switch_score_margin must be non-negative")
        if self.switch_confirmation_observations < 1:
            raise ValueError("switch_confirmation_observations must be positive")
        integer_fields = (
            self.primary_hold_updates,
            self.fallback_hold_updates,
            self.max_coast_warning_updates,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("selector hold windows must be non-negative")
        if self.min_coast_closing_speed_mps < 0.0:
            raise ValueError("min_coast_closing_speed_mps must be non-negative")


@dataclass(frozen=True, slots=True)
class TargetAssessment:
    track_id: int
    evidence_score: float
    priority_score: float
    path_score: float
    closing_score: float
    range_score: float
    maturity_score: float
    association_score: float
    uncertainty: float
    selected_ttc_s: float
    ttc_source: TTCSource
    confirmed: bool
    cold_start_eligible: bool
    eligible: bool
    dangerous: bool
    observed_this_update: bool
    hits: int
    missed_updates: int


@dataclass(frozen=True, slots=True)
class TargetSelection:
    primary_track_id: int | None
    primary_ttc_s: float
    warning: bool
    dangerous_track_ids: tuple[int, ...]
    assessments: tuple[TargetAssessment, ...]
    switched: bool
    held_by_hysteresis: bool


class DeployableTargetSelector:
    """Rank tracks and maintain one consistent primary collision target."""

    def __init__(self, config: TargetSelectorConfig | None = None) -> None:
        self.config = config or TargetSelectorConfig()
        self.reset()

    def reset(self) -> None:
        self._primary_track_id: int | None = None
        self._primary_absent_updates = 0
        self._challenger_track_id: int | None = None
        self._challenger_observations = 0
        self._warning_latched_tracks: set[int] = set()
        self._prewarning_latched_tracks: set[int] = set()
        self._cold_start_latched_tracks: set[int] = set()

    def select(self, candidates: Sequence[RuntimeTargetCandidate]) -> TargetSelection:
        """Score one causal frame and return a primary plus all dangers.

        Duplicate track identifiers are rejected because silently selecting one
        would make hysteresis dependent on input ordering.
        """

        candidate_by_id = {candidate.track_id: candidate for candidate in candidates}
        if len(candidate_by_id) != len(candidates):
            raise ValueError("candidate track_id values must be unique")
        assessments = tuple(self._assess(item) for item in candidates)
        live_ids = set(candidate_by_id)
        self._warning_latched_tracks.intersection_update(live_ids)
        self._prewarning_latched_tracks.intersection_update(live_ids)
        self._cold_start_latched_tracks.intersection_update(live_ids)
        for candidate, assessment in zip(candidates, assessments):
            if assessment.cold_start_eligible and candidate.observed_this_update:
                self._cold_start_latched_tracks.add(candidate.track_id)
            elif (
                candidate.missed_updates > self.config.cold_start_max_missed_updates
                or candidate.hits > self.config.cold_start_max_hits
            ):
                self._cold_start_latched_tracks.discard(candidate.track_id)
        assessment_by_id = {item.track_id: item for item in assessments}
        ranked = sorted(
            (item for item in assessments if item.eligible),
            key=lambda item: (
                not item.dangerous,
                -item.priority_score,
                item.selected_ttc_s,
                item.track_id,
            ),
        )
        best = ranked[0] if ranked else None

        previous_primary = self._primary_track_id
        held = False
        if not self.config.enable_hysteresis:
            self._primary_track_id = best.track_id if best is not None else None
            self._primary_absent_updates = 0
            self._clear_challenger()
        else:
            held = self._select_with_hysteresis(best, assessment_by_id)

        primary = assessment_by_id.get(self._primary_track_id)
        primary_candidate = candidate_by_id.get(self._primary_track_id)
        # During a bounded hysteresis hold, preserve the incumbent's finite
        # causal TTC even if one noisy corridor observation drops its path
        # gate.  Alert latching remains separate in ``_warning_for``.
        primary_ttc = (
            primary.selected_ttc_s
            if primary is not None
            and (primary.eligible or held)
            and _finite_positive(primary.selected_ttc_s)
            else float("inf")
        )
        warning = self._warning_for(primary, primary_candidate)
        switched = (
            previous_primary is not None
            and self._primary_track_id is not None
            and previous_primary != self._primary_track_id
        )
        dangerous_ids = tuple(
            item.track_id
            for item in sorted(
                (assessment for assessment in assessments if assessment.dangerous),
                key=lambda assessment: (
                    assessment.selected_ttc_s,
                    -assessment.priority_score,
                    assessment.track_id,
                ),
            )
        )
        ordered_assessments = tuple(
            sorted(assessments, key=lambda item: item.track_id)
        )
        return TargetSelection(
            primary_track_id=self._primary_track_id,
            primary_ttc_s=primary_ttc,
            warning=warning,
            dangerous_track_ids=dangerous_ids,
            assessments=ordered_assessments,
            switched=switched,
            held_by_hysteresis=held,
        )

    def _assess(self, candidate: RuntimeTargetCandidate) -> TargetAssessment:
        evidence = candidate.corridor
        current_overlap = _probability(evidence.corridor_overlap)
        predicted_overlap = _probability(evidence.predicted_corridor_overlap)
        in_path = _probability(evidence.in_path_probability)
        cut_in = _probability(evidence.cut_in_probability)
        path_score = _probability(
            0.55 * in_path + 0.20 * current_overlap + 0.25 * predicted_overlap
        )

        confirmed = candidate.hits >= self.config.min_confirmed_hits
        cold_start_eligible = self._cold_start_eligible(candidate)
        closing_score = _probability(
            max(0.0, candidate.closing_speed_mps)
            / self.config.closing_speed_scale_mps
        )
        raw_ttc_urgency = self._ttc_urgency(candidate.raw_physics_ttc_s)
        # A finite causal scale TTC is closing evidence even before an absolute
        # range slope has enough history.
        closing_score = max(closing_score, 0.70 * raw_ttc_urgency)
        if cold_start_eligible:
            closing_score = max(
                closing_score,
                0.55 * self._ttc_urgency(candidate.ego_motion_ttc_s),
            )
        if math.isfinite(candidate.range_m):
            range_score = _probability(
                (self.config.far_range_m - candidate.range_m)
                / (self.config.far_range_m - self.config.near_range_m)
            )
        else:
            range_score = 0.0
        stability = _probability(candidate.range_stability)
        maturity = _probability(candidate.hits / self.config.mature_track_hits)
        association = _probability(candidate.association_confidence)
        confidence = math.sqrt(
            _probability(candidate.confidence)
            * _probability(evidence.track_confidence)
        )
        uncertainty = self._combined_uncertainty(candidate)

        score = _probability(
            0.31 * path_score
            + 0.12 * cut_in
            + 0.13 * closing_score
            + 0.10 * range_score
            + 0.09 * stability
            + 0.09 * maturity
            + 0.09 * association
            + 0.07 * confidence
            - 0.14 * uncertainty
        )
        ttc, source = self._select_ttc(
            candidate,
            confirmed=confirmed,
            cold_start_eligible=cold_start_eligible,
        )
        urgency = self._ttc_urgency(ttc)
        priority = _probability(0.76 * score + 0.24 * urgency)
        path_gate = (
            in_path >= self.config.min_in_path_probability
            or cut_in >= self.config.min_cut_in_probability
            or max(current_overlap, predicted_overlap)
            >= self.config.min_corridor_overlap
            or source == "ego_motion_cold_start"
        )
        minimum_score = (
            self.config.cold_start_min_evidence_score
            if source == "ego_motion_cold_start"
            else self.config.min_evidence_score
        )
        eligible = (
            path_gate
            and score >= minimum_score
            and _finite_positive(ttc)
            and ttc <= self.config.monitor_ttc_s
            and (confirmed or source == "ego_motion_cold_start")
        )
        dangerous = (
            eligible
            and score >= (
                self.config.cold_start_min_evidence_score
                if source == "ego_motion_cold_start"
                else self.config.min_danger_score
            )
            and ttc < self.config.danger_ttc_s
        )
        return TargetAssessment(
            track_id=candidate.track_id,
            evidence_score=score,
            priority_score=priority,
            path_score=path_score,
            closing_score=closing_score,
            range_score=range_score,
            maturity_score=maturity,
            association_score=association,
            uncertainty=uncertainty,
            selected_ttc_s=ttc,
            ttc_source=source,
            confirmed=confirmed,
            cold_start_eligible=cold_start_eligible,
            eligible=eligible,
            dangerous=dangerous,
            observed_this_update=candidate.observed_this_update,
            hits=candidate.hits,
            missed_updates=candidate.missed_updates,
        )

    def _select_ttc(
        self,
        candidate: RuntimeTargetCandidate,
        *,
        confirmed: bool,
        cold_start_eligible: bool,
    ) -> tuple[float, TTCSource]:
        range_relative_uncertainty = self._relative_range_uncertainty(candidate)
        robust_reliable = (
            self.config.use_robust_ttc
            and candidate.robust_reason_code
            in {"ok", "within_safety_buffer", "jump_limited"}
            and _finite_positive(candidate.robust_ttc_s)
            and candidate.range_stability >= self.config.robust_min_range_stability
            and range_relative_uncertainty
            <= self.config.robust_max_relative_range_uncertainty
        )
        if confirmed and _finite_positive(candidate.raw_physics_ttc_s):
            return float(candidate.raw_physics_ttc_s), "physics"
        if confirmed and robust_reliable:
            return float(candidate.robust_ttc_s), "robust_range"
        if cold_start_eligible:
            return float(candidate.ego_motion_ttc_s), "ego_motion_cold_start"
        held_previous = (
            self.config.enable_last_good_fallback
            and candidate.robust_reason_code == "held_previous"
            and candidate.missed_updates <= self.config.fallback_hold_updates
            and self._combined_uncertainty(candidate)
            <= self.config.max_fallback_uncertainty
            and _finite_positive(candidate.robust_ttc_s)
        )
        if confirmed and held_previous:
            return float(candidate.robust_ttc_s), "last_good"
        if (
            self.config.enable_last_good_fallback
            and candidate.robust_reason_code == "held_previous"
        ):
            return float("inf"), "invalid"
        return float("inf"), "invalid"

    def _cold_start_eligible(self, candidate: RuntimeTargetCandidate) -> bool:
        """Conservative one/two-observation TTC from ego motion.

        The signal assumes no measured object closing rate yet, so it is
        admitted only on a real detector observation whose bbox footpoint is
        in or within its causal lateral-uncertainty envelope of the corridor.
        """

        if (
            not self.config.enable_cold_start_ttc
            or not _finite_positive(candidate.ego_motion_ttc_s)
            or candidate.ego_motion_ttc_s > self.config.monitor_ttc_s
            or not 1 <= candidate.hits <= self.config.cold_start_max_hits
            or candidate.confidence
            < self.config.cold_start_min_detection_confidence
            or candidate.corridor.track_confidence
            < self.config.cold_start_min_track_confidence
            or candidate.association_confidence
            < self.config.cold_start_min_association_confidence
            or _probability(candidate.uncertainty)
            > self.config.cold_start_max_uncertainty
            or self._combined_uncertainty(candidate)
            > self.config.cold_start_max_uncertainty
            or not _finite_positive(candidate.range_m)
        ):
            return False

        fresh_observation = (
            candidate.observed_this_update and candidate.missed_updates == 0
        )
        armed_continuation = (
            candidate.track_id in self._cold_start_latched_tracks
            and candidate.missed_updates <= self.config.cold_start_max_missed_updates
        )
        if not (fresh_observation or armed_continuation):
            return False

        evidence = candidate.corridor
        width = max(1.0, candidate.bbox[2] - candidate.bbox[0])
        if (
            not math.isfinite(evidence.lateral_uncertainty_px)
            or evidence.lateral_uncertainty_px < 0.0
            or evidence.lateral_uncertainty_px
            > self.config.cold_start_max_lateral_uncertainty_widths * width
        ):
            return False

        current_distance = self._distance_to_interval(
            evidence.footpoint[0], evidence.current_bounds
        )
        predicted_distance = self._distance_to_interval(
            evidence.predicted_footpoint[0], evidence.predicted_bounds
        )
        near_margin = (
            self.config.cold_start_near_uncertainty_scale
            * evidence.lateral_uncertainty_px
        )
        foot_near_corridor = min(current_distance, predicted_distance) <= near_margin
        spatial_probability = (
            evidence.in_path_probability
            >= self.config.cold_start_min_in_path_probability
            or evidence.cut_in_probability
            >= self.config.cold_start_min_cut_in_probability
            or max(
                evidence.corridor_overlap,
                evidence.predicted_corridor_overlap,
            )
            >= self.config.cold_start_min_corridor_overlap
        )
        return foot_near_corridor and spatial_probability

    @staticmethod
    def _distance_to_interval(
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        if not math.isfinite(value) or not all(math.isfinite(item) for item in bounds):
            return float("inf")
        left, right = sorted(bounds)
        if value < left:
            return left - value
        if value > right:
            return value - right
        return 0.0

    def _select_with_hysteresis(
        self,
        best: TargetAssessment | None,
        assessment_by_id: dict[int, TargetAssessment],
    ) -> bool:
        if self._primary_track_id is None:
            self._primary_track_id = best.track_id if best is not None else None
            self._primary_absent_updates = 0
            self._clear_challenger()
            return False

        current = assessment_by_id.get(self._primary_track_id)
        if current is None:
            self._primary_absent_updates += 1
        elif not current.eligible:
            # ``missed_updates`` advances only on a real detector call.  Do
            # not consume the hold window on the two intentional stride
            # frames between detector updates.
            if current.missed_updates > 0:
                self._primary_absent_updates = max(
                    self._primary_absent_updates, current.missed_updates
                )
            elif current.observed_this_update:
                self._primary_absent_updates += 1
        else:
            self._primary_absent_updates = 0

        if best is None or best.track_id == self._primary_track_id:
            self._clear_challenger()
            if (
                (current is None or not current.eligible)
                and self._primary_absent_updates > self.config.primary_hold_updates
            ):
                self._primary_track_id = None
                return False
            return current is None or not current.eligible

        # A confirmed, strongly in-path danger is already stable at the tracker
        # level.  Let it replace a safe incumbent immediately instead of
        # letting duplicate road-user hypotheses reset the challenger counter
        # during a cut-in.  Scheduled predictions never confirm a track.
        emergency_confirmed = (
            best.dangerous
            and (current is None or not current.dangerous)
            and best.confirmed
            and best.observed_this_update
            and best.missed_updates == 0
            and best.path_score >= 0.50
        )
        if emergency_confirmed:
            self._primary_track_id = best.track_id
            self._primary_absent_updates = 0
            self._clear_challenger()
            return False

        challenger_better = (
            current is None
            or not current.eligible
            or best.priority_score
            >= current.priority_score + self.config.switch_score_margin
        )
        if not challenger_better:
            self._clear_challenger()
            return True

        if self._challenger_track_id != best.track_id:
            self._challenger_track_id = best.track_id
            self._challenger_observations = 0
        if best.observed_this_update and best.missed_updates == 0:
            self._challenger_observations += 1
        if (
            self._challenger_observations
            >= self.config.switch_confirmation_observations
        ):
            self._primary_track_id = best.track_id
            self._primary_absent_updates = 0
            self._clear_challenger()
            return False
        return True

    def _warning_for(
        self,
        primary: TargetAssessment | None,
        candidate: RuntimeTargetCandidate | None,
    ) -> bool:
        if primary is None or candidate is None:
            return False
        if not primary.selected_ttc_s < self.config.warning_ttc_s:
            if candidate.observed_this_update:
                self._warning_latched_tracks.discard(candidate.track_id)
                if primary.dangerous:
                    self._prewarning_latched_tracks.add(candidate.track_id)
                else:
                    self._prewarning_latched_tracks.discard(candidate.track_id)
            return False
        was_warning = candidate.track_id in self._warning_latched_tracks
        if candidate.observed_this_update and candidate.missed_updates == 0:
            # A stale-but-bounded range fallback may preserve an existing
            # warning, but cannot arm a new warning after current physics has
            # become non-closing/uncertain.
            if primary.ttc_source == "last_good" and not was_warning:
                return False
            if primary.dangerous:
                self._warning_latched_tracks.add(candidate.track_id)
                self._prewarning_latched_tracks.discard(candidate.track_id)
                return True
            self._warning_latched_tracks.discard(candidate.track_id)
            return False

        # A detector-stride prediction with ``missed_updates == 0`` is not a
        # detector miss.  It may continue an already observed warning, but it
        # cannot arm a warning or confirm a primary-target switch.
        if candidate.missed_updates == 0:
            armed = (
                was_warning
                or candidate.track_id in self._prewarning_latched_tracks
            )
            if armed:
                self._warning_latched_tracks.add(candidate.track_id)
                self._prewarning_latched_tracks.discard(candidate.track_id)
            return armed

        # Missing data alone can never create a warning.  A warning may coast
        # only if it was established by a real detector observation and the
        # remaining causal motion/range evidence is still strong.
        cold_start_coast = (
            was_warning
            and primary.ttc_source == "ego_motion_cold_start"
            and primary.cold_start_eligible
            and candidate.missed_updates
            <= self.config.cold_start_max_missed_updates
            and primary.uncertainty <= self.config.max_coast_warning_uncertainty
        )
        strong_motion = (
            (
                candidate.closing_speed_mps
                >= self.config.min_coast_closing_speed_mps
                and candidate.range_stability
                >= self.config.min_coast_range_stability
            )
            or (
                primary.ttc_source == "physics"
                and primary.closing_score
                >= self.config.min_coast_physics_closing_score
            )
        )
        strong_coast = (
            candidate.missed_updates <= self.config.max_coast_warning_updates
            and candidate.corridor.in_path_probability
            >= self.config.min_coast_in_path_probability
            and strong_motion
            and primary.uncertainty <= self.config.max_coast_warning_uncertainty
        )
        if not (cold_start_coast or (was_warning and strong_coast)):
            self._warning_latched_tracks.discard(candidate.track_id)
            return False
        return True

    def _clear_challenger(self) -> None:
        self._challenger_track_id = None
        self._challenger_observations = 0

    def _combined_uncertainty(self, candidate: RuntimeTargetCandidate) -> float:
        width = max(1.0, candidate.bbox[2] - candidate.bbox[0])
        lateral = _probability(candidate.corridor.lateral_uncertainty_px / (2.0 * width))
        range_relative = self._relative_range_uncertainty(candidate)
        generic = _probability(candidate.uncertainty)
        coast = _probability(candidate.missed_updates / max(1, self.config.primary_hold_updates + 1))
        return _probability(
            0.30 * lateral + 0.30 * range_relative + 0.25 * generic + 0.15 * coast
        )

    @staticmethod
    def _relative_range_uncertainty(candidate: RuntimeTargetCandidate) -> float:
        if not math.isfinite(candidate.range_uncertainty_m):
            return 1.0
        if not _finite_positive(candidate.range_m):
            return 1.0
        return _probability(candidate.range_uncertainty_m / candidate.range_m)

    def _ttc_urgency(self, ttc_s: float) -> float:
        if not _finite_positive(ttc_s):
            return 0.0
        return _probability(
            (self.config.monitor_ttc_s - ttc_s) / self.config.monitor_ttc_s
        )
