"""Dependency-free IoU tracking and monocular scale-rate TTC estimation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Sequence

import numpy as np

from .types import BBox, Detection, TrackRisk


# Effective image heights were fitted only from the six labelled development
# trips by projecting their KITTI 3-D centres into the left camera.  They
# compensate for the fact that a COCO detector box does not cover the same
# vertical extent as a physical KITTI cuboid.  Runtime still uses RGB + bbox
# only; no depth, right camera or KITTI label enters this path.
_OBJECT_HEIGHT_M = {
    "person": 1.62,
    "bicycle": 1.80,
    "car": 2.84,
    "motorcycle": 2.07,
    "bus": 3.20,
    "truck": 2.80,
}

_SAFETY_BUFFER_M = {
    "person": 2.50,
    "bicycle": 3.50,
    "motorcycle": 3.50,
    "car": 4.50,
    "bus": 5.50,
    "truck": 5.50,
}


@dataclass(frozen=True)
class TrackerConfig:
    history_size: int = 12
    min_history: int = 4
    max_missed_frames: int = 5
    max_coast_updates: int = 2
    min_iou: float = 0.12
    max_center_distance: float = 0.65
    cross_class_min_iou: float = 0.18
    cross_class_max_center_distance: float = 0.55
    cross_class_penalty: float = 0.12
    min_inverse_ttc: float = 0.08
    min_scale_box_height_px: float = 1.0
    enable_range_ttc: bool = False
    range_recent_observations: int = 6
    min_range_history: int = 3
    min_range_closing_speed_mps: float = 0.50
    max_range_closing_speed_mps: float = 45.0
    min_range_box_height_px: float = 10.0
    min_range_confidence: float = 0.20
    min_range_decreasing_fraction: float = 0.70
    max_range_slope_relative_mad: float = 1.0
    max_range_distance_m: float = 100.0
    safety_buffer_scale: float = 1.0
    min_lateral_speed_px_s: float = 20.0
    min_lateral_box_height_px: float = 48.0
    min_lateral_ego_speed_kmh: float = 3.0
    lateral_recent_observations: int = 4
    max_lateral_shrink_rate: float = -0.35
    large_lateral_box_height_frac: float = 0.30
    max_ttc_s: float = 10.0
    collision_horizon_s: float = 2.5
    corridor_base_half_width: float = 0.10
    corridor_perspective_gain: float = 0.13
    enable_ego_fallback: bool = False
    min_fallback_box_height_px: float = 20.0
    enable_fast_attack: bool = True
    fast_attack_min_box_height_px: float = 48.0
    fast_attack_corridor_margin: float = 0.035
    fast_attack_max_observations: int = 2
    fast_attack_min_confidence: float = 0.35
    fast_attack_border_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.history_size < 2 or not 2 <= self.min_history <= self.history_size:
            raise ValueError("Cần 2 <= min_history <= history_size")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames phải >= 0")
        if not 0 <= self.max_coast_updates <= self.max_missed_frames:
            raise ValueError("Cần 0 <= max_coast_updates <= max_missed_frames")
        if self.max_ttc_s <= 0:
            raise ValueError("max_ttc_s phải > 0")
        if not 2 <= self.min_range_history <= self.history_size:
            raise ValueError("Cần 2 <= min_range_history <= history_size")
        if self.range_recent_observations < self.min_range_history:
            raise ValueError("range_recent_observations phải >= min_range_history")
        if self.min_range_closing_speed_mps <= 0:
            raise ValueError("min_range_closing_speed_mps phải > 0")
        if self.max_range_closing_speed_mps <= self.min_range_closing_speed_mps:
            raise ValueError("max_range_closing_speed_mps phải lớn hơn min")
        if not 0 <= self.min_range_decreasing_fraction <= 1:
            raise ValueError("min_range_decreasing_fraction phải thuộc [0, 1]")
        if self.max_range_slope_relative_mad < 0:
            raise ValueError("max_range_slope_relative_mad phải >= 0")
        if self.safety_buffer_scale < 0:
            raise ValueError("safety_buffer_scale phải >= 0")


@dataclass(frozen=True)
class _Observation:
    timestamp: float
    bbox: BBox
    label: str


@dataclass
class _Track:
    track_id: int
    class_id: int
    label: str
    confidence: float
    bbox: BBox
    history: Deque[_Observation]
    missed: int = 0


def _bbox_geometry(box: BBox) -> tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height, width * height


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _class_group(label: str) -> str:
    if label == "person":
        return "person"
    if label in {"bicycle", "motorcycle"}:
        return "two_wheeler"
    return "vehicle"


def _pairwise_slopes(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    slopes: list[float] = []
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            dx = xs[j] - xs[i]
            if dx > 1e-4:
                slopes.append((ys[j] - ys[i]) / dx)
    return slopes


def _median_pairwise_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    slopes = _pairwise_slopes(xs, ys)
    return float(np.median(slopes)) if slopes else 0.0


class MonocularTTCTracker:
    """Track road users and estimate TTC from image expansion.

    For an object with apparent image scale ``s = k / Z``, constant closing
    speed gives ``d(log(s))/dt = 1/TTC``.  A median pairwise slope over recent
    boxes is used instead of differentiating two noisy frames.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        focal_y_px: float = 320.0,
        focal_x_px: float | None = None,
        principal_x_px: float | None = None,
    ):
        self.config = config or TrackerConfig()
        self.focal_y_px = focal_y_px
        self.focal_x_px = focal_x_px if focal_x_px is not None else focal_y_px
        self.principal_x_px = principal_x_px
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: Sequence[Detection],
        *,
        timestamp: float,
        image_shape: tuple[int, int],
        ego_speed_kmh: float = 0.0,
    ) -> list[TrackRisk]:
        matches = self._match(detections, image_shape)
        matched_tracks = {track_index for track_index, _ in matches}
        matched_detections = {detection_index for _, detection_index in matches}

        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.missed += 1

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            track.class_id = detection.class_id
            track.label = detection.label
            track.confidence = detection.confidence
            track.bbox = detection.bbox
            track.missed = 0
            track.history.append(_Observation(timestamp, detection.bbox, detection.label))

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            history: Deque[_Observation] = deque(maxlen=self.config.history_size)
            history.append(_Observation(timestamp, detection.bbox, detection.label))
            self._tracks.append(_Track(
                track_id=self._next_track_id,
                class_id=detection.class_id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
                history=history,
            ))
            self._next_track_id += 1

        self._tracks = [
            track for track in self._tracks if track.missed <= self.config.max_missed_frames
        ]
        return self._emit_risks(image_shape, ego_speed_kmh, timestamp)

    def predict(
        self,
        *,
        timestamp: float,
        image_shape: tuple[int, int],
        ego_speed_kmh: float = 0.0,
    ) -> list[TrackRisk]:
        """Reuse active tracks when the detector intentionally skips a frame.

        No synthetic observation is appended: scale-rate is still fitted only
        from real detector boxes. TTC is counted down from the last observation.
        """
        return self._emit_risks(image_shape, ego_speed_kmh, timestamp)

    def _emit_risks(
        self,
        image_shape: tuple[int, int],
        ego_speed_kmh: float,
        timestamp: float,
    ) -> list[TrackRisk]:
        risks: list[TrackRisk] = []
        for track in self._tracks:
            if track.missed > self.config.max_coast_updates:
                continue
            risk = self._risk(track, image_shape, ego_speed_kmh, timestamp)
            # A short coast is essential when a lateral cut-in flickers, but
            # coasting scale-only tracks extended noisy warnings on T01/T06.
            # Normal detector-stride frames still emit because missed == 0.
            may_coast = (
                math.isfinite(risk.lateral_ttc_s)
                or math.isfinite(risk.range_ttc_s)
                or math.isfinite(risk.ego_fallback_ttc_s)
            )
            if track.missed == 0 or may_coast:
                risks.append(risk)
        return risks

    def _match(
        self,
        detections: Sequence[Detection],
        image_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        same_class_candidates: list[tuple[float, int, int]] = []
        cross_class_candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            tx, ty, tw, th, _ = _bbox_geometry(track.bbox)
            for di, detection in enumerate(detections):
                dx, dy, dw, dh, _ = _bbox_geometry(detection.bbox)
                overlap = _iou(track.bbox, detection.bbox)
                normalizer = max(8.0, math.sqrt(max(tw * th, dw * dh)))
                center_distance = math.hypot(dx - tx, dy - ty) / normalizer
                same_group = _class_group(track.label) == _class_group(detection.label)
                if same_group:
                    if (
                        overlap < self.config.min_iou
                        and center_distance > self.config.max_center_distance
                    ):
                        continue
                    class_penalty = 0.0
                    target = same_class_candidates
                else:
                    # A cut-in motorcycle is often alternately classified as
                    # car/person/motorcycle. Permit a strong geometric match
                    # across classes instead of resetting its motion history.
                    if (
                        overlap < self.config.cross_class_min_iou
                        and center_distance > self.config.cross_class_max_center_distance
                    ):
                        continue
                    class_penalty = self.config.cross_class_penalty
                    target = cross_class_candidates
                score = overlap - 0.08 * center_distance - class_penalty
                target.append((score, ti, di))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        # Preserve the original class-aware tracker wherever possible. Only
        # unmatched tracks/detections enter the cross-class recovery pass.
        for candidates in (same_class_candidates, cross_class_candidates):
            for _, ti, di in sorted(candidates, reverse=True):
                if ti in used_tracks or di in used_detections:
                    continue
                used_tracks.add(ti)
                used_detections.add(di)
                matches.append((ti, di))
        return matches

    def _risk(
        self,
        track: _Track,
        image_shape: tuple[int, int],
        ego_speed_kmh: float,
        timestamp: float,
    ) -> TrackRisk:
        observations = list(track.history)
        times = [obs.timestamp for obs in observations]
        centers_x: list[float] = []
        log_scales: list[float] = []
        for obs in observations:
            cx, _, _, _, area = _bbox_geometry(obs.bbox)
            centers_x.append(cx)
            log_scales.append(math.log(math.sqrt(area)))

        # Scale-rate must compare like-for-like boxes. Cross-class association
        # is useful for lateral cut-in motion, but person/motorcycle/car boxes
        # cover different pixels and would create a fake expansion signal.
        current_group = _class_group(track.label)
        scale_observations: list[_Observation] = []
        for observation in reversed(observations):
            if _class_group(observation.label) != current_group:
                break
            scale_observations.append(observation)
        scale_observations.reverse()
        scale_times = [obs.timestamp for obs in scale_observations]
        scale_log_sizes = [
            math.log(math.sqrt(_bbox_geometry(obs.bbox)[4]))
            for obs in scale_observations
        ]

        scale_ttc = float("inf")
        _, _, _, box_height, _ = _bbox_geometry(track.bbox)
        if (
            len(scale_observations) >= self.config.min_history
            and box_height >= self.config.min_scale_box_height_px
        ):
            inverse_ttc = _median_pairwise_slope(scale_times, scale_log_sizes)
            if inverse_ttc >= self.config.min_inverse_ttc:
                age_s = max(0.0, timestamp - observations[-1].timestamp)
                candidate = 1.0 / inverse_ttc - age_s
                if 0.1 <= candidate <= self.config.max_ttc_s:
                    scale_ttc = candidate

        # Absolute range from the class-calibrated apparent height.  Fitting a
        # robust slope in metres gives closing speed directly; subtracting the
        # combined safety envelope mirrors how the organizer's CARLA labels
        # define TTC near contact.  Pairwise-median slope resists bbox jitter.
        range_ttc = float("inf")
        range_closing_speed_mps = 0.0
        range_trend_confidence = 0.0
        range_n = max(self.config.min_range_history, self.config.range_recent_observations)
        range_observations = observations[-range_n:]
        range_times = [obs.timestamp for obs in range_observations]
        ranges_m = [self._estimated_distance(obs.label, obs.bbox) for obs in range_observations]
        if self.config.enable_range_ttc and len(ranges_m) >= self.config.min_range_history:
            slopes = _pairwise_slopes(range_times, ranges_m)
            closing_samples = [-value for value in slopes]
            range_closing_speed_mps = (
                float(np.median(closing_samples)) if closing_samples else 0.0
            )
            decreasing_fraction = (
                sum(value > 0.0 for value in closing_samples) / len(closing_samples)
                if closing_samples else 0.0
            )
            closing_mad = (
                float(np.median(np.abs(
                    np.asarray(closing_samples) - range_closing_speed_mps
                )))
                if closing_samples else float("inf")
            )
            relative_mad = closing_mad / max(range_closing_speed_mps, 0.1)
            range_trend_confidence = decreasing_fraction / (1.0 + relative_mad)
            reliable_trend = (
                self.config.min_range_closing_speed_mps
                <= range_closing_speed_mps
                <= self.config.max_range_closing_speed_mps
                and decreasing_fraction >= self.config.min_range_decreasing_fraction
                and relative_mad <= self.config.max_range_slope_relative_mad
                and box_height >= self.config.min_range_box_height_px
                and track.confidence >= self.config.min_range_confidence
                and ranges_m[-1] <= self.config.max_range_distance_m
            )
            if reliable_trend:
                age_s = max(0.0, timestamp - observations[-1].timestamp)
                buffer_m = (
                    _SAFETY_BUFFER_M.get(track.label, 4.0)
                    * self.config.safety_buffer_scale
                )
                candidate = (
                    (ranges_m[-1] - buffer_m) / range_closing_speed_mps - age_s
                )
                if 0.1 <= candidate <= self.config.max_ttc_s:
                    range_ttc = candidate

        velocity_x = 0.0
        lateral_ttc = float("inf")
        if len(observations) >= 2:
            recent_n = max(2, self.config.lateral_recent_observations)
            recent_times = times[-recent_n:]
            recent_centers_x = centers_x[-recent_n:]
            recent_log_scales = log_scales[-recent_n:]
            velocity_x = _median_pairwise_slope(recent_times, recent_centers_x)
            recent_scale_rate = _median_pairwise_slope(recent_times, recent_log_scales)
            cx = centers_x[-1]
            delta_to_ego_center = image_shape[1] / 2.0 - cx
            size_supports_cut_in = (
                recent_scale_rate >= self.config.max_lateral_shrink_rate
                or box_height >= self.config.large_lateral_box_height_frac * image_shape[0]
            )
            moving_toward_center = (
                abs(velocity_x) >= self.config.min_lateral_speed_px_s
                and delta_to_ego_center * velocity_x > 0.0
                and box_height >= self.config.min_lateral_box_height_px
                and ego_speed_kmh >= self.config.min_lateral_ego_speed_kmh
                and size_supports_cut_in
            )
            if moving_toward_center:
                age_s = max(0.0, timestamp - observations[-1].timestamp)
                candidate = delta_to_ego_center / velocity_x - age_s
                if 0.1 <= candidate <= self.config.max_ttc_s:
                    lateral_ttc = candidate

        distance_m = self._estimated_distance(track.label, track.bbox)
        ego_ttc = float("inf")
        ego_speed_mps = max(0.0, ego_speed_kmh) / 3.6
        fallback_candidate = float("inf")
        if ego_speed_mps > 0.5 and box_height >= self.config.min_fallback_box_height_px:
            age_s = max(0.0, timestamp - observations[-1].timestamp)
            candidate = distance_m / ego_speed_mps - age_s
            if 0.1 <= candidate <= self.config.max_ttc_s:
                fallback_candidate = candidate
                if self.config.enable_ego_fallback:
                    ego_ttc = candidate

        provisional_ttc = min(scale_ttc, range_ttc, lateral_ttc, ego_ttc)
        relevant = self._collision_relevant(
            track,
            image_shape,
            centers_x,
            times,
            provisional_ttc,
        )
        if (
            self.config.enable_fast_attack
            and relevant
            and box_height >= self.config.fast_attack_min_box_height_px
            and self._is_fast_attack_candidate(track, image_shape)
        ):
            ego_ttc = min(ego_ttc, fallback_candidate)
            provisional_ttc = min(provisional_ttc, ego_ttc)
        final_ttc = provisional_ttc if relevant else float("inf")
        return TrackRisk(
            track_id=track.track_id,
            label=track.label,
            confidence=track.confidence,
            bbox=track.bbox,
            collision_relevant=relevant,
            predicted_ttc_s=final_ttc,
            scale_ttc_s=scale_ttc,
            range_ttc_s=range_ttc,
            range_closing_speed_mps=range_closing_speed_mps,
            range_trend_confidence=range_trend_confidence,
            lateral_ttc_s=lateral_ttc,
            ego_fallback_ttc_s=ego_ttc,
            estimated_distance_m=distance_m,
        )

    def _estimated_distance(self, label: str, bbox: BBox) -> float:
        _, _, _, box_height, _ = _bbox_geometry(bbox)
        effective_height = _OBJECT_HEIGHT_M.get(label, 1.7)
        return self.focal_y_px * effective_height / max(box_height, 1.0)

    def _collision_relevant(
        self,
        track: _Track,
        image_shape: tuple[int, int],
        centers_x: Sequence[float],
        times: Sequence[float],
        ttc_s: float,
    ) -> bool:
        h, w = image_shape
        cx, _, _, _, _ = _bbox_geometry(track.bbox)
        bottom_y = track.bbox[3]
        y_ratio = min(1.0, max(0.0, bottom_y / h))
        half_width = w * (
            self.config.corridor_base_half_width
            + self.config.corridor_perspective_gain * y_ratio
        )
        corridor_left = w / 2.0 - half_width
        corridor_right = w / 2.0 + half_width

        velocity_x = 0.0
        if len(times) >= 2:
            velocity_x = _median_pairwise_slope(times, centers_x)
        horizon = self.config.collision_horizon_s
        if math.isfinite(ttc_s):
            horizon = min(horizon, max(0.0, ttc_s))
        x1, _, x2, _ = track.bbox
        future_left = x1 + velocity_x * horizon
        future_right = x2 + velocity_x * horizon
        motion_left = min(x1, future_left)
        motion_right = max(x2, future_right)
        margin = self.config.fast_attack_corridor_margin * w
        crosses_corridor = (
            motion_right >= corridor_left - margin
            and motion_left <= corridor_right + margin
        )
        # Ignore tiny detections very high in the image until their motion is measurable.
        return bottom_y >= 0.35 * h and crosses_corridor

    def _near_corridor(
        self,
        box: BBox,
        image_shape: tuple[int, int],
        *,
        extra_margin: float,
    ) -> bool:
        h, w = image_shape
        x1, _, x2, bottom_y = box
        y_ratio = min(1.0, max(0.0, bottom_y / h))
        half_width = w * (
            self.config.corridor_base_half_width
            + self.config.corridor_perspective_gain * y_ratio
        )
        margin = extra_margin * w
        return (
            x2 >= w / 2.0 - half_width - margin
            and x1 <= w / 2.0 + half_width + margin
        )

    def _is_fast_attack_candidate(
        self,
        track: _Track,
        image_shape: tuple[int, int],
    ) -> bool:
        """Conservative cold-start for a large object entering from one side.

        It deliberately excludes border-truncated detections and established
        objects already in the lane; those two cases caused long false-warning
        runs on T02 before the cut-in event.
        """
        if (
            len(track.history) > self.config.fast_attack_max_observations
            or track.confidence < self.config.fast_attack_min_confidence
        ):
            return False
        h, w = image_shape
        x1, _, x2, bottom_y = track.bbox
        border = self.config.fast_attack_border_margin * w
        if x1 <= border or x2 >= w - border:
            return False
        y_ratio = min(1.0, max(0.0, bottom_y / h))
        half_width = w * (
            self.config.corridor_base_half_width
            + self.config.corridor_perspective_gain * y_ratio
        )
        cx = (x1 + x2) / 2.0
        outside_inner_corridor = not (
            w / 2.0 - half_width <= cx <= w / 2.0 + half_width
        )
        return outside_inner_corridor and self._near_corridor(
            track.bbox,
            image_shape,
            extra_margin=self.config.fast_attack_corridor_margin,
        )
