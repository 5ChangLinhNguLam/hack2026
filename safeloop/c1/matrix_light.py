"""Virtual matrix-light targeting driven by C1 collision tracks.

This module produces an image/grid-space beam request for simulation and HMI
visualisation only.  It intentionally contains no CAN, GPIO or lamp actuator
code: illuminating another road user's eyes would require homologation,
eye-safety limits and an independent fail-safe controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .types import BBox, C1FramePrediction, TrackRisk
from .lane import LaneEstimate


@dataclass(frozen=True)
class MatrixLightConfig:
    columns: int = 32
    rows: int = 18
    max_targets: int = 3
    max_target_ttc_s: float = 3.0
    min_target_confidence: float = 0.25
    min_range_trend_confidence: float = 0.55
    strong_scale_ttc_s: float = 1.50
    strong_scale_min_box_height_px: float = 50.0
    bbox_margin_fraction: float = 0.06
    night_luma_threshold: float = 68.0
    low_visibility_contrast_threshold: float = 28.0
    day_intensity: float = 0.35
    night_intensity: float = 0.85
    low_visibility_intensity: float = 0.65
    lane_margin_fraction: float = 0.10
    cut_in_lane_margin_fraction: float = 0.35

    def __post_init__(self) -> None:
        if self.columns <= 0 or self.rows <= 0:
            raise ValueError("Kích thước matrix phải dương")
        if self.max_targets <= 0 or self.max_target_ttc_s <= 0:
            raise ValueError("max_targets/max_target_ttc_s phải dương")


@dataclass(frozen=True)
class MatrixLightTarget:
    track_id: int
    label: str
    bbox: BBox
    predicted_ttc_s: float
    priority: float
    azimuth_deg: float
    elevation_deg: float
    grid_cells: tuple[tuple[int, int], ...]
    geometric_coverage: float


@dataclass(frozen=True)
class MatrixLightFrame:
    mode: str
    ambient_luma: float
    contrast: float
    intensity: float
    grid_mask: np.ndarray
    targets: tuple[MatrixLightTarget, ...]
    simulation_only: bool = True


@dataclass(frozen=True)
class MatrixLightEvaluation:
    frames: int
    gt_danger_frames: int
    targeted_frames: int
    frame_precision: float
    frame_recall: float
    frame_f1: float
    predicted_targets: int
    matched_targets: int
    target_precision: float
    gt_objects: int
    gt_objects_covered_95: int
    gt_coverage_95_recall: float
    mean_gt_coverage: float
    metric_name: str = "projected_kitti_object_targeting"


class MatrixLightController:
    def __init__(
        self,
        config: MatrixLightConfig | None = None,
        *,
        focal_x_px: float = 320.0,
        focal_y_px: float = 320.0,
        principal_x_px: float = 320.0,
        principal_y_px: float = 180.0,
    ) -> None:
        self.config = config or MatrixLightConfig()
        self.fx = focal_x_px
        self.fy = focal_y_px
        self.cx = principal_x_px
        self.cy = principal_y_px

    def compute(
        self,
        image_bgr: np.ndarray,
        prediction: C1FramePrediction,
        lane: LaneEstimate | None = None,
    ) -> MatrixLightFrame:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Matrix light cần ảnh BGR HxWx3")
        mode, luma, contrast, intensity = self._ambient_mode(image_bgr)
        eligible = [risk for risk in prediction.risks if self._eligible(risk, lane)]
        eligible.sort(key=self._priority, reverse=True)
        eligible = eligible[: self.config.max_targets]

        mask = np.zeros((self.config.rows, self.config.columns), dtype=bool)
        targets: list[MatrixLightTarget] = []
        for risk in eligible:
            expanded = self._expanded_bbox(risk.bbox, image_bgr.shape[:2])
            cells = self._cells_for_bbox(expanded, image_bgr.shape[:2])
            for row, column in cells:
                mask[row, column] = True
            coverage = self._coverage(risk.bbox, cells, image_bgr.shape[:2])
            x1, y1, x2, y2 = risk.bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            targets.append(MatrixLightTarget(
                track_id=risk.track_id,
                label=risk.label,
                bbox=risk.bbox,
                predicted_ttc_s=risk.predicted_ttc_s,
                priority=self._priority(risk),
                azimuth_deg=math.degrees(math.atan2(center_x - self.cx, self.fx)),
                elevation_deg=math.degrees(math.atan2(center_y - self.cy, self.fy)),
                grid_cells=tuple(cells),
                geometric_coverage=coverage,
            ))
        return MatrixLightFrame(
            mode=mode,
            ambient_luma=luma,
            contrast=contrast,
            intensity=intensity,
            grid_mask=mask,
            targets=tuple(targets),
        )

    def _ambient_mode(self, image: np.ndarray) -> tuple[str, float, float, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Road/traffic occupy the lower 75%; excluding most sky prevents a
        # bright moon or sunset from making a dark road look like daytime.
        road = gray[gray.shape[0] // 4 :, :]
        luma = float(np.median(road))
        contrast = float(np.percentile(road, 90) - np.percentile(road, 10))
        if luma < self.config.night_luma_threshold:
            return "NIGHT", luma, contrast, self.config.night_intensity
        if contrast < self.config.low_visibility_contrast_threshold:
            return "LOW_VISIBILITY", luma, contrast, self.config.low_visibility_intensity
        return "DAY", luma, contrast, self.config.day_intensity

    def _eligible(self, risk: TrackRisk, lane: LaneEstimate | None) -> bool:
        box_height = max(0.0, risk.bbox[3] - risk.bbox[1])
        reliable_range = (
            math.isfinite(risk.range_ttc_s)
            and risk.range_trend_confidence >= self.config.min_range_trend_confidence
        )
        lateral_cut_in = math.isfinite(risk.lateral_ttc_s)
        strong_scale = (
            math.isfinite(risk.scale_ttc_s)
            and risk.scale_ttc_s <= self.config.strong_scale_ttc_s
            and box_height >= self.config.strong_scale_min_box_height_px
        )
        return (
            risk.collision_relevant
            and math.isfinite(risk.predicted_ttc_s)
            and risk.predicted_ttc_s <= self.config.max_target_ttc_s
            and risk.confidence >= self.config.min_target_confidence
            # An ego-speed fallback alone is deliberately insufficient to aim
            # a beam.  At least one image-motion cue must corroborate danger.
            and (reliable_range or lateral_cut_in or strong_scale)
            and self._inside_ego_lane(risk, lane)
        )

    def _inside_ego_lane(self, risk: TrackRisk, lane: LaneEstimate | None) -> bool:
        if lane is None or not lane.valid or not lane.left_points or not lane.right_points:
            return True
        bottom_y = float(risk.bbox[3])
        left_y = np.asarray([point[1] for point in lane.left_points], dtype=float)
        left_xs = np.asarray([point[0] for point in lane.left_points], dtype=float)
        right_y = np.asarray([point[1] for point in lane.right_points], dtype=float)
        right_xs = np.asarray([point[0] for point in lane.right_points], dtype=float)
        left_x = float(np.interp(bottom_y, left_y, left_xs))
        right_x = float(np.interp(bottom_y, right_y, right_xs))
        if right_x <= left_x:
            return True
        margin_fraction = (
            self.config.cut_in_lane_margin_fraction
            if math.isfinite(risk.lateral_ttc_s)
            else self.config.lane_margin_fraction
        )
        margin = (right_x - left_x) * margin_fraction
        return risk.bbox[2] >= left_x - margin and risk.bbox[0] <= right_x + margin

    @staticmethod
    def _priority(risk: TrackRisk) -> float:
        x1, y1, x2, y2 = risk.bbox
        area = max(1.0, (x2 - x1) * (y2 - y1))
        return risk.confidence * (1.0 / max(0.1, risk.predicted_ttc_s)) * math.log1p(area)

    def _expanded_bbox(self, bbox: BBox, shape: tuple[int, int]) -> BBox:
        h, w = shape
        x1, y1, x2, y2 = bbox
        dx = (x2 - x1) * self.config.bbox_margin_fraction
        dy = (y2 - y1) * self.config.bbox_margin_fraction
        return (
            max(0.0, x1 - dx),
            max(0.0, y1 - dy),
            min(float(w), x2 + dx),
            min(float(h), y2 + dy),
        )

    def _cells_for_bbox(
        self, bbox: BBox, shape: tuple[int, int]
    ) -> list[tuple[int, int]]:
        h, w = shape
        x1, y1, x2, y2 = bbox
        col0 = max(0, min(self.config.columns - 1, int(x1 / w * self.config.columns)))
        col1 = max(0, min(self.config.columns - 1, int(math.ceil(x2 / w * self.config.columns) - 1)))
        row0 = max(0, min(self.config.rows - 1, int(y1 / h * self.config.rows)))
        row1 = max(0, min(self.config.rows - 1, int(math.ceil(y2 / h * self.config.rows) - 1)))
        return [
            (row, column)
            for row in range(row0, row1 + 1)
            for column in range(col0, col1 + 1)
        ]

    def _coverage(
        self,
        target_bbox: BBox,
        cells: list[tuple[int, int]],
        shape: tuple[int, int],
    ) -> float:
        h, w = shape
        x1, y1, x2, y2 = target_bbox
        target_area = max(1.0, (x2 - x1) * (y2 - y1))
        covered = 0.0
        cell_w = w / self.config.columns
        cell_h = h / self.config.rows
        for row, column in cells:
            cx1, cy1 = column * cell_w, row * cell_h
            cx2, cy2 = cx1 + cell_w, cy1 + cell_h
            covered += max(0.0, min(x2, cx2) - max(x1, cx1)) * max(
                0.0, min(y2, cy2) - max(y1, cy1)
            )
        return min(1.0, covered / target_area)


class MatrixLightOfflineEvaluator:
    """Independent evaluator; never used by the runtime controller."""

    def __init__(self, image_shape: tuple[int, int], grid_shape: tuple[int, int]) -> None:
        self.image_shape = image_shape
        self.grid_shape = grid_shape
        self.frames = 0
        self.gt_danger_frames = 0
        self.targeted_frames = 0
        self.frame_tp = 0
        self.predicted_targets = 0
        self.matched_targets = 0
        self.gt_objects = 0
        self.gt_objects_covered_95 = 0
        self.gt_coverages: list[float] = []

    def add(self, frame: MatrixLightFrame, gt_danger_bboxes: list[BBox]) -> None:
        self.frames += 1
        has_gt = bool(gt_danger_bboxes)
        has_target = bool(frame.targets)
        self.gt_danger_frames += int(has_gt)
        self.targeted_frames += int(has_target)
        self.frame_tp += int(has_gt and has_target)
        self.predicted_targets += len(frame.targets)
        self.gt_objects += len(gt_danger_bboxes)

        unmatched = set(range(len(gt_danger_bboxes)))
        for target in frame.targets:
            if not unmatched:
                break
            best = max(unmatched, key=lambda index: _bbox_iou(target.bbox, gt_danger_bboxes[index]))
            if _bbox_iou(target.bbox, gt_danger_bboxes[best]) >= 0.05:
                self.matched_targets += 1
                unmatched.remove(best)
        for bbox in gt_danger_bboxes:
            coverage = _mask_bbox_coverage(
                frame.grid_mask, bbox, self.image_shape
            )
            self.gt_coverages.append(coverage)
            self.gt_objects_covered_95 += int(coverage >= 0.95)

    def report(self) -> MatrixLightEvaluation:
        precision = self.frame_tp / self.targeted_frames if self.targeted_frames else 0.0
        recall = self.frame_tp / self.gt_danger_frames if self.gt_danger_frames else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return MatrixLightEvaluation(
            self.frames,
            self.gt_danger_frames,
            self.targeted_frames,
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            self.predicted_targets,
            self.matched_targets,
            round(self.matched_targets / self.predicted_targets, 4)
            if self.predicted_targets else 0.0,
            self.gt_objects,
            self.gt_objects_covered_95,
            round(self.gt_objects_covered_95 / self.gt_objects, 4)
            if self.gt_objects else 0.0,
            round(float(np.mean(self.gt_coverages)), 4) if self.gt_coverages else 0.0,
        )


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _mask_bbox_coverage(
    mask: np.ndarray, bbox: BBox, image_shape: tuple[int, int]
) -> float:
    h, w = image_shape
    rows, columns = mask.shape
    x1, y1, x2, y2 = bbox
    target_area = max(1.0, (x2 - x1) * (y2 - y1))
    cell_w, cell_h = w / columns, h / rows
    covered = 0.0
    for row, column in zip(*np.nonzero(mask)):
        cx1, cy1 = column * cell_w, row * cell_h
        cx2, cy2 = cx1 + cell_w, cy1 + cell_h
        covered += max(0.0, min(x2, cx2) - max(x1, cx1)) * max(
            0.0, min(y2, cy2) - max(y1, cy1)
        )
    return min(1.0, covered / target_area)


def render_matrix_light(image: np.ndarray, frame: MatrixLightFrame) -> np.ndarray:
    panel = image.copy()
    h, w = panel.shape[:2]
    overlay = panel.copy()
    cell_w, cell_h = w / frame.grid_mask.shape[1], h / frame.grid_mask.shape[0]
    for row, column in zip(*np.nonzero(frame.grid_mask)):
        p1 = (int(column * cell_w), int(row * cell_h))
        p2 = (int((column + 1) * cell_w), int((row + 1) * cell_h))
        cv2.rectangle(overlay, p1, p2, (0, 220, 255), -1)
    alpha = 0.12 + 0.20 * frame.intensity
    panel = cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0)
    for target in frame.targets:
        x1, y1, x2, y2 = (int(value) for value in target.bbox)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            panel,
            f"BEAM #{target.track_id} cov={target.geometric_coverage:.0%}",
            (max(0, x1), max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return panel
