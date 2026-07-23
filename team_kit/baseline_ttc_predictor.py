"""
Baseline TTC Predictor
======================
A naive baseline that teams are expected to BEAT. Uses only OpenCV (no deep
learning models) — stereo disparity + ROI median depth + frame-to-frame
tracking to estimate Time-To-Collision.

This baseline is intentionally simple:
  - No object detection (uses a fixed ROI in front of ego)
  - Median depth in ROI → "closest obstacle" depth
  - Temporal smoothing over N frames
  - Closing speed via finite difference
  - TTC = depth / closing_speed

Beating this baseline is achievable with proper detection + tracking
(e.g., YOLOv8 + DeepSORT + monocular depth estimation, or stereo + segmentation).

Usage as library:
    from team_kit.baseline_ttc_predictor import BaselineTTCPredictor
    from team_kit.dataset_loader import TripDataset

    ds = TripDataset("./data/T01-Sample")  # or "./data/T01d" for a scored trip
    calib = ds.load_calibration()
    predictor = BaselineTTCPredictor(calib)

    predictions = []
    for frame in ds.iter_frames():
        left = ds.load_left(frame.frame_id)
        right = ds.load_right(frame.frame_id)
        pred_ttc = predictor.predict(frame.frame_id, frame.timestamp, left, right)
        predictions.append({"frame_id": frame.frame_id, "predicted_ttc": pred_ttc})

Usage as CLI:
    python team_kit/baseline_ttc_predictor.py --trip-dir ./data/T01-Sample --output predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import cv2
import numpy as np

# Add project root for team_kit.dataset_loader import when run as CLI
sys.path.insert(0, str(Path(__file__).parent.parent))

from team_kit.dataset_loader import TripDataset

logger = logging.getLogger(__name__)


# ----- Tunable hyperparameters (these are the "knobs" teams can tweak) -----
DEFAULT_ROI = {
    # Fraction of image: ROI is centered horizontally, lower-middle vertically
    "x_start_frac": 0.35,
    "x_end_frac":   0.65,
    "y_start_frac": 0.50,
    "y_end_frac":   0.85,
}
DEFAULT_SGBM = {
    # cv2.StereoSGBM_create's Python bindings are generated straight from
    # its C++ API, which uses camelCase param names -- there has never
    # been a snake_case variant in any OpenCV version. Keep these names
    # exactly as-is (don't "clean up" to snake_case) or StereoSGBM_create
    # will raise "invalid keyword argument".
    "minDisparity":     0,
    "numDisparities":   96,    # Must be divisible by 16
    "blockSize":        11,
    "P1": 8 * 3 * 11 ** 2,
    "P2": 32 * 3 * 11 ** 2,
    "disp12MaxDiff":    1,
    "uniquenessRatio":  10,
    "speckleWindowSize": 100,
    "speckleRange":     32,
    "mode":              cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}
MAX_VALID_DEPTH_M = 80.0
MIN_VALID_DEPTH_M = 1.5
TEMPORAL_WINDOW = 5    # Frames to smooth over
MIN_CLOSING_SPEED = 0.3  # m/s; below this → TTC = inf


class BaselineTTCPredictor:
    """Naive stereo + ROI median TTC predictor."""

    def __init__(
        self,
        calibration: Dict[str, Any],
        roi: Optional[Dict[str, float]] = None,
        sgbm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.calib = calibration
        self.roi_cfg = roi or DEFAULT_ROI
        self.sgbm_params = sgbm_params or DEFAULT_SGBM

        # Derived stereo params
        self.fx = None
        self.baseline = None
        if calibration:
            self.fx = float(calibration["K_left"][0][0])
            self.baseline = float(calibration["baseline_m"])

        # SGBM matcher (created lazily for thread safety in pickling scenarios)
        self._matcher = cv2.StereoSGBM_create(**self.sgbm_params)

        # State for temporal tracking
        self._depth_history: Deque[tuple[float, float]] = deque(maxlen=TEMPORAL_WINDOW)
        self._last_frame_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self,
        frame_id: int,
        timestamp: float,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> float:
        """
        Returns predicted TTC in seconds (inf if no obstacle detected or not approaching).
        """
        if self.fx is None:
            raise RuntimeError("Calibration not loaded — pass `calibration` to __init__.")

        # 1. Compute disparity map
        disparity = self._compute_disparity(left_bgr, right_bgr)

        # 2. Extract ROI median depth
        depth_in_roi = self._roi_median_depth(disparity)

        # 3. Update temporal history
        if depth_in_roi is None or not np.isfinite(depth_in_roi):
            # No reliable obstacle — return inf and keep history (don't reset)
            return float("inf")

        self._depth_history.append((timestamp, depth_in_roi))

        # 4. Estimate closing speed from history
        if len(self._depth_history) < 2:
            return float("inf")

        closing_speed = self._estimate_closing_speed()
        if closing_speed <= MIN_CLOSING_SPEED:
            return float("inf")

        # 5. TTC
        ttc = depth_in_roi / closing_speed
        return float(ttc)

    def reset(self) -> None:
        """Reset temporal state — call between trips."""
        self._depth_history.clear()
        self._last_frame_id = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _compute_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        # SGBM returns int16 with disparity scaled by 16
        disp = self._matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        return disp

    def _disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Convert disparity (pixels) to depth (meters)."""
        # Avoid division by zero / negative disparities
        depth = np.full_like(disparity, np.inf, dtype=np.float32)
        valid = disparity > 0.5
        depth[valid] = (self.fx * self.baseline) / disparity[valid]
        # Clip outliers
        depth[depth < MIN_VALID_DEPTH_M] = np.inf
        depth[depth > MAX_VALID_DEPTH_M] = np.inf
        return depth

    def _roi_median_depth(self, disparity: np.ndarray) -> Optional[float]:
        h, w = disparity.shape
        x0 = int(w * self.roi_cfg["x_start_frac"])
        x1 = int(w * self.roi_cfg["x_end_frac"])
        y0 = int(h * self.roi_cfg["y_start_frac"])
        y1 = int(h * self.roi_cfg["y_end_frac"])

        roi_disp = disparity[y0:y1, x0:x1]
        depth = self._disparity_to_depth(roi_disp)
        valid_depths = depth[np.isfinite(depth)]
        if valid_depths.size < 100:    # Need enough pixels for stable estimate
            return None
        return float(np.median(valid_depths))

    def _estimate_closing_speed(self) -> float:
        """
        Linear regression slope of -depth(t) over time = closing speed (m/s).
        Positive => approaching.
        """
        times = np.array([t for t, _ in self._depth_history])
        depths = np.array([d for _, d in self._depth_history])
        # slope of depth vs time
        # closing_speed = -d(depth)/dt
        if times.max() - times.min() < 1e-3:
            return 0.0
        # Least squares fit
        A = np.vstack([times, np.ones_like(times)]).T
        slope, _ = np.linalg.lstsq(A, depths, rcond=None)[0]
        return float(-slope)


# ====================================================================== #
# CLI: run baseline on a trip and write CSV
# ====================================================================== #
def predict_trip(trip_dir: Path, output_csv: Path, verbose: bool = False) -> None:
    ds = TripDataset(trip_dir)
    calib = ds.load_calibration()
    if not calib:
        raise RuntimeError(f"No calibration in {trip_dir}/kitti/calibration_info.txt")

    predictor = BaselineTTCPredictor(calib)

    rows = []
    n_total = len(ds)
    for i, frame in enumerate(ds.iter_frames()):
        left = ds.load_left(frame.frame_id)
        right = ds.load_right(frame.frame_id)
        try:
            ttc = predictor.predict(frame.frame_id, frame.timestamp, left, right)
        except Exception as e:
            logger.warning("Frame %d failed: %s", frame.frame_id, e)
            ttc = float("inf")

        rows.append({
            "frame_id": frame.frame_id,
            "timestamp": round(frame.timestamp, 3),
            "predicted_ttc": "inf" if not np.isfinite(ttc) else round(ttc, 3),
            # For local reference only when running against one of the 6
            # full-GT practice trips (T01-Sample..T06-Sample) --
            # evaluation.py always loads ground truth independently from a
            # trusted trip directory and IGNORES this column entirely, so
            # editing it has no effect on any score. On the 10 scored
            # trips (T0Xd) this will just show "inf" for every row (no GT
            # available here -- that's the redacted set).
            "ground_truth_ttc": "inf" if not np.isfinite(frame.min_ttc) else round(frame.min_ttc, 3),
        })

        if verbose and (i % 50 == 0 or i == n_total - 1):
            logger.info(
                "Frame %d/%d  pred=%.2f  gt=%.2f",
                i, n_total - 1,
                ttc if np.isfinite(ttc) else 99.0,
                frame.min_ttc if np.isfinite(frame.min_ttc) else 99.0,
            )

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "timestamp", "predicted_ttc", "ground_truth_ttc"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d predictions to %s", len(rows), output_csv)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline TTC predictor (stereo + ROI median).")
    parser.add_argument("--trip-dir", required=True, type=Path,
                        help="Trip directory (e.g. ./data/T01-Sample or ./data/T01d)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV with predicted TTC per frame")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    predict_trip(args.trip_dir, args.output, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
