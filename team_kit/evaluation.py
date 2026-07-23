"""
TTC Prediction Evaluator
========================
Compares team-submitted predictions against ground truth for all 3
hackathon challenges:
  - Challenge 1: Collision Risk Monitor (predicted_ttc)
  - Challenge 2: Driver Intelligence Platform (predicted_driver_state)
  - Challenge 3: Fleet Safe Driving Score (predicted_risk_score, aggregated per trip)

A team may attempt any subset of the 3 challenges (see README's "Chọn 1
(hoặc kết hợp)"). Each challenge is scored independently and only
reported if the corresponding CSV column is present with usable data --
attempting only Challenge 1 produces a report with no Challenge 2/3
sections, exactly like before this file grew multi-challenge support.

SECURITY NOTE: ground truth is ALWAYS loaded independently from a trusted
trip directory (--data-dir / --trip-dir), never from the submitted
predictions CSV, even if that CSV happens to contain a
'ground_truth_ttc' column (e.g. one written by baseline_ttc_predictor.py
for local reference). That column, if present, is ignored entirely by
this script. --data-dir/--trip-dir is therefore required.

Input format (CSV per trip):
    frame_id,timestamp,predicted_ttc,predicted_driver_state,predicted_risk_score
    0,0.000,inf,alert,5
    1,0.050,inf,alert,5
    ...
    1798,89.900,2.3,drowsy,67
(Only frame_id is strictly required. predicted_ttc defaults to inf if
missing/unparseable. predicted_driver_state/predicted_risk_score columns
are optional -- omit them entirely if you're only attempting Challenge 1.
Extra columns, including a ground_truth_ttc your own tooling may have
written, are fine to leave in the file -- they're simply ignored.)

Metrics computed:
  Challenge 1 (TTC):
    - MAE overall (Mean Absolute Error in seconds)
    - MAE in critical zone (gt_ttc < 3.0s — what matters for AEB)
    - RMSE in critical zone
    - Inverse-TTC MAE (1/TTC) — more stable metric, since 1/TTC ~ urgency
    - Classification: predicted_critical (pred<2s) vs actual_critical (gt<2s)
      → Precision, Recall, F1, FPR
    - composite_score = 40% MAE-critical + 30% F1 + 30% inv-TTC MAE
  Challenge 2 (driver state, 5-class):
    - Accuracy, per-class F1, macro-F1 (averaged over classes present in
      that trip's ground truth)
    - composite_score = 50% accuracy + 50% macro-F1
  Challenge 3 (fleet safe driving score, per trip):
    - Reconstructs the organizer's actual rule-based formula from
      package_organizer-v3-remote/src/analytics/behavior.py
      (`BehaviorScorer.aggregate()`), term by term:
        harsh_brake_count*3.0 + harsh_accel_count*2.0 + harsh_corner_count*2.0
        + near_miss_count*5.0 + speeding_pct_time*0.15
      * harsh_brake/accel/corner counts and speeding_pct_time are computed
        directly from the trusted trip's ego kinematics
        (longitudinal_accel/lateral_accel/speed_kmh) and
        metadata.speed_limit_kmh -- none of these are redacted at any
        trip (T0Xd included), and none of them depend on the team's
        model at all, so they're identical for every submission on a
        given trip.
      * near_miss_count is computed from the TEAM'S OWN predicted_ttc
        column (frames with predicted_ttc < NEAR_MISS_TTC_SEC) -- this is
        the one genuinely model-dependent term, and it's already required
        for Challenge 1, so no new column is needed for it either.
      * predicted_risk_score's actual values are NOT read into this
        formula -- the column's presence is still what gates whether
        Challenge 3 is reported at all (an explicit opt-in, matching the
        submission format/checklist), but the score itself is now fully
        determined by Challenge 1's predicted_ttc plus trip facts that
        are equal for every team.
    - The ONE term from the real formula this does NOT reconstruct is
      tailgating_pct_time*0.10, because it depends on `headway_sec`
      (redacted) and nothing in the current submission format lets a team
      predict it. composite_score will therefore be systematically a bit
      higher than a perfect predictor's true score whenever a trip has
      real tailgating -- a much smaller gap than the old mean-risk
      approximation, but not zero. Add an optional predicted_headway_sec
      column (and a matching tailgating reconstruction here) if closing
      this last gap ever matters.
    - composite_score = 100 - CHALLENGE3_ERROR_SCALE * abs_error

Usage:
    # Evaluate one trip
    python team_kit/evaluation.py \
        --predictions ./predictions/T01-Sample.csv \
        --trip-dir ./data/T01-Sample

    # Evaluate batch (one CSV per trip)
    python team_kit/evaluation.py \
        --predictions ./predictions/ \
        --data-dir ./data \
        --output ./evaluation_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root for team_kit imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# Thresholds (defaults — tunable for different hackathon scoring rules)
CRITICAL_TTC_SEC = 3.0       # gt_ttc < this = critical zone for MAE computation
DANGER_TTC_SEC = 2.0         # gt_ttc < this = "should trigger AEB" (binary classification)
INF_CLIP_VALUE = 99.0        # Replace inf with this for finite math (visualizations only)

DRIVER_STATE_CLASSES = ["alert", "drowsy", "yawning", "distracted", "microsleep"]
CHALLENGE3_ERROR_SCALE = 2.0  # abs_error (safe-score points) -> composite penalty; 0->100, 50->0

# Challenge 3 reconstruction constants -- these are exact copies of the
# thresholds in package_organizer-v3-remote/src/analytics/behavior.py
# (BehaviorScorer), kept in sync deliberately so the deterministic
# components below match the ground-truth formula bit-for-bit rather than
# approximating it.
HARSH_BRAKE_G = 0.40             # long_accel < -HARSH_BRAKE_G * G_MS2 = harsh braking
HARSH_ACCEL_G = 0.35             # long_accel > HARSH_ACCEL_G * G_MS2 = harsh acceleration
HARSH_LATERAL_G = 0.30           # |lat_accel| > HARSH_LATERAL_G * G_MS2 = harsh cornering
G_MS2 = 9.81
NEAR_MISS_TTC_SEC = 1.5          # min_ttc < this = a near-miss frame
SPEEDING_TOLERANCE_KMH = 5.0     # speed_kmh > speed_limit_kmh + this = speeding


# ====================================================================== #
# Result containers
# ====================================================================== #
@dataclass
class TripMetrics:
    trip_id: str
    n_frames: int
    n_finite_gt: int             # Frames where ground truth has a target
    n_critical_zone: int         # Frames where gt_ttc < CRITICAL_TTC_SEC
    n_danger_zone: int           # Frames where gt_ttc < DANGER_TTC_SEC
    mae_overall: float           # MAE on finite-gt frames
    mae_critical: float          # MAE on critical zone (the metric that matters)
    rmse_critical: float
    inv_ttc_mae: float           # MAE on 1/TTC (more robust)
    precision: float             # P(pred_danger | gt_danger)
    recall: float
    f1: float
    false_positive_rate: float
    accuracy: float
    composite_score: float       # Combined metric — see _composite_score()


@dataclass
class Challenge2Metrics:
    """Driver state classification (5-class) — per trip."""
    trip_id: str
    n_frames_scored: int
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    composite_score: float


@dataclass
class Challenge3Metrics:
    """Fleet Safe Driving Score (trip-level) -- see compute_challenge3_metrics()
    for how predicted_safe_score is reconstructed."""
    trip_id: str
    predicted_safe_score: float
    true_safe_score: float
    abs_error: float
    composite_score: float
    # Breakdown for transparency (all penalty terms except tailgating --
    # see module docstring):
    near_miss_count: int
    harsh_brake_count: int
    harsh_accel_count: int
    harsh_corner_count: int
    speeding_pct_time: float


@dataclass
class EvaluationReport:
    n_trips: int
    per_trip: List[TripMetrics]
    overall_mae_critical: float
    overall_inv_ttc_mae: float
    overall_f1: float
    overall_composite_score: float
    ranking: List[Tuple[str, float]]  # (trip_id, composite_score) sorted desc
    per_trip_challenge2: List[Challenge2Metrics]
    overall_challenge2_composite: Optional[float]
    per_trip_challenge3: List[Challenge3Metrics]
    overall_challenge3_composite: Optional[float]


# ====================================================================== #
# CSV parsing
# ====================================================================== #
def _parse_ttc(value: Any) -> float:
    """Parse a TTC value from CSV cell, handling 'inf', '-inf', empty."""
    if value is None or value == "":
        return float("inf")
    s = str(value).strip().lower()
    if s in ("inf", "infinity", "+inf"):
        return float("inf")
    if s in ("-inf", "-infinity", "nan"):
        return float("inf")  # Treat NaN as "no prediction" → inf
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _parse_float(value: Any) -> Optional[float]:
    """Parse an optional numeric CSV cell. Returns None if missing/blank/unparseable
    (as opposed to _parse_ttc, which always returns a number -- here None means
    "team didn't predict this", which callers need to distinguish from 0.0)."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class FramePrediction:
    predicted_ttc: float
    predicted_driver_state: Optional[str]
    predicted_risk_score: Optional[float]


def load_predictions(csv_path: Path) -> Dict[int, FramePrediction]:
    """Load predictions CSV. Returns {frame_id: FramePrediction}.

    predicted_driver_state / predicted_risk_score are optional columns --
    if a column is entirely absent from the CSV header, every frame gets
    None for that field (challenge not attempted for this trip). If the
    column is present but a specific cell is blank, that single frame is
    treated as "not predicted" for that field.

    SECURITY-CRITICAL: this deliberately does NOT read a
    'ground_truth_ttc' column from the submitted CSV, even if one is
    present (e.g. because baseline_ttc_predictor.py wrote one for the
    team's own local reference). Ground truth for scoring must ALWAYS
    come from an independently-held, organizer-controlled trip directory
    (--data-dir / --trip-dir), never from anything the team themselves
    produced -- see evaluate()'s docstring for the vulnerability this
    closes (found in project review: the previous version only fell back
    to trusted GT when EVERY value in the CSV's own ground_truth_ttc
    column was non-finite, so a team could submit any finite values
    there -- including a copy of their own predicted_ttc column -- and
    the evaluator would score against that instead of real data).
    """
    out: Dict[int, FramePrediction] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_driver_state = "predicted_driver_state" in fieldnames
        has_risk_score = "predicted_risk_score" in fieldnames

        for row in reader:
            try:
                fid = int(row["frame_id"])
            except (KeyError, ValueError):
                continue

            state: Optional[str] = None
            if has_driver_state:
                raw = (row.get("predicted_driver_state") or "").strip().lower()
                state = raw if raw else None

            risk: Optional[float] = None
            if has_risk_score:
                risk = _parse_float(row.get("predicted_risk_score"))

            out[fid] = FramePrediction(
                predicted_ttc=_parse_ttc(row.get("predicted_ttc")),
                predicted_driver_state=state,
                predicted_risk_score=risk,
            )
    return out


@dataclass
class TripGroundTruth:
    ttc: Dict[int, float]
    driver_state: Dict[int, str]
    safe_driving_score: Optional[float]
    # Deterministic Challenge 3 components -- computed straight from ego
    # kinematics + speed_limit_kmh, which stay unredacted at every trip
    # (T0Xd included). Same for every team on a given trip; not a
    # prediction of anything.
    harsh_brake_count: int
    harsh_accel_count: int
    harsh_corner_count: int
    speeding_pct_time: float


def load_ground_truth_from_trip(trip_dir: Path) -> TripGroundTruth:
    """Load GT for all 3 challenges directly from the trusted trip JSON."""
    from team_kit.dataset_loader import TripDataset
    ds = TripDataset(trip_dir)
    ttc: Dict[int, float] = {}
    driver_state: Dict[int, str] = {}
    frames = ds.frame_records
    for f in frames:
        ttc[f.frame_id] = f.min_ttc
        driver_state[f.frame_id] = f.driver_state

    harsh_brake = harsh_accel = harsh_corner = 0
    speeding_frames = 0
    speed_limit_kmh = float(ds.metadata.get("speed_limit_kmh", 0.0))
    for f in frames:
        if f.longitudinal_accel < -HARSH_BRAKE_G * G_MS2:
            harsh_brake += 1
        if f.longitudinal_accel > HARSH_ACCEL_G * G_MS2:
            harsh_accel += 1
        if abs(f.lateral_accel) > HARSH_LATERAL_G * G_MS2:
            harsh_corner += 1
        if f.speed_kmh > speed_limit_kmh + SPEEDING_TOLERANCE_KMH:
            speeding_frames += 1
    speeding_pct_time = 100.0 * speeding_frames / len(frames) if frames else 0.0

    return TripGroundTruth(
        ttc=ttc,
        driver_state=driver_state,
        safe_driving_score=ds.trip_aggregate.get("safe_driving_score"),
        harsh_brake_count=harsh_brake,
        harsh_accel_count=harsh_accel,
        harsh_corner_count=harsh_corner,
        speeding_pct_time=speeding_pct_time,
    )


# ====================================================================== #
# Challenge 1 — TTC metrics
# ====================================================================== #
def compute_trip_metrics(
    trip_id: str,
    pairs: Dict[int, Tuple[float, float]],
) -> TripMetrics:
    """Compute all metrics for one trip given (pred, gt) per frame."""
    preds = np.array([p for p, _ in pairs.values()])
    gts = np.array([g for _, g in pairs.values()])

    n_frames = len(pairs)
    gt_finite = np.isfinite(gts)
    n_finite_gt = int(gt_finite.sum())
    critical_mask = gts < CRITICAL_TTC_SEC
    n_critical = int(critical_mask.sum())
    danger_mask = gts < DANGER_TTC_SEC
    n_danger = int(danger_mask.sum())

    # ---- MAE Overall (finite GT frames; clip pred to range to avoid inf-inf) ----
    def _safe_diff(p: np.ndarray, g: np.ndarray) -> np.ndarray:
        # Replace inf with clip value to compute finite diff for MAE
        p_c = np.where(np.isfinite(p), p, INF_CLIP_VALUE)
        g_c = np.where(np.isfinite(g), g, INF_CLIP_VALUE)
        return np.abs(p_c - g_c)

    mae_overall = float(_safe_diff(preds[gt_finite], gts[gt_finite]).mean()) \
                  if n_finite_gt else float("nan")

    # ---- MAE in critical zone (the metric that matters for AEB) ----
    if n_critical > 0:
        diff_crit = _safe_diff(preds[critical_mask], gts[critical_mask])
        mae_critical = float(diff_crit.mean())
        rmse_critical = float(np.sqrt((diff_crit ** 2).mean()))
    else:
        mae_critical = float("nan")
        rmse_critical = float("nan")

    # ---- Inverse-TTC MAE (1/TTC is bounded and more well-behaved) ----
    inv_p = 1.0 / np.where(preds > 0.1, preds, 0.1)
    inv_g = 1.0 / np.where(gts > 0.1, gts, 0.1)
    # 1/inf → 0 naturally
    inv_p[~np.isfinite(preds)] = 0.0
    inv_g[~np.isfinite(gts)] = 0.0
    inv_ttc_mae = float(np.abs(inv_p - inv_g).mean())

    # ---- Binary classification (predicted danger vs actual danger) ----
    pred_danger = preds < DANGER_TTC_SEC
    gt_danger = gts < DANGER_TTC_SEC
    tp = int(((pred_danger) & (gt_danger)).sum())
    fp = int(((pred_danger) & (~gt_danger)).sum())
    fn = int(((~pred_danger) & (gt_danger)).sum())
    tn = int(((~pred_danger) & (~gt_danger)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if n_frames > 0 else 0.0

    # ---- Composite Score (0-100, higher = better) ----
    composite = _composite_score(mae_critical, inv_ttc_mae, f1)

    return TripMetrics(
        trip_id=trip_id,
        n_frames=n_frames,
        n_finite_gt=n_finite_gt,
        n_critical_zone=n_critical,
        n_danger_zone=n_danger,
        mae_overall=round(mae_overall, 3) if not math.isnan(mae_overall) else -1.0,
        mae_critical=round(mae_critical, 3) if not math.isnan(mae_critical) else -1.0,
        rmse_critical=round(rmse_critical, 3) if not math.isnan(rmse_critical) else -1.0,
        inv_ttc_mae=round(inv_ttc_mae, 4),
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        false_positive_rate=round(fpr, 3),
        accuracy=round(accuracy, 3),
        composite_score=round(composite, 1),
    )


def _composite_score(mae_critical: float, inv_ttc_mae: float, f1: float) -> float:
    """
    Combined score (0-100, higher = better).
    Weights: 40% MAE-critical, 30% F1, 30% inv-TTC MAE.
    """
    # Normalize MAE-critical: 0s MAE → 100, 5s MAE → 0
    if math.isnan(mae_critical):
        mae_score = 50.0  # No critical frames to evaluate
    else:
        mae_score = max(0, 100 - 20 * mae_critical)

    # F1 already in [0, 1]
    f1_score = 100 * f1

    # inv_ttc_mae: 0 → 100, 0.5 → 0
    inv_score = max(0, 100 - 200 * inv_ttc_mae)

    return 0.40 * mae_score + 0.30 * f1_score + 0.30 * inv_score


# ====================================================================== #
# Challenge 2 — driver state classification metrics
# ====================================================================== #
def compute_challenge2_metrics(
    trip_id: str,
    pairs: Dict[int, Tuple[Optional[str], str]],
) -> Optional[Challenge2Metrics]:
    """pairs: frame_id -> (predicted_state_or_None, true_state).

    Returns None if no frame in this trip had a (non-blank)
    predicted_driver_state -- i.e. the team didn't attempt Challenge 2 on
    this trip, so it's omitted from the report rather than scored as 0.
    """
    scored = {fid: (p, g) for fid, (p, g) in pairs.items() if p is not None}
    if not scored:
        return None

    n = len(scored)
    correct = sum(1 for p, g in scored.values() if p == g)
    accuracy = correct / n

    per_class_f1: Dict[str, float] = {}
    for c in DRIVER_STATE_CLASSES:
        tp = sum(1 for p, g in scored.values() if p == c and g == c)
        fp = sum(1 for p, g in scored.values() if p == c and g != c)
        fn = sum(1 for p, g in scored.values() if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[c] = round(f1, 3)

    # Macro-average only over classes that actually occur in this trip's
    # ground truth -- most trips don't hit all 5 driver states, and
    # averaging in classes with zero support would just inject noise.
    present_classes = {g for _, g in scored.values()}
    relevant_f1 = [per_class_f1[c] for c in present_classes if c in per_class_f1]
    macro_f1 = float(np.mean(relevant_f1)) if relevant_f1 else 0.0

    composite = 100.0 * (0.5 * accuracy + 0.5 * macro_f1)

    return Challenge2Metrics(
        trip_id=trip_id,
        n_frames_scored=n,
        accuracy=round(accuracy, 3),
        macro_f1=round(macro_f1, 3),
        per_class_f1=per_class_f1,
        composite_score=round(composite, 1),
    )


# ====================================================================== #
# Challenge 3 — Fleet Safe Driving Score
# ====================================================================== #
def compute_challenge3_metrics(
    trip_id: str,
    attempted: bool,
    predicted_ttc_values: List[float],
    gt: "TripGroundTruth",
    true_safe_score: Optional[float],
) -> Optional[Challenge3Metrics]:
    """Reconstructs the trip-level Fleet Safe Driving Score using the
    organizer's real formula (see module docstring for the full
    breakdown): deterministic penalty terms come straight from `gt`
    (harsh-brake/accel/corner counts, speeding%), and near_miss_count
    comes from the team's own predicted_ttc (already required for
    Challenge 1). predicted_risk_score's values are not used here --
    `attempted` (whether that column was present/non-blank) is what gates
    whether Challenge 3 is reported at all.

    Returns None if the team didn't attempt Challenge 3 for this trip, or
    if ground truth has no safe_driving_score (shouldn't happen against a
    trusted trip directory, but guards against a malformed/incomplete
    trip JSON).
    """
    if not attempted or true_safe_score is None:
        return None

    near_miss_count = sum(
        1 for t in predicted_ttc_values if math.isfinite(t) and t < NEAR_MISS_TTC_SEC
    )

    # Mirrors BehaviorScorer.aggregate()'s `penalties` sum, minus the
    # tailgating term (no headway/distance prediction is submitted -- see
    # module docstring).
    penalties = (
        gt.harsh_brake_count * 3.0 +
        gt.harsh_accel_count * 2.0 +
        gt.harsh_corner_count * 2.0 +
        near_miss_count * 5.0 +
        (gt.speeding_pct_time / 100.0) * 15.0
    )
    predicted_safe = max(0.0, min(100.0, 100.0 - penalties))
    abs_error = abs(predicted_safe - float(true_safe_score))
    composite = max(0.0, 100.0 - CHALLENGE3_ERROR_SCALE * abs_error)

    return Challenge3Metrics(
        trip_id=trip_id,
        predicted_safe_score=round(predicted_safe, 1),
        true_safe_score=round(float(true_safe_score), 1),
        abs_error=round(abs_error, 1),
        composite_score=round(composite, 1),
        near_miss_count=near_miss_count,
        harsh_brake_count=gt.harsh_brake_count,
        harsh_accel_count=gt.harsh_accel_count,
        harsh_corner_count=gt.harsh_corner_count,
        speeding_pct_time=round(gt.speeding_pct_time, 2),
    )


# ====================================================================== #
# Batch evaluation
# ====================================================================== #
def evaluate(
    predictions_dir: Path,
    data_dir: Optional[Path],
    output_json: Optional[Path],
) -> EvaluationReport:
    """Ground truth is ALWAYS loaded from `data_dir` (an organizer-trusted
    trip directory), never from the submitted predictions CSV -- see
    load_predictions()'s docstring for the vulnerability this closes.
    `data_dir` is therefore effectively required for any scoring that
    should be trusted; omitting it only makes sense for a quick informal
    smoke-test where you already know there's no real GT to compare to.
    """
    if data_dir is None:
        raise RuntimeError(
            "No --data-dir/--trip-dir given. Ground truth is always loaded "
            "from a trusted trip directory, never from the submitted CSV "
            "-- pass --trip-dir (single trip) or --data-dir (batch)."
        )

    # Discover prediction CSVs
    if predictions_dir.is_file():
        csv_files = [predictions_dir]
    else:
        csv_files = sorted(predictions_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {predictions_dir}")

    per_trip: List[TripMetrics] = []
    per_trip_c2: List[Challenge2Metrics] = []
    per_trip_c3: List[Challenge3Metrics] = []

    for csv_path in csv_files:
        # Extract trip_id from filename (e.g., T01d.csv -> T01d,
        # T01-Sample.csv -> T01-Sample)
        trip_id = csv_path.stem
        if not re.match(r"^T\d+d?(-Sample)?$", trip_id):
            logger.warning("Skipping non-trip CSV: %s", csv_path.name)
            continue

        preds = load_predictions(csv_path)
        if not preds:
            logger.warning("Empty predictions in %s", csv_path)
            continue

        gt = load_ground_truth_from_trip(data_dir / trip_id)

        # ---- Challenge 1: TTC ----
        pairs = {
            fid: (p.predicted_ttc, gt.ttc.get(fid, float("inf")))
            for fid, p in preds.items()
        }
        metrics = compute_trip_metrics(trip_id, pairs)
        per_trip.append(metrics)
        logger.info(
            "%s: [C1] MAE-crit=%.2fs  F1=%.2f  composite=%.1f  (n_critical=%d)",
            trip_id, metrics.mae_critical, metrics.f1,
            metrics.composite_score, metrics.n_critical_zone,
        )

        # ---- Challenge 2: driver state ----
        state_pairs = {
            fid: (p.predicted_driver_state, gt.driver_state.get(fid, "unknown"))
            for fid, p in preds.items()
        }
        c2 = compute_challenge2_metrics(trip_id, state_pairs)
        if c2 is not None:
            per_trip_c2.append(c2)
            logger.info(
                "%s: [C2] accuracy=%.2f  macro-F1=%.2f  composite=%.1f",
                trip_id, c2.accuracy, c2.macro_f1, c2.composite_score,
            )

        # ---- Challenge 3: fleet safe driving score ----
        # predicted_risk_score only gates whether Challenge 3 is attempted
        # here -- see compute_challenge3_metrics()'s docstring for why its
        # values aren't read into the reconstruction formula.
        attempted_c3 = any(p.predicted_risk_score is not None for p in preds.values())
        predicted_ttc_values = [p.predicted_ttc for p in preds.values()]
        c3 = compute_challenge3_metrics(
            trip_id, attempted_c3, predicted_ttc_values, gt, gt.safe_driving_score
        )
        if c3 is not None:
            per_trip_c3.append(c3)
            logger.info(
                "%s: [C3] predicted=%.1f  true=%.1f  composite=%.1f",
                trip_id, c3.predicted_safe_score, c3.true_safe_score, c3.composite_score,
            )

    if not per_trip:
        raise RuntimeError("No valid trip evaluations produced.")

    # Overall (macro-averaged, equal weight per trip)
    finite_mae_crit = [m.mae_critical for m in per_trip if m.mae_critical >= 0]
    overall_mae_crit = float(np.mean(finite_mae_crit)) if finite_mae_crit else -1.0
    overall_inv = float(np.mean([m.inv_ttc_mae for m in per_trip]))
    overall_f1 = float(np.mean([m.f1 for m in per_trip]))
    overall_composite = float(np.mean([m.composite_score for m in per_trip]))

    overall_c2 = float(np.mean([m.composite_score for m in per_trip_c2])) if per_trip_c2 else None
    overall_c3 = float(np.mean([m.composite_score for m in per_trip_c3])) if per_trip_c3 else None

    ranking = sorted(
        [(m.trip_id, m.composite_score) for m in per_trip],
        key=lambda x: -x[1],
    )

    report = EvaluationReport(
        n_trips=len(per_trip),
        per_trip=per_trip,
        overall_mae_critical=round(overall_mae_crit, 3),
        overall_inv_ttc_mae=round(overall_inv, 4),
        overall_f1=round(overall_f1, 3),
        overall_composite_score=round(overall_composite, 1),
        ranking=ranking,
        per_trip_challenge2=per_trip_c2,
        overall_challenge2_composite=round(overall_c2, 1) if overall_c2 is not None else None,
        per_trip_challenge3=per_trip_c3,
        overall_challenge3_composite=round(overall_c3, 1) if overall_c3 is not None else None,
    )

    if output_json is not None:
        doc = {
            "n_trips": report.n_trips,
            "overall_mae_critical": report.overall_mae_critical,
            "overall_inv_ttc_mae": report.overall_inv_ttc_mae,
            "overall_f1": report.overall_f1,
            "overall_composite_score": report.overall_composite_score,
            "ranking": [{"trip_id": tid, "score": s} for tid, s in report.ranking],
            "per_trip": [asdict(m) for m in per_trip],
            "challenge2": {
                "overall_composite_score": report.overall_challenge2_composite,
                "per_trip": [asdict(m) for m in per_trip_c2],
            } if per_trip_c2 else None,
            "challenge3": {
                "overall_composite_score": report.overall_challenge3_composite,
                "per_trip": [asdict(m) for m in per_trip_c3],
            } if per_trip_c3 else None,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(doc, indent=2))
        logger.info("Report written to %s", output_json)

    return report


# ====================================================================== #
# Pretty printing
# ====================================================================== #
def print_report(report: EvaluationReport) -> None:
    print()
    print("=" * 78)
    print("EVALUATION REPORT - Challenge 1: Collision Risk Monitor (TTC)")
    print("=" * 78)
    print(f"Trips evaluated:        {report.n_trips}")
    print(f"Overall MAE (critical): {report.overall_mae_critical:.3f}s")
    print(f"Overall inv-TTC MAE:    {report.overall_inv_ttc_mae:.4f}")
    print(f"Overall F1:             {report.overall_f1:.3f}")
    print(f"Overall composite:      {report.overall_composite_score:.1f} / 100")
    print()
    print(f"{'Trip':<6} {'n_crit':<8} {'MAE-crit':<10} {'F1':<7} {'FPR':<7} {'Composite':<10}")
    print("-" * 78)
    for m in report.per_trip:
        mae_str = f"{m.mae_critical:.3f}s" if m.mae_critical >= 0 else "  N/A "
        print(f"{m.trip_id:<6} {m.n_critical_zone:<8} {mae_str:<10} "
              f"{m.f1:<7.3f} {m.false_positive_rate:<7.3f} {m.composite_score:<10.1f}")
    print()
    print("RANKING (by composite score, higher = better):")
    for i, (tid, s) in enumerate(report.ranking, start=1):
        print(f"  #{i}  {tid}  ->  {s:.1f}")

    if report.per_trip_challenge2:
        print()
        print("=" * 78)
        print("Challenge 2: Driver Intelligence Platform (driver state)")
        print("=" * 78)
        print(f"Overall composite:      {report.overall_challenge2_composite:.1f} / 100")
        print()
        print(f"{'Trip':<14} {'n_scored':<10} {'Accuracy':<10} {'Macro-F1':<10} {'Composite':<10}")
        print("-" * 78)
        for m in report.per_trip_challenge2:
            print(f"{m.trip_id:<14} {m.n_frames_scored:<10} {m.accuracy:<10.3f} "
                  f"{m.macro_f1:<10.3f} {m.composite_score:<10.1f}")

    if report.per_trip_challenge3:
        print()
        print("=" * 78)
        print("Challenge 3: Fleet Safe Driving Score")
        print("=" * 78)
        print(f"Overall composite:      {report.overall_challenge3_composite:.1f} / 100")
        print()
        print(f"{'Trip':<14} {'Predicted':<10} {'True':<10} {'AbsErr':<10} {'Composite':<10}")
        print("-" * 78)
        for m in report.per_trip_challenge3:
            print(f"{m.trip_id:<14} {m.predicted_safe_score:<10.1f} {m.true_safe_score:<10.1f} "
                  f"{m.abs_error:<10.1f} {m.composite_score:<10.1f}")
        print()
        print("Breakdown (deterministic from trip facts, except near_miss = your own predicted_ttc):")
        print(f"{'Trip':<14} {'near_miss':<10} {'harsh_brk':<10} {'harsh_acc':<10} {'harsh_crn':<10} {'speeding%':<10}")
        print("-" * 78)
        for m in report.per_trip_challenge3:
            print(f"{m.trip_id:<14} {m.near_miss_count:<10} {m.harsh_brake_count:<10} "
                  f"{m.harsh_accel_count:<10} {m.harsh_corner_count:<10} {m.speeding_pct_time:<10.1f}")


# ====================================================================== #
# CLI
# ====================================================================== #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TTC / driver-state / safe-driving-score predictions vs ground truth.",
    )
    parser.add_argument("--predictions", required=True, type=Path,
                        help="CSV file (single trip) or directory of CSVs (batch)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Trip data root containing trusted ground truth (required)")
    parser.add_argument("--trip-dir", type=Path, default=None,
                        help="(Single-trip alias) trip directory containing trusted "
                             "ground truth (required if --data-dir not given)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve data dir for GT fallback
    data_dir = args.data_dir
    if data_dir is None and args.trip_dir is not None:
        data_dir = args.trip_dir.parent

    report = evaluate(args.predictions, data_dir, args.output)
    print_report(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
