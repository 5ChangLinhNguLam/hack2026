"""
Dataset Loader
==============
Lightweight loader for the CARLA + DMD hackathon dataset.
Teams use this as the starting point — no need to parse the JSON manually.

Note: driver-related fields (e.g. nthu_subject_id) are still named
"nthu_*" for historical reasons -- the actual driver footage is sourced
from DMD, not the original NTHU-DDD dataset. Schema is unchanged either
way; see the main project README's naming note for the full story.

Quick start:
    from team_kit.dataset_loader import TripDataset

    ds = TripDataset("./data/T01-Sample")  # or "./data/T01d" for a scored trip
    print(f"Frames: {len(ds)}, duration: {ds.metadata['duration_sec']}s")

    # Iterate frames
    for frame in ds.iter_frames():
        left = ds.load_left(frame.frame_id)
        driver = ds.load_driver(frame.frame_id)
        gt_ttc = frame.min_ttc
        # ... your code here

    # Pandas view for analytics
    df = ds.frames_df
    drowsy_frames = df[df["driver_state"] == "drowsy"]

    # Standard per-frame KITTI calibration (P0-P3, R0_rect, Tr_*)
    kitti_calib = ds.load_frame_calibration(frame.frame_id)
    P2 = kitti_calib["P2"]   # (3, 4)
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
import pandas as pd


# ====================================================================== #
# KITTI calibration parsing
# ====================================================================== #
def parse_kitti_calib(path: Path) -> Dict[str, np.ndarray]:
    """Parse a standard per-frame KITTI calib .txt file.

    Handles the usual KITTI keys -- P0..P3 (3x4 projection matrices),
    R0_rect (3x3 rectification), Tr_velo_to_cam / Tr_imu_to_velo (3x4
    extrinsics) -- reshaping each into its conventional matrix shape.
    Any other/unknown key is kept as a flat 1D array so unexpected fields
    don't get silently dropped.

    This is distinct from TripDataset.load_calibration(), which reads
    the synthetic-sim global `kitti/calibration_info.txt` (K_left /
    baseline_m) used by the stereo baseline predictor. Use this parser
    when you need the standard KITTI matrices themselves, e.g. to reuse
    third-party KITTI 3D-detection code against `kitti/label_2/`.
    """
    calib: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, values = line.partition(":")
        key = key.strip()
        nums = np.array([float(v) for v in values.split()], dtype=np.float64)
        if key.startswith("P") and nums.size == 12:
            calib[key] = nums.reshape(3, 4)
        elif key == "R0_rect" and nums.size == 9:
            calib[key] = nums.reshape(3, 3)
        elif key in ("Tr_velo_to_cam", "Tr_imu_to_velo") and nums.size == 12:
            calib[key] = nums.reshape(3, 4)
        else:
            calib[key] = nums
    return calib


# ====================================================================== #
# Data classes
# ====================================================================== #
@dataclass
class FrameRecord:
    """Per-frame structured view of the JSON record."""
    frame_id: int
    timestamp: float
    speed_kmh: float
    longitudinal_accel: float
    lateral_accel: float

    # Driver state (from NTHU compositing)
    driver_state: str
    alertness_score: float
    eye_state: str
    head_pose: str
    mouth_state: str
    nthu_subject_id: str

    # Ground truth metrics
    min_ttc: float                # ∞ if no target in collision cone
    headway_sec: float
    base_risk: float
    driver_factor: float
    final_risk_score: float

    # Behavior flags
    is_harsh_brake: bool
    is_harsh_accel: bool
    is_harsh_corner: bool
    is_speeding: bool
    is_tailgating: bool

    # Targets (full TTC info for each detected actor)
    targets: List[Dict[str, Any]]
    events_active: List[Dict[str, Any]]


# ====================================================================== #
# Single-trip loader
# ====================================================================== #
class TripDataset:
    """Loads one trip's full output: metadata, frames, images, driver, depth."""

    def __init__(self, trip_dir: str | Path) -> None:
        self.trip_dir = Path(trip_dir)
        if not self.trip_dir.is_dir():
            raise FileNotFoundError(f"Trip directory not found: {self.trip_dir}")

        self.trip_id = self.trip_dir.name
        self._doc = self._load_json()

        self.metadata: Dict[str, Any] = self._doc.get("metadata", {})
        self.trip_aggregate: Dict[str, Any] = self._doc.get("trip_aggregate", {})
        self.driver_summary: Dict[str, Any] = self._doc.get("driver_summary", {})
        self.events_log: List[Dict[str, Any]] = self._doc.get("events_log", [])

        self._raw_frames: List[Dict[str, Any]] = self._doc.get("frames", [])
        self._frame_records: Optional[List[FrameRecord]] = None  # Lazy
        self._frames_df_cache: Optional[pd.DataFrame] = None     # Lazy

        # Image directories
        self.image_left_dir = self.trip_dir / "kitti" / "image_2"
        self.image_right_dir = self.trip_dir / "kitti" / "image_3"
        self.depth_dir = self.trip_dir / "kitti" / "depth"
        self.driver_dir = self.trip_dir / "driver"
        self.label_dir = self.trip_dir / "kitti" / "label_2"
        self.calib_dir = self.trip_dir / "kitti" / "calib"

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._raw_frames)

    def __getitem__(self, idx: int) -> FrameRecord:
        return self.frame_records[idx]

    def iter_frames(self) -> Iterator[FrameRecord]:
        for fr in self.frame_records:
            yield fr

    @property
    def frame_records(self) -> List[FrameRecord]:
        """Lazy-parsed list of FrameRecord."""
        if self._frame_records is None:
            self._frame_records = [self._parse_frame(f) for f in self._raw_frames]
        return self._frame_records

    @property
    def frames_df(self) -> pd.DataFrame:
        """Flattened pandas DataFrame for analytics (one row per frame)."""
        if self._frames_df_cache is not None:
            return self._frames_df_cache

        rows = []
        for fr in self.frame_records:
            ttc_finite = fr.min_ttc if math.isfinite(fr.min_ttc) else 99.0
            head_finite = fr.headway_sec if math.isfinite(fr.headway_sec) else 99.0
            rows.append({
                "frame_id": fr.frame_id,
                "timestamp": fr.timestamp,
                "speed_kmh": fr.speed_kmh,
                "longitudinal_accel": fr.longitudinal_accel,
                "lateral_accel": fr.lateral_accel,
                "driver_state": fr.driver_state,
                "alertness_score": fr.alertness_score,
                "eye_state": fr.eye_state,
                "head_pose": fr.head_pose,
                "mouth_state": fr.mouth_state,
                "min_ttc": ttc_finite,
                "min_ttc_inf": not math.isfinite(fr.min_ttc),
                "headway_sec": head_finite,
                "base_risk": fr.base_risk,
                "driver_factor": fr.driver_factor,
                "final_risk_score": fr.final_risk_score,
                "is_harsh_brake": fr.is_harsh_brake,
                "is_harsh_accel": fr.is_harsh_accel,
                "is_harsh_corner": fr.is_harsh_corner,
                "is_speeding": fr.is_speeding,
                "is_tailgating": fr.is_tailgating,
                "n_targets": len(fr.targets),
                "active_event_types": [e["event_type"] for e in fr.events_active],
            })
        self._frames_df_cache = pd.DataFrame(rows)
        return self._frames_df_cache

    # ------------------------------------------------------------------ #
    # Image loaders
    # ------------------------------------------------------------------ #
    def load_left(self, frame_id: int) -> np.ndarray:
        """Load left RGB image. Returns HxWx3 BGR uint8 (OpenCV convention)."""
        return self._load_image(self.image_left_dir, f"{frame_id:06d}")

    def load_right(self, frame_id: int) -> np.ndarray:
        return self._load_image(self.image_right_dir, f"{frame_id:06d}")

    def load_depth(self, frame_id: int) -> Optional[np.ndarray]:
        """Load GT depth in meters (HxW float32). Returns None if not a keyframe."""
        path = self.depth_dir / f"{frame_id:06d}.npy"
        if not path.exists():
            return None
        return np.load(path)

    def load_driver(self, frame_id: int) -> np.ndarray:
        """Load NTHU-composited driver frame."""
        return self._load_image(self.driver_dir, f"frame_{frame_id:06d}")

    def load_calibration(self) -> Dict[str, Any]:
        """Load global calibration info (intrinsics, baseline).

        This is the synthetic-sim summary (K_left, baseline_m, P2_left,
        P3_right) used by the stereo baseline predictor -- constant
        across the whole trip. For the standard per-frame KITTI calib
        files (P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo), use
        load_frame_calibration() instead.
        """
        path = self.trip_dir / "kitti" / "calibration_info.txt"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def load_frame_calibration(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Load the standard KITTI calib file for one frame.

        Parses `kitti/calib/{frame_id:06d}.txt` into a dict of numpy
        matrices: P0..P3 (3x4), R0_rect (3x3), Tr_velo_to_cam /
        Tr_imu_to_velo (3x4). Use this if you need real KITTI
        projection/rectification matrices, e.g. to reuse third-party
        KITTI 3D-detection code against kitti/label_2/.
        """
        path = self.calib_dir / f"{frame_id:06d}.txt"
        if not path.exists():
            raise FileNotFoundError(f"No KITTI calib file for frame {frame_id}: {path}")
        return parse_kitti_calib(path)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """High-level summary dict for quick inspection."""
        return {
            "trip_id": self.trip_id,
            "duration_sec": self.metadata.get("duration_sec"),
            "fps": self.metadata.get("fps"),
            "map": self.metadata.get("map"),
            "driver_profile": self.metadata.get("driver_profile"),
            "n_frames": len(self),
            "n_events": len(self.events_log),
            "safe_driving_score": self.trip_aggregate.get("safe_driving_score"),
            "risk_classification": self.trip_aggregate.get("risk_classification"),
            "driver_subject": self.driver_summary.get("subject_id"),
            "driver_fatigue_score": self.driver_summary.get("fatigue_score"),
            "driver_state_distribution": self.driver_summary.get("state_distribution_pct"),
        }

    def _load_json(self) -> Dict[str, Any]:
        gz = self.trip_dir / f"{self.trip_id}.json.gz"
        plain = self.trip_dir / f"{self.trip_id}.json"
        if gz.exists():
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                return json.load(f)
        elif plain.exists():
            return json.loads(plain.read_text())
        else:
            raise FileNotFoundError(
                f"Trip JSON not found in {self.trip_dir} "
                f"(expected {self.trip_id}.json or .json.gz)"
            )

    def _parse_frame(self, raw: Dict[str, Any]) -> FrameRecord:
        ego = raw.get("ego", {})
        driver = raw.get("driver", {})
        risk = raw.get("risk", {})
        flags = raw.get("behavior_flags", {})

        # Handle ±Infinity sentinels from JSON
        def _f(x: Any) -> float:
            if x == "Infinity":
                return float("inf")
            if x == "-Infinity":
                return float("-inf")
            return float(x) if x is not None else float("inf")

        return FrameRecord(
            frame_id=raw["frame_id"],
            timestamp=float(raw["timestamp"]),
            speed_kmh=float(ego.get("speed_kmh", 0.0)),
            longitudinal_accel=float(ego.get("longitudinal_accel", 0.0)),
            lateral_accel=float(ego.get("lateral_accel", 0.0)),
            driver_state=driver.get("state", "unknown"),
            alertness_score=float(driver.get("alertness_score", 0.0)),
            eye_state=driver.get("eye_state", "unknown"),
            head_pose=driver.get("head_pose", "unknown"),
            mouth_state=driver.get("mouth_state", "unknown"),
            nthu_subject_id=driver.get("nthu_subject_id", "unknown"),
            min_ttc=_f(raw.get("min_ttc", float("inf"))),
            headway_sec=_f(raw.get("headway_sec", float("inf"))),
            base_risk=float(risk.get("base_risk", 0.0)),
            driver_factor=float(risk.get("driver_factor", 1.0)),
            final_risk_score=float(risk.get("final_risk_score", 0.0)),
            is_harsh_brake=bool(flags.get("harsh_brake", False)),
            is_harsh_accel=bool(flags.get("harsh_accel", False)),
            is_harsh_corner=bool(flags.get("harsh_corner", False)),
            is_speeding=bool(flags.get("speeding", False)),
            is_tailgating=bool(flags.get("tailgating", False)),
            targets=raw.get("targets", []),
            events_active=raw.get("events_active", []),
        )

    @staticmethod
    def _load_image(directory: Path, filename_stem: str) -> np.ndarray:
        """Load an image trying common extensions in order. kitti/image_2
        and image_3 are PNG for a trip at its original CARLA-render
        resolution, but become JPG once a trip has been through
        resize_kitti.sh (release-resolution downscale) -- extension
        varies per TRIP, not per file, so this tries both rather than
        assuming one."""
        for ext in (".png", ".jpg", ".jpeg"):
            path = directory / f"{filename_stem}{ext}"
            if path.exists():
                img = cv2.imread(str(path))
                if img is None:
                    raise IOError(f"OpenCV failed to read: {path}")
                return img
        raise FileNotFoundError(
            f"No image for '{filename_stem}' in {directory} (tried .png/.jpg/.jpeg)"
        )


# ====================================================================== #
# Multi-trip wrapper
# ====================================================================== #
class HackathonDataset:
    """Discovers and provides access to all trips under a data root."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        # Discover trip directories matching T<digits>, optionally suffixed
        # with "d" (scored trips, e.g. T01d) or "-Sample" (practice trips,
        # e.g. T01-Sample) -- see team_kit/README.md's dataset layout note.
        import re
        self._trip_ids: List[str] = sorted(
            p.name for p in self.data_root.iterdir()
            if p.is_dir() and re.match(r"^T\d+d?(-Sample)?$", p.name)
        )

    @property
    def trip_ids(self) -> List[str]:
        return list(self._trip_ids)

    def __len__(self) -> int:
        return len(self._trip_ids)

    def __iter__(self) -> Iterator[TripDataset]:
        for tid in self._trip_ids:
            yield self.get_trip(tid)

    def get_trip(self, trip_id: str) -> TripDataset:
        return TripDataset(self.data_root / trip_id)

    def summary_table(self) -> pd.DataFrame:
        """One-row-per-trip summary for fleet-level analytics."""
        rows = []
        for tid in self._trip_ids:
            try:
                ds = self.get_trip(tid)
                rows.append(ds.summary())
            except Exception as e:
                rows.append({"trip_id": tid, "error": str(e)})
        return pd.DataFrame(rows)
