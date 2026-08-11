"""Challenge 1 monocular TTC baseline.

The runtime contract is deliberately narrow: one forward-facing RGB image
(``FrameBundle.left()``) per frame.  Stereo, driver-camera and depth inputs are
not part of this package's inference path.
"""

from .detector import OpenCVDnnYoloDetector
from .ort_detector import OnnxRuntimeYoloDetector
from .lean_loader import (
    C1Image2IntegrityError,
    C1Image2IntegrityReport,
    C1InputFrame,
    C1LeanTripLoader,
)
from .pipeline import MonocularC1Pipeline, ReplayStats, run_replay
from .tracker import MonocularTTCTracker, TrackerConfig
from .types import C1FramePrediction, Detection, TrackRisk

__all__ = [
    "C1FramePrediction",
    "C1Image2IntegrityError",
    "C1Image2IntegrityReport",
    "C1InputFrame",
    "C1LeanTripLoader",
    "Detection",
    "MonocularC1Pipeline",
    "MonocularTTCTracker",
    "OpenCVDnnYoloDetector",
    "OnnxRuntimeYoloDetector",
    "ReplayStats",
    "TrackRisk",
    "TrackerConfig",
    "run_replay",
]
