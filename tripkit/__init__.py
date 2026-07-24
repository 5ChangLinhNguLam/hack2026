"""tripkit — Trip Loader & Replayer 20 FPS (WBS 1.2).

Module nền tảng phát lại 1 trip theo timestamp, đồng bộ ảnh stereo +
driver + kinematics theo frame_id. Dùng chung cho C1, C2, pipeline
runner 1.4 và HUD/dashboard.

API chính (xem docs/Task_1.2_TripReplayer_Spec_ClaudeCode.md mục 3):
    from tripkit import TripLoader, TripReplayer, Calib, FrameBundle
"""

__version__ = "0.1.0"

from .types import Calib
from .loader import TripLoader

__all__ = ["__version__", "Calib", "TripLoader"]
