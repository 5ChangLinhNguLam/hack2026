"""Phone detection for the distraction signal.

Phone use is the *actual* evidence for `distracted` in this dataset — the
frames are DMD phone-call recordings. Measured on every tenth frame of the
three mixed Practice trips, a COCO `cell phone` detection above 0.10
confidence covers 83 % / 90 % / 96 % of `distracted` frames against 0 % / 3 % /
20 % of the frames labelled otherwise. Nothing else in Phase 1 separates the
class that cleanly; the mouth-activity proxy it replaces managed roughly a
3:1 median ratio.

The cost is latency. YOLO11s at 640² runs 123 ms/frame on this CPU, six times
the 20 fps budget on its own. Two facts make that affordable:

* phone use persists for seconds, and the classifier consumes it as a *rate*
  over a 91-frame window, so sampling at 4 Hz loses nothing that matters;
* the smaller alternatives do not work. YOLO11n at 640² drops separation to
  73 %/66 %, and at 320² the phone is too few pixels to detect at all on T06
  (0 % — a silent failure, which is worse than a slow one).

So the full model runs on every `stride`-th frame and its result is held in
between. At the default stride of 5 that is 24.6 ms amortised, and combined
with MediaPipe's 8 ms leaves the pipeline at roughly 30 fps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from drive_state.vendor.models import FramePacket
from drive_state.vendor.objects import ObjectObservation, OnnxObjectDetector

DEFAULT_MODEL = Path("models/driver-objects.onnx")
DEFAULT_LABELS = Path("models/driver-objects.labels")
PHONE_LABELS = frozenset({"cell phone", "phone", "mobile"})

#: Detector confidence floor. Deliberately low: the phone is small, often
#: half-occluded by a hand, and the *window rate* is what the rules threshold
#: on, so a few weak false positives are absorbed while a missed phone is not.
DEFAULT_CONFIDENCE = 0.05


@dataclass(slots=True)
class PhoneObservation:
    confidence: float
    bbox: tuple[int, int, int, int]
    fresh: bool  # False when this is a held result from an earlier frame

    @property
    def found(self) -> bool:
        return self.confidence > 0.0


NOTHING = PhoneObservation(0.0, (0, 0, 0, 0), fresh=True)


class PhoneDetector:
    """Strided COCO detector that reports the best `cell phone` box.

    Not thread-safe and stateful across frames by design — it holds the last
    detection between strides, so one instance handles exactly one trip.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        labels_path: str | Path = DEFAULT_LABELS,
        *,
        stride: int = 5,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Phone detector model missing: {model_path}. Run "
                "`python scripts/download_models.py --phone-detector --phone-model yolo11s.pt`."
            )
        self.stride = stride
        self.confidence = confidence
        self._detector = OnnxObjectDetector(
            model_path,
            Path(labels_path),
            confidence_threshold=confidence,
            iou_threshold=0.45,
        )
        self._held = NOTHING
        self._seen = 0

    def detect(
        self, frame: npt.NDArray[np.uint8], frame_id: int, timestamp: float
    ) -> PhoneObservation:
        """Best phone box for this frame, running the model only every `stride`.

        Frames in between reuse the previous result with `fresh=False`, so a
        caller can tell a measurement from a hold.
        """
        if self._seen % self.stride:
            self._seen += 1
            return PhoneObservation(self._held.confidence, self._held.bbox, fresh=False)
        self._seen += 1

        packet = FramePacket(frame=frame, timestamp=timestamp, frame_index=frame_id)
        best: ObjectObservation | None = None
        for observation in self._detector.detect(packet):
            if observation.label.lower() not in PHONE_LABELS:
                continue
            if best is None or observation.confidence > best.confidence:
                best = observation

        self._held = (
            NOTHING if best is None else PhoneObservation(best.confidence, best.bbox, fresh=True)
        )
        return self._held


def create_phone_detector(
    model_path: str | Path | None,
    labels_path: str | Path = DEFAULT_LABELS,
    *,
    stride: int = 5,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PhoneDetector | None:
    """`None` model path disables detection; a missing file is an error.

    The distinction matters: turning the detector off is a deliberate choice
    that costs ~20 composite points, and should never happen because a path
    was mistyped.
    """
    if model_path is None:
        return None
    return PhoneDetector(model_path, labels_path, stride=stride, confidence=confidence)
