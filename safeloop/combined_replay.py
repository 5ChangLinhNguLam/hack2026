"""One-pass orchestration for all three challenge runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Iterator, Protocol


class C1Processor(Protocol):
    def process_bundle(self, bundle: Any) -> Any: ...


class C2Processor(Protocol):
    def process_bgr(
        self,
        image: Any,
        *,
        frame_id: int,
        timestamp: float,
    ) -> Any: ...


class C3Processor(Protocol):
    def update(
        self,
        bundle: Any,
        *,
        predicted_ttc_s: object,
    ) -> Any: ...


class DriveQualityProcessor(Protocol):
    def update(self, c3_frame: Any) -> Any: ...


class ContextualRiskProcessor(Protocol):
    def evaluate(self, c1: Any, c2: Any) -> Any: ...


@dataclass(frozen=True)
class CombinedFramePrediction:
    frame_id: int
    timestamp: float
    c1: Any
    c2: Any
    c3: Any
    drive_quality: Any
    contextual_risk: Any
    source_bundle: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, prediction in (
            ("C1", self.c1),
            ("C2", self.c2),
            ("C3", self.c3),
            ("Drive Quality", self.drive_quality),
        ):
            if int(prediction.frame_id) != self.frame_id:
                raise ValueError(
                    f"{name} frame mismatch: {prediction.frame_id} != {self.frame_id}"
                )
            if abs(float(prediction.timestamp) - self.timestamp) > 1e-6:
                raise ValueError(
                    f"{name} timestamp mismatch: "
                    f"{prediction.timestamp} != {self.timestamp}"
                )
        risk_score = float(self.contextual_risk.score_pct)
        if not math.isfinite(risk_score) or not 0.0 <= risk_score <= 100.0:
            raise ValueError(
                "contextual risk score must be finite and in [0, 100]"
            )

    def submission_row(self) -> dict[str, object]:
        c1_row = self.c1.submission_row()
        c2_row = self.c2.submission_row()
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "predicted_ttc": c1_row["predicted_ttc"],
            "predicted_driver_state": c2_row["predicted_driver_state"],
            # The evaluator currently uses this finite value as the explicit
            # C3 opt-in gate.  Its score reconstruction still comes from C1
            # raw TTC plus ego kinematics; see safeloop.c3.
            "predicted_risk_score": round(
                float(self.contextual_risk.score_pct), 3
            ),
        }


class CombinedModelReplay:
    """Consume one frame stream and run both stateful models in order."""

    def __init__(
        self,
        frames: Iterable[Any],
        *,
        c1: C1Processor,
        c2: C2Processor,
        c3: C3Processor,
        drive_quality: DriveQualityProcessor,
        contextual_risk: ContextualRiskProcessor,
    ) -> None:
        self.frames = frames
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.drive_quality = drive_quality
        self.contextual_risk = contextual_risk

    def __iter__(self) -> Iterator[CombinedFramePrediction]:
        for bundle in self.frames:
            collision = self.c1.process_bundle(bundle)
            driver = self.c2.process_bgr(
                bundle.driver(),
                frame_id=bundle.frame_id,
                timestamp=bundle.timestamp,
            )
            safe_score = self.c3.update(
                bundle,
                predicted_ttc_s=collision.predicted_ttc_s,
            )
            drive_quality = self.drive_quality.update(safe_score)
            risk = self.contextual_risk.evaluate(collision, driver)
            yield CombinedFramePrediction(
                frame_id=int(bundle.frame_id),
                timestamp=float(bundle.timestamp),
                c1=collision,
                c2=driver,
                c3=safe_score,
                drive_quality=drive_quality,
                contextual_risk=risk,
                source_bundle=bundle,
            )


__all__ = [
    "C1Processor",
    "C2Processor",
    "C3Processor",
    "CombinedFramePrediction",
    "CombinedModelReplay",
    "ContextualRiskProcessor",
    "DriveQualityProcessor",
]
