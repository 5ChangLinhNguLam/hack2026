"""One-pass orchestration for Challenge 1 and Challenge 2 runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class CombinedFramePrediction:
    frame_id: int
    timestamp: float
    c1: Any
    c2: Any
    source_bundle: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, prediction in (("C1", self.c1), ("C2", self.c2)):
            if int(prediction.frame_id) != self.frame_id:
                raise ValueError(
                    f"{name} frame mismatch: {prediction.frame_id} != {self.frame_id}"
                )
            if abs(float(prediction.timestamp) - self.timestamp) > 1e-6:
                raise ValueError(
                    f"{name} timestamp mismatch: "
                    f"{prediction.timestamp} != {self.timestamp}"
                )

    def submission_row(self) -> dict[str, object]:
        c1_row = self.c1.submission_row()
        c2_row = self.c2.submission_row()
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "predicted_ttc": c1_row["predicted_ttc"],
            "predicted_driver_state": c2_row["predicted_driver_state"],
        }


class CombinedModelReplay:
    """Consume one frame stream and run both stateful models in order."""

    def __init__(
        self,
        frames: Iterable[Any],
        *,
        c1: C1Processor,
        c2: C2Processor,
    ) -> None:
        self.frames = frames
        self.c1 = c1
        self.c2 = c2

    def __iter__(self) -> Iterator[CombinedFramePrediction]:
        for bundle in self.frames:
            collision = self.c1.process_bundle(bundle)
            driver = self.c2.process_bgr(
                bundle.driver(),
                frame_id=bundle.frame_id,
                timestamp=bundle.timestamp,
            )
            yield CombinedFramePrediction(
                frame_id=int(bundle.frame_id),
                timestamp=float(bundle.timestamp),
                c1=collision,
                c2=driver,
                source_bundle=bundle,
            )


__all__ = [
    "C1Processor",
    "C2Processor",
    "CombinedFramePrediction",
    "CombinedModelReplay",
]
