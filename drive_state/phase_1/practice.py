"""Reading the organiser's Practice_Dataset.

A trip is a directory ``<root>/<trip_id>/`` holding

* ``driver/frame_%06d.jpg`` — the cabin camera, one file per frame
* ``<trip_id>.json.gz`` — the full ground truth, of which Phase 1 reads only
  ``frames[i].driver``

The uncompressed ``<trip_id>.json`` sibling is ignored: it is a directory on
some checkouts and the ``.gz`` is authoritative.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

DEFAULT_FPS = 20.0

#: Cabin frames are 8-bit BGR. cv2.imread's annotation is looser than that, so
#: reads go through `imread_bgr` to pin the dtype once instead of casting at
#: every call site.
BGRImage = npt.NDArray[np.uint8]


def imread_bgr(path: str | Path) -> BGRImage:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unreadable frame: {path}")
    return cast(BGRImage, image)


@dataclass(slots=True)
class DriverLabel:
    frame_id: int
    timestamp: float
    state: str
    eye_state: str
    head_pose: str
    mouth_state: str


@dataclass(slots=True)
class Trip:
    trip_id: str
    root: Path
    fps: float
    description: str
    subject_id: str
    condition: str
    labels: list[DriverLabel]

    @property
    def frames_dir(self) -> Path:
        return self.root / "driver"

    def frame_paths(self) -> list[Path]:
        return sorted(self.frames_dir.glob("frame_*.jpg"))

    def iter_frames(self) -> list[tuple[int, float, BGRImage]]:
        """Decode every cabin frame. Frame ids come from the filename, so a
        gap in the sequence stays aligned with the label list."""
        out = []
        for path in self.frame_paths():
            frame_id = int(path.stem.split("_")[-1])
            out.append((frame_id, frame_id / self.fps, imread_bgr(path)))
        return out

    def states(self) -> dict[int, str]:
        return {label.frame_id: label.state for label in self.labels}


def load_trip(root: str | Path) -> Trip:
    root = Path(root)
    trip_id = root.name
    payload_path = root / f"{trip_id}.json.gz"
    if not payload_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {payload_path}")

    with gzip.open(payload_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata = payload.get("metadata", {})
    summary = payload.get("driver_summary", {})
    fps = float(metadata.get("fps") or DEFAULT_FPS)

    labels = []
    for frame in payload["frames"]:
        driver = frame["driver"]
        labels.append(
            DriverLabel(
                frame_id=int(frame["frame_id"]),
                timestamp=float(frame.get("timestamp", frame["frame_id"] / fps)),
                state=driver["state"],
                eye_state=driver["eye_state"],
                head_pose=driver["head_pose"],
                mouth_state=driver["mouth_state"],
            )
        )

    return Trip(
        trip_id=trip_id,
        root=root,
        fps=fps,
        description=str(metadata.get("description", "")),
        subject_id=str(summary.get("subject_id", "")),
        condition=str(summary.get("condition_subset", "")),
        labels=labels,
    )


def discover_trips(dataset_root: str | Path) -> list[Trip]:
    """Every subdirectory of `dataset_root` that carries a ground-truth blob."""
    dataset_root = Path(dataset_root)
    trips = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / f"{child.name}.json.gz").exists():
            trips.append(load_trip(child))
    if not trips:
        raise FileNotFoundError(f"No trips found under {dataset_root}")
    return trips
