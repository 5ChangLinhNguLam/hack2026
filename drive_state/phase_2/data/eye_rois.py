"""Eye-region geometry, cache serialization, and visibility-aware crops."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


@dataclass(frozen=True)
class EyeBox:
    """Pixel box with an exclusive lower-right corner."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"invalid eye box: {self}")


@dataclass(frozen=True)
class EyeRoiRecord:
    frame_id: int
    left: EyeBox | None
    right: EyeBox | None
    left_visibility: float = 0.0
    right_visibility: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("left_visibility", self.left_visibility),
            ("right_visibility", self.right_visibility),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.left is None and self.left_visibility != 0.0:
            raise ValueError("a missing left eye must have zero visibility")
        if self.right is None and self.right_visibility != 0.0:
            raise ValueError("a missing right eye must have zero visibility")


def box_from_normalized_landmarks(
    points: Sequence[tuple[float, float]],
    *,
    image_width: int,
    image_height: int,
    horizontal_margin: float = 0.35,
    vertical_margin: float = 1.0,
) -> EyeBox | None:
    """Convert normalized eye landmarks into a padded, image-clamped box."""
    if not points:
        return None
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    xs = [point[0] * image_width for point in points]
    ys = [point[1] * image_height for point in points]
    raw_width = max(xs) - min(xs)
    raw_height = max(ys) - min(ys)
    if raw_width <= 0.0 or raw_height <= 0.0:
        return None

    x1 = max(0, math.floor(min(xs) - raw_width * horizontal_margin))
    y1 = max(0, math.floor(min(ys) - raw_height * vertical_margin))
    x2 = min(image_width, math.ceil(max(xs) + raw_width * horizontal_margin))
    y2 = min(image_height, math.ceil(max(ys) + raw_height * vertical_margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return EyeBox(x1, y1, x2, y2)


_FIELDS = (
    "frame_id",
    "left_x1",
    "left_y1",
    "left_x2",
    "left_y2",
    "right_x1",
    "right_y1",
    "right_x2",
    "right_y2",
    "left_visibility",
    "right_visibility",
)


def _box_values(box: EyeBox | None) -> tuple[int | str, ...]:
    return ("", "", "", "") if box is None else (box.x1, box.y1, box.x2, box.y2)


def save_eye_roi_cache(path: Path | str, records: Iterable[EyeRoiRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(_FIELDS)
        for record in records:
            writer.writerow(
                (
                    record.frame_id,
                    *_box_values(record.left),
                    *_box_values(record.right),
                    f"{record.left_visibility:.6f}",
                    f"{record.right_visibility:.6f}",
                )
            )


def _box_from_row(row: dict[str, str], side: str) -> EyeBox | None:
    values = [row[f"{side}_{coordinate}"] for coordinate in ("x1", "y1", "x2", "y2")]
    if not any(values):
        return None
    if not all(values):
        raise ValueError(f"partially missing {side} eye box")
    return EyeBox(*(int(value) for value in values))


def load_eye_roi_cache(path: Path | str) -> dict[int, EyeRoiRecord]:
    path = Path(path)
    records: dict[int, EyeRoiRecord] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing cache columns: {sorted(missing)}")
        for row in reader:
            record = EyeRoiRecord(
                frame_id=int(row["frame_id"]),
                left=_box_from_row(row, "left"),
                right=_box_from_row(row, "right"),
                left_visibility=float(row["left_visibility"]),
                right_visibility=float(row["right_visibility"]),
            )
            if record.frame_id in records:
                raise ValueError(f"duplicate frame_id {record.frame_id} in {path}")
            records[record.frame_id] = record
    return records


def _crop_or_black(
    image: Image.Image, box: EyeBox | None, output_size: tuple[int, int]
) -> Image.Image:
    if box is None:
        return Image.new("RGB", output_size)
    return image.crop((box.x1, box.y1, box.x2, box.y2)).resize(
        output_size, Image.Resampling.BILINEAR
    )


def crop_eye_pair(
    image: Image.Image,
    record: EyeRoiRecord,
    *,
    output_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, tuple[float, float]]:
    """Crop a left/right eye pair, preserving missingness as visibility zero."""
    image = image.convert("RGB")
    left = _crop_or_black(image, record.left, output_size)
    right = _crop_or_black(image, record.right, output_size)
    return left, right, (record.left_visibility, record.right_visibility)
