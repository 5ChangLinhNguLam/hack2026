"""Two-view cabin/face dataset for partially labelled DMD primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .face_cache import FaceCropStore
from .primitive_labels import TASK_CLASS_COUNTS
from .primitive_records import PrimitiveFrameRecord


_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize to fit without deleting cabin edges, then pad with black."""
    image = image.convert("RGB")
    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", size)
    canvas.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return canvas


def normalized_image_tensor(image: Image.Image, size: tuple[int, int]) -> Tensor:
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


class PrimitiveFrameDataset(Dataset[dict[str, Tensor | str | int]]):
    def __init__(
        self,
        records: Sequence[PrimitiveFrameRecord],
        *,
        face_cache_dir: Path | str,
        cabin_size: tuple[int, int] = (320, 192),
        face_size: int = 224,
        strict_face_cache: bool = True,
    ) -> None:
        self.records = tuple(records)
        self.cabin_size = cabin_size
        self.face_size = face_size
        cache_dir = Path(face_cache_dir)
        self.face_stores: dict[str, FaceCropStore] = {}
        records_by_session: dict[str, list[PrimitiveFrameRecord]] = {}
        for record in self.records:
            records_by_session.setdefault(record.session_name, []).append(record)
        for session_name in sorted(records_by_session):
            try:
                store = FaceCropStore(
                    cache_dir, session_name, missing_size=face_size
                )
                if strict_face_cache:
                    available = {int(frame_id) for frame_id in store.frame_ids}
                    requested = {
                        (
                            record.src_frame_id
                            if store.rate == "native"
                            else record.frame_id
                        )
                        for record in records_by_session[session_name]
                    }
                    missing = requested.difference(available)
                    if missing:
                        raise ValueError(
                            f"face cache for {session_name} is missing "
                            f"{len(missing)} requested frame IDs"
                        )
                self.face_stores[session_name] = store
            except FileNotFoundError:
                if strict_face_cache:
                    raise

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        record = self.records[index]
        with Image.open(record.frame_path) as source:
            full_frame = source.convert("RGB")
        cabin_image = letterbox(full_frame, self.cabin_size)
        cabin = normalized_image_tensor(cabin_image, self.cabin_size)

        store = self.face_stores.get(record.session_name)
        if store is None:
            face_image = Image.new("RGB", (self.face_size, self.face_size))
            face_visible, pitch, yaw = False, 0.0, 0.0
        else:
            lookup_id = record.src_frame_id if store.rate == "native" else record.frame_id
            face_sample = store.get(lookup_id)
            face_image = face_sample.image
            face_visible = face_sample.visible
            pitch, yaw = face_sample.pitch, face_sample.yaw

        sample: dict[str, Tensor | str | int] = {
            "cabin": cabin,
            "face": normalized_image_tensor(
                face_image, (self.face_size, self.face_size)
            ),
            "face_visibility": torch.tensor(float(face_visible), dtype=torch.float32),
            "head_pose": torch.tensor((pitch, yaw), dtype=torch.float32),
            "session": record.session_name,
            "subject": record.subject_id,
            "protocol": record.protocol,
            "frame_id": record.frame_id,
        }
        sample.update(
            {
                task: torch.tensor(record.targets[task], dtype=torch.long)
                for task in TASK_CLASS_COUNTS
            }
        )
        return sample
