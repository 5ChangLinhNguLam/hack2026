"""Reader for the existing flat JPEG face-crop cache."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FaceCropSample:
    image: Image.Image
    visible: bool
    pitch: float
    yaw: float


@dataclass(frozen=True)
class FaceMetadata:
    visible: bool
    pitch: float
    yaw: float


def save_face_crop_cache(
    cache_dir: Path | str,
    session_name: str,
    *,
    images: Sequence[Image.Image],
    frame_ids: np.ndarray,
    visibility: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    rate: str = "20fps",
    jpeg_quality: int = 92,
) -> None:
    """Write a dense flat-JPEG cache with explicit visibility for every frame."""
    encoded: list[np.ndarray] = []
    for image in images:
        stream = BytesIO()
        image.convert("RGB").save(stream, format="JPEG", quality=jpeg_quality)
        encoded.append(np.frombuffer(stream.getvalue(), dtype=np.uint8))
    save_encoded_face_crop_cache(
        cache_dir,
        session_name,
        encoded=encoded,
        frame_ids=frame_ids,
        visibility=visibility,
        pitch=pitch,
        yaw=yaw,
        rate=rate,
    )


def save_encoded_face_crop_cache(
    cache_dir: Path | str,
    session_name: str,
    *,
    encoded: Sequence[np.ndarray],
    frame_ids: np.ndarray,
    visibility: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    rate: str = "20fps",
) -> None:
    """Write already JPEG-encoded crops without retaining decoded images."""
    count = len(encoded)
    frame_ids = np.asarray(frame_ids)
    visibility = np.asarray(visibility)
    pitch = np.asarray(pitch)
    yaw = np.asarray(yaw)
    if any(array.shape != (count,) for array in (frame_ids, visibility, pitch, yaw)):
        raise ValueError("face-cache metadata must have one value per image")
    if len(np.unique(frame_ids)) != count:
        raise ValueError("face-cache frame_ids must be unique")
    if np.any(visibility < 0.0) or np.any(visibility > 1.0):
        raise ValueError("face-cache visibility must be between 0 and 1")

    offsets = [0]
    buffers: list[np.ndarray] = []
    for value in encoded:
        buffer = np.asarray(value, dtype=np.uint8).reshape(-1)
        buffers.append(buffer)
        offsets.append(offsets[-1] + len(buffer))
    blob = np.concatenate(buffers) if buffers else np.empty(0, dtype=np.uint8)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{session_name}.blob.npy", blob, allow_pickle=False)
    np.savez(
        cache_dir / f"{session_name}.meta.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        frame_ids=frame_ids.astype(np.int32, copy=False),
        visibility=visibility.astype(np.float32, copy=False),
        pitch=pitch.astype(np.float32, copy=False),
        yaw=yaw.astype(np.float32, copy=False),
        rate=np.asarray([rate]),
        complete=np.asarray([True], dtype=np.bool_),
    )


class FaceCropStore:
    """Memory-map one ``.blob.npy`` plus offsets and pose metadata."""

    def __init__(
        self, cache_dir: Path | str, session_name: str, *, missing_size: int = 224
    ) -> None:
        cache_dir = Path(cache_dir)
        blob_path = cache_dir / f"{session_name}.blob.npy"
        metadata_path = cache_dir / f"{session_name}.meta.npz"
        if not blob_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"flat face cache is missing for {session_name}")
        self.blob = np.load(blob_path, mmap_mode="r", allow_pickle=False)
        with np.load(metadata_path, allow_pickle=False) as metadata:
            self.offsets = metadata["offsets"].astype(np.int64, copy=True)
            self.frame_ids = metadata["frame_ids"].astype(np.int64, copy=True)
            self.pitch = metadata["pitch"].astype(np.float32, copy=True)
            self.yaw = metadata["yaw"].astype(np.float32, copy=True)
            self.rate = str(metadata["rate"][0])
            self.visibility = metadata.get(
                "visibility", np.ones(len(self.frame_ids), dtype=np.float32)
            ).astype(np.float32, copy=True)
        count = len(self.frame_ids)
        if (
            len(self.offsets) != count + 1
            or self.pitch.shape != (count,)
            or self.yaw.shape != (count,)
            or self.visibility.shape != (count,)
        ):
            raise ValueError(f"inconsistent flat face cache arrays for {session_name}")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.blob):
            raise ValueError(f"invalid JPEG offsets for {session_name}")
        if len(np.unique(self.frame_ids)) != count:
            raise ValueError(f"duplicate frame IDs in face cache for {session_name}")
        self._rows = {int(frame_id): row for row, frame_id in enumerate(self.frame_ids)}
        self.missing_size = missing_size

    def metadata(self, frame_id: int) -> FaceMetadata:
        """Return visibility and pose without decoding the cached JPEG."""

        row = self._rows.get(int(frame_id))
        if row is None:
            return FaceMetadata(False, 0.0, 0.0)
        return FaceMetadata(
            visible=bool(self.visibility[row] > 0.0),
            pitch=float(self.pitch[row]),
            yaw=float(self.yaw[row]),
        )

    def get(self, frame_id: int) -> FaceCropSample:
        row = self._rows.get(int(frame_id))
        if row is None:
            return FaceCropSample(
                Image.new("RGB", (self.missing_size, self.missing_size)),
                False,
                0.0,
                0.0,
            )
        start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
        encoded = self.blob[start:stop].tobytes()
        with Image.open(BytesIO(encoded)) as image:
            decoded = image.convert("RGB")
        return FaceCropSample(
            decoded,
            bool(self.visibility[row] > 0.0),
            float(self.pitch[row]),
            float(self.yaw[row]),
        )
