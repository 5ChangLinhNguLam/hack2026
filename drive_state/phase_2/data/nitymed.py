"""Strict NITYMED video metadata without broadcasting recording labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import NamedTuple, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..face_detection import FaceDetection
from ..landmarks import LandmarkLike, evidence_regions_from_landmarks
from .evidence_regions import EvidenceRegionRecord
from .evidence_regions import EvidenceRegionStore, project_normalized_boxes_to_letterbox
from .five_state_labels import IGNORE_INDEX
from .primitive_dataset import letterbox, normalized_image_tensor


NITYMED_VIDEO_SCHEMA_VERSION = 1
NITYMED_FRAME_SCHEMA_VERSION = 2
NITYMED_TEACHER_SCHEMA_VERSION = 1
_CATEGORY_BY_COMPONENT = {
    "microsleep": "microsleep_recording",
    "yawning": "yawning_recording",
}


@dataclass(frozen=True)
class NitymedVideoRecord:
    video_id: str
    relative_path: Path
    category: str
    fps: float
    frames: int
    width: int
    height: int

    def __post_init__(self) -> None:
        relative = Path(self.relative_path)
        if (
            not self.video_id
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("NITYMED video IDs and relative paths must be safe")
        if self.category not in set(_CATEGORY_BY_COMPONENT.values()):
            raise ValueError("unknown NITYMED recording category")
        if self.fps <= 0.0 or min(self.frames, self.width, self.height) <= 0:
            raise ValueError("NITYMED video metadata must be positive")
        object.__setattr__(self, "relative_path", relative)


@dataclass(frozen=True)
class NitymedFrameRecord:
    video_id: str
    category: str
    frame_id: int
    timestamp: float
    frame_path: Path

    def __post_init__(self) -> None:
        path = Path(self.frame_path)
        if not self.video_id or self.frame_id < 0 or self.timestamp < 0.0:
            raise ValueError("NITYMED frame metadata must be non-negative")
        if self.category not in set(_CATEGORY_BY_COMPONENT.values()):
            raise ValueError("unknown NITYMED recording category")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("NITYMED frame paths must be safe and relative")
        object.__setattr__(self, "frame_path", path)


class NitymedPseudoTargets(NamedTuple):
    eye: Tensor
    yawn: Tensor
    eye_mask: Tensor
    yawn_mask: Tensor


@dataclass(frozen=True)
class NitymedTeacherRecord:
    embedding: np.ndarray
    region_visibility: np.ndarray
    eye_target: int
    yawn_target: int
    eye_mask: bool
    yawn_mask: bool


def _video_id(relative_path: Path) -> str:
    stem = "__".join(relative_path.with_suffix("").parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    if not safe:
        raise ValueError("cannot derive NITYMED video ID")
    return safe


def _category(relative_path: Path) -> str:
    categories = {
        _CATEGORY_BY_COMPONENT[part.casefold()]
        for part in relative_path.parts
        if part.casefold() in _CATEGORY_BY_COMPONENT
    }
    if len(categories) != 1:
        raise ValueError(
            f"NITYMED video path must contain one recording category: "
            f"{relative_path}"
        )
    return categories.pop()


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to probe NITYMED videos") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open NITYMED video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0.0 or min(frames, width, height) <= 0:
        raise ValueError(f"invalid NITYMED video metadata: {path}")
    return fps, frames, width, height


def scan_nitymed_videos(root: Path) -> tuple[NitymedVideoRecord, ...]:
    """Scan videos while keeping yawning/microsleep names as metadata only."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NITYMED root is missing: {root}")
    records: list[NitymedVideoRecord] = []
    seen_ids: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() != ".mp4":
            continue
        relative = path.relative_to(root)
        video_id = _video_id(relative)
        if video_id in seen_ids:
            raise ValueError(f"duplicate NITYMED video ID: {video_id}")
        seen_ids.add(video_id)
        fps, frames, width, height = _probe_video(path)
        records.append(
            NitymedVideoRecord(
                video_id=video_id,
                relative_path=relative,
                category=_category(relative),
                fps=fps,
                frames=frames,
                width=width,
                height=height,
            )
        )
    if not records:
        raise ValueError(f"NITYMED root contains no MP4 videos: {root}")
    return tuple(records)


def save_nitymed_video_manifest(
    path: Path,
    records: Sequence[NitymedVideoRecord],
    *,
    dataset_sha256: str,
) -> None:
    records = tuple(records)
    if not records or len(dataset_sha256) != 64:
        raise ValueError("NITYMED manifest requires records and SHA-256")
    if len({record.video_id for record in records}) != len(records):
        raise ValueError("NITYMED manifest video IDs must be unique")
    payload = {
        "schema_version": NITYMED_VIDEO_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "videos": [
            {
                **asdict(record),
                "relative_path": record.relative_path.as_posix(),
            }
            for record in records
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_nitymed_video_manifest(path: Path) -> tuple[NitymedVideoRecord, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != NITYMED_VIDEO_SCHEMA_VERSION:
        raise ValueError("incompatible NITYMED video manifest schema")
    raw = payload.get("videos")
    if not isinstance(raw, list) or not raw:
        raise ValueError("NITYMED video manifest is empty")
    records = tuple(
        NitymedVideoRecord(
            video_id=str(item["video_id"]),
            relative_path=Path(str(item["relative_path"])),
            category=str(item["category"]),
            fps=float(item["fps"]),
            frames=int(item["frames"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in raw
    )
    if len({record.video_id for record in records}) != len(records):
        raise ValueError("NITYMED video manifest contains duplicate IDs")
    return records


def save_nitymed_frame_manifest(
    path: Path,
    records: Sequence[NitymedFrameRecord],
    *,
    source_fps: float | Sequence[float],
    sample_fps: float,
) -> None:
    records = tuple(records)
    if isinstance(source_fps, (int, float)):
        source_rates = (float(source_fps),)
    else:
        source_rates = tuple(sorted({float(value) for value in source_fps}))
    if (
        not records
        or not source_rates
        or any(value <= 0.0 for value in source_rates)
        or sample_fps <= 0.0
    ):
        raise ValueError("NITYMED frame manifest metadata must be positive")
    keys = {(record.video_id, record.frame_id) for record in records}
    if len(keys) != len(records):
        raise ValueError("NITYMED frame manifest contains duplicate frames")
    payload = {
        "schema_version": NITYMED_FRAME_SCHEMA_VERSION,
        "source_fps": list(source_rates),
        "sample_fps": float(sample_fps),
        "frames": [
            {
                **asdict(record),
                "frame_path": record.frame_path.as_posix(),
            }
            for record in records
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_nitymed_frame_manifest(path: Path) -> tuple[NitymedFrameRecord, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != NITYMED_FRAME_SCHEMA_VERSION:
        raise ValueError("incompatible NITYMED frame manifest schema")
    raw_source_fps = payload.get("source_fps")
    if isinstance(raw_source_fps, (int, float)):
        source_rates = (float(raw_source_fps),)
    elif isinstance(raw_source_fps, list):
        source_rates = tuple(float(value) for value in raw_source_fps)
    else:
        source_rates = ()
    if (
        not source_rates
        or any(value <= 0.0 for value in source_rates)
        or float(payload.get("sample_fps", 0.0)) <= 0.0
    ):
        raise ValueError("NITYMED frame manifest FPS is invalid")
    raw = payload.get("frames")
    if not isinstance(raw, list) or not raw:
        raise ValueError("NITYMED frame manifest is empty")
    records = tuple(
        NitymedFrameRecord(
            video_id=str(item["video_id"]),
            category=str(item["category"]),
            frame_id=int(item["frame_id"]),
            timestamp=float(item["timestamp"]),
            frame_path=Path(str(item["frame_path"])),
        )
        for item in raw
    )
    keys = {(record.video_id, record.frame_id) for record in records}
    if len(keys) != len(records):
        raise ValueError("NITYMED frame manifest contains duplicate frames")
    return records


def nitymed_region_record(
    *,
    frame_id: int,
    image_size: tuple[int, int],
    detection: FaceDetection | None,
    landmarks: Sequence[LandmarkLike] | None,
) -> EvidenceRegionRecord:
    """Build honest Method A regions; a detector miss stays fully missing."""

    if detection is None:
        return EvidenceRegionRecord.missing(frame_id)
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("NITYMED image dimensions must be positive")
    return evidence_regions_from_landmarks(
        frame_id=frame_id,
        face_box=detection.box,
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        pitch=0.0,
        yaw=0.0,
        face_visibility=detection.confidence,
    )


def select_nitymed_pseudo_targets(
    *,
    eye_probabilities: Tensor,
    yawn_probabilities: Tensor,
    region_visibility: Tensor,
    confidence: float = 0.90,
    margin: float = 0.20,
) -> NitymedPseudoTargets:
    """Accept only visible, unambiguous frame-observable primitives."""

    if eye_probabilities.ndim != 2 or eye_probabilities.shape[1] != 3:
        raise ValueError("eye probabilities must have shape [frames, 3]")
    frames = eye_probabilities.shape[0]
    if yawn_probabilities.shape != (frames, 2):
        raise ValueError("yawn probabilities must have shape [frames, 2]")
    if region_visibility.shape != (frames, 4):
        raise ValueError("region visibility must have shape [frames, 4]")
    if not 0.0 < confidence <= 1.0 or not 0.0 <= margin <= 1.0:
        raise ValueError("pseudo-target confidence and margin are invalid")
    for values in (eye_probabilities, yawn_probabilities, region_visibility):
        if not torch.isfinite(values).all():
            raise ValueError("pseudo-target inputs must be finite")

    eye_top, eye_class = eye_probabilities.topk(2, dim=1)
    yawn_top, yawn_class = yawn_probabilities.topk(2, dim=1)
    eye_visible = (region_visibility[:, 1] > 0.0) & (
        region_visibility[:, 2] > 0.0
    )
    mouth_visible = region_visibility[:, 3] > 0.0
    eye_mask = (
        (eye_top[:, 0] >= confidence)
        & ((eye_top[:, 0] - eye_top[:, 1]) >= margin)
        & eye_visible
        & (eye_class[:, 0] != 1)
    )
    yawn_mask = (
        (yawn_top[:, 0] >= confidence)
        & ((yawn_top[:, 0] - yawn_top[:, 1]) >= margin)
        & mouth_visible
    )
    eye = torch.full(
        (frames,),
        IGNORE_INDEX,
        dtype=torch.long,
        device=eye_probabilities.device,
    )
    yawn = torch.full(
        (frames,),
        IGNORE_INDEX,
        dtype=torch.long,
        device=yawn_probabilities.device,
    )
    eye[eye_mask] = eye_class[eye_mask, 0]
    yawn[yawn_mask] = yawn_class[yawn_mask, 0]
    return NitymedPseudoTargets(eye, yawn, eye_mask, yawn_mask)


def _teacher_cache_path(cache_dir: Path, video_id: str) -> Path:
    if not video_id or "/" in video_id or "\\" in video_id:
        raise ValueError("NITYMED teacher video ID must be safe")
    return Path(cache_dir) / f"{video_id}.teacher.npz"


def save_nitymed_teacher_cache(
    cache_dir: Path,
    video_id: str,
    *,
    frame_ids: Tensor,
    embeddings: Tensor,
    eye_probabilities: Tensor,
    yawn_probabilities: Tensor,
    region_visibility: Tensor,
    confidence: float = 0.90,
    margin: float = 0.20,
) -> None:
    if frame_ids.ndim != 1:
        raise ValueError("NITYMED teacher frame IDs must be one-dimensional")
    count = frame_ids.numel()
    if embeddings.ndim != 2 or embeddings.shape[0] != count:
        raise ValueError("NITYMED teacher embeddings must align with frames")
    targets = select_nitymed_pseudo_targets(
        eye_probabilities=eye_probabilities,
        yawn_probabilities=yawn_probabilities,
        region_visibility=region_visibility,
        confidence=confidence,
        margin=margin,
    )
    ids = frame_ids.detach().to(dtype=torch.int64, device="cpu").numpy()
    if len(np.unique(ids)) != count or (count > 1 and np.any(np.diff(ids) <= 0)):
        raise ValueError("NITYMED teacher frame IDs must be unique and increasing")
    path = _teacher_cache_path(Path(cache_dir), video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            schema_version=np.asarray(
                [NITYMED_TEACHER_SCHEMA_VERSION], dtype=np.int32
            ),
            complete=np.asarray([True], dtype=np.bool_),
            frame_ids=ids,
            embeddings=embeddings.detach().to(
                dtype=torch.float16, device="cpu"
            ).numpy(),
            eye_probabilities=eye_probabilities.detach().to(
                dtype=torch.float32, device="cpu"
            ).numpy(),
            yawn_probabilities=yawn_probabilities.detach().to(
                dtype=torch.float32, device="cpu"
            ).numpy(),
            region_visibility=region_visibility.detach().to(
                dtype=torch.float32, device="cpu"
            ).numpy(),
            eye_targets=targets.eye.detach().to(device="cpu").numpy(),
            yawn_targets=targets.yawn.detach().to(device="cpu").numpy(),
            eye_masks=targets.eye_mask.detach().to(device="cpu").numpy(),
            yawn_masks=targets.yawn_mask.detach().to(device="cpu").numpy(),
        )
    temporary.replace(path)


class NitymedTeacherStore:
    def __init__(self, cache_dir: Path, video_id: str) -> None:
        path = _teacher_cache_path(Path(cache_dir), video_id)
        if not path.is_file():
            raise FileNotFoundError(f"NITYMED teacher cache is missing: {path}")
        with np.load(path, allow_pickle=False) as cache:
            if (
                int(cache["schema_version"][0])
                != NITYMED_TEACHER_SCHEMA_VERSION
                or not bool(cache["complete"][0])
            ):
                raise ValueError("incompatible NITYMED teacher cache")
            self.frame_ids = cache["frame_ids"].astype(np.int64, copy=True)
            self.embeddings = cache["embeddings"].astype(np.float32, copy=True)
            self.region_visibility = cache["region_visibility"].astype(
                np.float32,
                copy=True,
            )
            self.eye_targets = cache["eye_targets"].astype(np.int64, copy=True)
            self.yawn_targets = cache["yawn_targets"].astype(np.int64, copy=True)
            self.eye_masks = cache["eye_masks"].astype(np.bool_, copy=True)
            self.yawn_masks = cache["yawn_masks"].astype(np.bool_, copy=True)
        count = len(self.frame_ids)
        if (
            self.embeddings.ndim != 2
            or self.embeddings.shape[0] != count
            or self.region_visibility.shape != (count, 4)
            or self.eye_targets.shape != (count,)
            or self.yawn_targets.shape != (count,)
            or self.eye_masks.shape != (count,)
            or self.yawn_masks.shape != (count,)
        ):
            raise ValueError("NITYMED teacher cache arrays are misaligned")
        self._rows = {
            int(frame_id): row for row, frame_id in enumerate(self.frame_ids)
        }

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def get(self, frame_id: int) -> NitymedTeacherRecord:
        row = self._rows.get(int(frame_id))
        if row is None:
            raise KeyError(f"NITYMED teacher cache has no frame {frame_id}")
        return NitymedTeacherRecord(
            embedding=self.embeddings[row].copy(),
            region_visibility=self.region_visibility[row].copy(),
            eye_target=int(self.eye_targets[row]),
            yawn_target=int(self.yawn_targets[row]),
            eye_mask=bool(self.eye_masks[row]),
            yawn_mask=bool(self.yawn_masks[row]),
        )


class NitymedFrameDataset(Dataset[dict[str, Tensor | str | int]]):
    """Return NITYMED primitives while keeping recording categories non-target."""

    def __init__(
        self,
        *,
        root: Path,
        frame_manifest: Path,
        region_cache: Path,
        teacher_cache: Path,
        image_size: tuple[int, int] = (640, 384),
        training: bool = False,
        include_eye_targets: bool = True,
    ) -> None:
        if any(value <= 0 for value in image_size):
            raise ValueError("NITYMED image dimensions must be positive")
        self.root = Path(root)
        self.records = load_nitymed_frame_manifest(frame_manifest)
        self.image_size = tuple(int(value) for value in image_size)
        self.training = bool(training)
        self.include_eye_targets = bool(include_eye_targets)
        videos = sorted({record.video_id for record in self.records})
        self.region_stores = {
            video_id: EvidenceRegionStore(region_cache, video_id)
            for video_id in videos
        }
        self.teacher_stores = {
            video_id: NitymedTeacherStore(teacher_cache, video_id)
            for video_id in videos
        }
        dimensions = {
            store.embedding_dim for store in self.teacher_stores.values()
        }
        if len(dimensions) != 1:
            raise ValueError("NITYMED teacher embedding dimensions disagree")
        self.teacher_embedding_dim = dimensions.pop()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        from .five_state_visual import _augment_training_frame

        record = self.records[index]
        with Image.open(self.root / record.frame_path) as source:
            image = source.convert("RGB")
        source_size = image.size
        if self.training:
            image = _augment_training_frame(image)
        region = self.region_stores[record.video_id].get(record.frame_id)
        teacher = self.teacher_stores[record.video_id].get(record.frame_id)
        boxes = project_normalized_boxes_to_letterbox(
            region.boxes,
            source_size=source_size,
            target_size=self.image_size,
        )
        return {
            "image": normalized_image_tensor(
                letterbox(image, self.image_size),
                self.image_size,
            ),
            "region_boxes": torch.from_numpy(boxes.copy()),
            "region_visibility": torch.from_numpy(region.visibility.copy()),
            "head_pose_target": torch.zeros(2, dtype=torch.float32),
            "five_state_target": torch.tensor(IGNORE_INDEX, dtype=torch.long),
            "confidence": torch.tensor(0.0, dtype=torch.float32),
            "eye_phase_target": torch.tensor(IGNORE_INDEX, dtype=torch.long),
            "eye_aperture_target": torch.tensor(
                teacher.eye_target if self.include_eye_targets else IGNORE_INDEX,
                dtype=torch.long,
            ),
            "yawn_target": torch.tensor(teacher.yawn_target, dtype=torch.long),
            "yawn_mask": torch.tensor(teacher.yawn_mask, dtype=torch.bool),
            "distraction_target": torch.tensor(0, dtype=torch.long),
            "distraction_mask": torch.tensor(False, dtype=torch.bool),
            "pose_mask": torch.tensor(False, dtype=torch.bool),
            "source": "nitymed",
            "supervision_weight": torch.tensor(0.25, dtype=torch.float32),
            "teacher_embedding": torch.from_numpy(teacher.embedding.copy()),
            "consistency_mask": torch.tensor(
                bool(region.visibility[0] > 0.0),
                dtype=torch.bool,
            ),
            "session": record.video_id,
            "subject": record.video_id,
            "protocol": "nitymed",
            "frame_id": record.frame_id,
            "event_id": "",
            "recording_category": record.category,
        }


__all__ = [
    "NITYMED_FRAME_SCHEMA_VERSION",
    "NITYMED_TEACHER_SCHEMA_VERSION",
    "NITYMED_VIDEO_SCHEMA_VERSION",
    "NitymedFrameRecord",
    "NitymedFrameDataset",
    "NitymedPseudoTargets",
    "NitymedTeacherRecord",
    "NitymedTeacherStore",
    "NitymedVideoRecord",
    "load_nitymed_frame_manifest",
    "load_nitymed_video_manifest",
    "nitymed_region_record",
    "save_nitymed_frame_manifest",
    "save_nitymed_teacher_cache",
    "save_nitymed_video_manifest",
    "scan_nitymed_videos",
    "select_nitymed_pseudo_targets",
]
