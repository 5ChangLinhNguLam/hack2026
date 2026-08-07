"""Prepare chronological NITYMED candidates for Method A visual training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.evidence_regions import (
    EvidenceRegionStore,
    project_normalized_boxes_to_letterbox,
    save_evidence_region_cache,
)
from ..data.nitymed import (
    NitymedFrameRecord,
    NitymedVideoRecord,
    load_nitymed_frame_manifest,
    load_nitymed_video_manifest,
    nitymed_region_record,
    save_nitymed_frame_manifest,
    save_nitymed_teacher_cache,
    save_nitymed_video_manifest,
    scan_nitymed_videos,
    select_nitymed_pseudo_targets,
)
from ..data.primitive_dataset import letterbox, normalized_image_tensor
from ..face_detection import (
    PeriodicFaceDetector,
    TrackedFaceDetector,
    UltraLightFaceDetector,
)
from ..landmarks import FaceBoxLandmarkDetector, MediaPipeFaceLandmarker
from ..models.mobilenet_lstm import MobileNetV3LargeVisualEncoder


def sampled_source_frame_ids(
    *,
    source_fps: float,
    sample_fps: float,
    frame_count: int,
) -> tuple[int, ...]:
    """Return source IDs nearest a fixed timestamp grid."""

    if source_fps <= 0.0 or sample_fps <= 0.0 or frame_count <= 0:
        raise ValueError("frame sampling metadata must be positive")
    if sample_fps > source_fps:
        raise ValueError("sample FPS cannot exceed source FPS")
    samples = math.ceil(frame_count * sample_fps / source_fps)
    return tuple(
        sorted(
            {
                min(frame_count - 1, round(sample * source_fps / sample_fps))
                for sample in range(samples)
            }
        )
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dataset_sha256(root: Path, videos: Sequence[NitymedVideoRecord]) -> str:
    digest = hashlib.sha256()
    for video in videos:
        digest.update(video.relative_path.as_posix().encode("utf-8"))
        with (root / video.relative_path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _prepare_manifest(dataset_root: Path, output: Path) -> dict[str, object]:
    videos = scan_nitymed_videos(dataset_root)
    checksum = _dataset_sha256(dataset_root, videos)
    save_nitymed_video_manifest(
        output / "videos.json",
        videos,
        dataset_sha256=checksum,
    )
    categories: dict[str, int] = {}
    for video in videos:
        categories[video.category] = categories.get(video.category, 0) + 1
    report = {
        "stage": "manifest",
        "videos": len(videos),
        "categories": categories,
        "dataset_sha256": checksum,
        "fps": sorted({video.fps for video in videos}),
        "resolutions": sorted({f"{video.width}x{video.height}" for video in videos}),
    }
    _atomic_json(output / "manifest_audit.json", report)
    return report


def _decode_selected_frames(
    *,
    dataset_root: Path,
    output: Path,
    video: NitymedVideoRecord,
    sample_fps: float,
    ultralight_model: Path,
    landmark_model: Path,
) -> tuple[list[NitymedFrameRecord], dict[str, object]]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to prepare NITYMED") from error

    selected = set(
        sampled_source_frame_ids(
            source_fps=video.fps,
            sample_fps=sample_fps,
            frame_count=video.frames,
        )
    )
    frame_dir = output / "frames" / video.video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(dataset_root / video.relative_path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode NITYMED video: {video.relative_path}")
    detector = PeriodicFaceDetector(
        TrackedFaceDetector(
            UltraLightFaceDetector(ultralight_model),
            max_missed_frames=2,
        ),
        interval_frames=1,
    )
    landmarker = MediaPipeFaceLandmarker(
        landmark_model,
        min_detection_confidence=0.3,
    )
    remapper = FaceBoxLandmarkDetector(detector, landmarker)
    frames: list[NitymedFrameRecord] = []
    regions = []
    decoded = 0
    try:
        source_id = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            decoded += 1
            if source_id not in selected:
                source_id += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            path = frame_dir / f"frame_{source_id:06d}.jpg"
            image.save(path, format="JPEG", quality=92)
            landmarks = remapper.detect(image)
            regions.append(
                nitymed_region_record(
                    frame_id=source_id,
                    image_size=image.size,
                    detection=remapper.latest_detection,
                    landmarks=landmarks,
                )
            )
            frames.append(
                NitymedFrameRecord(
                    video_id=video.video_id,
                    category=video.category,
                    frame_id=source_id,
                    timestamp=source_id / video.fps,
                    frame_path=path.relative_to(output),
                )
            )
            source_id += 1
    finally:
        capture.release()
        remapper.close()
    if len(frames) != len(selected):
        raise ValueError(
            f"{video.video_id}: selected {len(selected)} frames but decoded "
            f"{len(frames)}"
        )
    save_evidence_region_cache(output / "regions", video.video_id, regions)
    visibility = np.stack([record.visibility for record in regions])
    return frames, {
        "video_id": video.video_id,
        "decoded": decoded,
        "selected": len(frames),
        "face_visible": int((visibility[:, 0] > 0.0).sum()),
        "both_eyes_visible": int(
            ((visibility[:, 1] > 0.0) & (visibility[:, 2] > 0.0)).sum()
        ),
        "mouth_visible": int((visibility[:, 3] > 0.0).sum()),
    }


def _prepare_frames(
    *,
    dataset_root: Path,
    output: Path,
    sample_fps: float,
    ultralight_model: Path,
    landmark_model: Path,
) -> dict[str, object]:
    videos = load_nitymed_video_manifest(output / "videos.json")
    source_fps = {round(video.fps, 6) for video in videos}
    records: list[NitymedFrameRecord] = []
    audits: list[dict[str, object]] = []
    for video in videos:
        video_records, audit = _decode_selected_frames(
            dataset_root=dataset_root,
            output=output,
            video=video,
            sample_fps=sample_fps,
            ultralight_model=ultralight_model,
            landmark_model=landmark_model,
        )
        records.extend(video_records)
        audits.append(audit)
        print(json.dumps(audit), flush=True)
    save_nitymed_frame_manifest(
        output / "frames.json",
        records,
        source_fps=tuple(sorted(source_fps)),
        sample_fps=sample_fps,
    )
    report = {
        "stage": "frames",
        "videos": len(videos),
        "frames": len(records),
        "sample_fps": sample_fps,
        "source_fps": sorted(source_fps),
        "video_audits": audits,
    }
    _atomic_json(output / "frames_audit.json", report)
    return report


class _TeacherFrameDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        *,
        output: Path,
        image_size: tuple[int, int],
    ) -> None:
        self.output = output
        self.image_size = image_size
        self.records = load_nitymed_frame_manifest(output / "frames.json")
        self.stores = {
            video_id: EvidenceRegionStore(output / "regions", video_id)
            for video_id in sorted({record.video_id for record in self.records})
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(self.output / record.frame_path) as source:
            image = source.convert("RGB")
        region = self.stores[record.video_id].get(record.frame_id)
        boxes = project_normalized_boxes_to_letterbox(
            region.boxes,
            source_size=image.size,
            target_size=self.image_size,
        )
        return {
            "video_id": record.video_id,
            "frame_id": record.frame_id,
            "image": normalized_image_tensor(
                letterbox(image, self.image_size),
                self.image_size,
            ),
            "region_boxes": torch.from_numpy(boxes.copy()),
            "region_visibility": torch.from_numpy(region.visibility.copy()),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def _prepare_teacher_cache(
    *,
    output: Path,
    checkpoint: Path,
    requested_image_size: tuple[int, int],
    batch_size: int,
    workers: int,
    amp: bool,
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("teacher checkpoint is missing model configuration")
    image_size = tuple(int(value) for value in config.get("image_size", ()))
    if image_size != requested_image_size:
        raise ValueError(
            f"teacher image size {image_size} does not match "
            f"{requested_image_size}"
        )
    model = MobileNetV3LargeVisualEncoder(
        embedding_dim=int(config["embedding_dim"]),
        region_dim=int(config["region_dim"]),
        pretrained=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = _TeacherFrameDataset(output=output, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    by_video: dict[str, dict[str, list[torch.Tensor]]] = {}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        boxes = batch["region_boxes"].to(device, non_blocking=True)
        visibility = batch["region_visibility"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            result = model(image, boxes, visibility)
        eye = result.eye_logits.float().softmax(dim=-1).cpu()
        yawn = result.yawn_logits.float().softmax(dim=-1).cpu()
        embedding = result.embedding.float().cpu()
        for row, video_id in enumerate(batch["video_id"]):
            target = by_video.setdefault(
                str(video_id),
                {
                    "frame_ids": [],
                    "embeddings": [],
                    "eye": [],
                    "yawn": [],
                    "visibility": [],
                },
            )
            target["frame_ids"].append(batch["frame_id"][row].reshape(1))
            target["embeddings"].append(embedding[row].unsqueeze(0))
            target["eye"].append(eye[row].unsqueeze(0))
            target["yawn"].append(yawn[row].unsqueeze(0))
            target["visibility"].append(
                batch["region_visibility"][row].unsqueeze(0)
            )
    audits: dict[str, dict[str, int]] = {}
    for video_id, values in sorted(by_video.items()):
        tensors = {
            key: torch.cat(parts, dim=0)
            for key, parts in values.items()
        }
        save_nitymed_teacher_cache(
            output / "teacher",
            video_id,
            frame_ids=tensors["frame_ids"].flatten(),
            embeddings=tensors["embeddings"],
            eye_probabilities=tensors["eye"],
            yawn_probabilities=tensors["yawn"],
            region_visibility=tensors["visibility"],
        )
        accepted = select_nitymed_pseudo_targets(
            eye_probabilities=tensors["eye"],
            yawn_probabilities=tensors["yawn"],
            region_visibility=tensors["visibility"],
        )
        audits[video_id] = {
            "frames": int(tensors["frame_ids"].numel()),
            "eye_open": int((accepted.eye == 0).sum()),
            "eye_closed": int((accepted.eye == 2).sum()),
            "eye_ignored": int((~accepted.eye_mask).sum()),
            "yawn_no": int((accepted.yawn == 0).sum()),
            "yawn_yes": int((accepted.yawn == 1).sum()),
            "yawn_ignored": int((~accepted.yawn_mask).sum()),
        }
        print(json.dumps({"video_id": video_id, **audits[video_id]}), flush=True)
    report = {
        "stage": "teacher-cache",
        "checkpoint_sha256": _file_sha256(checkpoint),
        "videos": len(audits),
        "frames": len(dataset),
        "video_audits": audits,
    }
    _atomic_json(output / "teacher_audit.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare NITYMED for Method A mixed visual training"
    )
    parser.add_argument(
        "--stage",
        choices=("manifest", "frames", "teacher-cache"),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--ultralight-model", type=Path)
    parser.add_argument("--landmark-model", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if args.sample_fps <= 0.0:
        raise ValueError("--sample-fps must be positive")
    if min(args.image_width, args.image_height, args.batch_size) <= 0:
        raise ValueError("image dimensions and batch size must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.stage == "manifest":
        report = _prepare_manifest(dataset_root, output)
    elif args.stage == "frames":
        if args.ultralight_model is None or args.landmark_model is None:
            raise ValueError("frames stage requires detector and landmark models")
        report = _prepare_frames(
            dataset_root=dataset_root,
            output=output,
            sample_fps=args.sample_fps,
            ultralight_model=args.ultralight_model.resolve(),
            landmark_model=args.landmark_model.resolve(),
        )
    else:
        if args.teacher_checkpoint is None:
            raise ValueError("teacher-cache stage requires --teacher-checkpoint")
        report = _prepare_teacher_cache(
            output=output,
            checkpoint=args.teacher_checkpoint.resolve(),
            requested_image_size=(args.image_width, args.image_height),
            batch_size=args.batch_size,
            workers=args.workers,
            amp=args.amp,
        )
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
