"""Precompute dense full-frame regions for the single-backbone evidence model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..data.evidence_regions import (
    EvidenceRegionStore,
    save_evidence_region_cache,
)
from ..data.face_cache import FaceCropStore
from ..data.manifest import SessionRecord, load_sessions
from ..data.primitive_records import load_primitive_records
from ..face_detection import (
    FaceDetection,
    PeriodicFaceDetector,
    TrackedFaceDetector,
    UltraLightFaceDetector,
)
from ..landmarks import (
    FaceBoxLandmarkDetector,
    MediaPipeFaceLandmarker,
    evidence_regions_from_landmarks,
)


class _HeldFaceDetector:
    """Expose an already-computed detection to the landmark remapper."""

    def __init__(self) -> None:
        self.current: FaceDetection | None = None

    def detect(self, _image: Image.Image) -> FaceDetection | None:
        return self.current


def _process_session(
    session: SessionRecord,
    *,
    face_cache_dir: Path,
    output_dir: Path,
    ultralight_model: Path,
    landmark_model: Path,
    detector_interval: int,
    landmark_interval: int,
    overwrite: bool,
) -> dict[str, object]:
    records = load_primitive_records((session,))
    expected_ids = np.asarray(
        [record.frame_id for record in records],
        dtype=np.int64,
    )
    if not overwrite:
        try:
            existing = EvidenceRegionStore(output_dir, session.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not np.array_equal(existing.frame_ids, expected_ids):
                raise ValueError(
                    f"region cache is misaligned for {session.name}"
                )
            return {
                "session": session.name,
                "status": "complete",
                "frames": len(expected_ids),
            }

    face_store = FaceCropStore(face_cache_dir, session.name)
    if face_store.rate != "20fps":
        raise ValueError(
            f"face cache for {session.name} must be 20fps, "
            f"got {face_store.rate!r}"
        )
    if not np.array_equal(face_store.frame_ids, expected_ids):
        raise ValueError(
            f"face cache is incomplete or misaligned for {session.name}"
        )

    ultralight = UltraLightFaceDetector(ultralight_model)
    tracked = TrackedFaceDetector(ultralight, max_missed_frames=10)
    detector = PeriodicFaceDetector(
        tracked,
        interval_frames=detector_interval,
    )
    landmarker = MediaPipeFaceLandmarker(
        landmark_model,
        min_detection_confidence=0.3,
    )
    held = _HeldFaceDetector()
    remapper = FaceBoxLandmarkDetector(held, landmarker)
    output = []
    latest_landmarks = None
    landmark_refreshes = 0
    try:
        for row, record in enumerate(records):
            with Image.open(record.frame_path) as source:
                image = source.convert("RGB")
            detection = detector.detect(image)
            held.current = detection
            if detection is None:
                latest_landmarks = None
            elif row % landmark_interval == 0 or latest_landmarks is None:
                latest_landmarks = remapper.detect(image)
                landmark_refreshes += 1
            metadata = face_store.metadata(record.frame_id)
            output.append(
                evidence_regions_from_landmarks(
                    frame_id=record.frame_id,
                    face_box=None if detection is None else detection.box,
                    landmarks=latest_landmarks,
                    image_width=image.width,
                    image_height=image.height,
                    pitch=metadata.pitch if metadata.visible else 0.0,
                    yaw=metadata.yaw if metadata.visible else 0.0,
                    face_visibility=(
                        0.0 if detection is None else detection.confidence
                    ),
                )
            )
    finally:
        remapper.close()

    save_evidence_region_cache(output_dir, session.name, output)
    visibility = np.stack([record.visibility for record in output])
    return {
        "session": session.name,
        "frames": len(output),
        "face_visible": int((visibility[:, 0] > 0.0).sum()),
        "both_eyes_visible": int(
            ((visibility[:, 1] > 0.0) & (visibility[:, 2] > 0.0)).sum()
        ),
        "mouth_visible": int((visibility[:, 3] > 0.0).sum()),
        "landmark_refreshes": landmark_refreshes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute dense full-frame face/eye/mouth regions using "
            "Ultra-Light and aligned face-cache pose"
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--face-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ultralight-model", type=Path)
    parser.add_argument("--landmark-model", type=Path)
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=("s1", "s2", "s3", "s5", "s6"),
    )
    parser.add_argument("--session", action="append", dest="sessions")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--detector-interval", type=int, default=5)
    parser.add_argument("--landmark-interval", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.detector_interval <= 0 or args.landmark_interval <= 0:
        raise ValueError("detector and landmark intervals must be positive")
    ultralight_model = (
        args.ultralight_model
        or args.root / "models" / "version-RFB-320.onnx"
    )
    landmark_model = (
        args.landmark_model
        or args.root / "models" / "face_landmarker.task"
    )
    selected = set(args.sessions or ())
    protocols = set(args.protocol)
    sessions = tuple(
        session
        for session in load_sessions(
            args.root / "labels_20fps" / "manifest_20fps.json"
        )
        if session.protocol in protocols
        and (not selected or session.name in selected)
    )
    jobs = tuple(
        {
            "session": session,
            "face_cache_dir": args.face_cache,
            "output_dir": args.output,
            "ultralight_model": ultralight_model,
            "landmark_model": landmark_model,
            "detector_interval": args.detector_interval,
            "landmark_interval": args.landmark_interval,
            "overwrite": args.overwrite,
        }
        for session in sessions
    )
    if args.workers == 1:
        for job in jobs:
            print(json.dumps(_process_session(**job)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_process_session, **job)
                for job in jobs
            ]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
