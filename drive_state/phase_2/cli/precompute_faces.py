from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..data.face_cache import FaceCropStore, save_encoded_face_crop_cache
from ..data.manifest import SessionRecord, load_sessions
from ..data.primitive_records import load_primitive_records
from ..face_detection import (
    FaceDetection,
    PeriodicFaceDetector,
    TrackedFaceDetector,
    UltraLightFaceDetector,
)
from ..landmarks import MediaPipeFaceLandmarker
from ..runtime import estimate_head_pose


def _crop(image: Image.Image, detection: FaceDetection | None, size: int = 224) -> Image.Image:
    if detection is None:
        return Image.new("RGB", (size, size))
    box = detection.box
    margin = round(max(box.x2 - box.x1, box.y2 - box.y1) * 0.35)
    x1, y1 = max(0, box.x1 - margin), max(0, box.y1 - margin)
    x2, y2 = min(image.width, box.x2 + margin), min(image.height, box.y2 + margin)
    return image.crop((x1, y1, x2, y2)).resize((size, size), Image.Resampling.BILINEAR)


def _encode_face(image: Image.Image) -> np.ndarray:
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=92)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def _leading_missing_rows(visibility: Sequence[float]) -> tuple[int, ...]:
    """Rows before the first valid detection, once a valid row exists."""
    first_visible = next(
        (index for index, value in enumerate(visibility) if value > 0.0), None
    )
    if first_visible is None:
        return ()
    return tuple(range(first_visible))


def _process_session(
    session: SessionRecord,
    cache_dir: Path,
    ultra_path: Path,
    landmark_path: Path,
    detector_interval: int,
    landmark_interval: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite:
        try:
            existing = FaceCropStore(cache_dir, session.name)
            expected_ids = np.arange(session.n_frames, dtype=np.int32)
            if (
                existing.rate == "20fps"
                and np.array_equal(existing.frame_ids, expected_ids)
            ):
                return {"session": session.name, "status": "complete"}
        except FileNotFoundError:
            pass

    ultra = UltraLightFaceDetector(ultra_path)
    landmarker = MediaPipeFaceLandmarker(landmark_path, min_detection_confidence=0.3)
    try:
        tracked = TrackedFaceDetector(ultra, max_missed_frames=10)
        detector = PeriodicFaceDetector(
            tracked, interval_frames=detector_interval
        )
        records = load_primitive_records((session,))
        encoded: list[np.ndarray] = []
        visibility: list[float] = []
        pitches: list[float] = []
        yaws: list[float] = []
        last_pitch = 0.0
        last_yaw = 0.0
        pose_refreshes = 0
        for row_index, record in enumerate(records):
            with Image.open(record.frame_path) as source:
                image = source.convert("RGB")
            detection = detector.detect(image)
            face = _crop(image, detection)
            if detection is not None and row_index % landmark_interval == 0:
                landmarks = landmarker.detect(face)
                if landmarks is not None:
                    last_pitch, last_yaw = estimate_head_pose(
                        landmarks, face.width, face.height
                    )
                pose_refreshes += 1
            encoded.append(_encode_face(face))
            visibility.append(detection.confidence if detection is not None else 0.0)
            pitches.append(last_pitch if detection is not None else 0.0)
            yaws.append(last_yaw if detection is not None else 0.0)
            if detection is not None:
                # If the very first frame(s) missed, use the first nearby
                # successful box to crop those same source images. This uses no
                # future label or pixels—only a location prior over a few cabin
                # frames—and prevents a scheduled startup miss from becoming
                # black training data.
                for missing_row in _leading_missing_rows(visibility):
                    with Image.open(records[missing_row].frame_path) as source:
                        missing_image = source.convert("RGB")
                    encoded[missing_row] = _encode_face(
                        _crop(missing_image, detection)
                    )
                    visibility[missing_row] = detection.confidence
                    pitches[missing_row] = last_pitch
                    yaws[missing_row] = last_yaw
        save_encoded_face_crop_cache(
            cache_dir,
            session.name,
            encoded=encoded,
            frame_ids=np.asarray([record.frame_id for record in records], dtype=np.int32),
            visibility=np.asarray(visibility, dtype=np.float32),
            pitch=np.asarray(pitches, dtype=np.float32),
            yaw=np.asarray(yaws, dtype=np.float32),
            rate="20fps",
        )
        return {
            "session": session.name,
            "frames": len(records),
            "detected_or_tracked": sum(value > 0 for value in visibility),
            "missing": sum(value == 0 for value in visibility),
            "pose_refreshes": pose_refreshes,
        }
    finally:
        landmarker.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild dense 20 FPS face crops with Ultra-Light (no dropped rows)"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ultralight-model", type=Path)
    parser.add_argument("--landmark-model", type=Path)
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=("s1", "s2", "s3", "s5", "s6"),
        help="DMD protocols to rebuild (default: all available protocols)",
    )
    parser.add_argument("--session", action="append", dest="sessions")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent session workers (use at most the available CPU cores)",
    )
    parser.add_argument(
        "--detector-interval",
        type=int,
        default=5,
        help="Refresh Ultra-Light every N frames and hold the tracked crop between",
    )
    parser.add_argument(
        "--landmark-interval",
        type=int,
        default=5,
        help="Estimate and hold head pose every N frames; face detection remains dense",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.detector_interval < 1:
        raise ValueError("--detector-interval must be at least 1")
    if args.landmark_interval < 1:
        raise ValueError("--landmark-interval must be at least 1")
    ultra_path = args.ultralight_model or args.root / "models/version-RFB-320.onnx"
    landmark_path = args.landmark_model or args.root / "models/face_landmarker.task"
    sessions = tuple(
        session
        for session in load_sessions(args.root / "labels_20fps/manifest_20fps.json")
        if session.protocol in set(args.protocol)
        and (not args.sessions or session.name in set(args.sessions))
    )
    jobs = (
        (
            session,
            args.cache_dir,
            ultra_path,
            landmark_path,
            args.detector_interval,
            args.landmark_interval,
            args.overwrite,
        )
        for session in sessions
    )
    if args.workers == 1:
        for job in jobs:
            print(json.dumps(_process_session(*job)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_session, *job) for job in jobs]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
