from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Sequence

from ..data.eye_sequences import load_eye_sequences, load_frame_sequences
from ..data.face_cache import FaceCropStore
from ..data.manifest import load_sessions
from ..face_detection import TrackedFaceDetector, UltraLightFaceDetector
from ..landmarks import (
    FaceBoxLandmarkDetector,
    MediaPipeFaceLandmarker,
    precompute_eye_crop_cache,
    precompute_eye_crop_cache_from_face_store,
)


def _process_sequence(
    sequence,
    *,
    cache_dir: Path,
    ultra_path: Path,
    landmark_path: Path,
    face_cache: Path | None,
    overwrite: bool,
) -> dict[str, object]:
    landmarker = MediaPipeFaceLandmarker(
        landmark_path, min_detection_confidence=0.3
    )
    try:
        if face_cache is not None:
            stats = precompute_eye_crop_cache_from_face_store(
                sequence,
                FaceCropStore(face_cache, sequence.session_name),
                landmarker,
                cache_dir,
                overwrite=overwrite,
            )
        else:
            ultra = UltraLightFaceDetector(ultra_path)
            tracked = TrackedFaceDetector(ultra, max_missed_frames=10)
            detector = FaceBoxLandmarkDetector(tracked, landmarker, padding=0.15)
            stats = precompute_eye_crop_cache(
                sequence,
                detector,
                cache_dir,
                overwrite=overwrite,
            )
        return {"session": sequence.session_name, **stats.__dict__}
    finally:
        landmarker.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute complete 20 FPS left/right eye crops with Ultra-Light + MediaPipe"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ultralight-model", type=Path)
    parser.add_argument("--landmark-model", type=Path)
    parser.add_argument(
        "--face-cache",
        type=Path,
        help="Complete dense 20 FPS face cache; avoids running Ultra-Light again",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--session", action="append", dest="sessions")
    parser.add_argument(
        "--all-protocols",
        action="store_true",
        help="Build dense ROI caches for s1/s2/s3/s5/s6 instead of s5 only",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    ultra_path = args.ultralight_model or args.root / "models/version-RFB-320.onnx"
    landmark_path = args.landmark_model or args.root / "models/face_landmarker.task"
    sessions = load_sessions(args.root / "labels_20fps/manifest_20fps.json")
    sequences = (
        load_frame_sequences(sessions) if args.all_protocols else load_eye_sequences(sessions)
    )
    if args.sessions:
        requested = set(args.sessions)
        sequences = tuple(sequence for sequence in sequences if sequence.session_name in requested)
        missing = requested.difference(sequence.session_name for sequence in sequences)
        if missing:
            scope = "all protocols" if args.all_protocols else "s5"
            raise SystemExit(f"unknown sessions in {scope} scope: {sorted(missing)}")

    jobs = tuple(
        {
            "sequence": sequence,
            "cache_dir": args.cache_dir,
            "ultra_path": ultra_path,
            "landmark_path": landmark_path,
            "face_cache": args.face_cache,
            "overwrite": args.overwrite,
        }
        for sequence in sequences
    )
    if args.workers == 1:
        for job in jobs:
            print(json.dumps(_process_sequence(**job)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_sequence, **job) for job in jobs]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
