"""Cache YOLO detections from image_2 for repeatable C1 experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import cv2

from tripkit import TripLoader

from .detector import OpenCVDnnYoloDetector

ROOT = Path(__file__).resolve().parents[2]


def cache_trip(
    loader: TripLoader,
    detector: OpenCVDnnYoloDetector,
    *,
    output: Path,
    stride: int,
    confidence: float,
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride phải >= 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for frame_id in range(0, loader.n_frames, stride):
        # Read image_2 directly so even the loader never touches depth/right/driver.
        image = cv2.imread(str(loader.left_path(frame_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(loader.left_path(frame_id))
        detections = detector.detect(image)
        rows.append({
            "frame_id": frame_id,
            "timestamp": float(loader.raw_frame(frame_id).get(
                "timestamp", frame_id / loader.fps
            )),
            "detections": [{
                "class_id": item.class_id,
                "label": item.label,
                "confidence": round(item.confidence, 6),
                "bbox": [round(value, 3) for value in item.bbox],
            } for item in detections],
        })
        if len(rows) % 200 == 0:
            print(f"{loader.trip_id}: {len(rows)} detector frame", flush=True)
    temporary = output.with_name(output.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump({
            "trip_id": loader.trip_id,
            "stride": stride,
            "confidence": confidence,
            "source_camera": "image_2",
            "rows": rows,
        }, stream, separators=(",", ":"))
    temporary.replace(output)
    elapsed = time.perf_counter() - started
    return {
        "trip_id": loader.trip_id,
        "frames": loader.n_frames,
        "detector_frames": len(rows),
        "elapsed_s": round(elapsed, 2),
        "equivalent_output_fps": round(loader.n_frames / max(elapsed, 1e-9), 2),
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache YOLO chỉ từ image_2.")
    parser.add_argument("trip_dirs", nargs="+")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "predictions/c1_detection_cache"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        detector = OpenCVDnnYoloDetector(
            ROOT / "models/yolo11s.onnx",
            ROOT / "models/driver-objects.labels",
            confidence_threshold=args.confidence,
            device=args.device,
        )
        reports = []
        for trip_dir in args.trip_dirs:
            loader = TripLoader(trip_dir)
            confidence_tag = f"{int(round(args.confidence * 100)):03d}"
            output = args.output_dir / (
                f"{loader.trip_id}.stride{args.stride}.conf{confidence_tag}.json.gz"
            )
            if output.exists() and not args.force:
                reports.append({"trip_id": loader.trip_id, "output": str(output), "skipped": True})
                continue
            reports.append(cache_trip(
                loader,
                detector,
                output=output,
                stride=args.stride,
                confidence=args.confidence,
            ))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Lỗi cache C1 detection: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
