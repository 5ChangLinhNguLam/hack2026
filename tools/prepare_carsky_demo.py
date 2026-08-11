#!/usr/bin/env python3
"""Build a truth-free trip bundle for the CarSky edge-runtime image.

The source practice trips contain labels, depth and outcome fields.  Those
files must never be present in the deployed inference image, even if the
runtime promises not to use them.  This tool copies only the two camera
streams used by C1/C2 and the causal ego/session fields required at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Mapping, Sequence


METADATA_FIELDS = (
    "trip_id",
    "description",
    "duration_sec",
    "fps",
    "weather",
    "speed_limit_kmh",
)
EGO_FIELDS = (
    "speed_kmh",
    "longitudinal_accel",
    "lateral_accel",
)
FORBIDDEN_FRAME_FIELDS = (
    "targets",
    "driver",
    "events_active",
    "min_ttc",
    "headway_sec",
    "behavior_flags",
    "risk",
)
FORBIDDEN_TOP_LEVEL_FIELDS = (
    "driver_summary",
    "trip_aggregate",
    "events_log",
)
FORBIDDEN_PATH_PARTS = (
    "depth",
    "image_3",
    "label_2",
    "calib",
)
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
TRIP_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)


def _safe_trip_id(value: object, *, directory_name: str) -> str:
    if not isinstance(value, str) or TRIP_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "trip_id chỉ được chứa chữ ASCII, số, dấu '-' hoặc '_'"
        )
    if value != directory_name:
        raise ValueError(
            f"trip_id {value!r} không khớp thư mục nguồn {directory_name!r}"
        )
    return value


def _contained_path(root: Path, candidate: Path, *, field: str) -> Path:
    root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} nằm ngoài thư mục cho phép") from exc
    return resolved


def _read_source_json(trip_dir: Path) -> tuple[Path, dict[str, Any]]:
    name = trip_dir.name
    candidates = (trip_dir / f"{name}.json", trip_dir / f"{name}.json.gz")
    for candidate in candidates:
        if candidate.is_file():
            candidate = _contained_path(
                trip_dir, candidate, field="JSON trip nguồn"
            )
            if candidate.suffix == ".gz":
                import gzip

                with gzip.open(candidate, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON trip nguồn phải là object")
            return candidate, payload
    raise FileNotFoundError(f"Không tìm thấy JSON trip trong {trip_dir}")


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} phải là số")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} phải hữu hạn")
    return number


def _image_path(directory: Path, stem: str, *, source_root: Path) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return _contained_path(
                source_root, candidate, field=f"Ảnh nguồn {candidate.name}"
            )
    raise FileNotFoundError(f"Thiếu ảnh {stem} trong {directory}")


def _sanitize_metadata(raw: Mapping[str, Any], *, trip_id: str) -> dict[str, Any]:
    metadata = {
        key: raw[key]
        for key in METADATA_FIELDS
        if key in raw and raw[key] is not None
    }
    metadata["trip_id"] = trip_id
    fps = _finite_number(metadata.get("fps", 20.0), field="metadata.fps")
    if fps <= 0.0:
        raise ValueError("metadata.fps phải > 0")
    metadata["fps"] = fps
    speed_limit = _finite_number(
        metadata.get("speed_limit_kmh", 0.0), field="metadata.speed_limit_kmh"
    )
    if speed_limit < 0.0:
        raise ValueError("metadata.speed_limit_kmh phải >= 0")
    metadata["speed_limit_kmh"] = speed_limit
    return metadata


def _sanitize_frame(raw: Mapping[str, Any], *, expected_id: int, fps: float) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"frames[{expected_id}] phải là object")
    frame_id = int(raw.get("frame_id", expected_id))
    if frame_id != expected_id:
        raise ValueError(
            f"frame_id không liên tiếp: gặp {frame_id}, cần {expected_id}"
        )
    timestamp = _finite_number(
        raw.get("timestamp", frame_id / fps), field=f"frames[{frame_id}].timestamp"
    )
    ego_raw = raw.get("ego")
    if not isinstance(ego_raw, Mapping):
        raise ValueError(f"frames[{frame_id}].ego phải là object")
    missing = [key for key in EGO_FIELDS if key not in ego_raw]
    if missing:
        raise ValueError(
            f"frames[{frame_id}].ego thiếu field bắt buộc: {', '.join(missing)}"
        )
    ego = {
        key: _finite_number(ego_raw[key], field=f"frames[{frame_id}].ego.{key}")
        for key in EGO_FIELDS
    }
    return {"frame_id": frame_id, "timestamp": timestamp, "ego": ego}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_truth_free(trip_dir: Path, payload: Mapping[str, Any]) -> None:
    forbidden_paths = sorted(
        str(path.relative_to(trip_dir))
        for path in trip_dir.rglob("*")
        if path.is_file()
        and any(part in FORBIDDEN_PATH_PARTS for part in path.relative_to(trip_dir).parts)
    )
    if forbidden_paths:
        raise RuntimeError(f"Bundle chứa modality cấm: {forbidden_paths[:5]}")
    present_top = sorted(set(payload).intersection(FORBIDDEN_TOP_LEVEL_FIELDS))
    if present_top:
        raise RuntimeError(f"Bundle chứa GT cấp trip: {present_top}")
    for index, frame in enumerate(payload.get("frames") or []):
        present = sorted(set(frame).intersection(FORBIDDEN_FRAME_FIELDS))
        if present:
            raise RuntimeError(f"Bundle frame {index} chứa field cấm: {present}")


def prepare_bundle(
    source_trip: Path,
    output_root: Path,
    *,
    limit: int | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    source_trip = source_trip.resolve()
    output_root = output_root.resolve()
    if not source_trip.is_dir():
        raise FileNotFoundError(f"Trip nguồn không tồn tại: {source_trip}")
    if limit is not None and limit <= 0:
        raise ValueError("limit phải > 0")

    directory_trip_id = _safe_trip_id(
        source_trip.name, directory_name=source_trip.name
    )
    _json_path, source = _read_source_json(source_trip)
    frames_raw = source.get("frames") or []
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ValueError("Trip nguồn không có frames")
    count = len(frames_raw) if limit is None else min(limit, len(frames_raw))
    trip_id = _safe_trip_id(
        source.get("trip_id"), directory_name=directory_trip_id
    )
    metadata_raw = source.get("metadata")
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("metadata phải là object")
    if "trip_id" in metadata_raw:
        _safe_trip_id(metadata_raw["trip_id"], directory_name=directory_trip_id)
    destination = _contained_path(
        output_root, output_root / trip_id, field="Bundle đích"
    )
    temporary = _contained_path(
        output_root,
        output_root / f".{trip_id}.tmp-{time.time_ns()}",
        field="Bundle tạm",
    )
    if destination.exists() and not replace_existing:
        raise FileExistsError(
            f"Bundle đã tồn tại: {destination}; dùng --force để lưu bản cũ rồi thay"
        )

    metadata = _sanitize_metadata(metadata_raw, trip_id=trip_id)
    frames = [
        _sanitize_frame(frames_raw[index], expected_id=index, fps=metadata["fps"])
        for index in range(count)
    ]
    payload = {"trip_id": trip_id, "metadata": metadata, "frames": frames}

    output_root.mkdir(parents=True, exist_ok=True)
    (temporary / "driver").mkdir(parents=True)
    (temporary / "kitti" / "image_2").mkdir(parents=True)
    try:
        for frame_id in range(count):
            road = _image_path(
                source_trip / "kitti" / "image_2",
                f"{frame_id:06d}",
                source_root=source_trip,
            )
            cabin = _image_path(
                source_trip / "driver",
                f"frame_{frame_id:06d}",
                source_root=source_trip,
            )
            shutil.copyfile(road, temporary / "kitti" / "image_2" / road.name)
            shutil.copyfile(cabin, temporary / "driver" / cabin.name)

        json_path = temporary / f"{trip_id}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        _assert_truth_free(temporary, payload)
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "schema": "safeloop.carsky.demo-bundle.v1",
            "trip_id": trip_id,
            "frames": count,
            "source_fps": metadata["fps"],
            "modalities": ["road_camera_left", "driver_camera", "ego_kinematics"],
            "truth_free": True,
            "forbidden_modalities_absent": list(FORBIDDEN_PATH_PARTS),
            "files": {
                str(path.relative_to(temporary)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            },
        }
        (temporary / "BUNDLE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        archived: Path | None = None
        if destination.exists():
            archive_root = _contained_path(
                output_root,
                output_root / "archive" / str(time.time_ns()),
                field="Thư mục archive",
            )
            archive_root.mkdir(parents=True, exist_ok=True)
            archived = _contained_path(
                output_root,
                archive_root / destination.name,
                field="Bundle archive",
            )
            destination.replace(archived)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "trip_id": trip_id,
        "frames": count,
        "destination": str(destination),
        "archived_previous": str(archived) if archived is not None else None,
        "truth_free": True,
        "manifest": str(destination / "BUNDLE_MANIFEST.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tạo bundle replay CarSky chỉ chứa input inference hợp lệ"
    )
    parser.add_argument("source_trip", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".carsky-build/demo"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="lưu bundle cũ vào archive rồi thay bằng bundle mới",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_bundle(
            args.source_trip,
            args.output_root,
            limit=args.limit,
            replace_existing=args.force,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
