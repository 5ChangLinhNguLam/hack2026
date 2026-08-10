"""CLI for the offline C1 source-JSON sanitization boundary."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .lean_loader import (
    C1_MANIFEST_SCHEMA_VERSION,
    C1_MANIFEST_SOURCE_CAMERA,
    _EGO_KINEMATICS_FIELDS,
    _validate_and_project_manifest,
)


def prepare_c1_manifest(
    trip_dir: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Create the sanitized C1 manifest as an explicit offline operation.

    ``output_path`` must not be inside the trip directory. Callers should use
    an external or Git-ignored run-artifact directory. ``.json.gz`` output is
    selected by a ``.gz`` suffix.
    """

    trip = Path(trip_dir)
    if not trip.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {trip}")
    output = Path(output_path)
    if output.resolve(strict=False).is_relative_to(trip.resolve()):
        raise ValueError(
            f"C1 manifest must be outside the source trip directory: {output}"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"C1 manifest already exists: {output}")

    source_path = _resolve_source_json_path(trip)
    source_document = _load_source_json(source_path)
    manifest = _sanitize_source_document(source_document, trip)
    del source_document
    # Use the runtime validator before publishing the boundary artifact. This
    # keeps preparation and inference on one fail-closed schema contract.
    _validate_and_project_manifest(manifest)
    _write_json_atomic(output, manifest)
    return output


def _resolve_source_json_path(trip_dir: Path) -> Path:
    """Resolve source JSON offline, tolerating a ``.json`` directory shadow."""

    name = trip_dir.name
    preferred = (trip_dir / f"{name}.json", trip_dir / f"{name}.json.gz")
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    fallback = sorted(
        path
        for pattern in ("*.json", "*.json.gz")
        for path in trip_dir.glob(pattern)
        if path.is_file()
    )
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"No source JSON file found for trip {name} in {trip_dir}")


def _load_source_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sanitize_source_document(document: Any, trip_dir: Path) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError("Trip source JSON root must be an object")
    source_trip_id = document.get("trip_id")
    if source_trip_id is None:
        trip_id = trip_dir.name
    elif isinstance(source_trip_id, str) and source_trip_id:
        trip_id = source_trip_id
    else:
        raise ValueError("Trip source JSON trip_id must be a non-empty string")
    if "frames" not in document:
        raise ValueError("Trip source JSON is missing frames")
    raw_frames = document["frames"]
    if not isinstance(raw_frames, list):
        raise ValueError("Trip source JSON frames must be a list")
    frames = [
        _sanitize_source_frame(raw_frame, position)
        for position, raw_frame in enumerate(raw_frames)
    ]
    return {
        "schema_version": C1_MANIFEST_SCHEMA_VERSION,
        "trip_id": trip_id,
        "source_camera": C1_MANIFEST_SOURCE_CAMERA,
        "expected_frames": len(frames),
        "frames": frames,
    }


def _sanitize_source_frame(raw_frame: Any, position: int) -> dict[str, Any]:
    if not isinstance(raw_frame, Mapping):
        raise ValueError(f"frames[{position}] must be an object")
    if "frame_id" not in raw_frame:
        raise ValueError(f"frames[{position}] is missing frame_id")
    if "timestamp" not in raw_frame:
        raise ValueError(f"frames[{position}] is missing timestamp")
    raw_ego_value = raw_frame.get("ego")
    raw_ego = {} if raw_ego_value is None else raw_ego_value
    if not isinstance(raw_ego, Mapping):
        raise ValueError(f"frames[{position}].ego must be an object or null")
    frame_id = raw_frame["frame_id"]
    if type(frame_id) is not int:
        raise ValueError(f"frames[{position}].frame_id must be a true int")
    timestamp = _source_finite_number(
        raw_frame["timestamp"], f"frames[{position}].timestamp"
    )
    ego: dict[str, float] = {}
    for field in _EGO_KINEMATICS_FIELDS:
        value = raw_ego.get(field)
        if value is not None:
            ego[field] = _source_finite_number(value, f"frames[{position}].ego.{field}")
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "ego": ego,
    }


def _source_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric and not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".json.gz" if path.suffix == ".gz" else ".json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if path.suffix == ".gz":
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
        else:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m safeloop.c1.prepare_manifest",
        description=(
            "Create a sanitized C1 runtime manifest containing only frame ID, "
            "timestamp and allowed ego kinematics. Output must stay outside "
            "the dataset and must not be committed."
        ),
    )
    parser.add_argument("trip_dir", type=Path, help="source trip directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="external/ignored .json or .json.gz manifest path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = prepare_c1_manifest(
            args.trip_dir,
            args.output,
            overwrite=args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"C1 manifest preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Sanitized C1 manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_c1_manifest"]
