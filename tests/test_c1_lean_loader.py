from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import pytest

from safeloop.c1 import (
    C1Image2IntegrityError,
    C1LeanTripLoader,
)
from safeloop.c1.prepare_manifest import main as prepare_manifest_main
from safeloop.c1.prepare_manifest import prepare_c1_manifest


def _write_source_trip(
    root: Path,
    trip_id: str,
    frames: list[Mapping[str, object]],
    image_frame_ids: tuple[int, ...],
    *,
    shadow_json_directory: bool = False,
) -> Path:
    trip_dir = root / "source_dataset" / trip_id
    image_dir = trip_dir / "kitti" / "image_2"
    image_dir.mkdir(parents=True)
    for frame_id in image_frame_ids:
        image = np.full((8, 12, 3), frame_id, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{frame_id:06d}.jpg"), image)

    if shadow_json_directory:
        (trip_dir / f"{trip_id}.json").mkdir()
    document = {
        "trip_id": trip_id,
        "metadata": {"fps": 20, "ground_truth_note": "must not be retained"},
        "events_log": [{"type": "lead_brake", "t": 0.1}],
        "trip_aggregate": {"score": 1.0},
        "driver_summary": {"state": "microsleep"},
        "frames": frames,
    }
    with gzip.open(trip_dir / f"{trip_id}.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(document, stream)
    return trip_dir


def _prepare(tmp_path: Path, trip_dir: Path, *, compressed: bool = False) -> Path:
    suffix = ".json.gz" if compressed else ".json"
    manifest = tmp_path / "runtime_manifests" / f"{trip_dir.name}{suffix}"
    return prepare_c1_manifest(trip_dir, manifest)


def _full_frame(frame_id: int) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp": frame_id / 20.0,
        "ego": {
            "speed_kmh": 31.0 + frame_id,
            "longitudinal_accel": -0.2,
            "lateral_accel": 0.1,
            "location": {"x": 999.0},
            "rotation": {"yaw": 10.0},
            "geolocation": {"latitude": 1.0},
        },
        "targets": [{"ttc_simple": 0.01, "rel_pos": [0, 0, 0]}],
        "events_active": [{"type": "ground_truth_event"}],
        "driver": {"state": "microsleep"},
        "min_ttc": 0.01,
        "headway_sec": 0.02,
        "behavior_flags": {"hard_brake": True},
        "risk": {"score": 1.0},
    }


def test_offline_preparation_handles_json_directory_shadow_and_sanitizes(
    tmp_path: Path,
) -> None:
    trip_dir = _write_source_trip(
        tmp_path,
        "T88d",
        [_full_frame(0)],
        (0,),
        shadow_json_directory=True,
    )
    manifest_path = _prepare(tmp_path, trip_dir, compressed=True)

    with gzip.open(manifest_path, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert set(manifest) == {
        "schema_version",
        "trip_id",
        "source_camera",
        "expected_frames",
        "frames",
    }
    assert manifest["schema_version"] == 1
    assert manifest["trip_id"] == "T88d"
    assert manifest["source_camera"] == "image_2"
    assert manifest["expected_frames"] == 1
    assert len(manifest["frames"]) == 1
    frame = manifest["frames"][0]
    assert set(frame) == {"frame_id", "timestamp", "ego"}
    assert set(frame["ego"]) == {
        "speed_kmh",
        "longitudinal_accel",
        "lateral_accel",
    }
    serialized = json.dumps(manifest)
    for forbidden in (
        "target",
        "event",
        "driver",
        "min_ttc",
        "headway",
        "behavior_flags",
        "risk",
        "location",
        "rotation",
        "geolocation",
        "trip_aggregate",
    ):
        assert forbidden not in serialized


def test_runtime_loader_never_opens_source_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trip_dir = _write_source_trip(tmp_path, "T77d", [_full_frame(0)], (0,))
    manifest_path = _prepare(tmp_path, trip_dir)
    source_json = trip_dir / "T77d.json.gz"
    original_gzip_open = gzip.open

    def forbid_source_gzip(path, *args, **kwargs):
        if Path(path) == source_json:
            raise AssertionError("runtime attempted to open forbidden source JSON")
        return original_gzip_open(path, *args, **kwargs)

    monkeypatch.setattr(gzip, "open", forbid_source_gzip)
    # Invalid source bytes make the test independent of filesystem permissions:
    # construction can only succeed if runtime consumes the manifest instead.
    source_json.write_bytes(b"not-readable-as-json")

    loader = C1LeanTripLoader(trip_dir, manifest_path)

    assert loader.trip_id == "T77d"
    assert loader.n_frames == 1
    assert loader.frame(0).timestamp == 0.0


def test_input_frame_exposes_only_monocular_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trip_dir = _write_source_trip(tmp_path, "T66d", [_full_frame(0)], (0,))
    manifest_path = _prepare(tmp_path, trip_dir)
    for relative in ("kitti/image_3", "kitti/depth", "kitti/label_2", "driver"):
        bait_dir = trip_dir / relative
        bait_dir.mkdir(parents=True)
        (bait_dir / "000000.bait").write_text("must not be read", encoding="utf-8")

    original_is_file = Path.is_file
    original_imread = cv2.imread
    forbidden_path_parts = {"image_3", "depth", "label_2", "driver"}

    def guarded_is_file(path: Path) -> bool:
        assert forbidden_path_parts.isdisjoint(path.parts), path
        return original_is_file(path)

    def guarded_imread(path: str, flags: int):
        assert "image_2" in Path(path).parts, path
        return original_imread(path, flags)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    monkeypatch.setattr(cv2, "imread", guarded_imread)

    loader = C1LeanTripLoader(trip_dir, manifest_path)
    report = loader.check_image_2_integrity()
    frame = loader.frame(0)

    assert report.ok
    assert set(frame.ego) == {
        "speed_kmh",
        "longitudinal_accel",
        "lateral_accel",
    }
    assert frame.ego["speed_kmh"] == 31.0
    with pytest.raises(TypeError):
        frame.ego["speed_kmh"] = 0.0  # type: ignore[index]

    public_api = {name for name in dir(frame) if not name.startswith("_")}
    assert public_api == {"ego", "frame_id", "left", "timestamp"}
    for forbidden in (
        "right",
        "driver",
        "depth",
        "targets",
        "events_active",
        "events_log",
        "labels",
        "gt",
        "min_ttc",
        "raw_frame",
    ):
        assert not hasattr(frame, forbidden)
        assert not hasattr(loader, forbidden)
    assert frame.left() is frame.left()


@pytest.mark.parametrize(
    ("forbidden_location", "forbidden_value"),
    (
        ("root", {"events_log": []}),
        ("frame", {"min_ttc": 0.1}),
        ("ego", {"location": {"x": 1.0}}),
    ),
)
def test_runtime_rejects_manifest_with_any_forbidden_field(
    tmp_path: Path,
    forbidden_location: str,
    forbidden_value: dict[str, object],
) -> None:
    trip_dir = _write_source_trip(tmp_path, "T55d", [_full_frame(0)], (0,))
    manifest_path = _prepare(tmp_path, trip_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if forbidden_location == "root":
        manifest.update(forbidden_value)
    elif forbidden_location == "frame":
        manifest["frames"][0].update(forbidden_value)
    else:
        manifest["frames"][0]["ego"].update(forbidden_value)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fields|forbidden"):
        C1LeanTripLoader(trip_dir, manifest_path)


@pytest.mark.parametrize(
    "invalid_case",
    (
        "schema_version",
        "source_camera",
        "expected_count",
        "expected_bool",
        "frame_id_bool",
        "frame_id_float",
        "frame_id_gap",
        "timestamp_bool",
        "timestamp_nan",
        "timestamp_nonmonotonic",
        "ego_bool",
        "ego_infinite",
    ),
)
def test_runtime_manifest_invariants_fail_closed(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    trip_dir = _write_source_trip(
        tmp_path,
        "T54d",
        [_full_frame(0), _full_frame(1)],
        (0, 1),
    )
    manifest_path = _prepare(tmp_path, trip_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if invalid_case == "schema_version":
        manifest["schema_version"] = 2
    elif invalid_case == "source_camera":
        manifest["source_camera"] = "image_3"
    elif invalid_case == "expected_count":
        manifest["expected_frames"] = 3
    elif invalid_case == "expected_bool":
        manifest["expected_frames"] = True
    elif invalid_case == "frame_id_bool":
        manifest["frames"][0]["frame_id"] = False
    elif invalid_case == "frame_id_float":
        manifest["frames"][0]["frame_id"] = 0.0
    elif invalid_case == "frame_id_gap":
        manifest["frames"][1]["frame_id"] = 2
    elif invalid_case == "timestamp_bool":
        manifest["frames"][0]["timestamp"] = False
    elif invalid_case == "timestamp_nan":
        manifest["frames"][0]["timestamp"] = float("nan")
    elif invalid_case == "timestamp_nonmonotonic":
        manifest["frames"][1]["timestamp"] = 0.0
    elif invalid_case == "ego_bool":
        manifest["frames"][0]["ego"]["speed_kmh"] = True
    elif invalid_case == "ego_infinite":
        manifest["frames"][0]["ego"]["speed_kmh"] = float("inf")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        C1LeanTripLoader(trip_dir, manifest_path)


def test_missing_image_2_is_reported_and_never_synthesized(tmp_path: Path) -> None:
    frames = [_full_frame(frame_id) for frame_id in range(3)]
    trip_dir = _write_source_trip(tmp_path, "T08d", frames, (0, 2))
    manifest_path = _prepare(tmp_path, trip_dir)
    loader = C1LeanTripLoader(trip_dir, manifest_path)

    report = loader.check_image_2_integrity()

    assert report.ok is False
    assert report.telemetry_frame_count == 3
    assert report.present_image_count == 2
    assert report.missing_frame_ids == (1,)
    assert report.duplicate_frame_ids == ()
    assert report.errors == ("trip T08d: missing image_2 for frame_id(s): 1",)
    with pytest.raises(C1Image2IntegrityError, match="missing image_2.*frame_id 1"):
        loader.frame(1)
    with pytest.raises(C1Image2IntegrityError, match=r"missing image_2.*frame_id\(s\): 1"):
        report.require_ok()
    assert not (trip_dir / "kitti" / "image_2" / "000001.jpg").exists()


def test_preparation_rejects_duplicate_frame_ids(tmp_path: Path) -> None:
    trip_dir = _write_source_trip(tmp_path, "T44d", [_full_frame(0), _full_frame(0)], (0,))

    with pytest.raises(ValueError, match="unique and contiguous"):
        _prepare(tmp_path, trip_dir)


def test_manifest_output_cannot_be_inside_source_trip(tmp_path: Path) -> None:
    trip_dir = _write_source_trip(tmp_path, "T33d", [_full_frame(0)], (0,))

    with pytest.raises(ValueError, match="outside the source trip"):
        prepare_c1_manifest(trip_dir, trip_dir / "c1-runtime.json")


def test_prepare_manifest_cli_writes_configurable_external_output(
    tmp_path: Path,
) -> None:
    trip_dir = _write_source_trip(tmp_path, "T22d", [_full_frame(0)], (0,))
    output = tmp_path / "external_runs" / "custom-name.json"

    assert prepare_manifest_main([str(trip_dir), "--output", str(output)]) == 0
    assert output.is_file()
    assert C1LeanTripLoader(trip_dir, output).n_frames == 1
