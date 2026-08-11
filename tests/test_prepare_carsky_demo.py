from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "prepare_carsky_demo.py"
    spec = importlib.util.spec_from_file_location("prepare_carsky_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_trip(root: Path, *, bad_frame_id: bool = False) -> Path:
    trip = root / "T01-Sample"
    (trip / "driver").mkdir(parents=True)
    (trip / "kitti" / "image_2").mkdir(parents=True)
    (trip / "kitti" / "image_3").mkdir(parents=True)
    (trip / "kitti" / "depth").mkdir(parents=True)
    (trip / "kitti" / "label_2").mkdir(parents=True)
    frames = []
    for index in range(2):
        (trip / "driver" / f"frame_{index:06d}.jpg").write_bytes(b"cabin")
        (trip / "kitti" / "image_2" / f"{index:06d}.jpg").write_bytes(b"road")
        (trip / "kitti" / "image_3" / f"{index:06d}.jpg").write_bytes(b"right")
        (trip / "kitti" / "depth" / f"{index:06d}.npy").write_bytes(b"depth")
        (trip / "kitti" / "label_2" / f"{index:06d}.txt").write_text("GT")
        frames.append(
            {
                "frame_id": 5 if bad_frame_id and index == 1 else index,
                "world_frame": 100 + index,
                "timestamp": index * 0.05,
                "ego": {
                    "speed_kmh": 40 + index,
                    "longitudinal_accel": -0.1,
                    "lateral_accel": 0.2,
                    "location": {"x": 1},
                },
                "targets": [{"id": "secret"}],
                "driver": {"state": "ground_truth"},
                "events_active": [{"type": "ground_truth"}],
                "min_ttc": 1.0,
                "risk": {"score": 100},
            }
        )
    payload = {
        "trip_id": trip.name,
        "metadata": {
            "trip_id": trip.name,
            "fps": 20,
            "speed_limit_kmh": 60,
            "weather": {"cloudiness": 10},
            "random_seed": 123,
        },
        "driver_summary": {"secret": True},
        "trip_aggregate": {"safe_score": 0},
        "events_log": [{"secret": True}],
        "frames": frames,
    }
    (trip / f"{trip.name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return trip


def _source_payload(trip: Path) -> tuple[Path, dict[str, object]]:
    path = trip / f"{trip.name}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_source_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_bundle_is_truth_free_and_manifested(tmp_path: Path) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    result = module.prepare_bundle(source, tmp_path / "output")
    destination = Path(result["destination"])
    payload = json.loads((destination / "T01-Sample.json").read_text())

    assert result["truth_free"] is True
    assert set(payload) == {"trip_id", "metadata", "frames"}
    assert set(payload["frames"][0]) == {"frame_id", "timestamp", "ego"}
    assert set(payload["frames"][0]["ego"]) == {
        "speed_kmh",
        "longitudinal_accel",
        "lateral_accel",
    }
    assert not (destination / "kitti" / "image_3").exists()
    assert not (destination / "kitti" / "depth").exists()
    assert not (destination / "kitti" / "label_2").exists()
    manifest = json.loads((destination / "BUNDLE_MANIFEST.json").read_text())
    assert manifest["truth_free"] is True
    assert manifest["frames"] == 2
    assert len(manifest["files"]) == 5


def test_prepare_bundle_refuses_overwrite_then_archives_on_force(tmp_path: Path) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    output = tmp_path / "output"
    module.prepare_bundle(source, output)
    with pytest.raises(FileExistsError):
        module.prepare_bundle(source, output)

    result = module.prepare_bundle(source, output, replace_existing=True)
    assert result["archived_previous"] is not None
    assert Path(result["archived_previous"]).is_dir()


def test_prepare_bundle_rejects_non_contiguous_frames(tmp_path: Path) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source", bad_frame_id=True)
    with pytest.raises(ValueError, match="frame_id không liên tiếp"):
        module.prepare_bundle(source, tmp_path / "output")


@pytest.mark.parametrize("trip_id", ("../escape", "bad trip", "-leading", "", None))
def test_prepare_bundle_rejects_unsafe_or_missing_trip_id(
    tmp_path: Path, trip_id: object
) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    json_path, payload = _source_payload(source)
    payload["trip_id"] = trip_id
    _write_source_payload(json_path, payload)

    with pytest.raises(ValueError, match="trip_id"):
        module.prepare_bundle(source, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_prepare_bundle_requires_trip_ids_to_match_source_directory(
    tmp_path: Path,
) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    json_path, payload = _source_payload(source)
    payload["trip_id"] = "T02-Sample"
    _write_source_payload(json_path, payload)
    with pytest.raises(ValueError, match="không khớp thư mục nguồn"):
        module.prepare_bundle(source, tmp_path / "output")

    payload["trip_id"] = source.name
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["trip_id"] = "T02-Sample"
    _write_source_payload(json_path, payload)
    with pytest.raises(ValueError, match="không khớp thư mục nguồn"):
        module.prepare_bundle(source, tmp_path / "output")


def test_prepare_bundle_resolves_destination_containment(tmp_path: Path) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / source.name).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="Bundle đích nằm ngoài"):
        module.prepare_bundle(source, output)
    assert list(outside.iterdir()) == []


def test_prepare_bundle_rejects_source_image_symlink_escape(tmp_path: Path) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    outside_image = tmp_path / "outside.jpg"
    outside_image.write_bytes(b"outside")
    road = source / "kitti" / "image_2" / "000000.jpg"
    road.unlink()
    road.symlink_to(outside_image)

    with pytest.raises(ValueError, match="Ảnh nguồn .* nằm ngoài"):
        module.prepare_bundle(source, tmp_path / "output")


@pytest.mark.parametrize(
    "missing_field",
    ("speed_kmh", "longitudinal_accel", "lateral_accel"),
)
def test_prepare_bundle_requires_every_ego_field(
    tmp_path: Path, missing_field: str
) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    json_path, payload = _source_payload(source)
    frames = payload["frames"]
    assert isinstance(frames, list) and isinstance(frames[0], dict)
    ego = frames[0]["ego"]
    assert isinstance(ego, dict)
    del ego[missing_field]
    _write_source_payload(json_path, payload)

    with pytest.raises(ValueError, match=f"thiếu field bắt buộc: {missing_field}"):
        module.prepare_bundle(source, tmp_path / "output")


@pytest.mark.parametrize("bad_value", (None, True, "40.0", float("inf")))
def test_prepare_bundle_requires_json_numeric_ego_values(
    tmp_path: Path, bad_value: object
) -> None:
    module = _module()
    source = _source_trip(tmp_path / "source")
    json_path, payload = _source_payload(source)
    frames = payload["frames"]
    assert isinstance(frames, list) and isinstance(frames[0], dict)
    ego = frames[0]["ego"]
    assert isinstance(ego, dict)
    ego["speed_kmh"] = bad_value
    _write_source_payload(json_path, payload)

    with pytest.raises(ValueError, match=r"ego\.speed_kmh phải (là số|hữu hạn)"):
        module.prepare_bundle(source, tmp_path / "output")
