from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from safeloop.aws_live_contract import (
    EXPECTED_MODALITIES,
    InputContractError,
    PreparedDemoBundle,
    parse_input_message,
)


def _jpeg() -> str:
    image = np.full((24, 32, 3), 17, dtype=np.uint8)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(payload).decode("ascii")


def valid_message() -> dict[str, object]:
    image = _jpeg()
    return {
        "schema": "safeloop.input.v1",
        "session_id": "vehicle-001",
        "seq": 0,
        "capture_ts_ms": 1_700_000_000_000,
        "source_kind": "LIVE_CAMERA",
        "telemetry_source": "THIRD_PARTY",
        "metadata": {"speed_limit_kmh": 60, "weather": "clear"},
        "ego": {
            "speed_kmh": 3,
            "longitudinal_accel": 0.1,
            "lateral_accel": -0.2,
        },
        "road_jpeg_b64": image,
        "cabin_jpeg_b64": image,
    }


def test_input_decodes_owned_read_only_contiguous_images() -> None:
    tick = parse_input_message(json.dumps(valid_message()))
    assert tick.road_bgr.flags.c_contiguous
    assert not tick.road_bgr.flags.writeable
    assert tick.road_bgr.base is None
    assert tick.cabin_bgr.base is None
    assert not np.shares_memory(tick.road_bgr, tick.cabin_bgr)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(seq=float("nan")),
        lambda value: value.update(source_kind="RECORDED_STREAM"),
        lambda value: value.update(road_jpeg_b64="%%%"),
        lambda value: value.update(telemetry_source="CARSKY_BROKER"),
    ],
)
def test_input_rejects_malformed_or_false_provenance(mutation) -> None:
    message = valid_message()
    mutation(message)
    with pytest.raises(InputContractError):
        parse_input_message(json.dumps(message))


def test_input_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    message = json.dumps(valid_message(), separators=(",", ":"))
    duplicate = message[:-1] + ',"seq":4}'
    with pytest.raises(InputContractError, match="duplicate"):
        parse_input_message(duplicate)
    with pytest.raises(InputContractError, match="non-finite"):
        parse_input_message(message.replace('"seq":0', '"seq":NaN'))


def test_recorded_bundle_rejects_truth_artifact_before_opening_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "T03-Sample"
    root.mkdir()
    manifest = {
        "schema": "safeloop.carsky.demo-bundle.v1",
        "trip_id": "T03-Sample",
        "frames": 1,
        "source_fps": 20,
        "modalities": list(EXPECTED_MODALITIES),
        "truth_free": True,
        "forbidden_modalities_absent": True,
        "files": {
            "T03-Sample.json": {"bytes": 0, "sha256": "0" * 64},
            "driver/frame_000000.jpg": {"bytes": 0, "sha256": "0" * 64},
            "kitti/image_2/000000.jpg": {"bytes": 0, "sha256": "0" * 64},
            "labels/ground_truth.json": {"bytes": 0, "sha256": "0" * 64},
        },
    }
    (root / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputContractError, match="contains extras"):
        PreparedDemoBundle.load(root)

    assert not (root / "labels/ground_truth.json").exists()
