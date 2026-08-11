#!/usr/bin/env python3
"""Reference source for the SafeLoop fast WSS ingest contract.

``live`` reads two camera/RTSP devices plus three causal ego values and labels
them LIVE_CAMERA + THIRD_PARTY. ``recorded-check`` verifies a truth-free bundle
for the server-side RECORDED_STREAM source; it never relabels recorded pixels
as a live camera and never reads labels, GT, depth, events or predictions.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
import uuid
from urllib.parse import urlsplit

import cv2
from websockets.sync.client import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeloop.aws_live_contract import PreparedDemoBundle  # noqa: E402


PERIOD_NS = 50_000_000


def _token() -> str:
    direct = os.getenv("SAFELOOP_INGEST_TOKEN", "")
    token_file = os.getenv("SAFELOOP_INGEST_TOKEN_FILE", "")
    if direct and token_file:
        raise SystemExit("set only one ingest token source")
    if token_file:
        path = Path(token_file)
        if path.is_symlink() or not path.is_file():
            raise SystemExit("ingest token file must be a regular file")
        direct = path.read_text(encoding="utf-8").strip()
    if len(direct) < 24:
        raise SystemExit("SAFELOOP_INGEST_TOKEN(_FILE) is required")
    return direct


def _capture(value: str) -> cv2.VideoCapture:
    source: str | int = int(value) if value.isdecimal() else value
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera source: {value}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _jpeg(frame: object, *, quality: int) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok or len(encoded) > 900_000:
        raise RuntimeError("JPEG encode failed or exceeds the service limit")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def send_live(args: argparse.Namespace) -> int:
    parsed = urlsplit(args.url)
    if parsed.scheme != "wss" or parsed.query or parsed.fragment:
        raise SystemExit("--url must be a credential-free wss:// URL")
    road = _capture(args.road)
    cabin = _capture(args.cabin)
    session_id = args.session_id or f"third-party-{uuid.uuid4().hex[:12]}"
    started = time.monotonic_ns()
    sent = 0
    try:
        with connect(
            args.url,
            additional_headers={"Authorization": f"Bearer {_token()}"},
            open_timeout=5,
            close_timeout=2,
            max_size=2_500_000,
        ) as websocket:
            while args.frames is None or sent < args.frames:
                deadline = started + sent * PERIOD_NS
                remaining = deadline - time.monotonic_ns()
                if remaining > 0:
                    time.sleep(remaining / 1_000_000_000.0)
                road_ok, road_frame = road.read()
                cabin_ok, cabin_frame = cabin.read()
                if not road_ok or not cabin_ok:
                    raise RuntimeError("one camera source stopped producing frames")
                capture_ms = time.time_ns() // 1_000_000
                message = {
                    "schema": "safeloop.input.v1",
                    "session_id": session_id,
                    "seq": sent,
                    "capture_ts_ms": capture_ms,
                    "source_kind": "LIVE_CAMERA",
                    "telemetry_source": "THIRD_PARTY",
                    "metadata": {
                        "speed_limit_kmh": args.speed_limit_kmh,
                        "weather": args.weather,
                    },
                    "ego": {
                        "speed_kmh": args.speed_kmh,
                        "longitudinal_accel": args.longitudinal_accel,
                        "lateral_accel": args.lateral_accel,
                    },
                    "road_jpeg_b64": _jpeg(road_frame, quality=args.jpeg_quality),
                    "cabin_jpeg_b64": _jpeg(cabin_frame, quality=args.jpeg_quality),
                }
                websocket.send(
                    json.dumps(
                        message,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                sent += 1
    finally:
        road.release()
        cabin.release()
    print(json.dumps({"sent": sent, "session_id": session_id, "source_kind": "LIVE_CAMERA"}))
    return 0


def recorded_check(args: argparse.Namespace) -> int:
    bundle = PreparedDemoBundle.load(args.bundle)
    print(
        json.dumps(
            {
                "demo_id": bundle.trip_id,
                "frames": len(bundle.frames),
                "manifest_sha256": bundle.manifest_sha256,
                "camera": "RECORDED_STREAM",
                "telemetry": "RECORDED_DATA",
                "inference": "LIVE_MODEL",
                "start_via": "POST /v1/admin/start",
            },
            separators=(",", ":"),
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="send real camera/RTSP frames over WSS")
    live.add_argument("--url", required=True)
    live.add_argument("--road", required=True, help="road camera index or RTSP URL")
    live.add_argument("--cabin", required=True, help="cabin camera index or RTSP URL")
    live.add_argument("--session-id")
    live.add_argument("--frames", type=int)
    live.add_argument("--speed-kmh", type=float, default=0.0)
    live.add_argument("--longitudinal-accel", type=float, default=0.0)
    live.add_argument("--lateral-accel", type=float, default=0.0)
    live.add_argument("--speed-limit-kmh", type=float, default=60.0)
    live.add_argument("--weather", choices=("clear", "cloudy", "rain", "wet", "fog"), default="clear")
    live.add_argument("--jpeg-quality", type=int, choices=range(50, 96), default=85)
    live.set_defaults(handler=send_live)
    recorded = commands.add_parser(
        "recorded-check", help="verify a server-side truth-free recorded source"
    )
    recorded.add_argument("--bundle", type=Path, required=True)
    recorded.set_defaults(handler=recorded_check)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
