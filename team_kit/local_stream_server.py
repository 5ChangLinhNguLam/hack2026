"""
local_stream_server.py
=======================
OPTIONAL local HTTP server wrapping team_kit/dataset_loader.py, for teams
who want to query trip data over HTTP instead of reading files directly
(e.g. a training pipeline in a non-Python stack, or a custom dashboard).

This is NOT required. team_kit/dataset_loader.py (direct file access)
remains the primary, recommended way to use this dataset -- most teams
don't need this at all. This server:
  - runs LOCALLY, on YOUR OWN machine, against data you already
    downloaded (e.g. ./data_public/, or wherever you point --data-dir)
  - is not provided or hosted by organizers -- you run it yourself, only
    if you want it; nobody else's team depends on your instance
  - reuses TripDataset/HackathonDataset directly, so its behavior (which
    fields are present/missing per trip, redacted vs. full-GT trips,
    etc.) is byte-for-byte identical to reading files yourself -- this
    is a pure pass-through, it adds no redaction logic of its own and
    removes none either
  - uses only the Python standard library (http.server) -- nothing new
    to install

Usage:
    python team_kit/local_stream_server.py --data-dir ./data_public --port 8765

Then, from any HTTP client (Python requests, curl, browser, another
language's HTTP library):

    GET /trips                                   -> {"trip_ids": [...]}
    GET /trips/{trip_id}/summary                  -> same dict as ds.summary()
    GET /trips/{trip_id}/calibration              -> same dict as ds.load_calibration()
    GET /trips/{trip_id}/events                   -> same list as ds.events_log
    GET /trips/{trip_id}/aggregate                -> {"trip_aggregate":.., "driver_summary":..}
    GET /trips/{trip_id}/frames                   -> {"n_frames":N, "frame_ids":[...]}
    GET /trips/{trip_id}/frames/{frame_id}        -> one FrameRecord as JSON
    GET /trips/{trip_id}/frames/{frame_id}/left   -> left RGB (raw PNG bytes)
    GET /trips/{trip_id}/frames/{frame_id}/right  -> right RGB (raw PNG bytes)
    GET /trips/{trip_id}/frames/{frame_id}/driver -> driver frame (raw JPG/PNG bytes)
    GET /trips/{trip_id}/frames/{frame_id}/depth  -> depth (raw .npy bytes; 404 if not a keyframe)

JSON field names match team_kit/dataset_loader.py's FrameRecord /
summary() exactly -- see team_kit/README.md's API reference. On the 10
scored trips (T0Xd, redacted), missing ground-truth fields show up as
their dataclass defaults (0.0 / "unknown" / inf), exactly as they would
reading the JSON directly.

Images/depth are served as raw file bytes (no decode/re-encode round
trip through OpenCV) -- byte-identical to the source file on disk, and
faster than going through TripDataset.load_left() etc. for this purpose.

Concurrency: ThreadingHTTPServer handles multiple requests in parallel
(e.g. your own training loop making several image requests at once)
without blocking. One TripDataset instance is cached per trip_id after
first access, so the (possibly large) trip JSON isn't re-parsed on every
request.

Security note: binds to 127.0.0.1 (localhost only) by default -- other
machines on your network can't reach it unless you explicitly pass
--host 0.0.0.0. There is no authentication; do not expose this to the
public internet.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

# Allow running as `python team_kit/local_stream_server.py` directly
# (not just as `python -m team_kit.local_stream_server`) by making
# dataset_loader importable as a plain top-level module either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import HackathonDataset, TripDataset  # noqa: E402


_TRIP_CACHE: dict[str, TripDataset] = {}


def _get_trip(data_root: Path, trip_id: str) -> Optional[TripDataset]:
    if trip_id not in _TRIP_CACHE:
        trip_dir = data_root / trip_id
        if not trip_dir.is_dir():
            return None
        _TRIP_CACHE[trip_id] = TripDataset(trip_dir)
    return _TRIP_CACHE[trip_id]


# trip_id: T<digits>, optionally suffixed with "d" (scored trips, e.g.
# T01d) or "-Sample" (practice trips, e.g. T01-Sample) -- matches the
# discovery pattern in dataset_loader.py's HackathonDataset.
_TRIP_ID_PATTERN = r"T\d+d?(?:-Sample)?"
ROUTE_TRIPS = re.compile(r"^/trips/?$")
ROUTE_TRIP_SUMMARY = re.compile(rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/summary/?$")
ROUTE_TRIP_CALIBRATION = re.compile(rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/calibration/?$")
ROUTE_TRIP_EVENTS = re.compile(rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/events/?$")
ROUTE_TRIP_AGGREGATE = re.compile(rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/aggregate/?$")
ROUTE_TRIP_FRAMES = re.compile(rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/frames/?$")
ROUTE_FRAME_IMAGE = re.compile(
    rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/frames/(?P<frame_id>\d+)/(?P<kind>left|right|driver|depth)/?$"
)
ROUTE_FRAME = re.compile(rf"^/trips/(?P<trip_id>{_TRIP_ID_PATTERN})/frames/(?P<frame_id>\d+)/?$")


class Handler(BaseHTTPRequestHandler):
    data_root: Path = None  # set in main() before serve_forever()

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path}")

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, default=str, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", ""):
            self._send_json({
                "service": "hackathon dataset local streaming server (OPTIONAL, "
                           "runs on your own machine against your own downloaded data)",
                "data_root": str(self.data_root),
                "routes": [
                    "/trips",
                    "/trips/{trip_id}/summary",
                    "/trips/{trip_id}/calibration",
                    "/trips/{trip_id}/events",
                    "/trips/{trip_id}/aggregate",
                    "/trips/{trip_id}/frames",
                    "/trips/{trip_id}/frames/{frame_id}",
                    "/trips/{trip_id}/frames/{frame_id}/left|right|driver|depth",
                ],
            })
            return

        if ROUTE_TRIPS.match(path):
            hd = HackathonDataset(self.data_root)
            self._send_json({"trip_ids": hd.trip_ids})
            return

        for route, fn in (
            (ROUTE_TRIP_SUMMARY, lambda ds: ds.summary()),
            (ROUTE_TRIP_CALIBRATION, lambda ds: ds.load_calibration()),
            (ROUTE_TRIP_EVENTS, lambda ds: ds.events_log),
            (ROUTE_TRIP_AGGREGATE, lambda ds: {
                "trip_aggregate": ds.trip_aggregate,
                "driver_summary": ds.driver_summary,
            }),
            (ROUTE_TRIP_FRAMES, lambda ds: {
                "n_frames": len(ds),
                "frame_ids": [fr.frame_id for fr in ds.iter_frames()],
            }),
        ):
            m = route.match(path)
            if m:
                self._handle_trip(m, fn)
                return

        m = ROUTE_FRAME_IMAGE.match(path)
        if m:
            self._handle_frame_image(m)
            return

        m = ROUTE_FRAME.match(path)
        if m:
            self._handle_frame_record(m)
            return

        self._error(404, f"No route matches {path}")

    def _handle_trip(self, m: re.Match, fn: Callable[[TripDataset], object]) -> None:
        trip_id = m.group("trip_id")
        ds = _get_trip(self.data_root, trip_id)
        if ds is None:
            self._error(404, f"Trip '{trip_id}' not found under {self.data_root}")
            return
        try:
            self._send_json(fn(ds))
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    def _handle_frame_record(self, m: re.Match) -> None:
        trip_id = m.group("trip_id")
        frame_id = int(m.group("frame_id"))
        ds = _get_trip(self.data_root, trip_id)
        if ds is None:
            self._error(404, f"Trip '{trip_id}' not found under {self.data_root}")
            return
        if frame_id < 0 or frame_id >= len(ds):
            self._error(404, f"Frame {frame_id} out of range for {trip_id} (0..{len(ds) - 1})")
            return
        self._send_json(asdict(ds[frame_id]))

    def _handle_frame_image(self, m: re.Match) -> None:
        trip_id = m.group("trip_id")
        frame_id = int(m.group("frame_id"))
        kind = m.group("kind")
        ds = _get_trip(self.data_root, trip_id)
        if ds is None:
            self._error(404, f"Trip '{trip_id}' not found under {self.data_root}")
            return

        path: Optional[Path] = None
        content_type: Optional[str] = None
        if kind in ("left", "right"):
            # kitti/image_2 and image_3 are PNG at original CARLA-render
            # resolution, but become JPG once a trip has been through
            # resize_kitti.sh (release-resolution downscale) -- extension
            # varies per TRIP, not per file, so try both (mirrors
            # dataset_loader.py's TripDataset._load_image).
            base_dir = ds.image_left_dir if kind == "left" else ds.image_right_dir
            for ext, ct in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")):
                candidate = base_dir / f"{frame_id:06d}{ext}"
                if candidate.exists():
                    path, content_type = candidate, ct
                    break
        elif kind == "driver":
            for ext, ct in ((".jpg", "image/jpeg"), (".png", "image/png"), (".jpeg", "image/jpeg")):
                candidate = ds.driver_dir / f"frame_{frame_id:06d}{ext}"
                if candidate.exists():
                    path, content_type = candidate, ct
                    break
        elif kind == "depth":
            path, content_type = ds.depth_dir / f"{frame_id:06d}.npy", "application/octet-stream"

        if path is None or not path.exists():
            self._error(404, f"No '{kind}' file for {trip_id} frame {frame_id} "
                              f"(depth is keyframe-only -- most frame_ids won't have one; "
                              f"check /trips/{trip_id}/frames for the frame count)")
            return

        self._send_bytes(path.read_bytes(), content_type)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                         help="Root containing T01d/, T02d/, ..., T01-Sample/, ... (your downloaded dataset)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                         help="Bind address. Default 127.0.0.1 (localhost only, "
                              "recommended). Only use 0.0.0.0 if you specifically "
                              "want other machines on your network to reach this.")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.is_dir():
        raise SystemExit(f"--data-dir not found: {data_root}")

    Handler.data_root = data_root
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local dataset server running at http://{args.host}:{args.port}")
    print(f"Serving data from: {data_root}")
    print(f"Try: curl http://{args.host}:{args.port}/trips")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
