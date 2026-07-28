r"""Build the local MP4 used by the lightweight C2 Streamlit demo.

The output is a raw concatenation of four curated DMD face-camera intervals.
It intentionally does not contain a pre-rendered prediction HUD: the web app
must run the C2 detector on every replayed frame.

Usage:
    python c2/make_web_test_video.py --dmd-root C:\DMD\dmd

The source DMD material is licensed for non-commercial research and includes a
NoDerivatives restriction. Keep the generated clip local; do not publish it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TestSegment:
    label: str
    relative_session: str
    start_seconds: float
    duration_seconds: float


# Each action interval starts with neutral driving so the complete clip begins
# cleanly and remains easy to explain during a live presentation.
SEGMENTS = (
    TestSegment("safe", "gA/1/s2", 0.0, 6.0),
    TestSegment("phone", "gA/1/s2", 48.0, 9.0),
    TestSegment("yawn", "gA/1/s5", 43.0, 8.0),
    TestSegment("eye-closure", "gA/1/s5", 66.0, 8.5),
)


def _video_for(root: Path, relative_session: str) -> Path:
    session = root.joinpath(*relative_session.split("/"))
    matches = sorted(session.glob("*_rgb_face.mp4"))
    if not matches:
        raise FileNotFoundError(f"no RGB face video under {session}")
    return matches[0]


def _valid_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if np.isfinite(fps) and fps > 1.0 else 29.76


def _open_writer(
    output: Path,
    fps: float,
    size: tuple[int, int],
    codec: str,
) -> tuple[cv2.VideoWriter, str]:
    candidates = [codec]
    if codec != "avc1":
        candidates.append("avc1")
    if "mp4v" not in candidates:
        candidates.append("mp4v")
    for candidate in candidates:
        fourcc = cv2.VideoWriter_fourcc(*candidate)
        if sys.platform == "win32" and candidate in {"avc1", "H264"}:
            writer = cv2.VideoWriter(
                str(output),
                cv2.CAP_MSMF,
                fourcc,
                fps,
                size,
            )
        else:
            writer = cv2.VideoWriter(str(output), fourcc, fps, size)
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise RuntimeError(
        "No MP4 encoder opened. Try --codec mp4v, or install an H.264 encoder."
    )


def build_video(
    root: Path,
    output: Path,
    *,
    codec: str = "avc1",
    width: int = 640,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    first_video = _video_for(root, SEGMENTS[0].relative_session)
    probe = cv2.VideoCapture(str(first_video))
    if not probe.isOpened():
        raise RuntimeError(f"cannot open {first_video}")
    fps = _valid_fps(probe)
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(
            f"invalid source dimensions: {source_width}x{source_height}"
        )
    output_width = min(max(320, width), source_width)
    output_width -= output_width % 2
    output_height = int(round(source_height * output_width / source_width))
    output_height -= output_height % 2

    writer, selected_codec = _open_writer(
        output,
        fps,
        (output_width, output_height),
        codec,
    )
    segment_frames: dict[str, int] = {}
    total_frames = 0
    try:
        for segment in SEGMENTS:
            video = _video_for(root, segment.relative_session)
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video}")
            source_fps = _valid_fps(capture)
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(round(segment.start_seconds * source_fps)),
            )
            requested = int(round(segment.duration_seconds * source_fps))
            written = 0
            try:
                for _ in range(requested):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if (
                        frame.shape[1] != output_width
                        or frame.shape[0] != output_height
                    ):
                        frame = cv2.resize(
                            frame,
                            (output_width, output_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    writer.write(frame)
                    written += 1
            finally:
                capture.release()
            if written < requested * 0.95:
                raise RuntimeError(
                    f"{segment.label}: decoded only {written}/{requested} frames"
                )
            segment_frames[segment.label] = written
            total_frames += written
            print(
                f"[OK] {segment.label:12s} "
                f"{written:4d} frames ({written / fps:4.1f}s)",
                flush=True,
            )
    finally:
        writer.release()

    check = cv2.VideoCapture(str(output))
    readable = check.isOpened()
    decoded = 0
    while readable:
        ok, _frame = check.read()
        if not ok:
            break
        decoded += 1
    check.release()
    if decoded < total_frames * 0.99:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"encoded MP4 failed verification: decoded {decoded}/{total_frames}"
        )

    return {
        "output": str(output),
        "codec": selected_codec,
        "fps": round(fps, 3),
        "size": f"{output_width}x{output_height}",
        "frames": total_frames,
        "seconds": round(total_frames / fps, 2),
        "bytes": output.stat().st_size,
        "segments": segment_frames,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dmd-root",
        type=Path,
        default=Path(r"C:\DMD\dmd"),
        help=r"extracted DMD root (default C:\DMD\dmd)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "cache"
        / "web_test_realtime.mp4",
    )
    parser.add_argument(
        "--codec",
        choices=["avc1", "H264", "mp4v"],
        default="avc1",
        help="avc1 is the browser-friendly default on the target Windows laptop",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="output width in pixels (default 640; aspect ratio is preserved)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.dmd_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: DMD root does not exist: {root}", file=sys.stderr)
        return 2
    try:
        result = build_video(root, output, codec=args.codec, width=args.width)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "[DONE] "
        f"{result['output']} | {result['seconds']}s | {result['fps']} FPS | "
        f"{result['size']} | {result['codec']} | "
        f"{result['bytes'] / 1024 / 1024:.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
