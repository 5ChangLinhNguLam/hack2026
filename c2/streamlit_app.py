"""Lightweight Streamlit web surface for the C2 realtime engine.

Run from the repository root:

    streamlit run c2/streamlit_app.py

The MVP intentionally uses MP4 replay in the browser.  It executes the same
Face Landmarker + phone detector + temporal state machine as ``demo.py`` and
updates an image placeholder at the source video's cadence.  Continuous
browser webcam capture needs a WebRTC component and is kept out of this
lightweight judging path.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    from .demo import render_hud
    from .live_detector import (
        CellPhoneDetector,
        FaceSignalExtractor,
        LiveConfig,
        PhoneDetection,
        TemporalStateMachine,
    )
except ImportError:
    try:
        from c2.demo import render_hud
        from c2.live_detector import (
            CellPhoneDetector,
            FaceSignalExtractor,
            LiveConfig,
            PhoneDetection,
            TemporalStateMachine,
        )
    except ImportError:
        from demo import render_hud
        from live_detector import (
            CellPhoneDetector,
            FaceSignalExtractor,
            LiveConfig,
            PhoneDetection,
            TemporalStateMachine,
        )


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = Path(__file__).resolve().parent / "cache" / "web_test_realtime.mp4"

st.set_page_config(
    page_title="C2 Driver Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _source_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if np.isfinite(fps) and fps > 1.0 else 20.0


def _run_realtime_replay(
    source: Path,
    *,
    phone_enabled: bool,
    phone_every: int,
    calibration_seconds: float,
    speed: float,
    max_seconds: float,
    target_fps: int,
) -> dict:
    """Decode one MP4 and update the Streamlit HUD at realtime cadence."""

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = _source_fps(capture)
    source_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    requested_frames = max(1, int(round(max_seconds * fps)))
    frame_limit = (
        min(requested_frames, source_total_frames)
        if source_total_frames > 0
        else requested_frames
    )
    frame_step = max(1, int(round(fps / max(1, target_fps))))
    replay_seconds = frame_limit / fps
    machine = TemporalStateMachine(
        LiveConfig(calibration_seconds=max(0.1, calibration_seconds))
    )
    face_model = FaceSignalExtractor()
    phone_model = CellPhoneDetector() if phone_enabled else None
    image_slot = st.empty()
    status_slot = st.empty()
    progress_slot = st.progress(0.0, text="Preparing replay…")
    events: deque[tuple[float, str, str]] = deque(maxlen=20)
    latest_phone = PhoneDetection()
    latest_phone_time = -1e9
    last_event_state: str | None = None
    state_counts: dict[str, int] = {}
    face_frames = 0
    face_ms = 0.0
    phone_runs = 0
    phone_ms = 0.0
    processed = 0
    wall_started = time.perf_counter()

    try:
        frame_index = 0
        while frame_index < frame_limit:
            loop_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            face = face_model.process(frame, int(round(timestamp * 1000.0)))
            phone_score = None
            if phone_model is not None and processed % max(1, phone_every) == 0:
                latest_phone = phone_model.process(frame)
                latest_phone_time = timestamp
                phone_score = latest_phone.score
                phone_runs += 1
                phone_ms += latest_phone.inference_ms

            snapshot = machine.update(timestamp, face, phone_score)
            if snapshot.calibrated and snapshot.state != last_event_state:
                events.append((timestamp, snapshot.state, snapshot.reason))
                last_event_state = snapshot.state
            if snapshot.calibrated:
                state_counts[snapshot.state] = state_counts.get(snapshot.state, 0) + 1

            face_frames += int(face.face_found)
            face_ms += face.inference_ms
            processed += 1
            phone_visible = (
                latest_phone.box is not None
                and timestamp - latest_phone_time <= 0.8
            )
            wall_elapsed = max(time.perf_counter() - wall_started, 1e-9)
            display_fps = processed / wall_elapsed
            hud = render_hud(
                frame,
                face,
                latest_phone,
                phone_visible,
                snapshot,
                display_fps=display_fps,
                events=events,
                source_name=source.name,
                phone_enabled=phone_model is not None,
            )
            image_slot.image(
                cv2.cvtColor(hud, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
            )
            status_slot.markdown(
                f"**{snapshot.state.upper()}** · {snapshot.reason}  \n"
                f"Source `{timestamp:05.1f}s / {replay_seconds:.1f}s` · "
                f"face `{face.inference_ms:.1f} ms` · display `{display_fps:.1f} FPS`"
            )
            progress_slot.progress(
                min(1.0, (frame_index + frame_step) / frame_limit),
                text=f"Replaying {timestamp:05.1f}s",
            )

            frame_budget = frame_step / fps / max(speed, 0.1)
            remaining = frame_budget - (time.perf_counter() - loop_started)
            if remaining > 0:
                time.sleep(remaining)
            for _ in range(frame_step - 1):
                if not capture.grab():
                    break
            frame_index += frame_step
    finally:
        capture.release()
        face_model.close()
        if phone_model is not None:
            phone_model.close()

    wall_elapsed = max(time.perf_counter() - wall_started, 1e-9)
    source_seconds = min(frame_index, frame_limit) / fps if processed else 0.0
    return {
        "frames": processed,
        "source_seconds": round(source_seconds, 2),
        "wall_fps": round(processed / wall_elapsed, 1),
        "realtime_factor": round(source_seconds / wall_elapsed, 2),
        "frame_step": frame_step,
        "face_found": round(face_frames / processed, 4) if processed else 0.0,
        "face_mean_ms": round(face_ms / processed, 2) if processed else 0.0,
        "phone_mean_ms": round(phone_ms / phone_runs, 2) if phone_runs else None,
        "state_counts": state_counts,
        "events": list(events),
    }


def _available_sample() -> bool:
    return SAMPLE_VIDEO.exists() and SAMPLE_VIDEO.stat().st_size > 0


def main() -> None:
    st.title("C2 · Driver Intelligence")
    st.caption(
        "Lightweight CPU realtime replay · Face + phone evidence + temporal state machine"
    )

    with st.sidebar:
        st.subheader("Replay controls")
        phone_enabled = st.checkbox("Cell-phone detector", value=True)
        phone_every = st.slider(
            "Phone detector interval (frames)",
            min_value=1,
            max_value=10,
            value=5,
            help="Face runs every frame; phone detection runs less often to keep CPU latency low.",
        )
        calibration_seconds = st.slider(
            "Neutral calibration (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )
        speed = st.select_slider(
            "Replay speed",
            options=[0.5, 1.0, 1.5, 2.0],
            value=1.0,
            format_func=lambda value: f"{value:g}×",
        )
        target_fps = st.slider(
            "Web display FPS",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="The engine samples the source at this rate so Streamlit stays realtime.",
        )
        max_seconds = st.slider(
            "Maximum replay duration",
            min_value=5,
            max_value=120,
            value=35,
            step=5,
            format="%d s",
        )

    st.info(
        "Upload an MP4 or use the bundled test clip. The browser receives rendered "
        "HUD frames in realtime; the same engine is used by the OpenCV live demo."
    )
    uploaded = st.file_uploader(
        "Choose a driver video",
        type=["mp4", "mov", "avi", "mkv"],
        help="A short 640p–720p MP4 is ideal for the CPU-only demo.",
    )

    source: Path | None = None
    if uploaded is not None:
        st.video(uploaded)
    elif _available_sample():
        st.success(f"Bundled test clip available: `{SAMPLE_VIDEO.name}`")
        source = SAMPLE_VIDEO
        st.video(str(SAMPLE_VIDEO))
    else:
        st.warning(
            "Upload an MP4. A bundled sample will appear after running "
            "`c2/make_web_test_video.py`."
        )

    start = st.button(
        "▶ Start realtime replay",
        type="primary",
        disabled=uploaded is None and source is None,
        width="stretch",
    )
    if not start:
        st.markdown(
            "The current lightweight web surface is MP4 replay. "
            "Continuous browser webcam capture can be added later with WebRTC."
        )
        return

    temporary_source: Path | None = None
    try:
        if uploaded is not None:
            import tempfile

            suffix = Path(uploaded.name).suffix or ".mp4"
            handle = tempfile.NamedTemporaryFile(
                prefix="c2_streamlit_",
                suffix=suffix,
                delete=False,
            )
            handle.write(uploaded.getbuffer())
            handle.close()
            temporary_source = Path(handle.name)
            source = temporary_source
        if source is None:
            raise RuntimeError("No source video selected.")
        with st.spinner("Loading MediaPipe models…"):
            summary = _run_realtime_replay(
                source,
                phone_enabled=phone_enabled,
                phone_every=phone_every,
                calibration_seconds=calibration_seconds,
                speed=speed,
                max_seconds=float(max_seconds),
                target_fps=target_fps,
            )
    except Exception as exc:
        st.error(f"Replay failed: {exc}")
        st.exception(exc)
        return
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)

    st.success(
        f"Replay finished · {summary['frames']} frames · "
        f"{summary['realtime_factor']}× realtime · "
        f"face coverage {summary['face_found']:.1%}"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Realtime factor", f"{summary['realtime_factor']}×")
    metric_columns[1].metric("Face coverage", f"{summary['face_found']:.1%}")
    metric_columns[2].metric("Face inference", f"{summary['face_mean_ms']} ms")
    metric_columns[3].metric(
        "Phone inference",
        f"{summary['phone_mean_ms']} ms"
        if summary["phone_mean_ms"] is not None
        else "off",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("State frames")
        st.json(summary["state_counts"])
    with right:
        st.subheader("Event timeline")
        if summary["events"]:
            st.dataframe(
                [
                    {"time_s": round(t, 2), "state": state, "reason": reason}
                    for t, state, reason in summary["events"]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No state transition after calibration.")


if __name__ == "__main__":
    main()
