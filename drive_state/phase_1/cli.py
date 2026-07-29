"""`driver-safety challenge ...` — the Phase 1 command surface."""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Annotated

import cv2
import typer
from rich.console import Console

from drive_state.phase_1.classifier import TUNABLE_KNOBS, ClassifierConfig
from drive_state.phase_1.demo import STREAMING_CONFIG, draw_hud, iter_demo_frames
from drive_state.phase_1.evaluate import evaluate_report, load_trip_data
from drive_state.phase_1.extract import extract_dataset
from drive_state.phase_1.practice import discover_trips
from drive_state.phase_1.predict import predict_trip_folder
from drive_state.phase_1.scoring import format_confusion
from drive_state.phase_1.tuning import grid_search, leave_one_trip_out

app = typer.Typer(no_args_is_help=True, help="Challenge 2 driver-state pipeline (Phase 1).")
console = Console()

DATASET_HELP = "Practice_Dataset root, containing one directory per trip."


def _load_classifier_config(path: Path | None) -> ClassifierConfig:
    if path is None:
        return ClassifierConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return replace(ClassifierConfig(), **{k: v for k, v in data.items() if k in TUNABLE_KNOBS})


def _dump_classifier_config(config: ClassifierConfig) -> dict[str, float | int]:
    return {knob: getattr(config, knob) for knob in TUNABLE_KNOBS}


@app.command()
def extract(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("runs/phase1/features"),
    model: Annotated[Path, typer.Option("--model")] = Path("models/face_landmarker.task"),
) -> None:
    """Landmark every cabin frame and cache per-frame features as CSV."""

    def progress(trip_id: str, done: int, total: int) -> None:
        console.print(f"[dim]{trip_id}[/dim] {done}/{total}", end="\r")

    paths = extract_dataset(dataset, out, model_path=model, progress=progress)
    console.print(f"[bold green]Extracted {len(paths)} trips[/bold green] -> {out}")


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    features: Annotated[Path, typer.Option("--features", file_okay=False)] = Path(
        "runs/phase1/features"
    ),
    config: Annotated[Path | None, typer.Option("--classifier-config", dir_okay=False)] = None,
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
) -> None:
    """Score the current thresholds against the Practice ground truth."""
    data = load_trip_data(discover_trips(dataset), features)
    report = evaluate_report(data, _load_classifier_config(config))

    for score in report.per_trip:
        console.print(
            f"{score.trip_id:<14} acc={score.accuracy:.3f}  macro-F1={score.macro_f1:.3f}  "
            f"composite={score.composite:5.1f}  present={list(score.present_classes)}"
        )
    console.print(f"[bold]Overall composite: {report.overall:.1f} / 100[/bold]")
    console.print(f"\n{format_confusion(report.per_trip)}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        console.print(f"\nwrote {out}")


@app.command()
def tune(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    features: Annotated[Path, typer.Option("--features", file_okay=False)] = Path(
        "runs/phase1/features"
    ),
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    loto: Annotated[bool, typer.Option("--loto/--no-loto")] = True,
) -> None:
    """Grid-search thresholds, and report the held-out score alongside.

    The fitted number is what the thresholds score on the trips they were
    chosen on; the leave-one-trip-out number withholds each trip from its own
    fit. Quote the second one — six trips over six subjects is not enough to
    fit six thresholds on.
    """
    data = load_trip_data(discover_trips(dataset), features)

    best = grid_search(data)
    console.print(
        f"[bold]Fitted composite: {best.composite:.1f}[/bold] ({best.n_candidates} candidates)"
    )
    console.print(f"  config: {_dump_classifier_config(best.config)}")
    for score in best.per_trip:
        console.print(f"    {score.trip_id:<14} composite={score.composite:5.1f}")

    payload: dict[str, object] = {
        "fitted": {
            "overall_composite": round(best.composite, 1),
            "config": _dump_classifier_config(best.config),
            "per_trip": [s.to_dict() for s in best.per_trip],
        }
    }

    if loto:
        result = leave_one_trip_out(data)
        console.print(
            f"\n[bold]Leave-one-trip-out composite: {result.overall:.1f}[/bold] "
            f"[dim](the honest number)[/dim]"
        )
        for score in result.held_out:
            console.print(f"    {score.trip_id:<14} composite={score.composite:5.1f}")
        payload["leave_one_trip_out"] = {
            "overall_composite": round(result.overall, 1),
            "per_trip": [s.to_dict() for s in result.held_out],
            "fold_configs": {
                tid: _dump_classifier_config(cfg) for tid, cfg in result.fold_configs.items()
            },
        }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"\nwrote {out}")


@app.command()
def predict(
    trip: Annotated[Path, typer.Option("--trip", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "runs/phase1/predictions.csv"
    ),
    features: Annotated[Path | None, typer.Option("--features", file_okay=False)] = Path(
        "runs/phase1/features"
    ),
    config: Annotated[Path | None, typer.Option("--classifier-config", dir_okay=False)] = None,
    model: Annotated[Path, typer.Option("--model")] = Path("models/face_landmarker.task"),
) -> None:
    """Write a submission CSV for one trip folder."""
    path = predict_trip_folder(
        trip,
        out,
        config=_load_classifier_config(config),
        features_dir=features,
        model_path=model,
    )
    console.print(f"[bold green]Wrote[/bold green] {path}")


@app.command()
def demo(
    trip: Annotated[Path, typer.Option("--trip", exists=True, file_okay=False)],
    mode: Annotated[str, typer.Option("--mode", help="realtime paces at 20 fps; fast does not")] = (
        "realtime"
    ),
    speed: Annotated[float, typer.Option("--speed", help="realtime multiplier, e.g. 2.0")] = 1.0,
    show: Annotated[bool, typer.Option("--show/--no-show")] = True,
    save: Annotated[Path | None, typer.Option("--save", dir_okay=False)] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    config: Annotated[Path | None, typer.Option("--classifier-config", dir_okay=False)] = None,
    model: Annotated[Path, typer.Option("--model")] = Path("models/face_landmarker.task"),
    phone_model: Annotated[Path, typer.Option("--phone-model")] = Path(
        "models/driver-objects.onnx"
    ),
    phone_stride: Annotated[
        int, typer.Option("--phone-stride", help="run the phone detector every Nth frame")
    ] = 5,
    no_phone: Annotated[
        bool, typer.Option("--no-phone", help="disable phone detection (costs ~20 composite)")
    ] = False,
    tripkit: Annotated[
        bool | None, typer.Option("--tripkit/--no-tripkit", help="force or forbid tripkit replay")
    ] = None,
) -> None:
    """Replay a trip with live driver-state inference and a HUD.

    Uses a *trailing* window, unlike `predict`: nothing from the future is
    read, so what you see is what a real vehicle could compute. That scores
    88.2 against the offline 96.2, and the gap is concentrated at state
    transitions -- expect a visible lag of a second or two after the driver
    changes behaviour.
    """
    if mode not in {"realtime", "fast"}:
        raise typer.BadParameter("--mode must be 'realtime' or 'fast'")

    classifier_config = STREAMING_CONFIG if config is None else _load_classifier_config(config)
    writer = None
    # Wall-clock starts at the first frame, not here: loading the MediaPipe
    # graph takes ~3 s, and folding that into the rate would understate a short
    # run badly (100 frames at 20 fps reads as 12 fps instead of 20).
    started: float | None = None
    n_frames = 0
    total_latency_ms = 0.0
    # Predictions are collected rather than scored inline: frames seen while the
    # buffer was still filling get backfilled with the first settled decision,
    # which is what a unit that stays quiet until it is warm would report for
    # them. Scoring them as they were guessed understates the design by ~3
    # composite -- see docs/drive_state.md.
    predicted: dict[int, str] = {}
    truth: dict[int, str] = {}
    pending: list[int] = []
    n_held = 0

    try:
        for frame in iter_demo_frames(
            trip,
            config=classifier_config,
            mode=mode,
            speed=speed,
            limit=limit,
            model_path=model,
            phone_model=None if no_phone else phone_model,
            phone_stride=phone_stride,
            use_tripkit=tripkit,
        ):
            if started is None:
                started = perf_counter()
            n_frames += 1
            total_latency_ms += frame.latency_ms
            if frame.truth is not None:
                truth[frame.frame_id] = frame.truth
            if frame.ready:
                for held in pending:
                    predicted[held] = frame.predicted
                pending.clear()
                predicted[frame.frame_id] = frame.predicted
            else:
                pending.append(frame.frame_id)
                n_held += 1

            elapsed = perf_counter() - started
            canvas = draw_hud(
                frame,
                config=classifier_config,
                fps=n_frames / elapsed if elapsed > 0 else None,
                landmarks=frame.landmarks,
                face_bbox=frame.face_bbox,
            )

            if save is not None and writer is None:
                save.parent.mkdir(parents=True, exist_ok=True)
                height, width = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(save), fourcc, 20.0, (width, height))
            if writer is not None:
                writer.write(canvas)

            if show:
                cv2.imshow("driver state - phase 1", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except cv2.error as exc:
        raise typer.BadParameter(
            f"OpenCV display failed (headless environment? use --no-show --save out.mp4): {exc}"
        ) from exc
    finally:
        if writer is not None:
            writer.release()
        if show:
            # Headless OpenCV builds raise from destroyAllWindows too; swallowing
            # it here keeps the teardown from masking the real exit status.
            with contextlib.suppress(cv2.error):
                cv2.destroyAllWindows()

    if not n_frames:
        console.print("[yellow]No frames replayed.[/yellow]")
        return

    elapsed = perf_counter() - (started or perf_counter())
    mean_latency = total_latency_ms / n_frames
    console.print(
        f"{n_frames} frames in {elapsed:.1f}s ({n_frames / elapsed:.1f} fps wall, mode={mode})"
    )
    # In realtime mode the wall figure is just the pacing target; the inference
    # rate is what says whether there is headroom.
    console.print(
        f"inference: {mean_latency:.1f} ms/frame ({1000 / mean_latency:.0f} fps capacity)"
    )
    # Anything still pending means the run ended before the buffer ever filled.
    for held in pending:
        predicted[held] = "alert"

    scored = sorted(set(predicted) & set(truth))
    if scored:
        correct = sum(1 for fid in scored if predicted[fid] == truth[fid])
        console.print(
            f"streaming accuracy against labels: {correct / len(scored):.3f} "
            f"({correct}/{len(scored)})"
        )
        if n_held:
            settled = [fid for fid in scored if fid >= n_held]
            held_right = sum(
                1 for fid in scored[:n_held] if predicted[fid] == truth[fid]
            )
            console.print(
                f"  first {n_held} frames held during buffering, backfilled: "
                f"{held_right}/{n_held} correct"
            )
            if settled:
                after = sum(1 for fid in settled if predicted[fid] == truth[fid])
                console.print(f"  after buffering: {after / len(settled):.3f}")
    if save is not None:
        console.print(f"[bold green]Wrote[/bold green] {save}")


if __name__ == "__main__":
    # Without this, `python -m drive_state.phase_1.cli` would import the module,
    # define the app, and exit 0 having printed nothing -- which reads exactly
    # like a command that ran and found no work to do.
    app()
