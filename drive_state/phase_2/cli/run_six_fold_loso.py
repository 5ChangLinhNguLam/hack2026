"""Restart-safe orchestration for the approved six-fold hybrid LOSO run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Mapping, Sequence

from ..metrics_events import aggregate_loso_metrics


SELECTED_SUBJECTS = (
    "gA_1",
    "gB_10",
    "gC_14",
    "gE_29",
    "gF_23",
    "gZ_33",
)
NEW_SUBJECTS = SELECTED_SUBJECTS[1:]


@dataclass(frozen=True)
class RunPaths:
    dmd_root: Path
    app_root: Path
    python: Path
    standardized_index: Path
    region_cache: Path
    nitymed_manifest: Path
    nitymed_regions: Path
    nitymed_teacher: Path
    base_output: Path
    hybrid_output: Path
    gate_output: Path
    completed_ga1: Path


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    required_artifacts: tuple[Path, ...]
    completion_marker: Path | None = None


def _marker(fold: Path, stage: str) -> Path:
    return fold / ".stage_complete" / f"{stage}.json"


def _command(
    stage: str,
    subject: str,
    output: Path,
    paths: RunPaths,
    *extra: str,
) -> tuple[str, ...]:
    return (
        str(paths.python),
        "-m",
        "driver_state.cli.train_mobilenet_lstm",
        "--stage",
        stage,
        "--root",
        str(paths.dmd_root),
        "--standardized-index",
        str(paths.standardized_index),
        "--region-cache",
        str(paths.region_cache),
        "--held-out-subject",
        subject,
        "--output",
        str(output),
        "--embedding-dim",
        "256",
        "--region-dim",
        "64",
        "--image-width",
        "640",
        "--image-height",
        "384",
        "--workers",
        "4",
        "--seed",
        "31",
        "--amp",
        *extra,
    )


def build_fold_stages(subject: str, paths: RunPaths) -> tuple[Stage, ...]:
    if subject not in NEW_SUBJECTS:
        raise ValueError(f"subject is not in the new stress-test folds: {subject}")
    base_fold = paths.base_output / subject
    hybrid_fold = paths.hybrid_output / subject
    gate_fold = paths.gate_output / subject
    visual_options = (
        "--visual-epochs",
        "12",
        "--visual-batch-size",
        "64",
        "--samples-per-epoch",
        "40000",
        "--max-windows-per-event",
        "8",
        "--patience",
        "4",
    )
    ocular_options = (
        "--ocular-sequence-length",
        "60",
        "--ocular-epochs",
        "20",
        "--ocular-batch-size",
        "256",
        "--ocular-channels",
        "64",
        "--ocular-samples-per-epoch",
        "20000",
        "--ocular-max-windows-per-event",
        "8",
        "--uncertain-eye-gap-seconds",
        "0.10",
        "--microsleep-seconds",
        "2.0",
        "--patience",
        "4",
    )
    return (
        Stage(
            "visual",
            _command("visual", subject, paths.base_output, paths, *visual_options),
            (base_fold / "visual/best.pt", base_fold / "visual/metrics.json"),
            _marker(base_fold, "visual"),
        ),
        Stage(
            "visual-mixed",
            _command(
                "visual-mixed",
                subject,
                paths.hybrid_output,
                paths,
                "--initial-visual-checkpoint",
                str(base_fold / "visual/best.pt"),
                "--nitymed-frame-manifest",
                str(paths.nitymed_manifest),
                "--nitymed-region-cache",
                str(paths.nitymed_regions),
                "--nitymed-teacher-cache",
                str(paths.nitymed_teacher),
                "--mixed-adaptation",
                "yawn",
                "--visual-epochs",
                "1",
                "--visual-batch-size",
                "128",
                "--patience",
                "4",
            ),
            (
                hybrid_fold / "visual/best.pt",
                hybrid_fold / "visual/metrics.json",
            ),
            _marker(hybrid_fold, "visual-mixed"),
        ),
        Stage(
            "cache",
            _command(
                "cache",
                subject,
                paths.hybrid_output,
                paths,
                "--visual-batch-size",
                "64",
            ),
            (hybrid_fold / "visual_cache/index.json",),
            _marker(hybrid_fold, "cache"),
        ),
        Stage(
            "ocular-clean",
            _command(
                "ocular",
                subject,
                paths.hybrid_output,
                paths,
                *ocular_options,
                "--ocular-input-corruption-probability",
                "0.0",
            ),
            (
                hybrid_fold / "ocular/best.pt",
                hybrid_fold / "ocular/metrics.json",
            ),
            _marker(hybrid_fold, "ocular-clean"),
        ),
        Stage(
            "ocular-cache-clean",
            _command(
                "ocular-cache",
                subject,
                paths.hybrid_output,
                paths,
                "--ocular-sequence-length",
                "60",
                "--microsleep-seconds",
                "2.0",
            ),
            (hybrid_fold / "ocular_cache/index.json",),
            _marker(hybrid_fold, "ocular-cache-clean"),
        ),
        Stage(
            "ocular-robust",
            _command(
                "ocular",
                subject,
                paths.gate_output,
                paths,
                *ocular_options,
                "--ocular-input-corruption-probability",
                "0.5",
            ),
            (gate_fold / "ocular/best.pt", gate_fold / "ocular/metrics.json"),
            _marker(gate_fold, "ocular-robust"),
        ),
        Stage(
            "ocular-cache-robust",
            _command(
                "ocular-cache",
                subject,
                paths.gate_output,
                paths,
                "--ocular-sequence-length",
                "60",
                "--microsleep-seconds",
                "2.0",
            ),
            (gate_fold / "ocular_cache/index.json",),
            _marker(gate_fold, "ocular-cache-robust"),
        ),
        Stage(
            "lstm",
            _command(
                "lstm",
                subject,
                paths.hybrid_output,
                paths,
                "--sequence-length",
                "200",
                "--lstm-epochs",
                "30",
                "--lstm-batch-size",
                "128",
                "--temporal-channels",
                "256",
                "--temporal-learning-rate",
                "3e-4",
                "--samples-per-epoch",
                "40000",
                "--max-windows-per-event",
                "8",
                "--patience",
                "4",
                "--microsleep-gate-ocular-cache",
                str(gate_fold / "ocular_cache"),
                "--microsleep-gate-ocular-checkpoint",
                str(gate_fold / "ocular/best.pt"),
            ),
            (
                hybrid_fold / "temporal/best.pt",
                hybrid_fold / "temporal/outer_metrics.json",
                hybrid_fold / "temporal/outer_predictions.jsonl",
            ),
            _marker(hybrid_fold, "lstm"),
        ),
    )


def stage_complete(stage: Stage) -> bool:
    return all(path.is_file() for path in stage.required_artifacts) and (
        stage.completion_marker is None or stage.completion_marker.is_file()
    )


def ensure_gate_visual_links(subject: str, paths: RunPaths) -> None:
    hybrid_fold = paths.hybrid_output / subject
    gate_fold = paths.gate_output / subject
    for name in ("visual", "visual_cache"):
        source = (hybrid_fold / name).resolve()
        if not source.is_dir():
            raise ValueError(f"hybrid {name} source is missing: {source}")
        target = gate_fold / name
        if target.is_symlink():
            if target.resolve() == source:
                continue
            raise ValueError(f"conflicting gate visual path: {target}")
        if target.exists():
            raise ValueError(f"conflicting gate visual path: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_stage(stage: Stage, log_path: Path, *, dry_run: bool) -> None:
    rendered = shlex.join(stage.command)
    if dry_run:
        print(rendered, flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment["PYTHONPATH"] = "src"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {rendered}\n")
        stream.flush()
        subprocess.run(
            stage.command,
            cwd=(None if not stage.command else Path.cwd()),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    missing = [path for path in stage.required_artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"stage {stage.name} exited successfully without artifacts: {missing}"
        )
    if stage.completion_marker is not None:
        _atomic_json(
            stage.completion_marker,
            {
                "command": list(stage.command),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "stage": stage.name,
            },
        )


def write_interim_report(paths: RunPaths) -> dict[str, object]:
    metric_paths = {
        "gA_1": paths.completed_ga1 / "temporal/outer_metrics.json",
        **{
            subject: (
                paths.hybrid_output
                / subject
                / "temporal/outer_metrics.json"
            )
            for subject in NEW_SUBJECTS
        },
    }
    fold_metrics: dict[str, Mapping[str, object]] = {}
    for subject in SELECTED_SUBJECTS:
        path = metric_paths[subject]
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"outer metrics are not a mapping: {path}")
            fold_metrics[subject] = payload
    if not fold_metrics:
        raise ValueError("interim aggregation requires one completed fold")
    report: dict[str, object] = {
        "schema_version": 1,
        "qualification": "stress_test_not_unbiased_loso",
        "selected_subjects": list(SELECTED_SUBJECTS),
        "completed_subjects": [
            subject for subject in SELECTED_SUBJECTS if subject in fold_metrics
        ],
        "per_fold": dict(fold_metrics),
        "aggregate": aggregate_loso_metrics(fold_metrics),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(paths.hybrid_output / "six_fold_interim.json", report)
    return report


def default_run_paths(dmd_root: Path) -> RunPaths:
    dmd_root = dmd_root.resolve()
    app_root = dmd_root / "temporal_driver_state"
    runs = dmd_root / "runs"
    nitymed = dmd_root / "nitymed_method_a_5fps"
    return RunPaths(
        dmd_root=dmd_root,
        app_root=app_root,
        python=dmd_root / "driver_state/.venv/bin/python",
        standardized_index=(
            app_root
            / "artifacts/five_state_standardized_20fps_confirmed/index.json"
        ),
        region_cache=dmd_root / "cache/evidence_regions_20fps_v3",
        nitymed_manifest=nitymed / "frames.json",
        nitymed_regions=nitymed / "regions",
        nitymed_teacher=nitymed / "teacher",
        base_output=runs / "evidence_ocular_lstm_v7_six_fold_base",
        hybrid_output=runs / "evidence_ocular_lstm_v7_six_fold_hybrid",
        gate_output=runs / "evidence_ocular_lstm_v7_six_fold_gate",
        completed_ga1=(
            runs / "evidence_ocular_lstm_v6_hybrid/gA_1"
        ),
    )


def _validate_inputs(paths: RunPaths) -> None:
    files = (
        paths.python,
        paths.standardized_index,
        paths.nitymed_manifest,
        paths.completed_ga1 / "temporal/outer_metrics.json",
    )
    directories = (
        paths.app_root,
        paths.region_cache,
        paths.nitymed_regions,
        paths.nitymed_teacher,
    )
    missing = [str(path) for path in files if not path.is_file()]
    missing.extend(str(path) for path in directories if not path.is_dir())
    if missing:
        raise ValueError("six-fold inputs are missing: " + ", ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved six-fold hybrid LOSO stress test"
    )
    parser.add_argument(
        "--dmd-root",
        type=Path,
        default=Path(
            "/home/ubuntu/workspaces/thiennh/dmd_extract_download"
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        choices=NEW_SUBJECTS,
        default=list(NEW_SUBJECTS),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subjects = tuple(str(subject) for subject in args.subjects)
    if len(set(subjects)) != len(subjects):
        raise ValueError("six-fold subjects must not be duplicated")
    paths = default_run_paths(args.dmd_root)
    _validate_inputs(paths)
    if args.dry_run:
        for subject in subjects:
            print(f"[{subject}]", flush=True)
            for stage in build_fold_stages(subject, paths):
                run_stage(
                    stage,
                    paths.hybrid_output / "logs" / f"{subject}_{stage.name}.log",
                    dry_run=True,
                )
        return 0

    os.chdir(paths.app_root)
    status_path = paths.hybrid_output / "status.json"
    started_at = datetime.now(timezone.utc).isoformat()
    completed_subjects: list[str] = []
    for subject in subjects:
        for stage in build_fold_stages(subject, paths):
            if stage.name in ("ocular-robust", "ocular-cache-robust"):
                ensure_gate_visual_links(subject, paths)
            if stage_complete(stage):
                print(f"[{subject}] skip complete {stage.name}", flush=True)
                continue
            _atomic_json(
                status_path,
                {
                    "active_subject": subject,
                    "active_stage": stage.name,
                    "completed_subjects": completed_subjects,
                    "last_error": None,
                    "started_at": started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            log_path = (
                paths.hybrid_output / "logs" / f"{subject}_{stage.name}.log"
            )
            print(f"[{subject}] run {stage.name}", flush=True)
            try:
                run_stage(stage, log_path, dry_run=False)
            except Exception as error:
                _atomic_json(
                    status_path,
                    {
                        "active_subject": subject,
                        "active_stage": stage.name,
                        "completed_subjects": completed_subjects,
                        "last_error": f"{type(error).__name__}: {error}",
                        "started_at": started_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                raise
        completed_subjects.append(subject)
        write_interim_report(paths)
    _atomic_json(
        status_path,
        {
            "active_subject": None,
            "active_stage": None,
            "completed_subjects": completed_subjects,
            "last_error": None,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


__all__ = [
    "NEW_SUBJECTS",
    "SELECTED_SUBJECTS",
    "RunPaths",
    "Stage",
    "build_fold_stages",
    "build_parser",
    "default_run_paths",
    "ensure_gate_visual_links",
    "run_stage",
    "stage_complete",
    "write_interim_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
