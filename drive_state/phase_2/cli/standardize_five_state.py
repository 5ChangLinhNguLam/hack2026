"""Materialize canonical causal five-state labels and nested-LOSO manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from ..data.five_state_labels import FiveState, WeakStateConfig
from ..data.five_state_loso import (
    build_nested_loso_folds,
    write_nested_loso_manifests,
)
from ..data.five_state_standardization import (
    load_standardized_index,
    select_target_rate_rows,
    standardize_session,
    write_standardized_index,
)
from ..data.manifest import load_sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize partial DMD labels into causal five-state events"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--slow-closure-seconds", type=float, default=0.5)
    parser.add_argument("--microsleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--microsleep-confirmation-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument("--alert-open-seconds", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _eye_phase(row: Mapping[str, str]) -> str:
    value = row.get("eyes_state", row.get("phase", ""))
    return str(value).strip().lower()


def _prepare_output(output: Path, *, root: Path, overwrite: bool) -> None:
    resolved = output.resolve()
    protected = {
        Path("/"),
        root.resolve(),
        (root / "labels_20fps").resolve(),
    }
    if resolved in protected:
        raise ValueError("standardized output cannot replace a protected dataset path")
    if output.exists() and not output.is_dir():
        raise ValueError("standardized output must be a directory")
    if output.is_dir() and any(output.iterdir()):
        if not overwrite:
            raise ValueError("standardized output is not empty; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _folds_are_disjoint(folds) -> bool:
    for fold in folds:
        train = set(fold.train_subjects)
        if train & {fold.validation_subject, fold.test_subject}:
            return False
        if fold.validation_subject == fold.test_subject:
            return False
    return True


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    manifest_path = root / "labels_20fps" / "manifest_20fps.json"
    sessions = load_sessions(manifest_path)
    config = WeakStateConfig(
        slow_closure_seconds=args.slow_closure_seconds,
        microsleep_seconds=args.microsleep_seconds,
        microsleep_confirmation_seconds=args.microsleep_confirmation_seconds,
        alert_open_seconds=args.alert_open_seconds,
    )

    standardized = []
    audit = {
        "drowsy_open_frames": 0,
        "drowsy_opening_frames": 0,
        "drowsy_closing_frames": 0,
        "perclos_hard_target_reasons": 0,
    }
    for session in sessions:
        with session.labels_csv.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            rows = tuple(csv.DictReader(stream))
        result = standardize_session(
            session=session.name,
            subject=session.subject_id,
            protocol=session.protocol,
            rows=rows,
            source_fps=session.fps,
            target_fps=args.target_fps,
            config=config,
        )
        selected_rows = select_target_rate_rows(
            rows,
            source_fps=session.fps,
            target_fps=args.target_fps,
        )
        for target, reason, row in zip(
            result.targets.targets,
            result.targets.reasons,
            selected_rows,
            strict=True,
        ):
            if "perclos" in reason.lower():
                audit["perclos_hard_target_reasons"] += 1
            if target != FiveState.DROWSY:
                continue
            phase = _eye_phase(row)
            if phase in {"open", "opening", "closing"}:
                audit[f"drowsy_{phase}_frames"] += 1
            if phase != "close":
                raise ValueError("drowsy target must have current close eye phase")
        standardized.append(result)

    if audit["perclos_hard_target_reasons"]:
        raise ValueError("PERCLOS cannot create a hard five-state target")

    output = args.output.resolve()
    _prepare_output(output, root=root, overwrite=args.overwrite)
    index_path = write_standardized_index(
        output_dir=output,
        target_fps=args.target_fps,
        config=config,
        source_manifest=manifest_path,
        sessions=tuple(standardized),
    )
    index = load_standardized_index(index_path, expected_fps=args.target_fps)
    folds = build_nested_loso_folds(index)
    write_nested_loso_manifests(index, output / "folds")
    disjoint = _folds_are_disjoint(folds)
    if not disjoint:
        raise ValueError("nested LOSO subject isolation failed")

    support = index["summary"]
    assert isinstance(support, Mapping)
    report = {
        "sessions": len(standardized),
        "target_fps": args.target_fps,
        **audit,
        "all_loso_folds_subject_disjoint": disjoint,
        "support": support,
    }
    _atomic_json(output / "summary.json", report)
    global_support = support["global"]
    assert isinstance(global_support, Mapping)
    print(
        "five-state support:",
        json.dumps(global_support["frames"], sort_keys=True),
    )
    print(f"standardized index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
