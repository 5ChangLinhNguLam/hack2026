#!/usr/bin/env python3
"""Offline-only top-1 target audit for C1 P2-C diagnostics.

This tool is intentionally outside :mod:`safeloop.c1`.  It joins a runtime
diagnostic primary bounding box to practice-only projected KITTI cuboids and
uses privileged target fields to define the oracle collision target.  Neither
ground truth nor projected oracle boxes are available to, or imported by, the
deployable runtime.

The image-space join is a diagnostic proxy, not identity ground truth.  A
primary box is mapped only when its best projected-box IoU is at least 0.10
and is separated from the runner-up by the configured ambiguity margin.
Mapping and oracle-projection coverage are therefore reported alongside every
target metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.c1_p2_oracle_diagnostics import (
    KittiObject,
    OfflineFrame,
    _oracle_target_id,
    load_offline_trip,
)


REPORT_SCHEMA = "safeloop.c1.p2c_target_audit.v1"
DEFAULT_MIN_IOU = 0.10
DEFAULT_AMBIGUITY_IOU_MARGIN = 0.05
DIAGNOSTIC_COLUMNS = (
    "trip_id",
    "frame_id",
    "primary_track_id",
    "primary_bbox_x1",
    "primary_bbox_y1",
    "primary_bbox_x2",
    "primary_bbox_y2",
)


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PrimaryDiagnostic:
    """Runtime-only fields needed by the offline identity audit."""

    trip_id: str
    frame_id: int
    primary_track_id: int | None
    primary_bbox: BBox | None


@dataclass(frozen=True, slots=True)
class TargetMapping:
    """Result of joining one primary bbox to projected oracle objects."""

    status: str
    target_id: int | None
    best_iou: float
    runner_up_iou: float

    @property
    def mapped(self) -> bool:
        return self.status == "mapped"


@dataclass(frozen=True, slots=True)
class TripTargetAudit:
    trip_id: str
    n_frames: int
    oracle_target_frames: int
    oracle_projected_frames: int
    oracle_projection_coverage: float | None
    primary_present_frames: int
    primary_presence_coverage: float
    mapped_primary_frames: int
    unknown_primary_frames: int
    mapped_coverage: float | None
    unknown_coverage: float | None
    unknown_missing_bbox_frames: int
    unknown_no_overlap_frames: int
    unknown_ambiguous_frames: int
    top1_evaluable_frames: int
    top1_correct_frames: int
    top1_target_agreement: float | None
    target_tp: int
    target_fp: int
    target_fn: int
    target_precision: float | None
    target_recall: float | None
    primary_track_switch_count: int
    mapped_target_switch_count: int
    oracle_target_switch_count: int


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _bbox_iou(left: BBox, right: BBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def map_primary_bbox(
    bbox: BBox,
    objects: Sequence[KittiObject],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
    ambiguity_iou_margin: float = DEFAULT_AMBIGUITY_IOU_MARGIN,
) -> TargetMapping:
    """Map a primary bbox to one projected target, preserving uncertainty.

    More than one projected cuboid may overlap a visible silhouette.  The
    highest-IoU object is accepted only when it clears both the absolute IoU
    gate and the configured margin over the runner-up.  Otherwise the mapping
    remains unknown and is excluded from conditional top-1 agreement.
    """

    if not 0.0 < min_iou <= 1.0:
        raise ValueError("min_iou must be in (0, 1]")
    if not 0.0 <= ambiguity_iou_margin <= 1.0:
        raise ValueError("ambiguity_iou_margin must be in [0, 1]")
    ranked = sorted(
        ((_bbox_iou(bbox, obj.bbox), int(obj.target_id)) for obj in objects),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] < min_iou:
        best = ranked[0][0] if ranked else 0.0
        return TargetMapping("unknown_no_overlap", None, best, 0.0)
    best_iou, best_target_id = ranked[0]
    runner_up_iou = ranked[1][0] if len(ranked) > 1 else 0.0
    if (
        runner_up_iou >= min_iou
        and best_iou - runner_up_iou <= ambiguity_iou_margin
    ):
        return TargetMapping("unknown_ambiguous", None, best_iou, runner_up_iou)
    return TargetMapping("mapped", best_target_id, best_iou, runner_up_iou)


def _optional_track_id(raw: str, *, path: Path, row_number: int) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"{path}: row {row_number}: invalid primary_track_id={raw!r}"
        ) from exc


def _primary_bbox(
    row: Mapping[str, str],
    track_id: int | None,
    *,
    path: Path,
    row_number: int,
) -> BBox | None:
    fields = tuple(row[column].strip() for column in DIAGNOSTIC_COLUMNS[3:])
    if track_id is None:
        if any(fields):
            raise ValueError(
                f"{path}: row {row_number}: primary bbox exists without a track ID"
            )
        return None
    if not any(fields):
        # A bounded hysteresis hold may retain a primary identity after the
        # live track (and therefore its current bbox) is gone.  Keep this as
        # explicit unknown coverage; never synthesize or carry a stale box.
        return None
    if not all(fields):
        raise ValueError(
            f"{path}: row {row_number}: primary track requires all four bbox fields"
        )
    try:
        bbox = tuple(float(value) for value in fields)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_number}: invalid primary bbox") from exc
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{path}: row {row_number}: primary bbox must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{path}: row {row_number}: primary bbox has non-positive area")
    return bbox  # type: ignore[return-value]


def load_primary_diagnostics(path: Path) -> tuple[PrimaryDiagnostic, ...]:
    """Load and validate the minimal P2-C target diagnostic schema."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = set(DIAGNOSTIC_COLUMNS) - fieldnames
        if missing:
            raise ValueError(f"{path}: missing diagnostic columns {sorted(missing)}")
        output: list[PrimaryDiagnostic] = []
        seen_frames: set[int] = set()
        expected_trip: str | None = None
        for row_number, row in enumerate(reader, start=2):
            trip_id = row["trip_id"].strip()
            if not trip_id:
                raise ValueError(f"{path}: row {row_number}: blank trip_id")
            if expected_trip is None:
                expected_trip = trip_id
            elif trip_id != expected_trip:
                raise ValueError(
                    f"{path}: row {row_number}: mixed trip IDs "
                    f"{expected_trip!r} and {trip_id!r}"
                )
            try:
                frame_id = int(row["frame_id"])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: row {row_number}: invalid frame_id={row['frame_id']!r}"
                ) from exc
            if frame_id in seen_frames:
                raise ValueError(f"{path}: duplicate frame_id={frame_id}")
            seen_frames.add(frame_id)
            track_id = _optional_track_id(
                row["primary_track_id"], path=path, row_number=row_number
            )
            output.append(
                PrimaryDiagnostic(
                    trip_id=trip_id,
                    frame_id=frame_id,
                    primary_track_id=track_id,
                    primary_bbox=_primary_bbox(
                        row,
                        track_id,
                        path=path,
                        row_number=row_number,
                    ),
                )
            )
    if not output:
        raise ValueError(f"{path}: diagnostic CSV is empty")
    output.sort(key=lambda item: item.frame_id)
    return tuple(output)


def _direct_switch_count(values: Sequence[int | None]) -> int:
    return sum(
        previous is not None and current is not None and previous != current
        for previous, current in zip(values, values[1:])
    )


def audit_trip(
    trip_id: str,
    diagnostics: Sequence[PrimaryDiagnostic],
    frames: Sequence[OfflineFrame],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
    ambiguity_iou_margin: float = DEFAULT_AMBIGUITY_IOU_MARGIN,
) -> TripTargetAudit:
    """Compute coverage-aware target metrics for one trip.

    ``target_precision`` uses unambiguously mapped primary selections on
    scorable reference frames.  ``target_recall`` uses every frame whose
    oracle target has a projected box; absent or unknown primary mappings are
    false negatives.  ``top1_target_agreement`` is conditional on both the
    oracle target being projectable and the runtime primary being mapped.
    """

    if not frames:
        raise ValueError(f"{trip_id}: offline trip is empty")
    diagnostic_by_id = {row.frame_id: row for row in diagnostics}
    frame_by_id = {frame.frame_id: frame for frame in frames}
    if len(diagnostic_by_id) != len(diagnostics):
        raise ValueError(f"{trip_id}: duplicate diagnostic frame IDs")
    if len(frame_by_id) != len(frames):
        raise ValueError(f"{trip_id}: duplicate offline frame IDs")
    if set(diagnostic_by_id) != set(frame_by_id):
        missing = sorted(set(frame_by_id) - set(diagnostic_by_id))
        unexpected = sorted(set(diagnostic_by_id) - set(frame_by_id))
        raise ValueError(
            f"{trip_id}: diagnostic/offline frame mismatch; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    ordered_ids = sorted(frame_by_id)
    if any(diagnostic_by_id[frame_id].trip_id != trip_id for frame_id in ordered_ids):
        raise ValueError(f"{trip_id}: diagnostic row has a different trip_id")

    oracle_target_ids: list[int | None] = []
    primary_track_ids: list[int | None] = []
    mapped_target_ids: list[int | None] = []
    oracle_target_frames = 0
    oracle_projected_frames = 0
    primary_present_frames = 0
    mapped_primary_frames = 0
    unknown_missing_bbox_frames = 0
    unknown_no_overlap_frames = 0
    unknown_ambiguous_frames = 0
    top1_evaluable_frames = 0
    top1_correct_frames = 0
    target_tp = 0
    target_fp = 0
    target_fn = 0

    for frame_id in ordered_ids:
        frame = frame_by_id[frame_id]
        diagnostic = diagnostic_by_id[frame_id]
        oracle_target_id = _oracle_target_id(frame)
        oracle_object_ids = {int(obj.target_id) for obj in frame.objects}
        oracle_projected = (
            oracle_target_id is not None and oracle_target_id in oracle_object_ids
        )
        oracle_target_ids.append(oracle_target_id)
        primary_track_ids.append(diagnostic.primary_track_id)
        oracle_target_frames += int(oracle_target_id is not None)
        oracle_projected_frames += int(oracle_projected)

        mapping: TargetMapping | None = None
        if diagnostic.primary_track_id is not None:
            primary_present_frames += 1
            if diagnostic.primary_bbox is None:
                unknown_missing_bbox_frames += 1
            else:
                mapping = map_primary_bbox(
                    diagnostic.primary_bbox,
                    frame.objects,
                    min_iou=min_iou,
                    ambiguity_iou_margin=ambiguity_iou_margin,
                )
                if mapping.mapped:
                    mapped_primary_frames += 1
                elif mapping.status == "unknown_no_overlap":
                    unknown_no_overlap_frames += 1
                elif mapping.status == "unknown_ambiguous":
                    unknown_ambiguous_frames += 1
                else:  # pragma: no cover - closed TargetMapping status set
                    raise AssertionError(
                        f"unexpected target mapping status {mapping.status}"
                    )
        mapped_target_id = mapping.target_id if mapping and mapping.mapped else None
        mapped_target_ids.append(mapped_target_id)

        correct = oracle_projected and mapped_target_id == oracle_target_id
        if oracle_projected and mapped_target_id is not None:
            top1_evaluable_frames += 1
            top1_correct_frames += int(correct)

        # A positive reference is scorable only while its projected bbox is
        # observable.  A no-target reference is always known.  This prevents
        # projection coverage gaps from being mislabeled as selector errors.
        reference_scorable = oracle_target_id is None or oracle_projected
        if mapped_target_id is not None and reference_scorable:
            if correct:
                target_tp += 1
            else:
                target_fp += 1
        if oracle_projected and not correct:
            target_fn += 1

    unknown_primary_frames = primary_present_frames - mapped_primary_frames
    classified_unknown = (
        unknown_missing_bbox_frames
        + unknown_no_overlap_frames
        + unknown_ambiguous_frames
    )
    if classified_unknown != unknown_primary_frames:
        raise AssertionError(
            "unknown primary mapping taxonomy does not cover every unknown frame"
        )
    return TripTargetAudit(
        trip_id=trip_id,
        n_frames=len(frames),
        oracle_target_frames=oracle_target_frames,
        oracle_projected_frames=oracle_projected_frames,
        oracle_projection_coverage=_ratio(
            oracle_projected_frames, oracle_target_frames
        ),
        primary_present_frames=primary_present_frames,
        primary_presence_coverage=primary_present_frames / len(frames),
        mapped_primary_frames=mapped_primary_frames,
        unknown_primary_frames=unknown_primary_frames,
        mapped_coverage=_ratio(mapped_primary_frames, primary_present_frames),
        unknown_coverage=_ratio(unknown_primary_frames, primary_present_frames),
        unknown_missing_bbox_frames=unknown_missing_bbox_frames,
        unknown_no_overlap_frames=unknown_no_overlap_frames,
        unknown_ambiguous_frames=unknown_ambiguous_frames,
        top1_evaluable_frames=top1_evaluable_frames,
        top1_correct_frames=top1_correct_frames,
        top1_target_agreement=_ratio(
            top1_correct_frames, top1_evaluable_frames
        ),
        target_tp=target_tp,
        target_fp=target_fp,
        target_fn=target_fn,
        target_precision=_ratio(target_tp, target_tp + target_fp),
        target_recall=_ratio(target_tp, target_tp + target_fn),
        primary_track_switch_count=_direct_switch_count(primary_track_ids),
        mapped_target_switch_count=_direct_switch_count(mapped_target_ids),
        oracle_target_switch_count=_direct_switch_count(oracle_target_ids),
    )


def _macro_mean(
    reports: Sequence[TripTargetAudit], field: str
) -> float | None:
    values = [getattr(report, field) for report in reports]
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def summarize_audits(
    reports: Sequence[TripTargetAudit],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
    ambiguity_iou_margin: float = DEFAULT_AMBIGUITY_IOU_MARGIN,
) -> dict[str, object]:
    if not reports:
        raise ValueError("target audit requires at least one trip")
    rate_fields = (
        "top1_target_agreement",
        "target_precision",
        "target_recall",
        "oracle_projection_coverage",
        "primary_presence_coverage",
        "mapped_coverage",
        "unknown_coverage",
    )
    macro: dict[str, object] = {
        "n_trips": len(reports),
        "n_frames": sum(report.n_frames for report in reports),
        **{field: _macro_mean(reports, field) for field in rate_fields},
    }
    count_fields = (
        "oracle_target_frames",
        "oracle_projected_frames",
        "primary_present_frames",
        "mapped_primary_frames",
        "unknown_primary_frames",
        "unknown_missing_bbox_frames",
        "unknown_no_overlap_frames",
        "unknown_ambiguous_frames",
        "top1_evaluable_frames",
        "top1_correct_frames",
        "target_tp",
        "target_fp",
        "target_fn",
        "primary_track_switch_count",
        "mapped_target_switch_count",
        "oracle_target_switch_count",
    )
    for field in count_fields:
        total = sum(int(getattr(report, field)) for report in reports)
        macro[f"{field}_total"] = total
        macro[f"{field}_mean_per_trip"] = total / len(reports)
    return {
        "report_schema": REPORT_SCHEMA,
        "protocol": (
            "offline development diagnostic; projected GT boxes and target fields "
            "are never runtime inputs"
        ),
        "mapping_policy": {
            "min_iou": min_iou,
            "ambiguity_iou_margin": ambiguity_iou_margin,
            "mapped_coverage_denominator": "frames with a runtime primary track",
            "top1_agreement_denominator": (
                "frames with a projectable oracle target and an unambiguously "
                "mapped runtime primary"
            ),
            "target_precision_denominator": (
                "unambiguously mapped runtime primaries on scorable reference frames"
            ),
            "target_recall_denominator": "frames with a projectable oracle target",
            "switch_definition": (
                "direct identity change between adjacent frames where both IDs "
                "are known; an absent or unknown frame breaks the chain"
            ),
            "caveats": (
                "Projected 3-D cuboids may be offset from visible silhouettes.",
                "A retained primary ID without a current bbox is unknown coverage; "
                "the audit never synthesizes or carries stale geometry.",
                "Unknown/ambiguous mappings are coverage failures, not precision false positives.",
                "Target recall counts absent and unknown runtime primaries as "
                "misses only when the oracle target is projectable.",
            ),
        },
        "per_trip": [asdict(report) for report in reports],
        "macro": macro,
    }


def run_audit(
    *,
    data_root: Path,
    diagnostics_root: Path,
    trips: Sequence[str] | None = None,
    min_iou: float = DEFAULT_MIN_IOU,
    ambiguity_iou_margin: float = DEFAULT_AMBIGUITY_IOU_MARGIN,
) -> dict[str, object]:
    selected_trips = tuple(trips or sorted(path.stem for path in diagnostics_root.glob("*.csv")))
    if not selected_trips:
        raise ValueError(f"no diagnostic CSVs found below {diagnostics_root}")
    if len(set(selected_trips)) != len(selected_trips):
        raise ValueError("trip list contains duplicates")
    reports: list[TripTargetAudit] = []
    for trip_id in selected_trips:
        diagnostic_path = diagnostics_root / f"{trip_id}.csv"
        diagnostics = load_primary_diagnostics(diagnostic_path)
        frames = load_offline_trip(data_root / trip_id)
        reports.append(
            audit_trip(
                trip_id,
                diagnostics,
                frames,
                min_iou=min_iou,
                ambiguity_iou_margin=ambiguity_iou_margin,
            )
        )
    return summarize_audits(
        reports,
        min_iou=min_iou,
        ambiguity_iou_margin=ambiguity_iou_margin,
    )


def _atomic_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit P2-C primary target agreement against offline projections."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="Trip ID to audit; repeat as needed. Defaults to discovered CSV stems.",
    )
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument(
        "--ambiguity-iou-margin",
        type=float,
        default=DEFAULT_AMBIGUITY_IOU_MARGIN,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_audit(
            data_root=args.data_root,
            diagnostics_root=args.diagnostics_root,
            trips=args.trips,
            min_iou=args.min_iou,
            ambiguity_iou_margin=args.ambiguity_iou_margin,
        )
        _atomic_json(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"P2-C target audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["macro"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
