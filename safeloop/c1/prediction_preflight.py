"""Strict completeness checks for C1 prediction CSV files.

This module intentionally sits in front of the organizer evaluator instead of
changing it.  The official evaluator accepts partial CSV files, which is useful
for exploratory work but unsafe for comparable C1 experiments.  Training and
evaluation workflows can call :func:`preflight_prediction_csv` before scoring
to require one prediction for every expected frame.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_EXPECTED_FRAMES = 600
REQUIRED_COLUMNS = ("frame_id", "timestamp", "predicted_ttc")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class PredictionPreflightError(ValueError):
    """Raised when a prediction CSV is incomplete or malformed."""

    def __init__(self, path: str | Path, errors: Sequence[str]) -> None:
        self.path = Path(path)
        self.errors = tuple(errors)
        detail = "; ".join(self.errors)
        super().__init__(f"{self.path}: {detail}")


@dataclass(frozen=True)
class PredictionPreflightReport:
    """Machine-readable summary returned after a successful check."""

    path: str
    trip_id: str
    expected_frames: int
    rows: int
    finite_ttc: int
    infinite_ttc: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _short_ids(values: Sequence[int], *, limit: int = 12) -> str:
    ordered = sorted(values)
    shown = ", ".join(str(value) for value in ordered[:limit])
    if len(ordered) > limit:
        shown += f", ... (+{len(ordered) - limit})"
    return shown


def _parse_frame_id(raw: str | None, row_number: int, errors: list[str]) -> int | None:
    value = "" if raw is None else raw.strip()
    if not _INTEGER_RE.fullmatch(value):
        errors.append(f"row {row_number}: frame_id is not an integer: {raw!r}")
        return None
    return int(value)


def _parse_finite_number(
    raw: str | None,
    *,
    column: str,
    row_number: int,
    errors: list[str],
) -> float | None:
    value = "" if raw is None else raw.strip()
    try:
        parsed = float(value)
    except ValueError:
        errors.append(f"row {row_number}: {column} is not numeric: {raw!r}")
        return None
    if not math.isfinite(parsed):
        errors.append(f"row {row_number}: {column} must be finite: {raw!r}")
        return None
    return parsed


def _parse_ttc(raw: str | None, row_number: int, errors: list[str]) -> str | None:
    """Return ``finite``/``infinite`` for valid TTC values, else ``None``.

    Positive infinity is the normal no-collision sentinel.  NaN and negative
    infinity are rejected because neither is a valid TTC prediction.
    """

    value = "" if raw is None else raw.strip()
    try:
        parsed = float(value)
    except ValueError:
        errors.append(f"row {row_number}: predicted_ttc is not numeric/inf: {raw!r}")
        return None
    if math.isnan(parsed) or parsed == float("-inf"):
        errors.append(f"row {row_number}: predicted_ttc must be finite or +inf: {raw!r}")
        return None
    return "infinite" if math.isinf(parsed) else "finite"


def preflight_prediction_csv(
    path: str | Path,
    *,
    expected_frames: int = DEFAULT_EXPECTED_FRAMES,
) -> PredictionPreflightReport:
    """Validate one C1 CSV before passing it to the official evaluator.

    The successful file has exactly ``expected_frames`` data rows and exactly
    the integer frame IDs ``0..expected_frames-1``.  Required values must be
    present, timestamps must be finite, and TTC must parse as a finite number
    or positive infinity.
    """

    csv_path = Path(path)
    if expected_frames <= 0:
        raise ValueError("expected_frames must be greater than zero")
    if not csv_path.is_file():
        raise PredictionPreflightError(csv_path, ["file does not exist"])

    errors: list[str] = []
    frame_ids: list[int] = []
    finite_ttc = 0
    infinite_ttc = 0
    rows = 0

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            if header is None:
                raise PredictionPreflightError(csv_path, ["missing CSV header"])

            duplicate_columns = sorted({name for name in header if header.count(name) > 1})
            if duplicate_columns:
                errors.append(f"duplicate columns: {duplicate_columns}")
            missing_columns = sorted(set(REQUIRED_COLUMNS) - set(header))
            if missing_columns:
                errors.append(f"missing required columns: {missing_columns}")
                raise PredictionPreflightError(csv_path, errors)

            for row_number, row in enumerate(reader, start=2):
                rows += 1
                if None in row:
                    errors.append(f"row {row_number}: too many CSV fields")

                frame_id = _parse_frame_id(row.get("frame_id"), row_number, errors)
                if frame_id is not None:
                    frame_ids.append(frame_id)
                _parse_finite_number(
                    row.get("timestamp"),
                    column="timestamp",
                    row_number=row_number,
                    errors=errors,
                )
                ttc_kind = _parse_ttc(row.get("predicted_ttc"), row_number, errors)
                finite_ttc += int(ttc_kind == "finite")
                infinite_ttc += int(ttc_kind == "infinite")
    except UnicodeError as exc:
        raise PredictionPreflightError(csv_path, [f"invalid UTF-8 CSV: {exc}"]) from exc
    except csv.Error as exc:
        raise PredictionPreflightError(csv_path, [f"malformed CSV: {exc}"]) from exc

    if rows != expected_frames:
        errors.append(f"expected {expected_frames} rows, found {rows}")

    counts: dict[int, int] = {}
    for frame_id in frame_ids:
        counts[frame_id] = counts.get(frame_id, 0) + 1
    duplicates = [frame_id for frame_id, count in counts.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate frame_ids: {_short_ids(duplicates)}")

    expected_ids = set(range(expected_frames))
    actual_ids = set(frame_ids)
    missing_ids = expected_ids - actual_ids
    out_of_range_ids = actual_ids - expected_ids
    if missing_ids:
        errors.append(f"missing frame_ids: {_short_ids(list(missing_ids))}")
    if out_of_range_ids:
        errors.append(f"out-of-range frame_ids: {_short_ids(list(out_of_range_ids))}")

    if errors:
        raise PredictionPreflightError(csv_path, errors)

    return PredictionPreflightReport(
        path=str(csv_path),
        trip_id=csv_path.stem,
        expected_frames=expected_frames,
        rows=rows,
        finite_ttc=finite_ttc,
        infinite_ttc=infinite_ttc,
    )


def discover_prediction_csvs(paths: Sequence[str | Path]) -> list[Path]:
    """Expand files/directories into a deterministic, de-duplicated CSV list."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidates = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for candidate in candidates:
            normalized = candidate.resolve()
            if normalized not in seen:
                seen.add(normalized)
                discovered.append(candidate)
    return discovered
