from __future__ import annotations

import csv
from pathlib import Path

import pytest

from safeloop.c1.prediction_preflight import (
    PredictionPreflightError,
    discover_prediction_csvs,
    preflight_prediction_csv,
)
from safeloop.c1.preflight_predictions import main


def _write_predictions(
    path: Path,
    frame_ids: list[int],
    *,
    headers: tuple[str, ...] = ("frame_id", "timestamp", "predicted_ttc"),
    ttc_values: list[str] | None = None,
) -> None:
    ttc_values = ttc_values or ["inf"] * len(frame_ids)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for index, (frame_id, ttc) in enumerate(zip(frame_ids, ttc_values)):
            row = {
                "frame_id": frame_id,
                "timestamp": index * 0.05,
                "predicted_ttc": ttc,
            }
            writer.writerow({key: row[key] for key in headers})


def test_preflight_accepts_complete_file_with_finite_and_inf_ttc(tmp_path: Path) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 1, 2], ttc_values=["inf", "+Infinity", "1.25"])

    report = preflight_prediction_csv(path, expected_frames=3)

    assert report.trip_id == "T01-Sample"
    assert report.rows == 3
    assert report.finite_ttc == 1
    assert report.infinite_ttc == 2


def test_preflight_defaults_to_600_frames(tmp_path: Path) -> None:
    path = tmp_path / "T02-Sample.csv"
    _write_predictions(path, list(range(600)))

    report = preflight_prediction_csv(path)

    assert report.expected_frames == 600
    assert report.rows == 600


@pytest.mark.parametrize("ttc", ["", "not-a-number", "nan", "-inf"])
def test_preflight_rejects_invalid_ttc(tmp_path: Path, ttc: str) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 1], ttc_values=["inf", ttc])

    with pytest.raises(PredictionPreflightError, match="predicted_ttc"):
        preflight_prediction_csv(path, expected_frames=2)


def test_preflight_rejects_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 1], headers=("frame_id", "predicted_ttc"))

    with pytest.raises(PredictionPreflightError, match="timestamp"):
        preflight_prediction_csv(path, expected_frames=2)


def test_preflight_reports_duplicate_missing_and_out_of_range_ids(tmp_path: Path) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 0, 3])

    with pytest.raises(PredictionPreflightError) as caught:
        preflight_prediction_csv(path, expected_frames=3)

    message = str(caught.value)
    assert "duplicate frame_ids: 0" in message
    assert "missing frame_ids: 1, 2" in message
    assert "out-of-range frame_ids: 3" in message


def test_preflight_rejects_partial_file_even_when_ids_are_unique(tmp_path: Path) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 1])

    with pytest.raises(PredictionPreflightError, match="expected 3 rows, found 2"):
        preflight_prediction_csv(path, expected_frames=3)


def test_preflight_rejects_non_integer_frame_id_and_nonfinite_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "T01-Sample.csv"
    path.write_text(
        "frame_id,timestamp,predicted_ttc\n"
        "0,0.0,inf\n"
        "1.0,nan,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(PredictionPreflightError) as caught:
        preflight_prediction_csv(path, expected_frames=2)

    assert "frame_id is not an integer" in str(caught.value)
    assert "timestamp must be finite" in str(caught.value)


def test_discovery_is_sorted_and_deduplicated(tmp_path: Path) -> None:
    first = tmp_path / "T01-Sample.csv"
    second = tmp_path / "T02-Sample.csv"
    _write_predictions(second, [0])
    _write_predictions(first, [0])

    paths = discover_prediction_csvs([tmp_path, first])

    assert paths == [first, second]


def test_cli_returns_success_for_complete_prediction(tmp_path: Path, capsys) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 1, 2])

    exit_code = main([str(path), "--expected-frames", "3"])

    assert exit_code == 0
    assert '"ok": true' in capsys.readouterr().out


def test_cli_returns_failure_for_incomplete_prediction(tmp_path: Path, capsys) -> None:
    path = tmp_path / "T01-Sample.csv"
    _write_predictions(path, [0, 2])

    exit_code = main([str(path), "--expected-frames", "3"])

    assert exit_code == 2
    assert "preflight failed" in capsys.readouterr().err
