from __future__ import annotations

import csv

from safeloop.c1.submission import build_submission


def test_submission_suppresses_low_confidence_pseudo_ttc(tmp_path) -> None:
    source = tmp_path / "pseudo.csv"
    source.write_text(
        "frame_id,timestamp,pseudo_ttc,confidence,support_count\n"
        "0,0.0,1.2,0.2,1\n"
        "1,0.05,1.0,0.8,3\n"
        "2,0.1,inf,0.8,0\n",
        encoding="utf-8",
    )
    output = tmp_path / "submission.csv"

    report = build_submission(source, output, minimum_confidence=0.3)
    rows = list(csv.DictReader(output.open()))

    assert rows[0]["predicted_ttc"] == "inf"
    assert rows[1]["predicted_ttc"] == "1.0"
    assert report["suppressed_low_confidence"] == 1
