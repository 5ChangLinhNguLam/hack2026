"""Convert confidence-aware pseudo labels into the organizer C1 CSV schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


MODEL_VERSION = "c1_rgb_physics_ensemble_v1"


def build_submission(
    pseudo_csv: str | Path,
    output_csv: str | Path,
    *,
    minimum_confidence: float = 0.30,
) -> dict[str, object]:
    pseudo_csv, output_csv = Path(pseudo_csv), Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    finite = danger = suppressed = frames = 0
    with pseudo_csv.open("r", encoding="utf-8", newline="") as source, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        required = {"frame_id", "timestamp", "pseudo_ttc", "confidence"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Pseudo CSV thiếu cột: {sorted(required - set(reader.fieldnames or ())) }")
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "frame_id", "timestamp", "predicted_ttc", "pseudo_confidence",
                "pseudo_support_count", "model_version",
            ],
        )
        writer.writeheader()
        for row in reader:
            frames += 1
            confidence = float(row["confidence"])
            value = row["pseudo_ttc"].strip().lower()
            is_finite = value not in {"inf", "infinity", ""} and math.isfinite(float(value))
            if is_finite and confidence < minimum_confidence:
                value = "inf"
                suppressed += 1
                is_finite = False
            finite += int(is_finite)
            danger += int(is_finite and float(value) < 2.0)
            writer.writerow({
                "frame_id": row["frame_id"],
                "timestamp": row["timestamp"],
                "predicted_ttc": value,
                "pseudo_confidence": row["confidence"],
                "pseudo_support_count": row.get("support_count", ""),
                "model_version": MODEL_VERSION,
            })
    return {
        "frames": frames,
        "finite_predictions": finite,
        "danger_predictions": danger,
        "suppressed_low_confidence": suppressed,
        "minimum_confidence": minimum_confidence,
        "output": str(output_csv),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pseudo_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--minimum-confidence", type=float, default=0.30)
    args = parser.parse_args(argv)
    try:
        report = build_submission(
            args.pseudo_csv,
            args.output_csv,
            minimum_confidence=args.minimum_confidence,
        )
    except (OSError, ValueError) as exc:
        print(f"Lỗi tạo submission C1: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
