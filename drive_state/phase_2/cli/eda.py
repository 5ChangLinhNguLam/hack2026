from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..data.manifest import load_sessions
from ..eda import analyze_dmd_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the local 20 FPS DMD labels")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("labels_20fps/manifest_20fps.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(analyze_dmd_labels(load_sessions(args.manifest)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
