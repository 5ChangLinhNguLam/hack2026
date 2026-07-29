"""Fitting the classifier thresholds, and reporting honestly about it.

The search is an exhaustive grid rather than anything cleverer. There are five
knobs, the objective is a step function of them (thresholds against a fixed set
of 3,600 windows), and a full sweep takes seconds once features are cached — so
a gradient-free grid is both the simplest and the most reliable choice.

`leave_one_trip_out` is the part that matters. With six trips over six
subjects, thresholds fitted on all six and scored on all six will look better
than the method really is. Refitting per fold with one trip withheld costs one
extra sweep per trip and yields a number that is not self-congratulatory.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from drive_state.phase_1.classifier import (
    TUNABLE_KNOBS,
    ClassifierConfig,
    classify_windows,
    compute_window_features,
)
from drive_state.phase_1.evaluate import TripData, evaluate
from drive_state.phase_1.scoring import TripScore, overall_composite, score_trip

#: Values swept per knob. Ranges are deliberately wider than the observed
#: per-trip statistics: a first pass with tighter ranges put four of the six
#: knobs on a grid endpoint, which means the search was clipped rather than
#: converged. If a re-tune lands on an endpoint again, widen that row before
#: trusting the result.
DEFAULT_GRID: dict[str, Sequence[float | int]] = {
    "window_frames": (61, 91, 121, 151, 181),
    "blink_closed": (0.25, 0.35, 0.45),
    "perclos_microsleep": (0.70, 0.80, 0.90),
    "perclos_drowsy": (0.02, 0.06, 0.10, 0.16, 0.24),
    "mar_yawning": (0.28, 0.35, 0.42),
    # 0.60 is effectively "off": no window reaches it, so the search can drop
    # the mouth fallback entirely when the phone detector covers the class.
    "mar_talking": (0.11, 0.14, 0.20, 0.30, 0.60),
    "phone_confidence": (0.03, 0.05, 0.10, 0.20),
    "phone_distracted": (0.10, 0.20, 0.35, 0.50, 0.65),
}

assert set(DEFAULT_GRID) <= set(TUNABLE_KNOBS), "grid sweeps a field the config does not declare"


@dataclass(slots=True)
class TuningResult:
    config: ClassifierConfig
    composite: float
    per_trip: list[TripScore]
    n_candidates: int


#: Knobs that change the window features themselves. Everything else only
#: moves a comparison, so windows are computed once per combination of these.
#: `phone_confidence` belongs here because it decides which frames count toward
#: `phone_frac`, not how that fraction is compared.
WINDOW_KNOBS = ("window_frames", "blink_closed", "phone_confidence")


def iter_candidates(
    base: ClassifierConfig, grid: dict[str, Sequence[float | int]]
) -> list[ClassifierConfig]:
    names = list(grid)
    return [
        # `replace` cannot be typed against a runtime-built kwargs dict, and the
        # grid keys are validated against TUNABLE_KNOBS at module import.
        replace(base, **cast(Any, dict(zip(names, values, strict=True))))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def grid_search(
    data: Sequence[TripData],
    *,
    base: ClassifierConfig | None = None,
    grid: dict[str, Sequence[float | int]] | None = None,
) -> TuningResult:
    """Best threshold set on `data` by mean per-trip composite.

    Ties are broken by keeping the first candidate seen, which — given the grid
    is ordered from smaller windows and lower thresholds upward — prefers the
    less aggressive setting.
    """
    base = base or ClassifierConfig()
    grid = grid or DEFAULT_GRID
    window_grid = {k: v for k, v in grid.items() if k in WINDOW_KNOBS}
    decision_grid = {k: v for k, v in grid.items() if k not in WINDOW_KNOBS}

    n_candidates = len(iter_candidates(base, grid))
    best: TuningResult | None = None

    for window_config in iter_candidates(base, window_grid):
        windows = {
            item.trip_id: compute_window_features(item.features, window_config) for item in data
        }
        truths = {item.trip_id: item.trip.states() for item in data}
        for config in iter_candidates(window_config, decision_grid):
            scores = [
                score_trip(tid, classify_windows(windows[tid], config), truths[tid])
                for tid in windows
            ]
            composite = overall_composite(scores)
            if best is None or composite > best.composite:
                best = TuningResult(config, composite, scores, n_candidates)

    assert best is not None  # grid is never empty
    return best


@dataclass(slots=True)
class LotoResult:
    """Per-fold held-out scores plus the config each fold chose."""

    held_out: list[TripScore]
    fold_configs: dict[str, ClassifierConfig]
    overall: float

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_composite": round(self.overall, 1),
            "per_trip": [s.to_dict() for s in self.held_out],
        }


def leave_one_trip_out(
    data: Sequence[TripData],
    *,
    base: ClassifierConfig | None = None,
    grid: dict[str, Sequence[float | int]] | None = None,
) -> LotoResult:
    """Refit on the other five trips, score the sixth. Repeat for all six."""
    held_out: list[TripScore] = []
    fold_configs: dict[str, ClassifierConfig] = {}
    for index, item in enumerate(data):
        train = [d for j, d in enumerate(data) if j != index]
        fold = grid_search(train, base=base, grid=grid)
        fold_configs[item.trip_id] = fold.config
        held_out.extend(evaluate([item], fold.config))
    return LotoResult(
        held_out=held_out,
        fold_configs=fold_configs,
        overall=overall_composite(held_out),
    )
