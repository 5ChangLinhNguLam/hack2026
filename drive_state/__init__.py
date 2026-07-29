"""Challenge 2 — driver state.

Organised by phase, because the two are different machines solving the same
task and will be compared against each other:

* `drive_state.phase_1` — MediaPipe face landmarks plus a YOLO11 phone
  detector, decided by thresholds over a time window. No training, no GPU.
  97.5 leave-one-trip-out on the six Practice trips (99.4 fitted), 94.8 running
  causally. See `docs/drive_state.md`.
* `drive_state.phase_2` — learned model trained on the native DMD labels. Not
  in this repo yet.

`drive_state.vendor` sits beside the phases rather than inside `phase_1`: the
HUD overlay, frame dataclasses and ONNX detector it holds are borrowed
infrastructure that phase 2 will render and score through as well.

The top-level namespace stays deliberately thin — import from the phase you
mean, so which one produced a number is never ambiguous:

    from drive_state.phase_1 import predict_trip
    python -m drive_state.phase_1.cli demo --trip data/T01-Sample
"""

__all__: list[str] = []
