"""Modules lifted from `Inferensys/ai-driver-safety`, unmodified except for imports.

`drive_state` reuses that project's frame/event dataclasses, risk scorer and
ONNX object detector. They are copied here rather than imported so this repo
installs on its own — a hackathon kit that needs a second repo cloned and on
`PYTHONPATH` is a kit that does not run on someone else's machine.

Do not edit these to fix a `drive_state` problem; the point of keeping them
byte-comparable to upstream is that a diff against `ai-driver-safety` stays
readable. Changes belong in the modules one level up.

One deliberate departure from upstream: `objects.py` drops
`create_object_detector`, which took a `DriverSafetyConfig` and pulled that
whole configuration tree in for a factory nothing here calls.
`OnnxObjectDetector` is constructed directly by `drive_state.phase_1.phone`.

Upstream's `overlay.py` is *not* vendored. Its HUD reports the realtime
pipeline's evidence signals, where this demo has to show the five states
Challenge 2 submits; `drive_state.phase_1.hud` draws that instead.

`scoring.py` upstream became `risk.py` here, because `drive_state.scoring` is
already the official Challenge 2 metric and two files by that name in one
import graph is a trap.
"""
