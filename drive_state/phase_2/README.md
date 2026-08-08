# Driver State Phase 2

This directory is the complete phase-2 DMD implementation copied from the
current `labels_20fps` pipeline. It includes:

- dataset preparation and precomputation under `data/` and `cli/`;
- model/loss/trainer code under `models/`, `losses/`, and `training/`;
- causal five-state runtime in `runtime_mobilenet_lstm.py`;
- native `tripkit.TripReplayer` integration in `replay.py` and `cli/replay.py`;
- retained v13 practice and 10-trip replay artifacts under `run_artifacts_v13/`.

The runtime bundle is `models/driver_state_phase_2_v13/` at repository root.
See `docs/DRIVER_STATE_PHASE_2_REPLAY.md` for installation and replay commands.
