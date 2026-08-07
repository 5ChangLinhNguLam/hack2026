# Driver State Phase 2 replay

The complete DMD training and inference pipeline is under
`drive_state/phase_2/`. The deployable v13 checkpoint bundle is under
`models/driver_state_phase_2_v13/`.

The deployable graph is causal and processes each driver-camera frame once:

```text
TripLoader -> TripReplayer -> MobileNetV3 visual encoder
  -> clean ocular LSTM + robust microsleep-gate ocular LSTM
  -> temporal four-state LSTM + exact two-second microsleep gate
  -> one of alert/drowsy/microsleep/yawning/distracted
```

## Install

```bash
pip install -e ".[dms-phase2]"
```

The checked bundle lives at `models/driver_state_phase_2_v13/` and contains four
fingerprint-linked checkpoints plus Ultra-Light and MediaPipe assets. Do not
mix checkpoints from different bundle directories; loading fails closed when
their fingerprints or schemas disagree.

## Replay one trip

```bash
python -m drive_state.phase_2.cli.replay \
  --dataset /path/to/Hackathon_Dataset_Redacted \
  --trip T01d \
  --device cuda \
  --output-dir predictions/driver_state_phase_2_v13
```

Omit `--trip` to replay `T01d..T10d`. Add `--mode realtime` to pace from trip
timestamps; the default `fast` mode is for batch inference.

Each trip writes two files:

- `<trip>.csv`: exact Challenge-2 submission columns;
- `diagnostics/<trip>.csv`: five probabilities, PERCLOS, ocular reliability,
  latency, and the five CarSky-facing VSS values.

The VSS mapping is a deterministic demo contract. In particular,
`Vehicle.Driver.IsEyesOnRoad` is inferred from distraction probability and eye
visibility; it is not a separately calibrated gaze-zone classifier. Publishing
to CarSky remains a transport concern outside the model/replayer adapter.

## Python API

```python
from drive_state.phase_2 import DMSBundle, GeneralDMS

bundle = DMSBundle.at("models/driver_state_phase_2_v13")
with GeneralDMS(bundle, device="auto") as dms:
    for result in dms.replay_trip("data/T01d"):
        print(result.state, result.confidence, result.vss_signals().as_vss_dict())
```

Create one `GeneralDMS` per independent trip, or call `reset()` before feeding
a new stream, so recurrent state and PERCLOS history never leak across trips.

## Training and retained run output

The port includes data preparation, cache builders, losses, model definitions,
trainers, evaluation utilities, webcam/demo entry points, and the causal runtime.
Installed command names use the `dms2-` prefix; for example:

```bash
dms2-standardize-five-state --help
dms2-precompute-embeddings --help
dms2-train-mobilenet-lstm --help
dms2-demo --help
```

The v13 output retained at `drive_state/phase_2/run_artifacts_v13/` includes
the original six-sample practice report and the complete T4 verification over
all ten redacted trips (18,000 frames, no failed trip). These files are archived
provenance; use the replay command above to regenerate them on the current host.
