# Challenge 2 — Driver State, Phase 1

Per-frame classification into `alert | drowsy | yawning | distracted |
microsleep` from the cabin camera alone: MediaPipe face landmarks for the eyes
and mouth, a YOLO11 COCO detector for the phone, and a threshold rule over a
time window. No training, no GPU, no DMD corpus — this is the baseline Phase 2
has to beat.

## Result

Scored against the organiser's `driver.state` labels on the six Practice trips
(3,600 frames), using the official metric
`composite = 100 * (0.5 * accuracy + 0.5 * macro_f1)`:

| | composite |
|---|---|
| offline, thresholds fitted on all six trips | 99.4 |
| offline, **leave-one-trip-out** (refit per fold) | **97.5** |
| streaming (causal, trailing window) | **94.8** |

**Quote 97.5.** The 99.4 is thresholds scored on the trips they were chosen on;
97.5 refits with each trip withheld, so nothing a trip contributed can
influence its own score.

Per trip, offline, at the shipped defaults:

| trip | states present | composite |
|---|---|---|
| T01 | alert, distracted | 100.0 |
| T02 | drowsy | 100.0 |
| T03 | yawning | 100.0 |
| T04 | alert, distracted | 100.0 |
| T05 | microsleep | 100.0 |
| T06 | drowsy, distracted | 96.7 |

Pooled confusion (rows = truth). Twenty frames of T06 are the entire error:

```
                 alert  drowsy  yawning  distracted  microsleep
alert              600       0        0           0           0
drowsy               0     880        0          20           0
yawning              0       0      600           0           0
distracted           0       0        0         900           0
microsleep           0       0        0           0         600
```

`scoring.py` is a reimplementation of the organiser's
`team_kit/evaluation.py::compute_challenge2_metrics`; it reproduces the
official accuracy, macro-F1 and composite to the printed precision on all six
trips.

### The held-out gap closed from 14 points to 2

Before phone detection, leave-one-trip-out cost **14 points** (96.2 fitted →
82.1 held out) and two folds collapsed outright — 66.6 and 41.9. Each trip was
the sole evidence pinning one threshold, so withholding it let that threshold
drift and the fold fell apart.

With the detector the same procedure costs **1.9 points**, and the worst fold
is 94.3:

| fold | knobs it moves | held-out composite |
|---|---|---|
| without T01 | `window_frames`, `blink_closed`, `mar_talking` | 95.7 |
| without T02 | — | 100.0 |
| without T03 | — | 100.0 |
| without T04 | `phone_confidence` 0.03 → 0.20 | 95.0 |
| without T05 | — | 100.0 |
| without T06 | `mar_yawning` 0.35 → 0.28 | 94.3 |

Three folds still pick a different config — the sample is still six trips over
six subjects, and that has not changed. What changed is that the disagreements
stopped mattering: the T01 fold moves *three* knobs at once and still scores
95.7, where previously a single knob moving cost 40 points. Direct evidence is
what makes a threshold set transfer; the mouth proxy was being bent to cover a
class it could barely see, and that bending was the overfit.

## How it works

```
driver/frame_%06d.jpg
  -> features.py    MediaPipe FaceLandmarker -> EAR, MAR, blendshapes, head angles
     phone.py       YOLO11 COCO, every 5th frame -> `cell phone` box
  -> classifier.py  centred 9 s window -> PERCLOS + MAR p75 + phone rate -> state
  -> scoring.py     official composite against driver.state
```

Features are cached to CSV per trip, so threshold tuning never re-runs either
model.

### Four findings that shaped the design

**The label triple is redundant, which makes the problem smaller.** Every frame
carries `state`, `eye_state`, `head_pose` and `mouth_state`. Across all 3,600
frames the last three determine the first exactly — five triples, five states,
no collisions. So the classifier estimates three sub-signals instead of
learning a 5-way decision. The table is in `states.py`.

**Distraction is the phone, and it has to be seen directly.** Every
`distracted` frame is labelled `head_pose = "side"`, but measured yaw does not
separate the classes at all — inside T01 the *alert* half has the wider spread
(median −5.5°) than the *distracted* half (+1.5°), and T04 is the same shape.
These are DMD phone-call recordings where the driver keeps their eyes on the
road, so `side` describes the scenario, not anything visible.

What the camera *does* see is the handset. A COCO `cell phone` detection above
0.10 confidence covers 83 % / 90 % / 96 % of `distracted` frames against
0 % / 3 % / 20 % of the rest — roughly 20:1.

An earlier version used mouth activity (the driver talking) as a proxy, which
manages about 3:1. Retuned from scratch, replacing that proxy with the detector
is worth **+3.2 offline (96.2 → 99.4)** and **+6.6 streaming (88.2 → 94.8)**;
T01 goes 88.7 → 100.0 and T04 93.0 → 100.0. Mouth activity survives as a
fallback for when the detector is disabled or misses.

It also removed the pipeline's worst fragility. With the mouth proxy,
`blink_closed` was a knife edge — 96.2 at 0.25 against 86.9 at 0.20. With the
phone detector the same knob is flat from 0.25 to 0.35. Distraction was
previously being inferred from a signal that fought with the eye thresholds;
giving it its own evidence let both settle.

**Decide over a window, not a frame.** The evidence is rate-shaped (PERCLOS is
a fraction of closed frames, a yawn is a sustained wide mouth, phone use is a
fraction of frames with a box), and the organiser holds each state for
contiguous 300-frame blocks. A 61-frame window scores 94.5 against 99.4 at 181.

**Eye closure comes from the blendshape, not EAR.** `eyeBlink*` is a trained
output and holds up under head rotation; geometric EAR on the same frames
overlaps badly between drowsy and alert (median 0.286 vs 0.343).

### Rule order

Ordered by risk, first match wins:

1. face lost for most of the window → `distracted` (not `microsleep`: no
   landmarks means no eye evidence, and a tracking dropout should not raise a
   microsleep alarm)
2. `perclos >= 0.80` → `microsleep`
3. `mar_p75 >= 0.35` → `yawning` (before drowsy: yawning squints the eyes too,
   PERCLOS 0.36 across T03, and would otherwise read as drowsiness)
4. `phone_frac >= 0.50` → `distracted` (before drowsy: a driver on a handset
   who is also blinking heavily is, first, on a handset)
5. `perclos >= 0.06` → `drowsy`
6. `mar_p75 >= 0.20` → `distracted` (the fallback, when no phone was seen)
7. otherwise → `alert`

## Threshold sensitivity

Offline composite against each knob, others held at their default:

| knob | default | behaviour |
|---|---|---|
| `window_frames` | 181 | flat above 181; 151 → 99.1, 121 → 96.6, 61 → 94.5 |
| `blink_closed` | 0.25 | flat 0.25–0.35; 0.20 → 96.3 |
| `perclos_microsleep` | 0.80 | flat across 0.6–0.95 |
| `perclos_drowsy` | 0.06 | plateau 0.06–0.10; 0.02 → 88.9, 0.24 → 93.1 |
| `mar_yawning` | 0.35 | flat across 0.32–0.50 |
| `mar_talking` | 0.20 | flat 0.20–0.60 — the fallback barely matters now |
| `phone_confidence` | 0.05 | flat 0.03–0.05; 0.20 → 99.0 |
| `phone_distracted` | 0.50 | **the sensitive one**: 0.35 → 97.2, 0.65 → 97.6, 0.10 → 86.2 |

Where the objective is flat, the default is the centre of the plateau rather
than whichever endpoint the search visited first — those are tie-break
artifacts, not fitted values.

`phone_distracted` is now the knob carrying the most weight, and it is the one
to re-measure on more subjects. It is a far better place to be fragile than
`blink_closed` was: a rate over an object detection rather than a threshold on
one person's eyelid geometry. The leave-one-trip-out result above bears that
out — no fold moved `phone_distracted` at all.

## Running it

```bash
# install (mediapipe + onnxruntime on top of tripkit)
pip install -e ".[drive-state]"

# model weights, not committed
python scripts/download_models.py --mediapipe-face
pip install -e ".[export]"   # ultralytics, only needed for this next line
python scripts/download_models.py --phone-detector --phone-model yolo11s.pt

# 1. cache features (~120 s for 3,600 frames, both models)
python -m drive_state.phase_1 extract \
    --dataset path/to/Practice_Dataset --out runs/phase1/features

# 2. score the shipped thresholds
python -m drive_state.phase_1 evaluate \
    --dataset path/to/Practice_Dataset --out runs/phase1/report.json

# 3. refit, with the held-out number alongside
python -m drive_state.phase_1 tune \
    --dataset path/to/Practice_Dataset --out runs/phase1/tuning.json

# 4. submission CSV for one trip
python -m drive_state.phase_1 predict \
    --trip path/to/Practice_Dataset/T06-Sample \
    --out runs/phase1/predictions/T06-Sample.csv
```

## Replay demo

`drive_state.phase_1 demo` replays a trip with live inference and a HUD: the
decided state, a confidence bar per submitted class, face landmarks, the phone
box, and the rule that fired. When the trip still carries labels it prints the
ground-truth state under the prediction, green when they agree.

The five bars are `alert | drowsy | yawning | distracted | microsleep` — the
classes actually submitted, not the internal evidence channels. They are **not
probabilities**: this is a rule cascade, not a model with a softmax, so each
value is that class's evidence as a fraction of the threshold that would fire
it. Several can read 1.00 at once — on T03 `yawning`, `drowsy` and `distracted`
all saturate, because a yawn closes the eyes and opens the mouth — and the
winner is the risk ordering in `classify_window`, not the tallest bar. Only the
selected class is drawn in colour, for exactly that reason.

```bash
# 20 fps HUD (q or ESC quits)
python -m drive_state.phase_1 demo \
    --trip path/to/Practice_Dataset/T06-Sample --mode realtime

# 4x speed
python -m drive_state.phase_1 demo \
    --trip path/to/Practice_Dataset/T02-Sample --mode realtime --speed 4

# headless: no window, write an mp4
python -m drive_state.phase_1 demo \
    --trip path/to/Practice_Dataset/T03-Sample \
    --mode fast --no-show --save runs/phase1/demo/T03.mp4
```

Measured: **32 ms/frame, ~31 fps capacity** against the 20 fps budget, with the
phone detector at stride 5. `--phone-stride` trades capacity for detection
freshness; `--no-phone` disables it entirely and costs about 13 composite.

The displayed latency is a rolling mean on purpose. Per-frame cost alternates
between ~8 ms and ~130 ms depending on whether the strided detector ran, so the
instantaneous number is true of no typical frame.

### It uses a trailing window, so it scores lower than the offline pipeline

The demo reads no future frames. Re-tuned for that constraint it scores
**94.8** against the offline **99.4**:

| trip | states | streaming | offline |
|---|---|---|---|
| T01 | alert → distracted | 80.7 | 100.0 |
| T02 | drowsy | 99.7 | 100.0 |
| T03 | yawning | 98.1 | 100.0 |
| T04 | alert → distracted | 99.2 | 100.0 |
| T05 | microsleep | 100.0 | 100.0 |
| T06 | drowsy → distracted | 91.3 | 96.7 |

#### Start-up: hold output, then backfill

A unit that has just powered on has no window to average over. Rather than
publish a guess made over three frames, the demo reports `INITIALISING` until
the buffer fills, and the frames it held are backfilled with the first settled
decision — which is what a device that stays quiet until warm would submit for
them.

That is worth 3 composite, and it beats the obvious alternative:

| approach | composite | alarm latency |
|---|---|---|
| report the partial-window guess immediately | 94.8 | 0 s |
| **hold during buffering, then backfill** | **97.8** | 0 s once warm |
| delay every decision by half a window | 97.5 | 2.2 s |

The third row is worth understanding, because it is the only way to make
streaming reproduce the offline result exactly: hold each decision back until
its centred window is complete. It scores no better here and costs 2.2 s
between an event and the alarm, which for a microsleep warning is most of the
point.

#### What no causal system can recover

On T01 the label says `distracted` from frame 0, but the driver does not raise
the handset until **frame 40** — `phone_conf` is 0.000 until then, and the
cabin images confirm it. The organiser assigns the state to a whole 300-frame
block; the footage inside that block starts two seconds late.

So the offline 100.0 on T01 is not a better classifier. It is a centred window
reading frames 40-90 in order to decide frame 0 — using the future. A perfect
causal classifier, correct the instant evidence exists, tops out at **93.3** on
that trip. Streaming with backfill reaches 93.3, i.e. the ceiling.

The remaining streaming gap is transition lag: after a state changes, the
window still holds the old evidence for up to `window_frames`. Firing
distraction on any detection in the last 20 frames instead of waiting for half
the window scores 95.8 rather than 94.8 (T01 80.7 → 87.2) and is the obvious
next change, not yet applied.

### Frame source

If `tripkit` (from the team's `hack2026` repo) is importable, the demo replays
through `tripkit.TripReplayer` and inherits its drift-free pacing; otherwise it
falls back to this package's own loader with equivalent pacing, driver camera
only. Force either with `--tripkit` / `--no-tripkit`.

One snag on some dataset copies: `TripLoader._resolve_json_path` prefers
`<trip>.json` over `<trip>.json.gz` and tests it with `Path.exists()`, which is
also true for a *directory*. Unzipping the practice set can leave a
`T01-Sample.json/` directory holding a nested `T01-Sample.json`, and tripkit
then tries to open the directory and fails with a bare `PermissionError`.

The demo detects that layout up front. In auto mode it falls back to the
built-in loader, which reads the `.gz` directly; with `--tripkit` it raises
`TripkitLayoutError` and names the fix, because asking for tripkit explicitly
means you want to know whether tripkit works. Fixes, best first:

1. in `tripkit/loader.py`, `cand.exists()` → `cand.is_file()`
2. delete the stray `<trip>.json/` directories — the `.gz` holds byte-identical
   content (verified on all six practice trips)
3. leave it; auto mode already handles it

## What Phase 1 does not do

- **`head_pose` and `eye_state` are never predicted**, only `state`. The
  sub-signals are internal.
- **Thresholds are fitted on the evaluation set.** Phase 2's plan fixes this by
  tuning on DMD with 14 subjects and touching Practice once, at the end.
- **No temporal model.** The window is a box filter; there is no hysteresis and
  no Viterbi smoothing.
- **The phone detector is COCO-generic.** It finds a handset held to the ear or
  in the lap, which is what these trips contain. A phone mounted on the
  dashboard would read the same, and a driver distracted by anything other than
  a phone — a passenger, the radio, food — falls back to the mouth heuristic.
- **The offline scorer uses a centred window** and is not a live monitor; see
  the replay demo above for the causal variant.
