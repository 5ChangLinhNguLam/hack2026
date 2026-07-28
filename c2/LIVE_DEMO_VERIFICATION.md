# C2 live-demo verification record

> Verified locally on 28/07/2026, Windows, Python 3.12, CPU/XNNPACK, Intel Iris
> Xe laptop. This record proves the functional MVP; it does not claim medical,
> regulatory or cross-subject accuracy.

## Requirement evidence

| Requirement | Evidence | Result |
|---|---|---|
| Webcam input | Camera 0 opened at 640×480; 60-frame full-pipeline smoke | PASS, 18,4 FPS |
| Video input | Four DMD RGB face segments decoded through the same engine | PASS |
| Five C2 states | `c2/validate_live_demo.py` integration gate | PASS all 5 |
| Temporal stability | 10 C2 state/HUD tests + 300-second soak | PASS |
| Warning/evidence/FPS HUD | Render test + visually inspected recorded HUD | PASS |
| Cell-phone evidence | EfficientDet bounding box and two-hit temporal confirmation | PASS |
| CPU-only runtime | Local inference benchmark and full-pipeline soak | PASS |
| Reproducible documentation | `LIVE_DEMO_RESEARCH.md` and commands below | PASS |

## Integration gate

Command:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $py c2/validate_live_demo.py --dmd-root C:\DMD\dmd `
    --json-out c2/cache/live_validation.json
```

Observed result:

```text
safe-alert          PASS  alert=119
phone-distraction   PASS  distracted=133
yawn                PASS  yawning=110
long-eye-closure    PASS  microsleep=39, drowsy=104
OVERALL: PASS
states=[alert, distracted, drowsy, microsleep, yawning]
face coverage=100% for all four cases
case throughput=41,9–56,4 FPS
```

The gate uses annotated segments from unrestricted DMD subject `gA/1`. It is a
functional regression test, not a cross-subject evaluation.

## Five-minute soak

Command:

```powershell
& $py c2/demo.py --video <DMD-s2-face-video> --headless --max-seconds 300
```

Observed result:

```text
frames=8.929
source duration=300 seconds
wall throughput=34,2 FPS
face coverage=99,4%
Face Landmarker mean=10,6 ms
phone detector mean=25,8 ms
crash=none
```

MediaPipe emitted non-fatal Clearcut telemetry upload warnings to stderr during
the long run. Inference and final exit status were unaffected.

## Automated tests

```text
pytest: 73 passed in 7,97s
C2 temporal/HUD tests: 10 passed
git diff --check: clean (apart from Git's Windows LF→CRLF notice)
```

The C2 tests cover:

- neutral calibration;
- sustained yawn;
- microsleep safety priority;
- yawn vs eye-squeeze arbitration;
- two independent phone hits;
- relative head pose;
- suppression of off-road flicker while eye closure develops;
- repeated long closures for drowsy;
- reset/recalibration;
- HUD rendering with face and phone boxes.

## Stage preflight still recommended

These are operational checks, not missing implementation:

1. Run `python c2/demo.py` once before judging so both model files are cached.
2. Place the camera near eye level and start with two neutral seconds.
3. Test the actual room lighting, glasses and camera distance.
4. Record a two-to-five-minute backup video with `--record`.
5. Do not describe the ten-second fatigue trend as clinical PERCLOS.
