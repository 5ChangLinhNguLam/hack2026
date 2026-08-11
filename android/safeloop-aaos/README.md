# SafeLoop Native AAOS HMI

This is the native Android Automotive display for SafeLoop. It is a framework-only
Java `Activity` with a custom `View`/`Canvas`; it contains no WebView, HTML,
JavaScript simulator, Compose, AndroidX, model, dataset, or A8 credential.

The app is intentionally read-only. It displays model decisions and emits local
warning tones, but it has no VHAL write permission and no vehicle-control API.
For legacy-v1 compatibility the parser accepts `EMERGENCY_BRAKE_REQUEST` and a
positive `brake_request_pct`, but the immutable snapshot normalizes either to
`VISUAL_AUDIO_HAPTIC_WARNING` with zero brake before UI, alert, or audio code
can observe it. The wire contract still requires `actuation_authorized=false`.

## Data path

```text
camera + ego telemetry
        |
        v
unified C1/C2/C3 replay or live runtime
        |
        v
DecisionEnvelopeBuilder -> UdpDecisionPublisher
        |                   one full JSON snapshot/datagram
        |                   UDP 48100, room-local Ethernet
        v
AAOS UdpDecisionReceiver -> strict parser -> freshness/order guard
        |                                      |
        |                              NO_DATA/DEGRADED/STALE
        v
immutable DashboardState -> Canvas dashboard + rate-limited local tones
```

The authoritative producer contract is
[`safeloop/carsky_decision.py`](../../safeloop/carsky_decision.py), and the
Python reference guard/publisher is
[`safeloop/carsky_hmi.py`](../../safeloop/carsky_hmi.py). The Android parser
accepts exactly `safeloop.decision.v1`; it does not invent missing values.

The current CarSky blueprint assigns `10.99.0.14` to the AAOS Ethernet pin.
Verify that address after each topology/deployment change with `ip addr`; the
producer must publish to the actual AAOS address, not assume that it is stable.
Keep the transport on the isolated room network. UDP v1 is not authenticated or
encrypted and must not be exposed to an untrusted network.

## Receiver safety rules

- Maximum datagram size is 1,472 UTF-8 bytes (1,500-byte Ethernet MTU minus
  IPv4 and UDP headers). The receiver allocates one extra byte only to detect
  and reject an oversized packet; it never parses a truncated packet.
  Malformed, non-finite, extra-key, or incomplete JSON is also rejected.
- The first packet observed for an unknown `session_id` establishes its
  baseline at any non-negative sequence, so a late-starting app can join a
  stream. Later sequences must strictly increase; gaps are counted and
  duplicates/out-of-order packets are rejected.
- Replay protection is a bounded LRU window over the 64 most recently retired
  sessions. This intentionally bounds memory: identifiers older than that
  finite window may be accepted as new. A recent retired identifier (or the
  active identifier) can restart at sequence 0 only after the current local TTL
  has expired. A fresh-stream replay remains rejected. Generated replay addons
  also use a best-effort per-process boot token and wait longer than one TTL at
  startup, so a Script Node restart cannot get stuck cycling old identifiers.
- Producer timestamps and `expires_at_ms` are validated for internal contract
  consistency but are not compared with the AAOS wall clock. On receipt,
  `ttl_ms` starts on Android `elapsedRealtime`; container and AAOS clocks do
  not need to be synchronized and wall-clock changes cannot resurrect data.
- At exact expiry the UI becomes `STALE`, hides decision values, and suppresses
  audio. `NO_DATA` and an invalid contextual decision are also silent. If only
  a non-safety metric such as Drive Quality is degraded, the health badge stays
  `DEGRADED` but a still-valid contextual collision decision can warn.
- The source badge always says `LIVE`, `REPLAY`, or `SIMULATION`.
- C3 Safe Estimate and product Drive Quality remain separate. A null/unavailable
  score is shown as `—`, never as zero.

Default product TTL is 200 ms and the expected publisher cadence is 20 Hz. The
app redraws at 4 Hz; each packet also triggers an immediate redraw.

`health.model_versions` carries compact 12-hex SHA-256 prefixes on the wire so
the complete snapshot stays below the non-fragmenting UDP limit. The edge run
report keeps the corresponding full SHA-256, digest method, and source path for
audit; the short values are identifiers, not a replacement for that report.

## Build

Required toolchain:

- JDK 17
- Android SDK Platform 35 and Build Tools
- Gradle 8.9 (Android Gradle Plugin 8.7.3)

The repository includes a Gradle 8.9 wrapper. From this directory, with JDK 17
and an accepted Android SDK license:

```bash
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The output is `app/build/outputs/apk/debug/app-debug.apk`. No Gradle wrapper
generation step or globally installed Gradle is required.

This build path was verified on JDK 17 and Android SDK 35: unit tests passed,
lint completed with zero errors and zero warnings, and `assembleDebug` produced
the installable debug APK. Build outputs remain local generated artifacts.

The dependency-free core smoke test can be run with the JDK compiler module
without an Android SDK or Gradle installation:

```bash
check_dir=$(mktemp -d)
java --module jdk.compiler/com.sun.tools.javac.Main -d "$check_dir" \
  app/src/main/java/com/fptautomotive/safeloop/DecisionSnapshot.java \
  app/src/main/java/com/fptautomotive/safeloop/DashboardState.java \
  app/src/main/java/com/fptautomotive/safeloop/DecisionStateMachine.java \
  app/src/main/java/com/fptautomotive/safeloop/AlertPolicy.java \
  app/src/test/java/com/fptautomotive/safeloop/DecisionFixtures.java \
  app/src/test/java/com/fptautomotive/safeloop/CoreSelfTest.java
java -cp "$check_dir" com.fptautomotive.safeloop.CoreSelfTest
```

## Install and launch

With a normal ADB connection:

```bash
adb install -r -t app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop --user current com.fptautomotive.safeloop
adb shell am start --user current -W \
  -n com.fptautomotive.safeloop/.MainActivity \
  --ei udp_port 48100
```

The CarSky deployment's Conduit service is currently not configured, so its REST
`adb-exec`/VM shell routes return HTTP 502. The working fallback is the CarSky
ADB Widget interactive shell. Generate paste-safe commands locally:

```bash
python3 safeloop_aaos_tools/apk_payload.py \
  app/build/outputs/apk/debug/app-debug.apk \
  > /tmp/safeloop-aaos-install.sh
```

Review `/tmp/safeloop-aaos-install.sh`, then paste it into the ADB Widget shell.
It transfers via base64, verifies byte count and SHA-256 on the VM, installs,
launches, and removes only its two `/data/local/tmp/safeloop-aaos.*` files.
No `run-as` is needed (the WebView Shell package was not debuggable).

Run one real model replay from a process that is attached to the same CarSky
Ethernet network as AAOS:

```bash
python3 -m safeloop.carsky_edge \
  --dataset data \
  --trip T01-Sample \
  --device cpu \
  --mode realtime \
  --udp-host 10.99.0.14 \
  --udp-port 48100 \
  --no-kuksa
```

This command uses the unified C1/C2/C3 inference path and sends its actual
decision envelopes; it is not the old HTML/JavaScript mock. Remove `--no-kuksa`
only when the producer also has a working route to the configured KUKSA broker.
Running the command on a laptop outside the CarSky room will not make
`10.99.0.14` reachable.

Useful non-mutating probes in that shell:

```sh
getprop sys.boot_completed
wm size
ip addr
pm path com.fptautomotive.safeloop
ss -lun | grep 48100
logcat -d -t 300 | grep -E 'SafeLoop|AndroidRuntime'
```

## Acceptance gates

1. `testDebugUnitTest`, `lintDebug`, and `assembleDebug` pass on JDK 17/SDK 35.
2. APK manifest contains only `INTERNET`; no A8 secret, model, dataset, HTML,
   cleartext HTTP client, VHAL write permission, or `distractionOptimized=true`.
3. App cold-starts as `NO_DATA`; valid 20 Hz packets become `LIVE` without a
   blank/white WebView screen.
4. Fault injection covers oversize/malformed JSON, inconsistent expiry fields,
   missing sequence, gaps, duplicate/out-of-order, session restart, retired
   session replay, degraded component flags, stream stop, and restart.
5. At exact TTL expiry, all decision values disappear and warning audio stops.
6. `REPLAY`/`SIMULATION` is unmistakable on screen. C3 and Drive Quality null
   values remain unavailable rather than becoming `0`.
7. Run a 30-minute 20 Hz soak and verify bounded memory, no crash/ANR, stable
   packet-age behavior, and expected gap/reject counters.
8. Test AAOS UX restrictions: while driving, this ordinary sideloaded Activity
   must be blocked/paused and audio must stop. A production HMI intended to stay
   visible while moving requires OEM/system integration and safety approval; do
   not mark this custom Activity distraction-optimized merely to bypass AAOS.

This project is a CarSky/hackathon sideload target, not a Google Play-certified
car app. The manifest deliberately omits `distractionOptimized` metadata.
See Android's official
[AAOS platform/UX restriction guidance](https://developer.android.com/training/cars/platforms/automotive-os)
and [parked-app distraction requirements](https://developer.android.com/training/cars/parked/automotive-os)
before adapting it for a production vehicle.
