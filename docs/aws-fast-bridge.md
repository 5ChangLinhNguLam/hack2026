# SafeLoop AWS fast bridge

This is a single-server demonstration path, not the production AEB data plane.
It runs the existing C1, C2 and C3 models for every accepted input tick and
publishes warning-only `safeloop.decision.v1` snapshots.

## Provenance

- `LIVE_CAMERA + THIRD_PARTY + LIVE_MODEL` is accepted only through the WSS
  input contract. A recorded file must never use these labels.
- `RECORDED_STREAM + RECORDED_DATA + LIVE_MODEL` is generated only from a
  checksum-verified output of `tools/prepare_carsky_demo.py` selected by the
  server allowlist.
- `TRANSPORT_PROBE` is hidden from the demo list and is always `NO_MODEL`.
- `CARSKY_BROKER` is disabled until a real CarSky Cloud Bridge exists.
- There is no model-output replay path in this service.

Decision v1 starts its TTL when Android receives a packet. It does not carry a
trusted capture-age/clock-health contract. This is therefore a fast HMI demo,
not a production freshness contract and not an autonomous-braking interface.

## Local secrets and Compose

Create `.secrets/ingest_token`, `.secrets/android_token`, and
`.secrets/cloudflared_token` as three independent values of at least 24
characters. The directory is ignored by Git. Do not put any value in `.env`.

```bash
docker compose -f docker-compose.aws.yml config --quiet
docker compose -f docker-compose.aws.yml build app
docker compose -f docker-compose.aws.yml up -d app
docker compose -f docker-compose.aws.yml ps
docker compose -f docker-compose.aws.yml logs --no-color app
docker compose -f docker-compose.aws.yml down
```

The `app` service exposes port 8000 only inside the Compose network and runs
one Uvicorn worker as UID 10001 with a read-only root filesystem. The demo root
is a read-only mount. Start `cloudflared` only after the named tunnel and DNS
route are configured:

```bash
docker compose -f docker-compose.aws.yml up -d
```

## Cloudflare configuration still required

Use a Named Tunnel whose origin is `http://app:8000` and public hostname is
`safeloop.sonnet.io.vn`. Enable WebSockets, bypass cache for `/v1/*`, and allow
outbound TCP/UDP 7844 from this server. Configure Cloudflare Access with one
explicit admin email and deny-by-default for `/admin/*` and `/v1/admin/*`.
Do not place interactive Access in front of `/v1/ingest` or
`/v1/decisions/stream`; those endpoints use independent application bearer
tokens. Enable HSTS in the app only after the TLS route is verified.

Public routes are:

- `https://safeloop.sonnet.io.vn/admin/`
- `https://safeloop.sonnet.io.vn/v1/admin/*`
- `wss://safeloop.sonnet.io.vn/v1/ingest`
- `https://safeloop.sonnet.io.vn/v1/decisions/stream`
- `https://safeloop.sonnet.io.vn/healthz`
- `https://safeloop.sonnet.io.vn/readyz`

## Third-party sender

Tokens are loaded from `SAFELOOP_INGEST_TOKEN` or
`SAFELOOP_INGEST_TOKEN_FILE`, never from a URL:

```bash
SAFELOOP_INGEST_TOKEN_FILE=.secrets/ingest_token \
  .venv-dms2/bin/python tools/aws_live_sender.py live \
  --url wss://safeloop.sonnet.io.vn/v1/ingest \
  --road 'rtsp://ROAD_SOURCE' --cabin 'rtsp://CABIN_SOURCE' \
  --speed-limit-kmh 60 --weather clear
```

Use `recorded-check` to verify a mounted bundle. Start recorded inference from
the Admin console; the WSS endpoint intentionally refuses recorded provenance.

## Android build and launch

Build with JDK 17 and Android SDK 35:

```bash
cd android/safeloop-aaos
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
ANDROID_HOME="$PWD/../../android-local/sdk" \
  ./gradlew testDebugUnitTest assembleDebug --no-daemon
```

Install and launch from a local shell. The token is a separate Intent extra and
must not be saved in the repository, URL, preferences, or a shared script:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop com.fptautomotive.safeloop
adb shell am start -n com.fptautomotive.safeloop/.MainActivity \
  --es cloud_stream_url \
  'https://safeloop.sonnet.io.vn/v1/decisions/stream' \
  --es cloud_stream_token "$SAFELOOP_ANDROID_TOKEN"
```

With no `cloud_stream_url`, the app retains the room-local UDP 48100 receiver.
The two transports never run together.
