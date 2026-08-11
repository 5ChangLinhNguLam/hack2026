"""FastAPI vertical slice for the single-GPU SafeLoop demonstration.

The process intentionally owns exactly one model pipeline and one inference
worker.  It is designed to sit only behind a named Cloudflare Tunnel; no app
port is published by the supplied Compose file.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from types import MappingProxyType
from typing import Any, AsyncIterator, Mapping
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from .aws_live_contract import (
    InputContractError,
    MAX_INPUT_MESSAGE_BYTES,
    PreparedDemoBundle,
    parse_input_message,
)
from .aws_live_runtime import (
    FastBridgeController,
    FastBridgeError,
    ModelPipeline,
    RealModelPipeline,
)


LOGGER = logging.getLogger("safeloop.aws")
PUBLIC_HOST = "safeloop.sonnet.io.vn"
ADMIN_EMAIL_HEADER = "cf-access-authenticated-user-email"
ADMIN_ACTION_HEADER = "x-safeloop-admin-action"
DEMO_ID = re.compile(r"T(?:0[1-9]|10)-Sample", re.ASCII)
SSE_HEARTBEAT_SECONDS = 5.0


def _secret(name: str, *, required: bool = True) -> str:
    file_value = os.getenv(f"{name}_FILE", "").strip()
    direct = os.getenv(name, "")
    if file_value and direct:
        raise RuntimeError(f"set only one of {name} or {name}_FILE")
    if file_value:
        path = Path(file_value)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{name}_FILE must be a regular file")
        direct = path.read_text(encoding="utf-8").strip()
    if required and len(direct) < 24:
        raise RuntimeError(f"{name} must contain at least 24 characters")
    return direct


@dataclass(frozen=True)
class ServiceSettings:
    public_base_url: str
    admin_origin: str
    demo_root: Path
    ingest_token: str
    android_token: str
    c1_checkpoint: Path
    c2_bundle: Path
    device: str = "cuda"
    allow_local_admin: bool = False
    hsts: bool = False

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        public = os.getenv(
            "SAFELOOP_PUBLIC_BASE_URL", "https://safeloop.sonnet.io.vn"
        ).rstrip("/")
        admin_origin = os.getenv("SAFELOOP_ADMIN_ORIGIN", public).rstrip("/")
        for name, value in (("public base URL", public), ("admin origin", admin_origin)):
            parsed = urlsplit(value)
            if parsed.scheme != "https" or not parsed.hostname or parsed.path:
                raise RuntimeError(f"{name} must be an HTTPS origin")
        ingest = _secret("SAFELOOP_INGEST_TOKEN")
        android = _secret("SAFELOOP_ANDROID_TOKEN")
        if hmac.compare_digest(ingest, android):
            raise RuntimeError("ingest and Android bearer tokens must be independent")
        return cls(
            public_base_url=public,
            admin_origin=admin_origin,
            demo_root=Path(os.getenv("SAFELOOP_DEMO_ROOT", "/demo-data")).resolve(),
            ingest_token=ingest,
            android_token=android,
            c1_checkpoint=Path(
                os.getenv("SAFELOOP_C1_CHECKPOINT", "/app/C1/student_ttc.pth")
            ),
            c2_bundle=Path(
                os.getenv(
                    "SAFELOOP_C2_BUNDLE", "/app/models/driver_state_phase_2_v13"
                )
            ),
            device=os.getenv("SAFELOOP_DEVICE", "cuda"),
            allow_local_admin=os.getenv("SAFELOOP_ALLOW_LOCAL_ADMIN") == "1",
            hsts=os.getenv("SAFELOOP_HSTS") == "1",
        )


class LatestOnlyHub:
    """Fan out without history: every subscriber owns one pending item."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[bytes]] = set()
        self._lock = threading.Lock()
        self.last_write_ms: int | None = None

    def subscribe(self) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        with self._lock:
            self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes]) -> None:
        with self._lock:
            self._queues.discard(queue)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._queues)

    def publish(self, payload: bytes) -> None:
        with self._lock:
            queues = tuple(self._queues)
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


class ControlRateLimiter:
    def __init__(self, *, limit: int = 12, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> None:
        now = time.monotonic()
        with self._lock:
            calls = self._calls[identity]
            while calls and calls[0] <= now - self.window_seconds:
                calls.popleft()
            if len(calls) >= self.limit:
                raise HTTPException(429, "admin control rate limit exceeded")
            calls.append(now)


def load_demo_allowlist(root: Path) -> Mapping[str, PreparedDemoBundle]:
    if not root.is_dir() or root.is_symlink():
        return MappingProxyType({})
    bundles: dict[str, PreparedDemoBundle] = {}
    for number in range(1, 11):
        demo_id = f"T{number:02d}-Sample"
        candidate = root / demo_id
        if candidate.is_dir() and not candidate.is_symlink():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            bundles[demo_id] = PreparedDemoBundle.load(resolved)
    return MappingProxyType(bundles)


def _bearer(headers: Mapping[str, str], expected: str) -> bool:
    value = headers.get("authorization", "")
    if not value.startswith("Bearer "):
        return False
    supplied = value[7:]
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def _strict_body(body: bytes) -> Mapping[str, Any]:
    if not body or len(body) > 4_096:
        raise HTTPException(400, "invalid JSON body size")
    try:
        decoded = json.loads(
            body.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, "body must be strict JSON") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(400, "body must be a JSON object")
    return decoded


def _audit(identity: str, action: str, demo_id: str | None, controller: FastBridgeController, result: str) -> None:
    status = controller.status()
    LOGGER.info(
        "admin_audit %s",
        json.dumps(
            {
                "timestamp_ms": time.time_ns() // 1_000_000,
                "admin_identity": identity,
                "action": action,
                "demo_id": demo_id,
                "session": status.get("session_id"),
                "result": result,
            },
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def create_app(
    *,
    settings: ServiceSettings | None = None,
    models: ModelPipeline | None = None,
    bundles: Mapping[str, PreparedDemoBundle] | None = None,
) -> FastAPI:
    supplied_settings = settings
    supplied_models = models
    supplied_bundles = bundles

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = supplied_settings or ServiceSettings.from_env()
        active_bundles = supplied_bundles or load_demo_allowlist(
            active_settings.demo_root
        )
        active_models = supplied_models or RealModelPipeline(
            c1_checkpoint=active_settings.c1_checkpoint,
            c2_bundle=active_settings.c2_bundle,
            device=active_settings.device,
        )
        decision_hub = LatestOnlyHub()
        event_hub = LatestOnlyHub()
        loop = asyncio.get_running_loop()
        controller_box: list[FastBridgeController] = []

        def publish_decision(publication: Any) -> None:
            loop.call_soon_threadsafe(decision_hub.publish, publication.payload)

        def publish_event(event: Mapping[str, Any]) -> None:
            payload = json.dumps(
                dict(event), separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            loop.call_soon_threadsafe(event_hub.publish, payload)

        controller = FastBridgeController(
            active_models,
            on_decision=publish_decision,
            on_event=publish_event,
        )
        controller_box.append(controller)
        app.state.settings = active_settings
        app.state.controller = controller
        app.state.bundles = active_bundles
        app.state.decision_hub = decision_hub
        app.state.event_hub = event_hub
        app.state.rate_limiter = ControlRateLimiter()
        app.state.idempotency = {}
        app.state.started_ms = time.time_ns() // 1_000_000
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            controller.close()

    app = FastAPI(
        title="SafeLoop AWS Fast Bridge",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if getattr(getattr(request.app.state, "settings", None), "hsts", False):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response

    def admin_identity(request: Request) -> str:
        active_settings: ServiceSettings = request.app.state.settings
        if active_settings.allow_local_admin and request.client and request.client.host in {
            "127.0.0.1",
            "::1",
            "testclient",
        }:
            return request.headers.get(ADMIN_EMAIL_HEADER, "local-admin")[:254]
        if request.headers.get("x-forwarded-proto", "").lower() != "https":
            raise HTTPException(403, "admin requires the trusted tunnel")
        if request.url.hostname != urlsplit(active_settings.public_base_url).hostname:
            raise HTTPException(403, "unexpected admin host")
        identity = request.headers.get(ADMIN_EMAIL_HEADER, "").strip()
        if not identity or len(identity) > 254:
            raise HTTPException(403, "Cloudflare Access identity is required")
        return identity

    async def control_request(
        request: Request,
        identity: str = Depends(admin_identity),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> tuple[str, str, Mapping[str, Any]]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(415, "application/json is required")
        if request.headers.get(ADMIN_ACTION_HEADER) != "1":
            raise HTTPException(403, "admin action confirmation header is required")
        origin = request.headers.get("origin", "").rstrip("/")
        if origin != request.app.state.settings.admin_origin:
            raise HTTPException(403, "admin Origin is not allowed")
        if not idempotency_key or len(idempotency_key) > 128:
            raise HTTPException(400, "Idempotency-Key is required")
        request.app.state.rate_limiter.check(identity)
        body = await request.body()
        decoded = _strict_body(body)
        digest = hashlib.sha256(body).hexdigest()
        cache_key = f"{identity}:{idempotency_key}"
        return cache_key, digest, decoded

    @app.get("/healthz")
    async def healthz() -> Mapping[str, Any]:
        return {"status": "ok", "service": "safeloop-fast-bridge"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        ready = bool(getattr(request.app.state, "ready", False))
        return JSONResponse(
            {"status": "ready" if ready else "not_ready"},
            status_code=200 if ready else 503,
        )

    @app.websocket("/v1/ingest")
    async def ingest(websocket: WebSocket) -> None:
        active_settings: ServiceSettings = websocket.app.state.settings
        if not _bearer(websocket.headers, active_settings.ingest_token):
            await websocket.close(code=4401)
            return
        controller: FastBridgeController = websocket.app.state.controller
        try:
            token = controller.live_connect()
        except FastBridgeError:
            await websocket.close(code=4409)
            return
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                raw = message.get("bytes")
                if text is None and raw is None:
                    await websocket.close(code=1003)
                    break
                payload: str | bytes = text if text is not None else raw
                if len(payload.encode("utf-8") if isinstance(payload, str) else payload) > MAX_INPUT_MESSAGE_BYTES:
                    await websocket.close(code=1009)
                    break
                try:
                    tick = await asyncio.to_thread(parse_input_message, payload)
                    controller.ingest(tick)
                except InputContractError:
                    await websocket.close(code=1007)
                    break
                except FastBridgeError:
                    await websocket.close(code=1011)
                    break
        except WebSocketDisconnect:
            pass
        finally:
            controller.live_disconnect(token)

    async def _sse(
        request: Request, hub: LatestOnlyHub, *, decisions: bool
    ) -> AsyncIterator[bytes]:
        queue = hub.subscribe()
        controller: FastBridgeController = request.app.state.controller
        try:
            if decisions:
                controller.update_sse_metrics(
                    subscribers=hub.subscriber_count, last_write_ms=hub.last_write_ms
                )
            while True:
                if await request.is_disconnected():
                    return
                try:
                    payload = await asyncio.wait_for(
                        queue.get(), timeout=SSE_HEARTBEAT_SECONDS
                    )
                    hub.last_write_ms = time.time_ns() // 1_000_000
                    if decisions:
                        controller.update_sse_metrics(
                            subscribers=hub.subscriber_count,
                            last_write_ms=hub.last_write_ms,
                        )
                    yield b"data: " + payload + b"\n\n"
                except asyncio.TimeoutError:
                    yield b": heartbeat\n\n"
        finally:
            hub.unsubscribe(queue)
            if decisions:
                controller.update_sse_metrics(
                    subscribers=hub.subscriber_count, last_write_ms=hub.last_write_ms
                )

    @app.get("/v1/decisions/stream")
    async def decisions(request: Request) -> StreamingResponse:
        if not _bearer(request.headers, request.app.state.settings.android_token):
            raise HTTPException(401, "bearer token required")
        response = StreamingResponse(
            _sse(request, request.app.state.decision_hub, decisions=True),
            media_type="text/event-stream",
        )
        response.headers["Cache-Control"] = "no-store, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.get("/v1/admin/events")
    async def admin_events(
        request: Request, _identity: str = Depends(admin_identity)
    ) -> StreamingResponse:
        response = StreamingResponse(
            _sse(request, request.app.state.event_hub, decisions=False),
            media_type="text/event-stream",
        )
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.get("/v1/admin/status")
    async def admin_status(
        request: Request, _identity: str = Depends(admin_identity)
    ) -> Mapping[str, Any]:
        return request.app.state.controller.status()

    @app.get("/v1/admin/demos")
    async def demos(
        request: Request, _identity: str = Depends(admin_identity)
    ) -> Mapping[str, Any]:
        recorded = [
            {
                "demo_id": demo_id,
                "camera": "RECORDED_STREAM",
                "telemetry": "RECORDED_DATA",
                "inference": "LIVE_MODEL",
                "frames": len(bundle.frames),
            }
            for demo_id, bundle in sorted(request.app.state.bundles.items())
        ]
        return {
            "demos": [
                {
                    "demo_id": "LIVE_FULL_SYSTEM",
                    "camera": "LIVE_CAMERA",
                    "telemetry": "THIRD_PARTY",
                    "inference": "LIVE_MODEL",
                },
                *recorded,
            ],
            "unavailable": [
                {
                    "demo_id": "CARSKY_BROKER",
                    "reason": "Cloud Bridge is not available",
                }
            ],
            "hidden": {"TRANSPORT_PROBE": "NO_MODEL"},
        }

    def _idempotent_result(
        request: Request, cache_key: str, digest: str
    ) -> Mapping[str, Any] | None:
        cached = request.app.state.idempotency.get(cache_key)
        if cached is None:
            return None
        if cached[0] != digest:
            raise HTTPException(409, "Idempotency-Key was reused with another body")
        return cached[1]

    @app.post("/v1/admin/start")
    async def start(
        request: Request,
        control: tuple[str, str, Mapping[str, Any]] = Depends(control_request),
        identity: str = Depends(admin_identity),
    ) -> Mapping[str, Any]:
        cache_key, digest, body = control
        cached = _idempotent_result(request, cache_key, digest)
        if cached is not None:
            return cached
        if set(body) != {"demo_id", "expected_revision"}:
            raise HTTPException(400, "start body keys are invalid")
        demo_id = body["demo_id"]
        revision = body["expected_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise HTTPException(400, "expected_revision must be an integer")
        controller: FastBridgeController = request.app.state.controller
        try:
            if demo_id == "LIVE_FULL_SYSTEM":
                result = controller.start_live(expected_revision=revision)
            elif isinstance(demo_id, str) and DEMO_ID.fullmatch(demo_id):
                bundle = request.app.state.bundles.get(demo_id)
                if bundle is None:
                    raise HTTPException(404, "recorded demo is unavailable")
                result = controller.start_recorded(
                    bundle, expected_revision=revision
                )
            else:
                raise HTTPException(400, "demo_id is not allow-listed")
        except FastBridgeError as exc:
            raise HTTPException(409, str(exc)) from exc
        response = dict(result.__dict__)
        request.app.state.idempotency[cache_key] = (digest, response)
        _audit(identity, "START", str(demo_id), controller, "OK")
        return response

    @app.post("/v1/admin/stop")
    async def stop(
        request: Request,
        control: tuple[str, str, Mapping[str, Any]] = Depends(control_request),
        identity: str = Depends(admin_identity),
    ) -> Mapping[str, Any]:
        cache_key, digest, body = control
        cached = _idempotent_result(request, cache_key, digest)
        if cached is not None:
            return cached
        if set(body) != {"expected_revision"}:
            raise HTTPException(400, "stop body keys are invalid")
        revision = body["expected_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise HTTPException(400, "expected_revision must be an integer")
        controller: FastBridgeController = request.app.state.controller
        try:
            result = controller.stop(expected_revision=revision)
        except FastBridgeError as exc:
            raise HTTPException(409, str(exc)) from exc
        response = dict(result.__dict__)
        request.app.state.idempotency[cache_key] = (digest, response)
        _audit(identity, "STOP", None, controller, "OK")
        return response

    @app.post("/v1/admin/analyze")
    async def analyze(
        request: Request,
        control: tuple[str, str, Mapping[str, Any]] = Depends(control_request),
        identity: str = Depends(admin_identity),
    ) -> Mapping[str, Any]:
        cache_key, digest, body = control
        cached = _idempotent_result(request, cache_key, digest)
        if cached is not None:
            return cached
        if set(body) != {"demo_id"} or not isinstance(body["demo_id"], str):
            raise HTTPException(400, "analyze body keys are invalid")
        demo_id = body["demo_id"]
        bundle = request.app.state.bundles.get(demo_id)
        if bundle is None:
            raise HTTPException(404, "recorded demo is unavailable")
        try:
            result = await asyncio.to_thread(
                request.app.state.controller.analyze_bundle, bundle
            )
        except FastBridgeError as exc:
            raise HTTPException(409, str(exc)) from exc
        response = dict(result)
        request.app.state.idempotency[cache_key] = (digest, response)
        _audit(identity, "ANALYZE", demo_id, request.app.state.controller, "OK")
        return response

    static_root = Path(__file__).with_name("aws_admin")

    @app.get("/admin/")
    async def admin_page(
        request: Request, _identity: str = Depends(admin_identity)
    ) -> FileResponse:
        return FileResponse(static_root / "index.html", media_type="text/html")

    @app.get("/admin/app.css")
    async def admin_css(
        request: Request, _identity: str = Depends(admin_identity)
    ) -> FileResponse:
        return FileResponse(static_root / "app.css", media_type="text/css")

    @app.get("/admin/app.js")
    async def admin_js(
        request: Request, _identity: str = Depends(admin_identity)
    ) -> FileResponse:
        return FileResponse(
            static_root / "app.js", media_type="application/javascript"
        )

    return app


app = create_app()


__all__ = [
    "LatestOnlyHub",
    "ServiceSettings",
    "app",
    "create_app",
    "load_demo_allowlist",
]
