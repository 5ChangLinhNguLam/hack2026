from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from safeloop.aws_live_service import LatestOnlyHub, ServiceSettings, create_app
from test_aws_live_contract import valid_message
from test_aws_live_runtime import FakeModels


SETTINGS = ServiceSettings(
    public_base_url="https://safeloop.sonnet.io.vn",
    admin_origin="https://safeloop.sonnet.io.vn",
    demo_root=Path("/missing"),
    ingest_token="i" * 32,
    android_token="a" * 32,
    c1_checkpoint=Path("unused"),
    c2_bundle=Path("unused"),
    device="cpu",
    allow_local_admin=True,
)


def action_headers(key="one"):
    return {
        "Origin": SETTINGS.admin_origin,
        "Content-Type": "application/json",
        "X-SafeLoop-Admin-Action": "1",
        "Idempotency-Key": key,
    }


def test_latest_only_hub_has_no_replay_and_capacity_one() -> None:
    async def scenario():
        hub = LatestOnlyHub()
        hub.publish(b"before")
        queue = hub.subscribe()
        assert queue.empty()
        hub.publish(b"one")
        hub.publish(b"two")
        assert queue.qsize() == 1
        assert queue.get_nowait() == b"two"
    asyncio.run(scenario())


def test_health_admin_security_allowlist_and_revision_conflict() -> None:
    app = create_app(settings=SETTINGS, models=FakeModels(), bundles={})
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/docs").status_code == 404
        demos = client.get("/v1/admin/demos").json()
        assert demos["hidden"] == {"TRANSPORT_PROBE": "NO_MODEL"}
        assert demos["unavailable"][0]["demo_id"] == "CARSKY_BROKER"
        traversal = client.post(
            "/v1/admin/start",
            headers=action_headers(),
            json={"demo_id": "../T01-Sample", "expected_revision": 0},
        )
        assert traversal.status_code == 400
        started = client.post(
            "/v1/admin/start",
            headers=action_headers("two"),
            json={"demo_id": "LIVE_FULL_SYSTEM", "expected_revision": 0},
        )
        assert started.status_code == 200
        conflict = client.post(
            "/v1/admin/stop",
            headers=action_headers("three"),
            json={"expected_revision": 0},
        )
        assert conflict.status_code == 409


def test_machine_endpoint_requires_independent_bearer() -> None:
    app = create_app(settings=SETTINGS, models=FakeModels(), bundles={})
    with TestClient(app) as client:
        assert client.get("/v1/decisions/stream").status_code == 401
        assert SETTINGS.android_token != SETTINGS.ingest_token


def test_wss_ingest_auth_and_contract_fixture_reach_inference() -> None:
    app = create_app(settings=SETTINGS, models=FakeModels(), bundles={})
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect("/v1/ingest"):
                pass
        assert denied.value.code == 4401

        started = client.post(
            "/v1/admin/start",
            headers=action_headers("wss-start"),
            json={"demo_id": "LIVE_FULL_SYSTEM", "expected_revision": 0},
        )
        assert started.status_code == 200
        with client.websocket_connect(
            "/v1/ingest",
            headers={"Authorization": f"Bearer {SETTINGS.ingest_token}"},
        ) as socket:
            socket.send_text(json.dumps(valid_message()))
            deadline = time.monotonic() + 2.0
            while app.state.controller.status()["counts"].get("decisions", 0) != 1:
                assert time.monotonic() < deadline
                time.sleep(0.005)


def test_idempotency_replays_same_control_result() -> None:
    app = create_app(settings=SETTINGS, models=FakeModels(), bundles={})
    with TestClient(app) as client:
        body = {"demo_id": "LIVE_FULL_SYSTEM", "expected_revision": 0}
        first = client.post("/v1/admin/start", headers=action_headers("same"), json=body)
        second = client.post("/v1/admin/start", headers=action_headers("same"), json=body)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()


def test_concurrent_admin_controls_are_serialized_by_revision() -> None:
    app = create_app(settings=SETTINGS, models=FakeModels(), bundles={})
    with TestClient(app) as client:
        def start(key: str) -> int:
            response = client.post(
                "/v1/admin/start",
                headers=action_headers(key),
                json={"demo_id": "LIVE_FULL_SYSTEM", "expected_revision": 0},
            )
            return response.status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            status_codes = sorted(pool.map(start, ("concurrent-a", "concurrent-b")))

        assert status_codes == [200, 409]
