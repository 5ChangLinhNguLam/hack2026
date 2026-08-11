from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tools import carsky_native_ctl as native
from tools import carsky_scheduled_retry as retry


CANDIDATE_ID = "e6528451-f14d-4b4f-bd1b-f21691507a5f"
DEPLOYMENT_ID = "6331dc72-83ed-4df8-ab6f-1f51080f7e24"
TARGET_ID = retry.LOCKED_TARGET_DEVICE_ID


def make_config() -> native.NativeConfig:
    return native.parse_config(
        {
            "schema_version": native.SCHEMA_VERSION,
            "base_blueprint_id": native.DEFAULT_BASE_BLUEPRINT_ID,
            "candidate_blueprint_id": CANDIDATE_ID,
            "target_device_id": TARGET_ID,
            "blueprint_name": "SafeLoop Native Candidate",
            "deployment_name": "SafeLoop-Native-T01",
            "gateway": {
                "node_label": "IVI Gateway",
                "addon_contract": native.ADDON_CONTRACT,
                "kuksa": {
                    "contract": native.KUKSA_CONTRACT,
                    "reference_room_id": TARGET_ID,
                    "reference_node_key": "central-broker-vss",
                },
            },
            "hmi": {
                "transport": native.HMI_TRANSPORT,
                "switch_label": "IVI Switch",
                "android_node_label": "IVI - Android",
                "target_host": "10.99.0.14",
                "port": 48100,
            },
        }
    )


class FakeApi:
    def __init__(
        self,
        states: list[str],
        *,
        row_overrides: dict[str, str] | None = None,
        keep_after_delete: bool = False,
    ) -> None:
        self.states = list(states)
        self.row = {
            "id": DEPLOYMENT_ID,
            "blueprintId": CANDIDATE_ID,
            "roomId": TARGET_ID,
            "name": "SafeLoop-Native-T01",
            "status": states[0],
        }
        self.row.update(row_overrides or {})
        self.keep_after_delete = keep_after_delete
        self.deleted = 0

    def get(self, path: str):
        if path.startswith("/api/v1/deployments/find?"):
            return [dict(self.row)] if self.row is not None else []
        if path == f"/api/v1/deployments/{TARGET_ID}/status":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return {"status": state}
        raise AssertionError(path)

    def delete_locked_room_deployment(self, room_id: str):
        assert room_id == TARGET_ID
        self.deleted += 1
        if not self.keep_after_delete:
            self.row = None
        return {"ok": True}


def invoke(api: FakeApi, deploy_fn):
    return retry.retry_once(
        make_config(),
        object(),  # retry wrapper treats the addon opaquely; native deploy validates it.
        api,
        expected_deployment_id=DEPLOYMENT_ID,
        confirm_candidate_id=CANDIDATE_ID,
        confirm_device_id=TARGET_ID,
        checked_at="2026-08-11T01:00:00+07:00",
        poll_attempts=1,
        poll_interval_s=0,
        deploy_fn=deploy_fn,
    )


def test_running_is_noop() -> None:
    api = FakeApi(["RUNNING"])
    result = invoke(api, lambda *_args, **_kwargs: pytest.fail("must not deploy"))
    assert result["action"] == "NOOP_ALREADY_RUNNING"
    assert result["mutation_calls"] == 0
    assert api.deleted == 0


def test_exact_deploying_row_is_deleted_then_deployed_once() -> None:
    api = FakeApi(["DEPLOYING", "DEPLOYING"])
    calls = []

    def deploy_fn(*args, **kwargs):
        calls.append((args, kwargs))
        return {"deployed": True, "running": True, "mutation_calls": 1}

    result = invoke(api, deploy_fn)
    assert result["action"] == "RETRIED_ONCE"
    assert result["running"] is True
    assert result["mutation_calls"] == 2
    assert api.deleted == 1
    assert len(calls) == 1
    assert calls[0][1]["confirm_candidate_id"] == CANDIDATE_ID
    assert calls[0][1]["confirm_device_id"] == TARGET_ID


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "different-deployment"},
        {"blueprintId": "different-blueprint"},
        {"roomId": "different-room"},
        {"name": "different-name"},
    ],
)
def test_identity_mismatch_fails_closed(overrides: dict[str, str]) -> None:
    api = FakeApi(["DEPLOYING"], row_overrides=overrides)
    with pytest.raises(retry.ScheduledRetryRefused):
        invoke(api, lambda *_args, **_kwargs: pytest.fail("must not deploy"))
    assert api.deleted == 0


def test_unexpected_state_fails_closed() -> None:
    api = FakeApi(["PENDING"])
    with pytest.raises(retry.ScheduledRetryRefused, match="không thuộc"):
        invoke(api, lambda *_args, **_kwargs: pytest.fail("must not deploy"))
    assert api.deleted == 0


def test_race_to_running_is_preserved() -> None:
    api = FakeApi(["DEPLOYING", "RUNNING"])
    result = invoke(api, lambda *_args, **_kwargs: pytest.fail("must not deploy"))
    assert result["action"] == "NOOP_BECAME_RUNNING"
    assert api.deleted == 0


def test_row_remaining_after_delete_blocks_new_deployment() -> None:
    api = FakeApi(["FAILED", "FAILED"], keep_after_delete=True)
    with pytest.raises(retry.ScheduledRetryRefused, match="vẫn còn"):
        invoke(api, lambda *_args, **_kwargs: pytest.fail("must not deploy"))
    assert api.deleted == 1


def test_schedule_window_uses_vietnam_time() -> None:
    # 18:05 UTC on the previous date is 01:05 in Vietnam.
    instant = datetime(2026, 8, 10, 18, 5, tzinfo=timezone.utc)
    assert retry.assert_schedule_window(instant) == "2026-08-11T01:05:00+07:00"

    with pytest.raises(retry.ScheduledRetryRefused, match="01:00-01:59"):
        retry.assert_schedule_window(datetime(2026, 8, 10, 17, 59, tzinfo=timezone.utc))
