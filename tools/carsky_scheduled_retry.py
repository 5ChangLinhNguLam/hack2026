#!/usr/bin/env python3
"""One-shot, fail-closed retry for the approved 01:00 CarSky deployment.

The scheduler is deliberately kept outside this module.  This command accepts
only the locked SafeLoop room, checks the exact existing deployment identity,
and behaves as follows:

* exact deployment is ``RUNNING``: return success without mutation;
* exact deployment is ``DEPLOYING`` or ``FAILED``: delete that room deployment
  once, verify that it disappeared, then call the normal guarded deploy once;
* any missing, duplicate, mismatched, or unexpected state: fail closed.

The command also refuses to run outside 01:00--01:59 Asia/Ho_Chi_Minh.  It
reads ``A8_API_KEY`` from the environment; it never reads or embeds an env file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib import parse
from zoneinfo import ZoneInfo

try:
    from tools import carsky_native_ctl as native
except ModuleNotFoundError:  # Direct execution: python3 tools/<script>.py
    import carsky_native_ctl as native  # type: ignore[no-redef]


LOCKED_TARGET_DEVICE_ID = "h5sjc3jtzl8vzl9wyqokv"
SCHEDULE_TIMEZONE_NAME = "Asia/Ho_Chi_Minh"
SCHEDULE_HOUR = 1
RETRYABLE_STATES = frozenset({"DEPLOYING", "FAILED"})
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


class ScheduledRetryRefused(native.MutationRefused):
    """The one-shot retry did not satisfy every destructive-action guard."""


class ScheduledRetryApi(native.CarSkyApi):
    """CarSky client with one narrowly scoped teardown operation."""

    def delete_locked_room_deployment(self, room_id: str) -> Any:
        if room_id != LOCKED_TARGET_DEVICE_ID:
            raise ScheduledRetryRefused("Từ chối teardown ngoài room đã khóa")
        return self._request(
            "DELETE", f"/api/v1/deployments/{parse.quote(room_id, safe='')}"
        )


def assert_schedule_window(now: datetime | None = None) -> str:
    """Require the scheduled 01:00 local-hour window and return its timestamp."""

    timezone = ZoneInfo(SCHEDULE_TIMEZONE_NAME)
    current = datetime.now(timezone) if now is None else now
    if current.tzinfo is None:
        raise ScheduledRetryRefused("Thời gian kiểm tra phải có timezone")
    local = current.astimezone(timezone)
    if local.hour != SCHEDULE_HOUR:
        raise ScheduledRetryRefused(
            "Lệnh retry chỉ được chạy trong 01:00-01:59 Asia/Ho_Chi_Minh"
        )
    return local.isoformat(timespec="seconds")


def _items(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, Mapping):
        for key in ("data", "items", "deployments"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _target_rows(api: Any, target_device_id: str) -> list[dict[str, Any]]:
    path = "/api/v1/deployments/find?" + parse.urlencode(
        {"device": target_device_id}
    )
    return [
        item
        for item in _items(api.get(path))
        if item.get("roomId") == target_device_id
    ]


def _status(api: Any, target_device_id: str) -> str:
    value = api.get(
        f"/api/v1/deployments/{parse.quote(target_device_id, safe='')}/status"
    )
    if not isinstance(value, Mapping) or not isinstance(value.get("status"), str):
        raise ScheduledRetryRefused("CarSky không trả deployment status hợp lệ")
    state = value["status"].strip().upper()
    if not state:
        raise ScheduledRetryRefused("CarSky trả deployment status rỗng")
    return state


def _require_exact_row(
    rows: list[dict[str, Any]],
    *,
    expected_deployment_id: str,
    candidate_blueprint_id: str,
    target_device_id: str,
    deployment_name: str,
) -> dict[str, Any]:
    if len(rows) != 1:
        raise ScheduledRetryRefused(
            f"Cần đúng 1 deployment trên target, thực tế có {len(rows)}"
        )
    row = rows[0]
    expected = {
        "id": expected_deployment_id,
        "blueprintId": candidate_blueprint_id,
        "roomId": target_device_id,
        "name": deployment_name,
    }
    mismatches = [
        key for key, expected_value in expected.items() if row.get(key) != expected_value
    ]
    if mismatches:
        raise ScheduledRetryRefused(
            "Deployment identity không khớp: " + ", ".join(mismatches)
        )
    return {key: row.get(key) for key in expected}


def retry_once(
    config: native.NativeConfig,
    addon: native.GatewayAddon,
    api: Any,
    *,
    expected_deployment_id: str,
    confirm_candidate_id: str,
    confirm_device_id: str,
    checked_at: str,
    poll_attempts: int = 120,
    poll_interval_s: float = 15.0,
    deploy_fn: Callable[..., dict[str, Any]] = native.deploy_candidate,
    openapi_path: Path = native.DEFAULT_OPENAPI,
) -> dict[str, Any]:
    """No-op on success, otherwise replace the exact retryable row once."""

    candidate_id = config.candidate_blueprint_id
    if candidate_id is None:
        raise ScheduledRetryRefused("Config chưa có candidate_blueprint_id")
    if config.target_device_id != LOCKED_TARGET_DEVICE_ID:
        raise ScheduledRetryRefused("Config không trỏ tới room retry đã khóa")
    if confirm_candidate_id != candidate_id:
        raise ScheduledRetryRefused("Xác nhận candidate không khớp config")
    if confirm_device_id != LOCKED_TARGET_DEVICE_ID:
        raise ScheduledRetryRefused("Xác nhận device không khớp room đã khóa")
    if not _SAFE_ID.fullmatch(expected_deployment_id):
        raise ScheduledRetryRefused("Expected deployment id không an toàn")

    identity_args = {
        "expected_deployment_id": expected_deployment_id,
        "candidate_blueprint_id": candidate_id,
        "target_device_id": config.target_device_id,
        "deployment_name": config.deployment_name,
    }
    identity = _require_exact_row(
        _target_rows(api, config.target_device_id), **identity_args
    )
    initial_state = _status(api, config.target_device_id)
    if initial_state == "RUNNING":
        return {
            "action": "NOOP_ALREADY_RUNNING",
            "checked_at": checked_at,
            "state": initial_state,
            "deployment": identity,
            "mutation_calls": 0,
        }
    if initial_state not in RETRYABLE_STATES:
        raise ScheduledRetryRefused(
            f"State {initial_state!r} không thuộc DEPLOYING/FAILED; không teardown"
        )

    # Close the inspection-to-delete race.  A deployment that becomes RUNNING
    # while this command is checking must be preserved.
    identity = _require_exact_row(
        _target_rows(api, config.target_device_id), **identity_args
    )
    delete_state = _status(api, config.target_device_id)
    if delete_state == "RUNNING":
        return {
            "action": "NOOP_BECAME_RUNNING",
            "checked_at": checked_at,
            "state": delete_state,
            "deployment": identity,
            "mutation_calls": 0,
        }
    if delete_state not in RETRYABLE_STATES:
        raise ScheduledRetryRefused(
            f"State đổi thành {delete_state!r}; từ chối teardown"
        )

    deleted = api.delete_locked_room_deployment(config.target_device_id)
    if not isinstance(deleted, Mapping) or deleted.get("ok") is not True:
        raise ScheduledRetryRefused(
            "CarSky không xác nhận teardown; từ chối tạo deployment mới"
        )
    remaining = _target_rows(api, config.target_device_id)
    if remaining:
        raise ScheduledRetryRefused(
            "Deployment cũ vẫn còn sau teardown; từ chối tạo deployment mới"
        )

    deployed = deploy_fn(
        config,
        addon,
        api,
        confirm_candidate_id=confirm_candidate_id,
        confirm_device_id=confirm_device_id,
        poll_attempts=poll_attempts,
        poll_interval_s=poll_interval_s,
        openapi_path=openapi_path,
    )
    return {
        "action": "RETRIED_ONCE",
        "checked_at": checked_at,
        "previous_state": delete_state,
        "removed_deployment": identity,
        "deployment": deployed,
        "mutation_calls": 1 + int(deployed.get("mutation_calls", 0)),
        "running": deployed.get("running") is True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="01:00 fail-closed one-shot retry cho SafeLoop CarSky"
    )
    parser.add_argument("--url", default=os.getenv("A8_URL", native.DEFAULT_URL))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--openapi", type=Path, default=native.DEFAULT_OPENAPI)
    parser.add_argument("config", type=Path)
    parser.add_argument("--gateway-addon", type=Path, required=True)
    parser.add_argument("--expected-deployment-id", required=True)
    parser.add_argument("--confirm-candidate-id", required=True)
    parser.add_argument("--confirm-device-id", required=True)
    parser.add_argument("--poll-attempts", type=int, default=120)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[..., Any] = ScheduledRetryApi,
    now: datetime | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        checked_at = assert_schedule_window(now)
        config = native.load_config(args.config)
        addon = native.load_gateway_addon(args.gateway_addon, config)
        api = api_factory(
            args.url,
            os.getenv("A8_API_KEY", ""),
            timeout_s=args.timeout,
        )
        result = retry_once(
            config,
            addon,
            api,
            expected_deployment_id=args.expected_deployment_id,
            confirm_candidate_id=args.confirm_candidate_id,
            confirm_device_id=args.confirm_device_id,
            checked_at=checked_at,
            poll_attempts=args.poll_attempts,
            poll_interval_s=args.poll_interval,
            openapi_path=args.openapi,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["action"].startswith("NOOP_"):
            return 0
        return 0 if result.get("running") is True else 6
    except (
        ScheduledRetryRefused,
        native.NativeConfigError,
        native.CarSkyApiError,
        native.MutationRefused,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
