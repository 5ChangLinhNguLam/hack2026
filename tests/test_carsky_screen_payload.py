"""Offline tests for the CarSky ADB Widget dashboard payload."""

from __future__ import annotations

import base64
import re

import pytest

from tools.carsky_screen_payload import REMOTE_PAYLOAD, build_adb_widget_commands


def test_commands_reconstruct_dashboard_and_launch_explicit_webview():
    html = b"<!doctype html><title>SafeLoop</title>" * 40

    commands = build_adb_widget_commands(html, chunk_size=256)

    chunks = [
        match.group(1)
        for command in commands
        if (match := re.search(r"printf '%s' '([A-Za-z0-9+/=]+)'", command))
    ]
    assert base64.b64decode("".join(chunks)) == html
    assert commands[0] == f"rm -f {REMOTE_PAYLOAD}"
    assert "SAFELOOP_UPLOAD_OK" in commands[-1]
    assert f"Content-Length: {len(html)}" in commands[-1]
    assert "Cache-Control: no-store, no-cache, must-revalidate" in commands[-1]
    assert "toybox nc -l -s 127.0.0.1 -p 8765" in commands[-1]
    assert "am force-stop org.chromium.webview_shell" in commands[-1]
    assert "org.chromium.webview_shell/.WebViewBrowserActivity" in commands[-1]
    assert "http://127.0.0.1:8765/safeloop?build=" in commands[-1]
    assert "run-as" not in commands[-1]


@pytest.mark.parametrize("chunk_size", [0, 255, 4097])
def test_rejects_unsafe_chunk_sizes(chunk_size):
    with pytest.raises(ValueError):
        build_adb_widget_commands(b"dashboard", chunk_size=chunk_size)


def test_rejects_empty_dashboard():
    with pytest.raises(ValueError):
        build_adb_widget_commands(b"")
