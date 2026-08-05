#!/usr/bin/env python3
"""Generate paste-ready ADB Widget commands for the SafeLoop dashboard.

CarSky's REST ADB endpoints require the optional Conduit service.  The ADB
Widget can still provide an interactive shell, so this helper encodes the
standalone dashboard and emits bounded-size shell commands for that terminal.
No credential or network access is used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = ROOT / "carsky" / "screen" / "safeloop_dashboard.html"
REMOTE_PAYLOAD = "/data/local/tmp/safeloop_dashboard.b64"
WEBVIEW_PACKAGE = "org.chromium.webview_shell"
WEBVIEW_ACTIVITY = "org.chromium.webview_shell/.WebViewBrowserActivity"
LOCAL_HTTP_PORT = 8765


def build_adb_widget_commands(html: bytes, chunk_size: int = 1200) -> list[str]:
    """Return commands that upload and launch ``html`` in AAOS WebView Shell."""

    if chunk_size < 256 or chunk_size > 4096:
        raise ValueError("chunk_size must be between 256 and 4096")
    encoded = base64.b64encode(html).decode("ascii")
    chunks = [encoded[index : index + chunk_size] for index in range(0, len(encoded), chunk_size)]
    if not chunks:
        raise ValueError("dashboard HTML is empty")

    commands = [f"rm -f {REMOTE_PAYLOAD}"]
    for index, chunk in enumerate(chunks):
        redirect = ">" if index == 0 else ">>"
        commands.append(f"printf '%s' '{chunk}' {redirect} {REMOTE_PAYLOAD}")

    expected_size = len(encoded)
    html_size = len(html)
    build_id = hashlib.sha256(html).hexdigest()[:12]
    http_uri = (
        f'"http://127.0.0.1:{LOCAL_HTTP_PORT}/safeloop?build={build_id}"'
    )
    http_header = (
        "HTTP/1.1 200 OK\\r\\n"
        "Content-Type: text/html; charset=utf-8\\r\\n"
        f"Content-Length: {html_size}\\r\\n"
        "Cache-Control: no-store, no-cache, must-revalidate\\r\\n"
        "Connection: close\\r\\n\\r\\n"
    )
    commands.append(
        f'if [ "$(wc -c < {REMOTE_PAYLOAD})" -eq {expected_size} ] '
        f"&& base64 -d {REMOTE_PAYLOAD} >/dev/null; then "
        f"echo SAFELOOP_UPLOAD_OK:{expected_size}; "
        f"(printf '{http_header}'; base64 -d {REMOTE_PAYLOAD}) | "
        f"toybox nc -l -s 127.0.0.1 -p {LOCAL_HTTP_PORT} & "
        "sleep 1; "
        f"am force-stop {WEBVIEW_PACKAGE}; "
        "am start -W -a android.intent.action.VIEW "
        f"-n {WEBVIEW_ACTIVITY} -d {http_uri}; "
        "else echo SAFELOOP_UPLOAD_FAILED; fi"
    )
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tạo lệnh dán vào CarSky ADB Widget để mở SafeLoop dashboard"
    )
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--chunk-size", type=int, default=1200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        html = args.html.read_bytes()
        commands = build_adb_widget_commands(html, args.chunk_size)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Lỗi: {exc}") from exc
    print("\n".join(commands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
