#!/usr/bin/env python3
"""Generate paste-safe AAOS shell commands for installing the SafeLoop APK.

CarSky's Conduit-backed REST ADB route is not configured in the current room,
but its interactive ADB Widget shell works.  This tool converts an APK into
small base64 commands that can be pasted into that shell without putting the
APK on a public web server.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
import sys


PACKAGE = "com.fptautomotive.safeloop"
COMPONENT = f"{PACKAGE}/.MainActivity"
REMOTE_B64 = "/data/local/tmp/safeloop-aaos.apk.b64"
REMOTE_APK = "/data/local/tmp/safeloop-aaos.apk"


def install_script(apk: bytes, *, chunk_size: int, port: int) -> str:
    if not apk.startswith(b"PK"):
        raise ValueError("input does not look like an APK/ZIP archive")
    if not 256 <= chunk_size <= 4_096:
        raise ValueError("chunk size must be in [256, 4096]")
    if not 1_024 <= port <= 65_535:
        raise ValueError("UDP port must be in [1024, 65535]")
    encoded = base64.b64encode(apk).decode("ascii")
    digest = hashlib.sha256(apk).hexdigest()
    chunks = [encoded[offset : offset + chunk_size]
              for offset in range(0, len(encoded), chunk_size)]
    lines = [
        "set -eu",
        f"rm -f {REMOTE_B64} {REMOTE_APK}",
    ]
    for index, chunk in enumerate(chunks):
        redirect = ">" if index == 0 else ">>"
        lines.append(f"printf '%s' '{chunk}' {redirect} {REMOTE_B64}")
    lines.extend(
        [
            f"base64 -d {REMOTE_B64} > {REMOTE_APK}",
            f"test \"$(wc -c < {REMOTE_APK})\" -eq {len(apk)}",
            f"test \"$(toybox sha256sum {REMOTE_APK} | cut -d' ' -f1)\" = '{digest}'",
            f"pm install -r -t {REMOTE_APK}",
            f"pm path {PACKAGE}",
            f"am force-stop --user current {PACKAGE}",
            f"am start --user current -W -n {COMPONENT} --ei udp_port {port}",
            f"rm -f {REMOTE_B64} {REMOTE_APK}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit commands for sideloading SafeLoop through an AAOS shell"
    )
    parser.add_argument("apk", type=Path, help="path to app-debug.apk or signed release APK")
    parser.add_argument("--chunk-size", type=int, default=1_200)
    parser.add_argument("--port", type=int, default=48_100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        payload = args.apk.read_bytes()
        script = install_script(payload, chunk_size=args.chunk_size, port=args.port)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
