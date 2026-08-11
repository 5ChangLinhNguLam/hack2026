from __future__ import annotations

import base64
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safeloop_aaos_tools.apk_payload import COMPONENT, PACKAGE, install_script


def test_script_round_trips_payload_and_uses_fixed_component() -> None:
    apk = b"PK\x03\x04" + bytes(range(256)) * 5
    script = install_script(apk, chunk_size=256, port=48_100)
    chunks = re.findall(r"printf '%s' '([A-Za-z0-9+/=]+)'", script)

    assert base64.b64decode("".join(chunks)) == apk
    assert f"pm path {PACKAGE}" in script
    assert f"am start --user current -W -n {COMPONENT} --ei udp_port 48100" in script
    assert str(len(apk)) in script
    assert "rm -f /data/local/tmp/safeloop-aaos.apk.b64" in script


@pytest.mark.parametrize("chunk", [0, 255, 4097])
def test_invalid_chunk_size_is_rejected(chunk: int) -> None:
    with pytest.raises(ValueError, match="chunk size"):
        install_script(b"PKdata", chunk_size=chunk, port=48_100)


def test_non_apk_and_privileged_port_are_rejected() -> None:
    with pytest.raises(ValueError, match="APK"):
        install_script(b"not-a-zip", chunk_size=256, port=48_100)
    with pytest.raises(ValueError, match="UDP port"):
        install_script(b"PKdata", chunk_size=256, port=80)
