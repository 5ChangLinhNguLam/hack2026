"""Fix "WinError 1114 ... c10.dll" when importing torch inside a Jupyter kernel.

The problem
-----------
Windows limits how many dynamically-loaded DLLs may use *static* TLS
(`__declspec(thread)`). torch's `c10.dll` is one of them.

At startup an IPython kernel calls `platform.win32_ver()`, which queries WMI and
pulls in ~28 extra system DLLs (measured: 39 -> 67 loaded modules). By the time
your first cell runs, the static-TLS slots are exhausted, so `LoadLibrary` on
`c10.dll` fails with ERROR_DLL_INIT_FAILED (1114).

Plain `python -c "import torch"` works because nothing called WMI first -- which
is why the same code runs fine as a script and dies in a notebook.

The fix
-------
Load `c10.dll` during interpreter startup, before WMI eats the slots. Python's
`site` module imports `usercustomize` if it can be found on `sys.path`, so a tiny
`usercustomize.py` plus `PYTHONPATH` does it -- no torch import, ~0 startup cost.

This script writes that file and registers a **separate Jupyter kernel** that sets
PYTHONPATH to it, so nothing else on the machine changes. Select
"Python 3 (torch fix)" as the notebook kernel in VS Code / Jupyter.

    python scripts/fix_torch_jupyter_windows.py            # install
    python scripts/fix_torch_jupyter_windows.py --remove   # undo

Colab and Linux are unaffected -- there is nothing to fix there.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

KERNEL_NAME = "python3-torchfix"
DISPLAY_NAME = "Python 3 (torch fix)"

PRELOAD_SRC = '''\
"""Load torch's c10.dll before WMI exhausts Windows' static-TLS slots.

Imported automatically by `site` at interpreter startup when this directory is on
PYTHONPATH. Deliberately does NOT import torch -- loading the one DLL is enough to
reserve its TLS slot, and keeps startup cost near zero.
"""
import os

try:
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("torch")       # does not import torch
    if spec and spec.origin:
        lib = os.path.join(os.path.dirname(spec.origin), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            ctypes.CDLL(os.path.join(lib, "c10.dll"))
except Exception:
    pass          # never break interpreter startup over this
'''


def kernel_dir() -> Path:
    base = os.environ.get("APPDATA")
    if not base:
        raise SystemExit("APPDATA not set -- is this Windows?")
    return Path(base) / "jupyter" / "kernels" / KERNEL_NAME


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remove", action="store_true", help="uninstall the kernel")
    ap.add_argument("--preload-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / ".torch_preload")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("Chỉ cần trên Windows — bỏ qua.")
        return

    kd = kernel_dir()

    if args.remove:
        for path in (kd, args.preload_dir):
            if path.exists():
                shutil.rmtree(path)
                print(f"đã xoá {path}")
            else:
                print(f"không có {path}")
        return

    args.preload_dir.mkdir(parents=True, exist_ok=True)
    (args.preload_dir / "usercustomize.py").write_text(PRELOAD_SRC, encoding="utf-8")
    print(f"viết {args.preload_dir / 'usercustomize.py'}")

    kd.mkdir(parents=True, exist_ok=True)
    (kd / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": DISPLAY_NAME,
        "language": "python",
        "env": {"PYTHONPATH": str(args.preload_dir)},
    }, indent=1), encoding="utf-8")
    print(f"đăng ký kernel {DISPLAY_NAME!r} tại {kd}")

    print(f"\nTrong VS Code / Jupyter, chọn kernel {DISPLAY_NAME!r} rồi chạy lại notebook.")
    print(f"Gỡ bằng: python {Path(__file__).name} --remove")


if __name__ == "__main__":
    main()
