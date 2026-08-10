"""Giải nén CÓ CHỌN LỌC các gói DeepAccident_train_*.zip.

Vì sao không giải nén cả gói: hai file zip cộng lại 53 GB, mà pipeline TTC
(scripts/build_deepaccident_ttc.py) chỉ đọc đúng ba thứ:

    <scenario_type>/meta/<scenario>.txt              ~1 KB / scenario
    <scenario_type>/<agent>/label/<scenario>/*.txt   box 3D trong hệ ego
    <scenario_type>/<agent>/Camera_Front/<scenario>/*.jpg

Phần còn lại là lidar01 (13 GB/gói), 5 camera khác, BEV_instance_camera và
calib — không dùng đến. Lọc lại còn ~3.4 GB, vừa ổ đĩa và nhanh hơn nhiều.

Cũng bỏ luôn agent `infrastructure`: đó là camera CỐ ĐỊNH bên đường, "ego" của
nó không di chuyển nên min_ttc theo góc nhìn lái xe vô nghĩa (cùng lý do
build_deepaccident_ttc.py chỉ giữ 4 agent xe).

Đầu ra giữ NGUYÊN cây thư mục trong zip, nên thư mục đích dùng thẳng được làm
`--src` cho build_deepaccident_ttc.py — y hệt cách raw/mini được dùng trước đây.

Chạy lại được: file nào đã có và đúng kích thước thì bỏ qua, nên lỡ ngắt giữa
chừng cứ chạy lại lệnh cũ.
"""

import argparse
import sys
import zipfile
from pathlib import Path

# console Windows mặc định cp1252, không in nổi tiếng Việt trong log
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

AGENTS = ("ego_vehicle", "ego_vehicle_behind", "other_vehicle", "other_vehicle_behind")
MODALITIES = ("Camera_Front", "label")


def wanted(name: str, agents, modalities) -> bool:
    """True nếu member này thuộc phần dữ liệu pipeline TTC thực sự đọc."""
    q = name.split("/")
    if name.endswith("/"):
        return False
    if len(q) == 3 and q[1] == "meta" and q[2].endswith(".txt"):
        return True
    return len(q) == 5 and q[1] in agents and q[2] in modalities


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", type=Path, nargs="*", default=None,
                    help="mặc định: datasets/DeepAccident_train_*.zip")
    ap.add_argument("--out", type=Path,
                    default=root / "datasets" / "deepaccident" / "raw" / "train")
    ap.add_argument("--agents", nargs="*", default=list(AGENTS))
    ap.add_argument("--modalities", nargs="*", default=list(MODALITIES))
    ap.add_argument("--dry-run", action="store_true", help="chỉ đếm, không ghi")
    args = ap.parse_args()

    zips = args.zips or sorted((root / "datasets").glob("DeepAccident_train_*.zip"))
    if not zips:
        raise SystemExit("không thấy DeepAccident_train_*.zip trong datasets/")
    agents, mods = set(args.agents), set(args.modalities)
    args.out.mkdir(parents=True, exist_ok=True)

    n_all = n_new = n_skip = 0
    b_new = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            members = [i for i in z.infolist() if wanted(i.filename, agents, mods)]
            total = sum(i.file_size for i in members)
            print(f"{zp.name}: {len(members)} member cần lấy, {total / 1e9:.2f} GB")
            if args.dry_run:
                n_all += len(members)
                continue

            for k, info in enumerate(members, 1):
                dst = args.out / info.filename
                # đã có đúng kích thước -> coi như xong (cho phép chạy lại)
                if dst.exists() and dst.stat().st_size == info.file_size:
                    n_skip += 1
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as fsrc, dst.open("wb") as fdst:
                        while True:
                            chunk = fsrc.read(1 << 20)
                            if not chunk:
                                break
                            fdst.write(chunk)
                    n_new += 1
                    b_new += info.file_size
                n_all += 1
                if k % 2000 == 0 or k == len(members):
                    print(f"  [{k}/{len(members)}] mới {n_new} · bỏ qua {n_skip} "
                          f"· {b_new / 1e9:.2f} GB")

    scen = sorted(p.stem for p in args.out.glob("*/meta/*.txt"))
    print(f"\n{n_all} member · giải nén mới {n_new} ({b_new / 1e9:.2f} GB) · "
          f"có sẵn {n_skip}")
    print(f"{len(scen)} scenario -> {args.out}")


if __name__ == "__main__":
    main()
