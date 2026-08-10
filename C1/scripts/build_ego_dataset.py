"""MỘT lệnh: từ Practice_dataset + mọi DeepAccident_train_*.zip -> file .zip upload Colab.

Chạy sau khi tải xong toàn bộ zip:

    python scripts/build_ego_dataset.py

Nó gọi lại đúng các script đã có, theo thứ tự, và CHẠY LẠI ĐƯỢC: thêm zip thứ 3,
4, 5 rồi chạy lại đúng lệnh trên thì chỉ phần mới bị xử lý, video cũ không mã hoá
lại (bước nặng nhất — 416 clip mất ~35 phút).

    1. extract_deepaccident_zips.py   lọc meta + label + Camera_Front (4 agent xe)
                                      ra khỏi zip; bỏ lidar/5 camera kia/BEV,
                                      53 GB còn ~3.4 GB mỗi 2 gói
    2. build_practice_trips.py        6 trip GT thật -> datasets/practice
    3. build_deepaccident_ttc.py      tính lại min_ttc từ box 3D + dựng mp4
                                      (--skip-existing: giữ mp4 đã có)
    4. pack_hackathon_dataset.py      trộn hai nguồn, áp CHÍNH SÁCH EGO
    5. verify_hackathon_dataset.py    giải mã lại toàn bộ video, đối chiếu nhãn
    6. zip_dataset.py                 đóng gói + kiểm lại

CHÍNH SÁCH EGO (mặc định `no-label`, đúng yêu cầu "chỉ trip mà chính xe gắn cam
bị va chạm mới có nhãn TTC"):

    ego_involved = 1   trip giữ nguyên nhãn min_ttc
    ego_involved = 0   TOÀN BỘ frame -> ttc_s = -1 (an toàn), trip thành negative

`ego_involved` suy từ meta của scenario: `agents id:` cho id của 4 agent theo
đúng thứ tự, dòng đầu cho id hai vật va chạm. Agent nào có id trùng một trong
hai thì chính nó bị đâm. Lưu ý ĐÃ ĐO: 77/104 scenario KHÔNG hề có va chạm dù tên
thư mục là `*_accident` (`obj1_id = -1`, `colliding agents: none none`) — nên
không được suy va chạm từ tên thư mục.

⚠️ ĐÁNH ĐỔI CỦA `no-label`, đo trên 2 gói zip đầu: frame nguy hiểm tụt từ 3.465
xuống ~511 (1,2% tổng số frame), và một xe bám đuôi ở TTC 0.8s mà không đâm sẽ
mang nhãn "an toàn" trong khi frame nhìn y hệt ở trip có đâm lại là "nguy hiểm".
Muốn so sánh thì chạy lại với `--ego-policy keep` ra một bộ khác rồi train cả hai.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, *a) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, a)]
    print(f"\n{'=' * 78}\n$ {script} {' '.join(map(str, a))}\n{'=' * 78}", flush=True)
    # PYTHONUNBUFFERED: cả chạy mất hơn tiếng, log bị đệm 8 KB thì nhìn như treo
    r = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                 "PYTHONUNBUFFERED": "1"})
    if r.returncode != 0:
        raise SystemExit(f"\n{script} thất bại (mã {r.returncode}) — dừng tại đây")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="hackathon_ttc_ego",
                    help="tên bộ dữ liệu -> datasets/<name>/ và datasets/<name>.zip")
    ap.add_argument("--ego-policy", choices=("no-label", "keep", "drop"),
                    default="no-label")
    ap.add_argument("--carry-from", default="hackathon_ttc",
                    help="bộ cũ để lấy lại trip không dựng lại được; '' = tắt")
    ap.add_argument("--reencode-all", action="store_true",
                    help="mã hoá lại mọi video kể cả đã có (chậm; chỉ khi nghi hỏng)")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    data = ROOT / "datasets"
    raw = data / "deepaccident" / "raw" / "train"
    out = data / args.name

    zips = sorted(data.glob("DeepAccident_train_*.zip"))
    n_scen = len(list(raw.glob("*/meta/*.txt")))
    print(f"{len(zips)} gói zip · {n_scen} scenario đã giải nén sẵn ở {raw}")

    # Xoá zip sau khi giải nén là chuyện bình thường (53 GB). Đã có dữ liệu giải
    # nén thì thiếu zip KHÔNG phải lỗi — chỉ là không có gì mới để lấy ra.
    if not zips and not n_scen:
        raise SystemExit(f"không có zip trong {data} lẫn dữ liệu giải nén ở {raw}")
    if not args.skip_extract and zips:
        run("extract_deepaccident_zips.py", "--out", raw)
    elif not zips:
        print("(không có zip mới — dùng dữ liệu đã giải nén)")

    run("build_practice_trips.py", "--src", data / "Practice_dataset",
        "--out", data / "practice")

    da_args = ["--src", raw, "--out", data / "deepaccident_ttc"]
    if not args.reencode_all:
        da_args.append("--skip-existing")
    run("build_deepaccident_ttc.py", *da_args)

    pack = ["--out", out, "--ego-policy", args.ego_policy]
    if args.carry_from:
        carry = data / args.carry_from
        if carry.exists() and carry.resolve() != out.resolve():
            pack += ["--carry-from", carry]
        else:
            print(f"(bỏ qua --carry-from: {carry} không dùng được)")
    run("pack_hackathon_dataset.py", *pack)

    run("verify_hackathon_dataset.py", "--data", out)

    if not args.no_zip:
        run("zip_dataset.py", "--data", out)

    print(f"\n{'=' * 78}")
    print(f"XONG -> {out}.zip")
    print(f"Upload lên Drive, rồi trong cell 1 của notebook đặt: DATASET = \"{args.name}\"")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
