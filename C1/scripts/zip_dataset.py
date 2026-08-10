"""Zip một bộ đã đóng gói để upload lên Drive.

Hai chi tiết quyết định file zip có dùng được trên Colab hay không:

1. KHÔNG bọc thêm thư mục. Trong zip, `annotations/trips.csv` nằm ngay gốc, nên
   giải nén vào /content/datasets/<DATASET> là ra đúng cây notebook mong đợi.
   (Cell 2 của notebook có rglob nên bọc thêm lớp vẫn chạy — nhưng lúc đó
   DATA_ROOT thành .../<DATASET>/<DATASET>, nhìn log rất dễ tưởng sai.)

2. mp4 lưu STORED, không deflate. Video đã nén rồi; ép deflate lên 690 MB mp4
   tốn vài phút CPU để giảm chưa tới 1%. CSV thì ngược lại — nhãn per-frame là
   text lặp, deflate còn khoảng 1/5. Nên trộn: nén thứ đáng nén thôi.

Kiểm lại sau khi ghi: mở zip đọc danh sách và đối chiếu số file + tổng byte với
thư mục nguồn. Zip hỏng chỉ lộ ra sau khi upload xong mới là đắt.

3. GỘP LUÔN datasets/depth_aux (bộ đích cho head depth phụ) vào cùng zip, dưới
   tiền tố `depth_aux/`. Nó chỉ vài MB, và bắt người dùng nhớ upload đúng hai
   file zip rồi giải nén đúng hai chỗ là cách chắc chắn để một hôm nào đó train
   thiếu mất phần regularize mà không ai nhận ra — notebook chỉ in một dòng
   "depth phụ: TẮT" giữa hàng chục dòng log khác.
"""

import argparse
import sys
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

STORE_EXT = {".mp4", ".jpg", ".jpeg", ".png", ".npz", ".gz"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=root / "datasets" / "hackathon_ttc")
    ap.add_argument("--out", type=Path, default=None,
                    help="mặc định: <data>.zip cạnh thư mục dữ liệu")
    ap.add_argument("--depth-aux", type=Path, default=root / "datasets" / "depth_aux",
                    help="bộ depth phụ gộp vào cùng zip; --no-depth-aux để bỏ")
    ap.add_argument("--no-depth-aux", action="store_true")
    args = ap.parse_args()

    data: Path = args.data
    if not (data / "annotations" / "trips.csv").exists():
        raise SystemExit(f"không thấy {data / 'annotations' / 'trips.csv'}")
    out: Path = args.out or data.with_suffix(".zip")

    # (đường dẫn trong zip, file trên đĩa)
    items = [(p.relative_to(data).as_posix(), p)
             for p in sorted(data.rglob("*")) if p.is_file()]
    dx = args.depth_aux
    if not args.no_depth_aux and dx and dx.is_dir():
        extra = [("depth_aux/" + p.relative_to(dx).as_posix(), p)
                 for p in sorted(dx.rglob("*")) if p.is_file()]
        items += extra
        print(f"gộp thêm {len(extra)} file depth phụ từ {dx}")
    elif not args.no_depth_aux:
        print(f"KHÔNG có {dx} -> zip này train được nhưng head depth phụ sẽ TẮT "
              f"(chạy scripts/build_depth_aux.py trước nếu muốn bật)")

    total = sum(p.stat().st_size for _, p in items)
    print(f"{len(items)} file · {total / 1e6:.0f} MB -> {out}")

    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for k, (name, p) in enumerate(items, 1):
            ct = (zipfile.ZIP_STORED if p.suffix.lower() in STORE_EXT
                  else zipfile.ZIP_DEFLATED)
            zf.write(p, name, compress_type=ct)
            if k % 200 == 0 or k == len(items):
                print(f"  {k}/{len(items)}", flush=True)
    tmp.replace(out)

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        got = sum(i.file_size for i in zf.infolist())
    assert len(names) == len(items), f"zip có {len(names)} entry vs {len(items)} file"
    assert got == total, f"zip {got} byte vs nguồn {total} byte"
    assert "annotations/trips.csv" in names, "annotations/trips.csv không ở gốc zip"

    print(f"\nOK · {out.stat().st_size / 1e6:.0f} MB "
          f"({out.stat().st_size / total:.0%} so với nguồn) · {len(names)} entry")
    print(f"upload {out.name} lên Drive rồi đặt DATASET = \"{data.name}\" trong cell 1")


if __name__ == "__main__":
    main()
