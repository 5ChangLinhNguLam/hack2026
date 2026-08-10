"""Bộ dữ liệu phụ cho auxiliary task DỰ ĐOÁN ĐỘ SÂU.

Vì sao làm task phụ chứ không đưa depth vào input
-------------------------------------------------
`kitti/depth/*.npy` là depth GT dày đặc (mét), có ở CẢ Practice_Dataset lẫn
Hackathon_Dataset_Redacted — tức lúc chấm vẫn còn (đã kiểm: T01d có 360 file,
min 1.79 m, max 999.9). Nhưng DeepAccident KHÔNG có cặp stereo và cũng chưa giải
nén lidar, nên không thể tạo kênh depth cho ~99% dữ liệu train. Đưa depth thành
kênh input thứ 4 sẽ khiến kênh đó rỗng ở gần như toàn bộ tập train — vô dụng.

Ngược lại, làm ĐÍCH của một head phụ thì chỉ cần depth ở phần dữ liệu có nó:
loss được mask, DeepAccident đóng góp 0 vào nhánh này. Head phụ ép backbone mã
hoá THANG ĐO MÉT — đúng thứ TTC cần mà ảnh RGB đơn thuần không ràng buộc — rồi
bị vứt bỏ lúc export, nên không tốn một mili-giây latency nào.

Bộ này TỰ CHỨA (ảnh + depth trong cùng file .npz), cố ý không dựa vào
datasets/hackathon_ttc_*: nhờ vậy nó nạp được cả trip của Hackathon_Dataset_Redacted,
nơi có depth thật nhưng KHÔNG có nhãn TTC nào để nằm trong bộ train chính.

⚠️  --sources redacted: 10 trip chấm điểm có depth GT hợp lệ và dùng nó cho task
    phụ KHÔNG rò rỉ nhãn TTC (nhãn đó đã bị xoá khỏi bộ redacted). Nhưng nhiều
    cuộc thi CẤM chạm vào tập chấm điểm dưới mọi hình thức. Mặc định TẮT; bật
    hay không là quyết định của bạn theo thể lệ, không phải mặc định kỹ thuật.

Đầu ra
  datasets/depth_aux/index.csv           1 dòng / mẫu
  datasets/depth_aux/<trip>.npz          jpg (uint8, đã mã hoá) + depth (h,w) f16

Khớp thời gian với bộ chính: bộ chính lấy stride 2 (20 fps -> 10 fps) nên chỉ
giữ frame CHẴN; depth chỉ có ở frame chia hết cho 5. Giao của hai điều kiện là
frame chia hết cho 10 -> 60 mẫu/trip practice, 180 mẫu/trip redacted.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

IMG_SIZE = (224, 224)        # (W, H) — phải trùng input backbone lúc train
DEPTH_HW = (28, 28)          # (H, W) lưới đích của head phụ
DEPTH_MIN, DEPTH_MAX = 1.0, 80.0   # mét; ngoài dải này thì nghịch đảo bão hoà
STRIDE = 2                   # 20 fps -> 10 fps, y hệt pack_hackathon_dataset.py


def downsample_depth(d):
    """(360,640) mét -> (28,28) mét.

    INTER_AREA chứ không phải resize mặc định: lấy TRUNG BÌNH vùng. Nhưng trung
    bình trên trời (~1000 m) lẫn với xe (~10 m) cho ra một con số chẳng tả gì.
    Nên chặn trước rồi mới thu nhỏ TRÊN NGHỊCH ĐẢO — trung bình của 1/d mới là
    đại lượng tuyến tính theo disparity, và đó cũng là thứ head phụ hồi quy.
    """
    inv = 1.0 / np.clip(d, DEPTH_MIN, DEPTH_MAX)
    small = cv2.resize(inv.astype(np.float32), DEPTH_HW[::-1],
                       interpolation=cv2.INTER_AREA)
    return (1.0 / np.clip(small, 1.0 / DEPTH_MAX, 1.0 / DEPTH_MIN)).astype(np.float16)


def trip_dirs(root: Path):
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "kitti").is_dir())


def build_trip(d: Path, out: Path, tag: str):
    depth_dir, img_dir = d / "kitti" / "depth", d / "kitti" / "image_2"
    if not depth_dir.is_dir() or not img_dir.is_dir():
        return []

    jpgs, deps, rows = [], [], []
    for dp in sorted(depth_dir.glob("*.npy")):
        fid = int(dp.stem)
        if fid % STRIDE:                       # frame lẻ không tồn tại ở 10 fps
            continue
        ip = img_dir / f"{fid:06d}.jpg"
        if not ip.exists():
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        im = cv2.resize(im, IMG_SIZE, interpolation=cv2.INTER_AREA)
        # nén lại thành jpg: bộ này chỉ để regularize, mất mát nén ở q=92 nhỏ hơn
        # nhiều so với sai số của chính head phụ, mà đĩa thì giảm ~20 lần
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        jpgs.append(buf.astype(np.uint8))
        deps.append(downsample_depth(np.load(dp)))
        rows.append({"trip_id": f"{tag}__{d.name}", "source": tag,
                     "frame_id": fid, "frame_idx": fid // STRIDE,
                     "idx": len(jpgs) - 1})

    if not jpgs:
        return []
    np.savez_compressed(
        out / f"{tag}__{d.name}.npz",
        # ảnh jpg dài ngắn khác nhau -> object array, không phải mảng chữ nhật
        jpg=np.array(jpgs, dtype=object),
        depth=np.stack(deps),
        frame_id=np.array([r["frame_id"] for r in rows], dtype=np.int32),
    )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=root / "datasets" / "depth_aux")
    ap.add_argument("--sources", nargs="*", default=["practice"],
                    choices=["practice", "redacted"],
                    help="'redacted' dùng depth của 10 trip CHẤM ĐIỂM — hợp lệ về "
                         "mặt nhãn nhưng có thể trái thể lệ; đọc cảnh báo ở đầu file")
    ap.add_argument("--practice-src", type=Path,
                    default=root / "datasets" / "Practice_dataset")
    ap.add_argument("--redacted-src", type=Path,
                    default=root / "datasets" / "Hackathon_Dataset_Redacted")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = []
    if "practice" in args.sources:
        todo.append(("practice", args.practice_src))
    if "redacted" in args.sources:
        todo.append(("redacted", args.redacted_src))

    all_rows = []
    for tag, src in todo:
        if not src.is_dir():
            print(f"  ! bỏ qua {tag}: không thấy {src}")
            continue
        for d in trip_dirs(src):
            rows = build_trip(d, args.out, tag)
            all_rows += rows
            print(f"  {tag}/{d.name}: {len(rows)} mẫu depth")

    if not all_rows:
        raise SystemExit("không dựng được mẫu nào — kiểm tra --practice-src")

    with (args.out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trip_id", "source", "frame_id",
                                           "frame_idx", "idx"])
        w.writeheader()
        w.writerows(all_rows)

    mb = sum(p.stat().st_size for p in args.out.glob("*.npz")) / 1e6
    per = {}
    for r in all_rows:
        per[r["source"]] = per.get(r["source"], 0) + 1
    print(f"\n{len(all_rows)} mẫu ({' · '.join(f'{k} {v}' for k, v in per.items())})"
          f" · {mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
