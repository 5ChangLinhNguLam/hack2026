#!/usr/bin/env python3
"""
safe_extract.py — Giải nén file .zip lớn một cách BỀN VỮNG.

Vì sao cần script này?
  `unzip` của Ubuntu (và trình giải nén GUI) sẽ ABORT toàn bộ khi gặp 1 file
  hỏng CRC. Nếu file hỏng nằm ở giữa archive (vd ~1.7 GB), bạn luôn fail đúng
  chỗ đó và mất toàn bộ phần còn lại — dù 99.99% file vẫn tốt.

Script này:
  - Giải nén TỪNG entry độc lập; 1 file lỗi không làm hỏng cả mẻ.
  - RESUME: chạy lại sẽ bỏ qua file đã giải nén đúng kích thước.
  - --salvage: cố cứu file hỏng CRC bằng cách ghi bytes giải nén được (bỏ CRC).
  - Báo cáo rõ: OK / skipped / FAILED, và ghi danh sách file hỏng ra log.
  - Chống Zip-Slip (path traversal) an toàn.

Ví dụ:
  python3 tools/safe_extract.py data/Hackathon_Dataset_Redacted.zip -d data/extracted
  python3 tools/safe_extract.py data/Hackathon_Dataset_Redacted.zip -d data/extracted --salvage
  python3 tools/safe_extract.py data/Hackathon_Dataset_Redacted.zip --list-bad   # chỉ kiểm tra
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
import zlib


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def safe_join(dest_dir: str, name: str) -> str:
    """Ghép path an toàn, chặn Zip-Slip (../ thoát ra ngoài dest)."""
    dest_dir = os.path.realpath(dest_dir)
    target = os.path.realpath(os.path.join(dest_dir, name))
    if target != dest_dir and not target.startswith(dest_dir + os.sep):
        raise ValueError(f"Bị chặn (đường dẫn thoát ra ngoài dest): {name!r}")
    return target


def common_top_dir(infos: list[zipfile.ZipInfo]) -> str | None:
    """Trả về tên thư mục gốc chung nếu MỌI entry đều nằm trong 1 folder duy
    nhất (vd 'Hackathon_Dataset_Redacted'); ngược lại None (kiểu 'tarbomb')."""
    tops = set()
    for i in infos:
        first = i.filename.replace("\\", "/").lstrip("/").split("/", 1)[0]
        if first:
            tops.add(first)
        if len(tops) > 1:
            return None
    return next(iter(tops)) if tops else None


def already_ok(target: str, info: zipfile.ZipInfo) -> bool:
    """Đã có file đúng kích thước chưa? (để resume, không giải nén lại)."""
    if info.is_dir():
        return os.path.isdir(target)
    try:
        return os.path.getsize(target) == info.file_size
    except OSError:
        return False


def salvage_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: str) -> int:
    """
    Cứu 1 entry hỏng CRC: đọc raw compressed stream và giải nén thủ công,
    ghi mọi byte lấy được (bỏ qua kiểm tra CRC). Trả về số byte ghi được.
    """
    # Đọc thẳng từ compressed bytes trong file zip để bỏ qua CRC hoàn toàn.
    with open(zf.filename, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError("Local header sai, không salvage được")
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        comp = fh.read(info.compress_size)

    if info.compress_type == zipfile.ZIP_STORED:
        data = comp
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        data = d.decompress(comp) + d.flush()
    else:
        raise ValueError(f"Không hỗ trợ salvage compress_type={info.compress_type}")

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "wb") as out:
        out.write(data)
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Giải nén zip lớn, bỏ qua file hỏng.")
    ap.add_argument("zip_path", help="Đường dẫn file .zip")
    ap.add_argument("-d", "--dest", default=None,
                    help="Thư mục đích (mặc định: cạnh file zip, tên = tên zip)")
    ap.add_argument("--salvage", action="store_true",
                    help="Cố cứu file hỏng CRC (ghi bytes giải nén được, bỏ CRC)")
    ap.add_argument("--list-bad", action="store_true",
                    help="Chỉ liệt kê file hỏng, KHÔNG giải nén")
    ap.add_argument("--no-resume", action="store_true",
                    help="Không bỏ qua file đã có (giải nén lại từ đầu)")
    args = ap.parse_args()

    zip_path = args.zip_path
    if not os.path.isfile(zip_path):
        print(f"[LỖI] Không thấy file zip: {zip_path}", file=sys.stderr)
        return 2

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        print(f"[LỖI] Zip hỏng nặng ở phần mục lục (central directory): {e}",
              file=sys.stderr)
        print("      -> Có thể file tải về bị thiếu. Hãy tải/copy lại.",
              file=sys.stderr)
        return 2

    infos = zf.infolist()

    # Chọn thư mục đích:
    #  - Nếu user chỉ định -d thì dùng nguyên.
    #  - Nếu archive có 1 thư mục gốc chung -> giải nén vào thư mục CHỨA zip
    #    (tránh lồng 2 lần: .../Name/Name/...).
    #  - Ngược lại (file rải ở gốc) -> tạo thư mục bao theo tên zip.
    if args.dest:
        dest = args.dest
    else:
        zip_dir = os.path.dirname(os.path.abspath(zip_path))
        if common_top_dir(infos) is not None:
            dest = zip_dir
        else:
            dest = os.path.join(
                zip_dir, os.path.splitext(os.path.basename(zip_path))[0])
    total = len(infos)
    total_bytes = sum(i.file_size for i in infos)
    print(f"Archive : {zip_path}  ({human(os.path.getsize(zip_path))})")
    print(f"Entries : {total:,}   Uncompressed: {human(total_bytes)}")

    if args.list_bad:
        print("\nĐang kiểm tra CRC toàn bộ (có thể mất vài phút)...")
        bad = []
        for i, info in enumerate(infos, 1):
            if info.is_dir():
                continue
            try:
                with zf.open(info) as fh:
                    while fh.read(1 << 20):
                        pass
            except Exception as e:  # noqa: BLE001
                bad.append((info.filename, str(e)))
                print(f"  [BAD] {info.filename}  ({e})")
            if i % 10000 == 0:
                print(f"  ...đã kiểm tra {i:,}/{total:,}")
        print(f"\nKết quả: {len(bad)} file hỏng / {total:,} entries.")
        return 0 if not bad else 1

    os.makedirs(dest, exist_ok=True)
    print(f"Dest    : {dest}")
    print(f"Resume  : {'OFF' if args.no_resume else 'ON'}   Salvage: {'ON' if args.salvage else 'OFF'}\n")

    ok = skipped = failed = salvaged = 0
    failed_list: list[tuple[str, str]] = []
    done_bytes = 0
    t0 = time.time()
    last = t0

    for idx, info in enumerate(infos, 1):
        try:
            target = safe_join(dest, info.filename)
        except ValueError as e:
            failed += 1
            failed_list.append((info.filename, str(e)))
            continue

        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue

        if not args.no_resume and already_ok(target, info):
            skipped += 1
            done_bytes += info.file_size
        else:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            try:
                with zf.open(info) as src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                ok += 1
                done_bytes += info.file_size
            except Exception as e:  # noqa: BLE001  (CRC/bad data -> đừng abort)
                if args.salvage:
                    try:
                        n = salvage_member(zf, info, target)
                        salvaged += 1
                        done_bytes += n
                        failed_list.append((info.filename, f"CRC lỗi -> đã salvage {human(n)}"))
                    except Exception as e2:  # noqa: BLE001
                        failed += 1
                        failed_list.append((info.filename, f"{e} | salvage fail: {e2}"))
                        if os.path.exists(target):
                            try:
                                os.remove(target)
                            except OSError:
                                pass
                else:
                    failed += 1
                    failed_list.append((info.filename, str(e)))
                    # Xoá file ghi dở để resume lần sau sạch sẽ.
                    if os.path.exists(target):
                        try:
                            os.remove(target)
                        except OSError:
                            pass

        now = time.time()
        if now - last >= 1.0 or idx == total:
            pct = 100.0 * idx / total
            rate = done_bytes / max(now - t0, 1e-6)
            sys.stdout.write(
                f"\r  {idx:,}/{total:,} ({pct:5.1f}%) | "
                f"OK {ok:,} skip {skipped:,} salv {salvaged} fail {failed} | "
                f"{human(done_bytes)} | {human(rate)}/s   "
            )
            sys.stdout.flush()
            last = now

    dt = time.time() - t0
    print(f"\n\n=== XONG trong {dt:.1f}s ===")
    print(f"  Giải nén mới : {ok:,}")
    print(f"  Bỏ qua (resume): {skipped:,}")
    if args.salvage:
        print(f"  Salvage được : {salvaged}")
    print(f"  THẤT BẠI     : {failed}")

    if failed_list:
        log = os.path.join(dest, "_extract_errors.log")
        with open(log, "w") as fh:
            for name, err in failed_list:
                fh.write(f"{name}\t{err}\n")
        print(f"\n  Chi tiết file lỗi/salvage -> {log}")
        for name, err in failed_list[:20]:
            print(f"    - {name}  ({err})")
        if len(failed_list) > 20:
            print(f"    ... và {len(failed_list) - 20} dòng nữa (xem log).")

    # Exit 0 nếu không còn file THẤT BẠI thực sự (salvage/known-bad không tính).
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
