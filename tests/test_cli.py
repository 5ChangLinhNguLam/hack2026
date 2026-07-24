"""Phase 4 — CLI replay + validate.

DoD (spec mục 4): validate chạy trên cả 6 trip mẫu pass.
"""

import gzip
import re

import pytest

from conftest import DATA_DIR, N_FAKE_FRAMES, make_full_trip
from tripkit import replay as replay_cli
from tripkit import validate as validate_cli


# ---------------------------------------------------------------------- #
# python -m tripkit.replay
# ---------------------------------------------------------------------- #
def test_replay_cli_fast_limit_stats(t01_dir, capsys):
    rc = replay_cli.main([str(t01_dir), "--mode", "fast", "--limit", "50", "--stats"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "50/600" in out            # số frame đã phát / tổng
    assert "fps đọc" in out           # fps đọc được
    assert re.search(r"có GT\s*: có", out)   # đúng cực tính GT, không chỉ có nhãn
    assert "T01-Sample" in out


def test_replay_cli_start_end(t01_dir, capsys):
    rc = replay_cli.main([str(t01_dir), "--start", "10", "--end", "20"])
    assert rc == 0
    assert "10/600" in capsys.readouterr().out


def test_replay_cli_limit_clamped_to_trip_end(t01_dir, capsys):
    # --limit vượt cuối trip phải được kẹp lại, không nổ "Khoảng phát không hợp lệ"
    rc = replay_cli.main([str(t01_dir), "--start", "580", "--limit", "50"])
    assert rc == 0
    assert "20/600" in capsys.readouterr().out


def test_replay_cli_limit_end_min_wins(t01_dir, capsys):
    rc = replay_cli.main([str(t01_dir), "--limit", "30", "--end", "20"])
    assert rc == 0
    assert "20/600" in capsys.readouterr().out


def test_replay_cli_bad_dir(tmp_path, capsys):
    rc = replay_cli.main([str(tmp_path / "khong-ton-tai")])
    assert rc == 2


def test_replay_cli_corrupt_json_no_traceback(tmp_path, capsys):
    trip = tmp_path / "T91d"
    trip.mkdir()
    (trip / "T91d.json").write_text("{hong json")
    assert replay_cli.main([str(trip)]) == 2
    # gz cụt (download hỏng) cũng phải báo gọn, không traceback
    trip2 = tmp_path / "T90d"
    trip2.mkdir()
    with gzip.open(trip2 / "T90d.json.gz", "wb") as f:
        f.write(b'{"trip_id"')
    (trip2 / "T90d.json.gz").write_bytes(
        (trip2 / "T90d.json.gz").read_bytes()[:10]
    )
    assert replay_cli.main([str(trip2)]) == 2


def test_replay_cli_stats_without_calibration_info(redacted_trip_dir, capsys):
    # --stats trên trip thiếu calibration_info.txt: vẫn rc 0, calib báo gọn
    rc = replay_cli.main([str(redacted_trip_dir), "--stats"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "không đọc được" in out


def test_replay_cli_bad_args(t01_dir, capsys):
    assert replay_cli.main([str(t01_dir), "--start", "-5"]) == 2
    assert replay_cli.main([str(t01_dir), "--speed", "0"]) == 2


# ---------------------------------------------------------------------- #
# python -m tripkit.validate — trên dữ liệu thật
# ---------------------------------------------------------------------- #
def test_validate_trip_t01(t01_dir):
    v = validate_cli.validate_trip(t01_dir)
    assert v.ok, v.errors
    assert v.n_frames == 600
    assert v.has_gt is True
    assert v.counts["image_2"] == 600
    assert v.counts["image_3"] == 600
    assert v.counts["driver"] == 600
    assert v.counts["depth"] == 120
    assert v.counts["calib"] == 600
    assert v.counts["label_nonempty"] == 57       # T01: 57 frame có nhãn Pedestrian
    assert any("label rỗng" in w for w in v.warnings)


def test_validate_cli_all_sample_trips(t01_dir, capsys):
    # DoD: validate cả data/ phải pass. Số trip lấy động — sau này thêm
    # T0Xd vào data/ thì test vẫn đúng.
    n = len(validate_cli._find_trip_dirs(DATA_DIR))
    rc = validate_cli.main([str(DATA_DIR)])
    out = capsys.readouterr().out
    assert rc == 0
    assert n >= 6
    assert f"{n}/{n} trip đạt" in out


def test_validate_cli_redacted_trip_fails(redacted_trip_dir, capsys):
    # Trip giả lập không có thư mục ảnh → FAIL nhưng không crash
    rc = validate_cli.main([str(redacted_trip_dir)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out


def test_validate_cli_empty_dir(tmp_path, capsys):
    assert validate_cli.main([str(tmp_path)]) == 2


# ---------------------------------------------------------------------- #
# validate — từng check riêng lẻ trên trip đầy đủ 7 frame (khác 600/1800!)
# ---------------------------------------------------------------------- #
def test_validate_full_trip_nonstandard_frame_count(full_trip_dir):
    v = validate_cli.validate_trip(full_trip_dir)
    assert v.ok, v.errors
    assert v.n_frames == N_FAKE_FRAMES
    assert v.counts["image_2"] == N_FAKE_FRAMES
    assert v.counts["depth"] == (N_FAKE_FRAMES + 4) // 5   # ceil(7/5) = 2
    assert v.has_gt is False                                # GT=không, không phải lỗi


def test_validate_detects_divergent_calib(full_trip_dir):
    (full_trip_dir / "kitti" / "calib" / "000003.txt").write_text("KHAC BIET\n")
    v = validate_cli.validate_trip(full_trip_dir)
    assert not v.ok
    assert any("calib không đồng nhất" in e for e in v.errors)


def test_validate_detects_single_missing_image(full_trip_dir):
    (full_trip_dir / "kitti" / "image_2" / "000003.jpg").unlink()
    v = validate_cli.validate_trip(full_trip_dir)
    assert not v.ok
    assert any("image_2" in e and "thiếu" in e for e in v.errors)


def test_validate_stray_file_does_not_mask_missing(full_trip_dir):
    # File lạ không được che file thiếu (check theo tên, không phải đếm)
    (full_trip_dir / "kitti" / "image_2" / "000003.jpg").unlink()
    (full_trip_dir / "kitti" / "image_2" / "stray.jpg").write_bytes(b"x")
    v = validate_cli.validate_trip(full_trip_dir)
    assert not v.ok
    assert any("image_2" in e and "000003" in e for e in v.errors)
    assert any("file lạ" in w for w in v.warnings)


def test_validate_detects_wrong_image_size(full_trip_dir):
    import cv2
    import numpy as np
    cv2.imwrite(str(full_trip_dir / "kitti" / "image_2" / "000000.jpg"),
                np.zeros((100, 200, 3), np.uint8))
    v = validate_cli.validate_trip(full_trip_dir)
    assert not v.ok
    assert any("kích thước" in e for e in v.errors)


def test_validate_empty_frames_does_not_crash_run(tmp_path, full_trip_dir, capsys):
    # 1 trip JSON rỗng frames (nhưng có ảnh + calib_info) không được làm sập
    # cả run — trip đó FAIL, trip lành bên cạnh vẫn được validate
    import shutil
    root = tmp_path / "mixed"
    root.mkdir()
    shutil.move(str(full_trip_dir), str(root / full_trip_dir.name))
    broken = make_full_trip(root, "T89d", 3)
    with gzip.open(broken / "T89d.json.gz", "wt", encoding="utf-8") as f:
        f.write('{"trip_id": "T89d", "metadata": {"fps": 20}, "frames": []}')
    rc = validate_cli.main([str(root)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "1/2 trip đạt" in out
    assert "JSON không có frame nào" in out


def test_validate_json_named_differently(tmp_path):
    # JSON không trùng tên thư mục: loader đọc được thì validate cũng phải thấy
    trip = make_full_trip(tmp_path, "T88d", 3, json_name="telemetry.json.gz")
    v = validate_cli.validate_trip(trip)
    assert v.ok, v.errors
    dirs = validate_cli._find_trip_dirs(tmp_path)
    assert trip in dirs


def test_validate_dot_inside_trip_dir(full_trip_dir, capsys, monkeypatch):
    # `python -m tripkit.validate .` từ trong thư mục trip phải chạy được
    monkeypatch.chdir(full_trip_dir)
    rc = validate_cli.main(["."])
    assert rc == 0
    assert "1/1 trip đạt" in capsys.readouterr().out
