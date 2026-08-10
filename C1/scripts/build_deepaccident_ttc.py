"""Chuyển DeepAccident sang ĐÚNG schema nhãn của hackathon (min_ttc động học).

Vì sao dùng được: DeepAccident cũng là CARLA, cùng họ map với bộ chấm điểm, và
label của nó có box 3D + id ổn định trong hệ quy chiếu ego — đủ để tính lại toàn
bộ chuỗi rel_pos -> closing_speed -> ttc_simple -> min_ttc.

BỐN ĐIỀU ĐÃ KIỂM CHỨNG BẰNG DỮ LIỆU trước khi viết file này:

1. Quy tắc cone suy ngược từ Practice_Dataset: `x > 0 và |y| <= 2.5 m`
   — khớp 100% trên 21.810 target-frame, và tái tạo cột min_ttc lệch 0/3600 frame.
2. Trục của DeepAccident: x là DỌC (vận tốc ego trong hệ ego có vx trung vị
   +9.25 m/s, |vy| 0.04) — trùng quy ước hackathon.
3. Vận tốc trong label DeepAccident nằm ở hệ WORLD, KHÔNG phải hệ ego
   (cos giữa vận tốc ego và dịch chuyển vật tĩnh ≈ 0, không phải −1). Vì vậy
   KHÔNG trừ trực tiếp vận tốc để ra vận tốc tương đối.
4. Thay vào đó sai phân vị trí trong hệ ego. Đo trên Practice (nơi có cả rel_pos
   lẫn rel_velocity thật): với target TRONG CONE sai số dọc chỉ 0.016 m/s
   (p90 0.27) — trong khi tính trên MỌI target thì sai tới 1.21 (p90 22.1) vì
   số hạng quay của ego. Đúng những target quyết định min_ttc thì cách này chuẩn.

Đầu ra khớp cột với datasets/practice để nối thẳng được:
  datasets/deepaccident_ttc/trips/<trip>.mp4
  datasets/deepaccident_ttc/annotations/{trips,ttc_per_frame}.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

import weather_features as wx

# console Windows mặc định cp1252, không in nổi tiếng Việt trong log
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

FPS = 10.0
DT = 1.0 / FPS
CONE_HALF_WIDTH = 2.5
EGO_LEN_DEFAULT = 4.7          # chỉ dùng khi label thiếu dòng ego (id -100)
POS_TTC, NEG_TTC = 2.0, 4.0
TTC_FLOOR = 0.5
OUT_SIZE = (640, 360)          # cùng độ phân giải với practice; 16:9 -> chỉ thu nhỏ

# camera cố định bên đường: "ego" không di chuyển nên TTC không có nghĩa lái xe
AGENTS = ["ego_vehicle", "ego_vehicle_behind", "other_vehicle", "other_vehicle_behind"]


def parse_label(path: Path):
    """-> (ego_vel, ego_len, {id: (x, y, length)}). x dọc, y ngang, hệ ego.

    Trường: class x y z l w h yaw vx vy id num_lidar_pts visibility
    (kiểm bằng dữ liệu: car 4.18x1.99x1.38, truck 8.47x2.89x3.83 -> index 4 = DÀI)
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return (0.0, 0.0), EGO_LEN_DEFAULT, {}
    head = lines[0].split(" ")
    ego = (float(head[0]), float(head[1]))
    ego_len = EGO_LEN_DEFAULT
    objs = {}
    for ln in lines[1:]:
        f = ln.split(" ")
        if len(f) <= 1:
            continue
        vid = int(f[-3])
        if vid == -100:                 # chính agent đang ghi hình
            ego_len = float(f[4])
            continue
        objs[vid] = (float(f[1]), float(f[2]), float(f[4]))
    return ego, ego_len, objs


def parse_meta(path: Path):
    """-> (weather, has_collision, {agent: ego_involved}, mô tả cặp đâm nhau).

    Dòng đầu: <weather+time> <id1> <type1> <id2> <type2> <impulse> <dir> <impact> <end>
    `id1 == -1` = kịch bản kết thúc mà KHÔNG có va chạm nào — và điều này xảy ra
    với cả scenario tên `*_accident` (đo được: 77/104 scenario không hề va chạm),
    nên KHÔNG được suy ra va chạm từ tên thư mục.

    `agents id:` liệt kê id theo ĐÚNG thứ tự AGENTS. Một agent "có dính" khi id
    của nó là một trong hai vật va chạm. Va chạm với xe nền hoặc vật tĩnh thì id
    kia không nằm trong danh sách agent — lúc đó chỉ một agent được đánh dấu,
    khớp với dòng `colliding agents: ego none` của chính bộ dữ liệu.
    """
    ls = path.read_text(encoding="utf-8").splitlines()
    t = ls[0].split(" ")
    o1, o2 = int(t[1]), int(t[3])
    ids = []
    for ln in ls[1:]:
        if "agents id:" in ln:
            ids = [int(x) for x in ln.split(": ")[1].split(" ")]
    has = o1 != -1
    inv = {a: int(has and (ids[i] if i < len(ids) else None) in (o1, o2))
           for i, a in enumerate(AGENTS)}
    pair = "" if not has else "+".join(a for a in AGENTS if inv[a]) or "non_agent"
    return t[0], has, inv, pair


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def frame_targets(min_ttc: float):
    if not math.isfinite(min_ttc):
        return 0.0, 1.0, 0.0
    inv = 1.0 / max(min_ttc, TTC_FLOOR)
    if min_ttc <= POS_TTC:
        return 1.0, 1.0, inv
    if min_ttc >= NEG_TTC:
        return 0.0, 1.0, inv
    return 0.0, 0.0, inv


def lateral_accels(vels):
    """[(vx, vy) world] -> gia tốc NGANG mỗi frame (m/s²), a_lat = v · dpsi/dt.

    Practice/Redacted có sẵn `ego.lateral_accel` từ IMU của CARLA; DeepAccident
    không có trường nào tương đương nên phải suy từ hướng vector vận tốc. Hai
    cách không cho ra cùng một con số ở mức chi tiết, nên phía model chỉ dùng
    |a_lat| đã bị tanh nén — "đang vào cua gắt hay không" thì hai cách đồng ý,
    còn dấu và biên độ tuyệt đối thì không.

    Ba lớp khử nhiễu, tất cả đều cần thiết — bản đầu tiên KHÔNG có chúng cho ra
    |a_lat| p95 = 12.9 m/s² (tức 1.3 g ngang, xe con không làm nổi), trong khi
    practice đo bằng IMU chỉ quanh 0.02-0.46 m/s²:
      - MIN_SPEED: atan2 của vector gần 0 là nhiễu thuần tuý (xe đứng yên vẫn ra
        yaw_rate hàng radian/s).
      - Sai phân hướng qua LAG frame thay vì 1: nhiễu hướng chia cho khoảng thời
        gian dài gấp LAG lần. Vẫn LÙI hoàn toàn, không mượn frame tương lai.
      - Chặn ±A_MAX: giới hạn bám đường thực tế của lốp.
    """
    MIN_SPEED, LAG, A_MAX = 2.0, 3, 10.0
    out = [0.0] * len(vels)
    for i in range(LAG, len(vels)):
        v0, v1 = vels[i - LAG], vels[i]
        s0, s1 = math.hypot(*v0), math.hypot(*v1)
        if s0 < MIN_SPEED or s1 < MIN_SPEED:
            continue
        d = math.atan2(v1[1], v1[0]) - math.atan2(v0[1], v0[0])
        d = (d + math.pi) % (2 * math.pi) - math.pi        # về [-pi, pi]
        out[i] = max(-A_MAX, min(A_MAX, s1 * d / (LAG * DT)))
    return out


def compute_min_ttc(prev, cur, nxt, ego_len):
    """min_ttc tại frame hiện tại. Sai phân TRUNG TÂM khi có đủ 2 phía.

    PHẢI TRỪ ĐỆM. `ttc_simple` của ban tổ chức không phải x/closing mà là
    (x - đệm)/closing, với đệm = nửa dài ego + nửa dài target — tức khoảng cách
    giữa hai đầu xe chứ không phải giữa hai tâm. Đo trên Practice: đệm đúng bằng
    4.731 m cho vehicle, 3.491 cho bike, 2.584 cho walker.

    Bỏ số hạng này thì TTC luôn LỚN HƠN thật và nhãn nguy hiểm bị bỏ sót 75%
    (F1 0.394). Trừ vào rồi thì F1 lên 0.946 và sai số ở vùng TTC<3s giảm từ
    0.909s còn 0.121s. (Khử thêm phần quay của hệ ego không cải thiện: 0.944.)
    """
    best = float("inf")
    n_cone = 0
    for vid, (x, y, olen) in cur.items():
        if x <= 0 or abs(y) > CONE_HALF_WIDTH:
            continue
        n_cone += 1
        if nxt is not None and vid in nxt and prev is not None and vid in prev:
            dxdt = (nxt[vid][0] - prev[vid][0]) / (2 * DT)
        elif nxt is not None and vid in nxt:
            dxdt = (nxt[vid][0] - x) / DT
        elif prev is not None and vid in prev:
            dxdt = (x - prev[vid][0]) / DT
        else:
            continue                     # mới xuất hiện, chưa suy được tốc độ
        closing = -dxdt                  # >0 nghĩa là đang lại gần
        if closing <= 0:
            continue
        gap = x - 0.5 * (ego_len + olen)
        best = min(best, gap / closing if gap > 0 else 0.0)
    return best, n_cone


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=root / "datasets" / "deepaccident" / "raw" / "mini")
    ap.add_argument("--out", type=Path, default=root / "datasets" / "deepaccident_ttc")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="giữ nguyên mp4 đã có ĐỦ số frame; nhãn vẫn tính lại hết. "
                         "Dùng khi thêm zip mới: khỏi mã hoá lại hàng trăm clip cũ")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    (args.out / "trips").mkdir(parents=True, exist_ok=True)
    ann = args.out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)

    metas = sorted(args.src.glob("*/meta/*.txt"))
    if args.limit:
        metas = metas[:args.limit]
    if not metas:
        raise SystemExit(f"không thấy scenario nào trong {args.src}")

    tf = (ann / "trips.csv").open("w", newline="", encoding="utf-8")
    ff = (ann / "ttc_per_frame.csv").open("w", newline="", encoding="utf-8")
    tw, fw = csv.writer(tf), csv.writer(ff)
    tw.writerow(["trip_id", "video", "num_frames", "fps", "width", "height",
                 "map", "weather_desc", "speed_limit_kmh", "events",
                 "n_finite_ttc", "n_ttc_lt2", "n_ttc_lt15", "min_ttc_overall",
                 "driver_states", "safe_driving_score",
                 "ego_involved", "scenario_has_collision", "collision_pair",
                 *wx.COLS])
    fw.writerow(["trip_id", "frame_id", "timestamp", "min_ttc_s", "inv_ttc",
                 "cls", "keep", "headway_s", "ego_speed_kmh",
                 "ego_long_accel", "ego_lat_accel", "n_targets", "n_in_cone",
                 "driver_state", "alertness", "risk_final", "events_active"])

    n_trip = n_frame = n_kept = n_inv = 0
    for m in metas:
        stype, scen = m.parts[-3], m.stem
        weather, has_coll, involved, pair = parse_meta(m)
        for agent in AGENTS:
            lab_dir = args.src / stype / agent / "label" / scen
            cam_dir = args.src / stype / agent / "Camera_Front" / scen
            labs = sorted(lab_dir.glob("*.txt"))
            imgs = sorted(cam_dir.glob("*.jpg"))
            if not labs or not imgs:
                continue

            parsed = [parse_label(p) for p in labs]
            vels = [e for e, _, _ in parsed]
            speeds = [math.hypot(*e) for e in vels]
            lat_acc = lateral_accels(vels)
            ego_lens = [el for _, el, _ in parsed]
            objs = [o for _, _, o in parsed]

            trip_id = f"{stype}__{scen}__{agent}"
            vid_path = args.out / "trips" / f"{trip_id}.mp4"
            n_expect = min(len(labs), len(imgs))

            # mp4 cũ chỉ được giữ khi giải mã ra ĐỦ n_expect frame — file dở dang
            # từ lần chạy bị ngắt mà vẫn tin là xong thì nhãn lệch âm thầm
            skip = (args.skip_existing and vid_path.exists()
                    and video_frame_count(vid_path) == n_expect)
            n_kept += skip
            writer = None
            if not args.no_video and not skip:
                # unlink TRƯỚC: mp4 ở đây được hardlink sang các bộ đã đóng gói,
                # ghi đè lên hardlink là sửa thẳng inode dùng chung -> hỏng luôn
                # video bên bộ kia mà không có dấu hiệu gì
                vid_path.unlink(missing_ok=True)
                writer = cv2.VideoWriter(str(vid_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         FPS, OUT_SIZE)

            n_fin = n_lt2 = n_lt15 = 0
            best_all = float("inf")
            n = min(len(labs), len(imgs))
            for i in range(n):
                prev = objs[i - 1] if i > 0 else None
                nxt = objs[i + 1] if i + 1 < len(objs) else None
                mt, n_cone = compute_min_ttc(prev, objs[i], nxt, ego_lens[i])
                cls, keep, inv = frame_targets(mt)
                accel = ((speeds[i] - speeds[i - 1]) / DT) if i > 0 else 0.0

                n_fin += math.isfinite(mt)
                n_lt2 += mt < POS_TTC
                n_lt15 += mt < 1.5
                best_all = min(best_all, mt)

                fw.writerow([
                    trip_id, i, round(i / FPS, 3),
                    "inf" if not math.isfinite(mt) else round(mt, 4), round(inv, 6),
                    int(cls), int(keep), "inf",
                    round(speeds[i] * 3.6, 3), round(accel, 3),
                    round(lat_acc[i], 3),
                    len(objs[i]), n_cone, "", "", "", "",
                ])
                if writer is not None:
                    im = cv2.imread(str(imgs[i]))
                    if im is not None:
                        writer.write(cv2.resize(im, OUT_SIZE, interpolation=cv2.INTER_AREA))
                n_frame += 1

            if writer is not None:
                writer.release()

            tw.writerow([
                trip_id, f"trips/{trip_id}.mp4", n, FPS, OUT_SIZE[0], OUT_SIZE[1],
                scen.split("_")[0], weather, "", stype,
                n_fin, n_lt2, n_lt15,
                "inf" if not math.isfinite(best_all) else round(best_all, 3), "", "",
                involved[agent], int(has_coll), pair,
                *(wx.from_preset(weather)[c] for c in wx.COLS),
            ])
            n_trip += 1
            n_inv += involved[agent]

        print(f"  {stype}/{scen}: xong {len(AGENTS)} agent"
              + (f" · va chạm: {pair}" if has_coll else " · không va chạm"))

    tf.close()
    ff.close()
    print(f"\n{n_trip} trip · {n_frame} frame -> {args.out}")
    print(f"{n_inv} trip có CHÍNH xe mang cam nằm trong vụ va chạm (ego_involved=1)")
    if args.skip_existing:
        print(f"{n_kept} video giữ nguyên, không mã hoá lại")


if __name__ == "__main__":
    main()
