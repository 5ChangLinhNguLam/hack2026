"""Fine-tune student_ttc trên nhãn min_ttc của hackathon.

Vì sao phải fine-tune lại chứ không dùng thẳng checkpoint cũ: hai nhãn khác NGHĨA.
Checkpoint cũ học "còn bao lâu tới vụ va chạm ĐÃ XẢY RA trong clip". Hackathon
chấm `min_ttc` = TTC động học tới target trong collision cone — hữu hạn cả khi
lái bình thường tiến gần xe trước, và không đòi hỏi có tai nạn nào. Đo trên 4 trip
chấm điểm, checkpoint cũ im lặng ở 9/11 event.

Dữ liệu train:
  datasets/practice          6 trip có GT thật, 20 fps -> lấy stride 2 còn 10 fps
  datasets/deepaccident_ttc  80 trip, 10 fps, min_ttc tính lại từ box 3D

Cả hai đều ở 10 Hz, khớp giả định thời gian của TSM + TCN (receptive field đo
bằng đơn vị frame).

Đích:
  hồi quy  inv_ttc = 1/max(min_ttc, 0.5), = 0 khi min_ttc vô hạn   (mọi frame)
  phân loại cls = min_ttc <= 2.0, bỏ qua vùng biên (2, 4)          (frame keep=1)
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import cv2

IMG_SIZE = (224, 224)


def load_student_defs(notebook: Path):
    """Lấy định nghĩa model NGUYÊN VĂN từ notebook để không lệch kiến trúc."""
    import nbformat
    nb = nbformat.read(str(notebook), as_version=4)
    src = next((c.source for c in nb.cells
                if c.cell_type == "code" and "class StudentTTC" in c.source), None)
    if src is None:
        raise SystemExit(f"không thấy class StudentTTC trong {notebook}")
    for cut in ("student = StudentTTC(", "\n_clip = torch.rand"):
        if cut in src:
            src = src.split(cut)[0]
            break
    ns = {"torch": torch, "np": np, "cv2": cv2, "Path": Path,
          "pip": lambda *a, **k: None, "__name__": "student_defs"}
    exec(src, ns)
    return ns


def read_ann(root: Path, stride: int):
    """-> {trip_id: dict(video=Path, inv=np.array, cls=, keep=, idx=)}"""
    ann = root / "annotations"
    if not (ann / "trips.csv").exists():
        return {}
    with (ann / "trips.csv").open(encoding="utf-8") as fh:
        trips = {r["trip_id"]: r for r in csv.DictReader(fh)}
    per = defaultdict(list)
    with (ann / "ttc_per_frame.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per[r["trip_id"]].append(r)
    out = {}
    for tid, rows in per.items():
        rows.sort(key=lambda r: int(r["frame_id"]))
        rows = rows[::stride]
        out[tid] = {
            "video": root / trips[tid]["video"],
            "idx": np.array([int(r["frame_id"]) for r in rows]),
            "inv": np.array([float(r["inv_ttc"]) for r in rows], np.float32),
            "cls": np.array([float(r["cls"]) for r in rows], np.float32),
            "keep": np.array([float(r["keep"]) for r in rows], np.float32),
        }
    return out


class TripSeq(Dataset):
    """Cắt đoạn L frame. Cache ảnh đã resize ra .npy: giải mã video bằng cv2 trên
    CPU mới là nút cổ chai, không phải phép nhân ma trận."""

    def __init__(self, items, cache: Path, L=24, full=False):
        self.items = items
        self.keys = sorted(items)
        self.cache = cache
        self.L = L
        self.full = full
        cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.keys)

    def _frames(self, tid):
        p = self.cache / f"{tid}.npy"
        if not p.exists():
            it = self.items[tid]
            cap = cv2.VideoCapture(str(it["video"]))
            all_f = []
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                all_f.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), IMG_SIZE))
            cap.release()
            sel = [all_f[i] for i in it["idx"] if i < len(all_f)]
            tmp = p.with_suffix(f".{id(self)}.tmp.npy")
            np.save(tmp, np.stack(sel))
            tmp.replace(p)
        return np.load(p, mmap_mode="r")

    def __getitem__(self, i):
        tid = self.keys[i]
        it = self.items[tid]
        ims = self._frames(tid)
        n = min(len(ims), len(it["inv"]))
        if self.full:
            sl = np.arange(n)
        else:
            s = int(np.random.randint(0, max(1, n - self.L)))
            sl = np.arange(s, min(s + self.L, n))
            if len(sl) < self.L:
                sl = np.concatenate([np.full(self.L - len(sl), sl[0]), sl])
        clip = torch.from_numpy(np.ascontiguousarray(ims[sl])).permute(0, 3, 1, 2).float() / 255.0
        return {"clip": clip,
                "inv": torch.from_numpy(it["inv"][sl]),
                "cls": torch.from_numpy(it["cls"][sl]),
                "keep": torch.from_numpy(it["keep"][sl]),
                "trip_id": tid}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    errs = []
    for b in loader:
        logit, inv = model(b["clip"].to(device))
        p = torch.sigmoid(logit)[0].cpu()
        inv = inv[0].cpu()
        cls, keep, inv_t = b["cls"][0], b["keep"][0], b["inv"][0]
        pred = (p >= 0.5) & (keep > 0)
        true = (cls > 0) & (keep > 0)
        tp += int((pred & true).sum())
        fp += int((pred & ~true).sum())
        fn += int((~pred & true).sum())
        errs.append((inv - inv_t).abs())
    model.train()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, float(torch.cat(errs).mean()), tp, fp, fn


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", type=Path,
                    default=root / "notebooks" / "explore_combined.ipynb")
    ap.add_argument("--init", type=Path, default=root / "student_ttc.pth")
    ap.add_argument("--out", type=Path, default=root / "student_ttc_hackathon.pth")
    ap.add_argument("--cache", type=Path, default=root / ".frame_cache_ttc")
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--val-trips", nargs="*", default=["T05-Sample", "T06-Sample"])
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ns = load_student_defs(args.notebook)

    practice = read_ann(root / "datasets" / "practice", stride=2)      # 20 -> 10 fps
    deepacc = read_ann(root / "datasets" / "deepaccident_ttc", stride=1)
    print(f"practice {len(practice)} trip · deepaccident {len(deepacc)} trip")

    val_items = {k: v for k, v in practice.items() if k in args.val_trips}
    train_items = {k: v for k, v in practice.items() if k not in args.val_trips}
    train_items.update(deepacc)
    if not val_items:
        raise SystemExit(f"không thấy trip val {args.val_trips}")

    tr = TripSeq(train_items, args.cache, L=args.seq_len)
    va = TripSeq(val_items, args.cache, full=True)
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=1, shuffle=False)

    n1 = sum(float((v["cls"] * v["keep"]).sum()) for v in train_items.values())
    n0 = sum(float(((1 - v["cls"]) * v["keep"]).sum()) for v in train_items.values())
    pos_w = torch.tensor(max(n0, 1.0) / max(n1, 1.0), device=device)
    print(f"train {len(tr)} trip · val {len(va)} trip · "
          f"frame nguy hiểm {n1:.0f} / an toàn {n0:.0f} -> pos_weight {float(pos_w):.1f}")

    model = ns["StudentTTC"](pretrained=False).to(device)
    if args.init.exists():
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"khởi tạo từ {args.init.name}")
    model.train()

    opt = torch.optim.AdamW([
        {"params": model.net.parameters(), "lr": 5e-5},
        {"params": [*model.tcn.parameters(), *model.head_cls.parameters(),
                    *model.head_ttc.parameters()], "lr": 5e-4},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, args.epochs * len(tr_dl)))

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        tot = c_s = t_s = nb = 0
        for b in tr_dl:
            clip = b["clip"].to(device)
            cls, keep, inv = (b[k].to(device) for k in ("cls", "keep", "inv"))
            logit, pred_inv = model(clip)
            l_cls = (F.binary_cross_entropy_with_logits(
                logit, cls, pos_weight=pos_w, reduction="none") * keep
            ).sum() / keep.sum().clamp(min=1)
            l_ttc = ((pred_inv - inv) ** 2).mean()
            loss = l_cls + l_ttc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()); c_s += float(l_cls.detach())
            t_s += float(l_ttc.detach()); nb += 1; step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"smoke test: {step} step OK, {time.time() - t0:.0f}s")
                return
        f1, mae, tp, fp, fn = evaluate(model, va_dl, device)
        print(f"epoch {ep + 1}/{args.epochs} · loss {tot / nb:.4f} "
              f"(cls {c_s / nb:.3f} · ttc {t_s / nb:.4f}) | "
              f"val F1 {f1:.3f} (tp{tp} fp{fp} fn{fn}) · MAE 1/TTC {mae:.4f} "
              f"| {time.time() - t0:.0f}s")

    torch.save({"state": model.state_dict(), "backbone": ns["BACKBONE"],
                "shift_at": ns["SHIFT_AT"], "tcn_dil": ns["TCN_DIL"],
                "pos_ttc": 2.0, "neg_ttc": 4.0,
                "label": "hackathon_min_ttc", "fps": 10},
               args.out)
    print("đã lưu", args.out)


if __name__ == "__main__":
    main()
