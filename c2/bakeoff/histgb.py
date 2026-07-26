"""LOTO bake-off: HistGradientBoostingClassifier on c2 driver-state features.

Protocol:
- Leave-one-trip-out over the 6 sample trips.
- Train on the other 5 trips, dropping rows where face_found==0 (train only).
- No imputer/scaler: HistGB handles NaN natively.
- Predict all 600 frames of the held-out trip (NaN rows included).
- Metrics: accuracy over all 600; macro-F1 averaged only over classes present
  in the held-out trip's ground truth; composite = 100*(0.5*acc + 0.5*mf1).
- Smoothed variant: sliding-window majority vote, window +/-25 frames.

Grid: max_iter in {100, 300} x max_depth in {None, 3}. Deterministic
(random_state=0). Best combo chosen by mean smoothed composite.
"""

import os
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
TRIPS = [f"T{i:02d}-Sample" for i in range(1, 7)]
CLASSES = ["alert", "distracted", "drowsy", "microsleep", "yawning"]


def load_trip(trip):
    d = np.load(os.path.join(CACHE, f"features_{trip}.npz"), allow_pickle=True)
    X = d["features"].astype(np.float64)
    names = [str(n) for n in d["names"]]
    y = np.load(os.path.join(CACHE, f"labels_{trip}.npy"), allow_pickle=True)
    y = np.array([str(v) for v in y])
    return X, y, names


def smooth_majority(pred, half_window=25):
    """Sliding-window majority vote over +/-half_window frames.

    Ties broken deterministically by first occurrence order within the window
    (Counter.most_common preserves insertion order for equal counts).
    """
    n = len(pred)
    out = np.empty(n, dtype=pred.dtype)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = Counter(pred[lo:hi]).most_common(1)[0][0]
    return out


def metrics(y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    present = sorted(set(y_true))
    mf1 = float(f1_score(y_true, y_pred, labels=present, average="macro",
                         zero_division=0))
    comp = 100.0 * (0.5 * acc + 0.5 * mf1)
    return acc, mf1, comp


def main():
    data = {t: load_trip(t) for t in TRIPS}
    names = data[TRIPS[0]][2]
    ff_idx = names.index("face_found")

    grid = [(100, None), (100, 3), (300, None), (300, 3)]
    results = {}  # (max_iter, max_depth) -> per-trip dict + fold predictions

    for max_iter, max_depth in grid:
        per_trip = []
        fold_preds = {}
        for held in TRIPS:
            Xte, yte, _ = data[held]
            Xtr_parts, ytr_parts = [], []
            for t in TRIPS:
                if t == held:
                    continue
                X, y, _ = data[t]
                mask = X[:, ff_idx] == 1  # drop face_found==0 from TRAINING only
                Xtr_parts.append(X[mask])
                ytr_parts.append(y[mask])
            Xtr = np.vstack(Xtr_parts)
            ytr = np.concatenate(ytr_parts)

            clf = HistGradientBoostingClassifier(
                max_iter=max_iter, max_depth=max_depth, random_state=0)
            clf.fit(Xtr, ytr)
            pred_raw = clf.predict(Xte)  # all 600 frames, NaNs handled natively
            pred_sm = smooth_majority(pred_raw, half_window=25)

            acc_r, mf1_r, comp_r = metrics(yte, pred_raw)
            acc_s, mf1_s, comp_s = metrics(yte, pred_sm)
            per_trip.append(dict(trip=held, acc_raw=acc_r, mf1_raw=mf1_r,
                                 comp_raw=comp_r, acc_smooth=acc_s,
                                 mf1_smooth=mf1_s, comp_smooth=comp_s))
            fold_preds[held] = (pred_raw, pred_sm)

        results[(max_iter, max_depth)] = (per_trip, fold_preds)
        mean_raw = np.mean([r["comp_raw"] for r in per_trip])
        mean_sm = np.mean([r["comp_smooth"] for r in per_trip])
        print(f"\n=== max_iter={max_iter}, max_depth={max_depth} ===")
        print(f"{'trip':<12}{'acc_raw':>9}{'mf1_raw':>9}{'comp_raw':>10}"
              f"{'acc_sm':>9}{'mf1_sm':>9}{'comp_sm':>10}")
        for r in per_trip:
            print(f"{r['trip']:<12}{r['acc_raw']:>9.4f}{r['mf1_raw']:>9.4f}"
                  f"{r['comp_raw']:>10.2f}{r['acc_smooth']:>9.4f}"
                  f"{r['mf1_smooth']:>9.4f}{r['comp_smooth']:>10.2f}")
        print(f"{'MEAN':<12}{'':>9}{'':>9}{mean_raw:>10.2f}{'':>9}{'':>9}"
              f"{mean_sm:>10.2f}")

    # Pick best config by mean smoothed composite (tie-break: raw composite).
    def key(cfg):
        per_trip, _ = results[cfg]
        return (np.mean([r["comp_smooth"] for r in per_trip]),
                np.mean([r["comp_raw"] for r in per_trip]))

    best = max(grid, key=key)
    best_trips, best_preds = results[best]
    print(f"\nBEST CONFIG: max_iter={best[0]}, max_depth={best[1]}")

    # Pooled smoothed confusion across all 6 held-out trips for best config.
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    for t in TRIPS:
        _, yte, _ = data[t]
        pred_sm = best_preds[t][1]
        for yt, yp in zip(yte, pred_sm):
            conf[cidx[yt], cidx[yp]] += 1
    print("\nPooled smoothed confusion (rows=true, cols=pred):")
    print(f"{'':<12}" + "".join(f"{c:>12}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:<12}" + "".join(f"{conf[i, j]:>12}" for j in range(len(CLASSES))))

    # Most-confused off-diagonal pair (symmetric sum).
    best_pair, best_count = None, -1
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            c = conf[i, j] + conf[j, i]
            if c > best_count:
                best_count, best_pair = c, (CLASSES[i], CLASSES[j])
    print(f"\nMost-confused pair (smoothed, symmetric): {best_pair[0]} <-> "
          f"{best_pair[1]} ({best_count} frames)")

    # Feature-group importance via permutation on a full-sample-fit model is
    # leaky; instead use a cheap proxy: retrain best config per fold on
    # geometric-only vs blendshape-only features and compare mean composites.
    geo_cols = list(range(11))
    bs_cols = list(range(11, 63))
    group_means = {}
    for gname, cols in [("geometric-only", geo_cols),
                        ("blendshapes-only", bs_cols)]:
        comps = []
        for held in TRIPS:
            Xte, yte, _ = data[held]
            Xtr_parts, ytr_parts = [], []
            for t in TRIPS:
                if t == held:
                    continue
                X, y, _ = data[t]
                mask = X[:, ff_idx] == 1
                Xtr_parts.append(X[mask][:, cols])
                ytr_parts.append(y[mask])
            clf = HistGradientBoostingClassifier(
                max_iter=best[0], max_depth=best[1], random_state=0)
            clf.fit(np.vstack(Xtr_parts), np.concatenate(ytr_parts))
            pred_sm = smooth_majority(clf.predict(Xte[:, cols]), 25)
            comps.append(metrics(yte, pred_sm)[2])
        group_means[gname] = float(np.mean(comps))
        print(f"Ablation {gname}: mean smoothed composite = "
              f"{group_means[gname]:.2f}")

    # Summary block for the caller to parse if desired.
    print("\nFINAL SUMMARY")
    print(f"best_config: max_iter={best[0]}, max_depth={best[1]}")
    print(f"mean_composite_raw: {np.mean([r['comp_raw'] for r in best_trips]):.4f}")
    print(f"mean_composite_smooth: "
          f"{np.mean([r['comp_smooth'] for r in best_trips]):.4f}")
    for r in best_trips:
        print(f"  {r['trip']}: comp_raw={r['comp_raw']:.4f} "
              f"comp_smooth={r['comp_smooth']:.4f} "
              f"acc_smooth={r['acc_smooth']:.4f} "
              f"mf1_smooth={r['mf1_smooth']:.4f}")


if __name__ == "__main__":
    main()
