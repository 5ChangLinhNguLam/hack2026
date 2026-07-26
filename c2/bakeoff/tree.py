"""C2 bakeoff: interpretable prototype-rule tree (DecisionTreeClassifier).

Leave-one-trip-out (LOTO) over the 6 labeled sample trips.
Pipeline: SimpleImputer(median) -> DecisionTreeClassifier(random_state=0),
max_depth tried over {3, 4, 5}. No scaler (trees are scale-invariant).

Metrics per held-out trip:
  - accuracy over all 600 frames
  - macro-F1 averaged ONLY over classes present in the trip's ground truth
  - composite = 100 * (0.5*acc + 0.5*mf1)
Smoothed variant: sliding-window majority vote, window +/-25 frames, then
recompute the same metrics.

Deterministic: random_state=0 everywhere applicable.
"""

import json
import os
from collections import Counter

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.normpath(os.path.join(HERE, "..", "cache"))

TRIPS = [f"T{i:02d}-Sample" for i in range(1, 7)]
CLASSES = ["alert", "distracted", "drowsy", "microsleep", "yawning"]
DEPTHS = [3, 4, 5]
HALF_WIN = 25  # +/- 25 frames = 2.5 s at 20 FPS


def load_trip(trip):
    npz = np.load(os.path.join(CACHE, f"features_{trip}.npz"), allow_pickle=True)
    X = npz["features"].astype(np.float64)
    names = [str(n) for n in npz["names"]]
    y = np.load(os.path.join(CACHE, f"labels_{trip}.npy"), allow_pickle=True)
    y = np.array([str(v) for v in y])
    return X, y, names


def majority_smooth(pred, half_win):
    """Sliding-window majority vote over +/-half_win frames (ties: keep the
    label that appears first in the window for determinism)."""
    n = len(pred)
    out = np.empty(n, dtype=object)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        window = pred[lo:hi]
        counts = Counter(window)
        best = max(counts.values())
        # deterministic tie-break: first label in window order reaching best
        for lab in window:
            if counts[lab] == best:
                out[i] = lab
                break
    return np.array([str(v) for v in out])


def trip_metrics(y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    present = sorted(set(y_true))
    # macro-F1 only over classes present in ground truth of this trip
    f1s = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    mf1 = float(np.mean(f1s))
    comp = 100.0 * (0.5 * acc + 0.5 * mf1)
    return acc, mf1, comp


def run_loto(depth):
    """Return per-trip results and pooled smoothed predictions for one depth."""
    results = {}
    pooled_true, pooled_smooth = [], []
    for held in TRIPS:
        # ---- assemble training data from the other 5 trips ----
        Xtr_list, ytr_list = [], []
        for t in TRIPS:
            if t == held:
                continue
            X, y, _ = load_trip(t)
            ff = X[:, FACE_FOUND_IDX]
            keep = ff == 1  # drop face_found==0 rows from TRAINING only
            Xtr_list.append(X[keep])
            ytr_list.append(y[keep])
        Xtr = np.vstack(Xtr_list)
        ytr = np.concatenate(ytr_list)

        # ---- fit imputer + tree on training folds only ----
        imp = SimpleImputer(strategy="median")
        Xtr_i = imp.fit_transform(Xtr)
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(Xtr_i, ytr)

        # ---- predict all 600 frames of held-out trip ----
        Xte, yte, _ = load_trip(held)
        Xte_i = imp.transform(Xte)  # face_found==0 rows imputed, never dropped
        pred_raw = clf.predict(Xte_i)
        pred_smooth = majority_smooth(pred_raw, HALF_WIN)

        acc_r, mf1_r, comp_r = trip_metrics(yte, pred_raw)
        acc_s, mf1_s, comp_s = trip_metrics(yte, pred_smooth)
        results[held] = dict(
            acc_raw=acc_r, mf1_raw=mf1_r, comp_raw=comp_r,
            acc_smooth=acc_s, mf1_smooth=mf1_s, comp_smooth=comp_s,
        )
        pooled_true.append(yte)
        pooled_smooth.append(pred_smooth)
    return results, np.concatenate(pooled_true), np.concatenate(pooled_smooth)


def main():
    global FACE_FOUND_IDX
    _, _, names = load_trip(TRIPS[0])
    FACE_FOUND_IDX = names.index("face_found")

    all_results = {}
    pooled = {}
    for d in DEPTHS:
        res, pt, ps = run_loto(d)
        all_results[d] = res
        pooled[d] = (pt, ps)
        mean_raw = np.mean([r["comp_raw"] for r in res.values()])
        mean_smooth = np.mean([r["comp_smooth"] for r in res.values()])
        print(f"depth={d}: mean composite raw={mean_raw:.2f}  "
              f"smooth={mean_smooth:.2f}")

    # pick best depth by mean smoothed composite (tie: smaller depth)
    best_depth = max(DEPTHS,
                     key=lambda d: (round(np.mean(
                         [r["comp_smooth"] for r in all_results[d].values()]), 6),
                         -d))
    print(f"\nBest depth: {best_depth}\n")

    res = all_results[best_depth]
    header = (f"{'trip':<12}{'acc_raw':>9}{'mf1_raw':>9}{'comp_raw':>10}"
              f"{'acc_sm':>9}{'mf1_sm':>9}{'comp_sm':>10}")
    print(header)
    print("-" * len(header))
    for t in TRIPS:
        r = res[t]
        print(f"{t:<12}{r['acc_raw']:>9.4f}{r['mf1_raw']:>9.4f}"
              f"{r['comp_raw']:>10.2f}{r['acc_smooth']:>9.4f}"
              f"{r['mf1_smooth']:>9.4f}{r['comp_smooth']:>10.2f}")
    mean_raw = float(np.mean([res[t]["comp_raw"] for t in TRIPS]))
    mean_smooth = float(np.mean([res[t]["comp_smooth"] for t in TRIPS]))
    print("-" * len(header))
    print(f"{'MEAN':<12}{'':>9}{'':>9}{mean_raw:>10.2f}{'':>9}{'':>9}"
          f"{mean_smooth:>10.2f}")

    # ---- confusion matrix over pooled smoothed predictions (all 6 folds) ----
    pt, ps = pooled[best_depth]
    labels = sorted(set(pt) | set(ps))
    cm = confusion_matrix(pt, ps, labels=labels)
    print("\nPooled smoothed confusion matrix (rows=true, cols=pred):")
    print(f"{'':<12}" + "".join(f"{l[:10]:>11}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<12}" + "".join(f"{c:>11d}" for c in cm[i]))

    # most-confused (off-diagonal) pair
    off = cm.astype(float).copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"\nMost-confused pair (smoothed): true={labels[i]} -> "
          f"pred={labels[j]} ({int(off[i, j])} frames)")

    # ---- fit final tree at best depth on ALL 6 trips for rule inspection ----
    Xall, yall = [], []
    for t in TRIPS:
        X, y, _ = load_trip(t)
        keep = X[:, FACE_FOUND_IDX] == 1
        Xall.append(X[keep])
        yall.append(y[keep])
    Xall = np.vstack(Xall)
    yall = np.concatenate(yall)
    imp = SimpleImputer(strategy="median")
    Xall_i = imp.fit_transform(Xall)
    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=0)
    clf.fit(Xall_i, yall)
    print(f"\nLearned tree rules (depth={best_depth}, fit on all 6 sample "
          "trips, face_found frames only):")
    print(export_text(clf, feature_names=names, decimals=3))

    imps = clf.feature_importances_
    order = np.argsort(imps)[::-1]
    print("Top feature importances:")
    for k in order[:10]:
        if imps[k] <= 0:
            break
        print(f"  {names[k]:<28}{imps[k]:.4f}")

    # machine-readable summary for the caller
    summary = dict(
        best_depth=best_depth,
        per_trip=res,
        mean_composite_raw=mean_raw,
        mean_composite_smooth=mean_smooth,
        most_confused=[labels[i], labels[j], int(off[i, j])],
        top_features=[(names[k], float(imps[k])) for k in order[:10]
                      if imps[k] > 0],
    )
    print("\nJSON_SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
