"""C2 bakeoff: Multinomial Logistic Regression, leave-one-trip-out (LOTO).

Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression(max_iter=2000).
C tuned over {0.1, 1, 10}: all three are evaluated with full LOTO; the best mean
smoothed composite picks the winning config. Deterministic (random_state=0).
"""
import os
import numpy as np
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
TRIPS = [f"T{i:02d}-Sample" for i in range(1, 7)]
CS = [0.1, 1.0, 10.0]
WIN = 25  # +/- frames for majority-vote smoothing (2.5 s @ 20 FPS)


def load_trip(trip):
    d = np.load(os.path.join(CACHE, f"features_{trip}.npz"), allow_pickle=True)
    X = d["features"].astype(np.float64)
    names = [str(n) for n in d["names"]]
    y = np.load(os.path.join(CACHE, f"labels_{trip}.npy"), allow_pickle=True)
    y = np.array([str(v) for v in y])
    return X, y, names


def smooth_majority(pred, win=WIN):
    """Sliding-window majority vote over +/- win frames. Deterministic ties:
    most common wins; ties broken by first occurrence order in Counter
    (insertion order = order encountered in window), stable across runs."""
    n = len(pred)
    out = np.empty(n, dtype=pred.dtype)
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        c = Counter(pred[lo:hi])
        # deterministic tie-break: highest count, then alphabetical
        best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out[i] = best
    return out


def metrics(y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    present = sorted(set(y_true))  # macro-F1 only over classes present in GT
    mf1 = float(f1_score(y_true, y_pred, labels=present, average="macro",
                         zero_division=0))
    comp = 100.0 * (0.5 * acc + 0.5 * mf1)
    return acc, mf1, comp


def main():
    data = {t: load_trip(t) for t in TRIPS}
    names = data[TRIPS[0]][2]
    ff_idx = names.index("face_found")

    all_labels = sorted(set(np.concatenate([data[t][1] for t in TRIPS])))
    results = {}  # C -> list of per-trip dicts
    confusions = {}  # C -> summed smoothed confusion matrix
    coefs = {}  # C -> list of |coef| aggregated per fold

    for C in CS:
        rows = []
        cm_total = np.zeros((len(all_labels), len(all_labels)), dtype=np.int64)
        fold_coefs = []
        for hold in TRIPS:
            Xtr_list, ytr_list = [], []
            for t in TRIPS:
                if t == hold:
                    continue
                X, y, _ = data[t]
                keep = X[:, ff_idx] == 1  # drop no-face rows from TRAINING only
                Xtr_list.append(X[keep])
                ytr_list.append(y[keep])
            Xtr = np.vstack(Xtr_list)
            ytr = np.concatenate(ytr_list)
            Xte, yte, _ = data[hold]  # keep ALL 600 test frames

            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, C=C,
                                          random_state=0)),
            ])
            pipe.fit(Xtr, ytr)
            pred_raw = pipe.predict(Xte)
            pred_sm = smooth_majority(pred_raw)

            acc_r, mf1_r, comp_r = metrics(yte, pred_raw)
            acc_s, mf1_s, comp_s = metrics(yte, pred_sm)
            rows.append(dict(trip=hold, acc_raw=acc_r, mf1_raw=mf1_r,
                             comp_raw=comp_r, acc_sm=acc_s, mf1_sm=mf1_s,
                             comp_sm=comp_s))
            cm_total += confusion_matrix(yte, pred_sm, labels=all_labels)
            fold_coefs.append(np.abs(pipe.named_steps["lr"].coef_).mean(axis=0))
        results[C] = rows
        confusions[C] = cm_total
        coefs[C] = np.mean(fold_coefs, axis=0)

    # --- report ---
    for C in CS:
        rows = results[C]
        print(f"\n=== LogisticRegression C={C} ===")
        print(f"{'trip':<12}{'acc_raw':>9}{'mf1_raw':>9}{'comp_raw':>10}"
              f"{'acc_sm':>9}{'mf1_sm':>9}{'comp_sm':>10}")
        for r in rows:
            print(f"{r['trip']:<12}{r['acc_raw']:>9.4f}{r['mf1_raw']:>9.4f}"
                  f"{r['comp_raw']:>10.2f}{r['acc_sm']:>9.4f}"
                  f"{r['mf1_sm']:>9.4f}{r['comp_sm']:>10.2f}")
        mr = np.mean([r["comp_raw"] for r in rows])
        ms = np.mean([r["comp_sm"] for r in rows])
        print(f"{'MEAN':<12}{'':>9}{'':>9}{mr:>10.2f}{'':>9}{'':>9}{ms:>10.2f}")

    best_C = max(CS, key=lambda c: np.mean([r["comp_sm"] for r in results[c]]))
    print(f"\nBest config: C={best_C} "
          f"(mean smoothed composite "
          f"{np.mean([r['comp_sm'] for r in results[best_C]]):.2f})")

    # most-confused pair from summed smoothed confusion (off-diagonal, best C)
    cm = confusions[best_C]
    print("\nSmoothed confusion (rows=true, cols=pred), all 6 held-out trips, "
          f"C={best_C}:")
    print(f"{'':<12}" + "".join(f"{l:>12}" for l in all_labels))
    for i, l in enumerate(all_labels):
        print(f"{l:<12}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(all_labels))))
    off = [(cm[i, j], all_labels[i], all_labels[j])
           for i in range(len(all_labels)) for j in range(len(all_labels)) if i != j]
    off.sort(key=lambda t: (-t[0], t[1], t[2]))
    cnt, a, b = off[0]
    print(f"\nMost-confused pair (smoothed): true={a} -> pred={b} ({cnt} frames)")

    # feature-group importance from mean |coef| (standardized inputs, best C)
    w = coefs[best_C]
    geo_names = names[:11]
    order = np.argsort(-w)
    print("\nTop 15 features by mean |coef| across classes/folds "
          f"(C={best_C}):")
    for k in order[:15]:
        grp = "geometric" if names[k] in geo_names else "blendshape"
        print(f"  {names[k]:<28}{w[k]:.3f}  [{grp}]")
    geo_mean = float(np.mean([w[i] for i in range(len(names))
                              if names[i] in geo_names]))
    bs_mean = float(np.mean([w[i] for i in range(len(names))
                             if names[i] not in geo_names]))
    print(f"Group mean |coef|: geometric={geo_mean:.3f} "
          f"blendshape={bs_mean:.3f}")

    return results, best_C, all_labels, confusions[best_C], names, coefs[best_C]


if __name__ == "__main__":
    main()
