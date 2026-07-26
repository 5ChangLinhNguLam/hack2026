"""C2 bakeoff — Small MLP family (leave-one-trip-out over the 6 sample trips).

Pipeline: SimpleImputer(median) -> StandardScaler -> MLPClassifier
Architectures tried: (32,) and (64, 32). Deterministic: random_state=0.

Protocol:
  - Train on the other 5 trips only, dropping face_found==0 rows from TRAIN.
  - Predict all 600 frames of the held-out trip (imputer handles NaN test rows).
  - accuracy over all 600 frames; macro-F1 over classes present in GT only;
    composite = 100*(0.5*acc + 0.5*mf1).
  - Smoothed variant: sliding-window majority vote, window +/-25 frames.
"""
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CACHE = r"c:\HackathonFPT\hack2026\c2\cache"
TRIPS = [f"T{i:02d}-Sample" for i in range(1, 7)]
CLASSES = ["alert", "distracted", "drowsy", "microsleep", "yawning"]
ARCHS = [(32,), (64, 32)]
HALF_WIN = 25  # +/- 25 frames = 2.5 s @ 20 FPS


def load_trip(trip):
    d = np.load(rf"{CACHE}\features_{trip}.npz")
    X = d["features"].astype(np.float64)
    names = [str(n) for n in d["names"]]
    y = np.load(rf"{CACHE}\labels_{trip}.npy")
    return X, y, names


def majority_smooth(pred, half_win=HALF_WIN):
    """Sliding-window majority vote; ties broken by alphabetical class order."""
    n = len(pred)
    idx = np.array([CLASSES.index(p) for p in pred])
    onehot = np.zeros((n, len(CLASSES)), dtype=np.int32)
    onehot[np.arange(n), idx] = 1
    cs = np.vstack([np.zeros((1, len(CLASSES)), dtype=np.int32), np.cumsum(onehot, axis=0)])
    out = np.empty(n, dtype=object)
    for i in range(n):
        lo, hi = max(0, i - half_win), min(n, i + half_win + 1)
        counts = cs[hi] - cs[lo]
        out[i] = CLASSES[int(np.argmax(counts))]  # argmax -> first (alphabetical) on tie
    return np.array(out, dtype="<U10")


def metrics(y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    present = np.unique(y_true)
    mf1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    comp = 100.0 * (0.5 * acc + 0.5 * mf1)
    return acc, mf1, comp


def make_pipe(arch):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=arch, max_iter=800,
                              random_state=0, early_stopping=False)),
    ])


def feature_groups(names):
    groups = {}
    for j, n in enumerate(names):
        nl = n.lower()
        if n in ("ear_l", "ear_r"):
            g = "geo_eye(EAR)"
        elif n == "mar":
            g = "geo_mouth(MAR)"
        elif n in ("yaw", "pitch", "roll", "nose_dx", "nose_dy"):
            g = "geo_headpose"
        elif n in ("face_w", "face_h", "face_found"):
            g = "geo_facebox"
        elif nl.startswith("eyeblink") or nl.startswith("eyesquint") or nl.startswith("eyewide"):
            g = "bs_eye_openness"
        elif nl.startswith("eyelook"):
            g = "bs_gaze"
        elif nl.startswith("jaw") or nl.startswith("mouth"):
            g = "bs_mouth_jaw"
        elif nl.startswith("brow"):
            g = "bs_brow"
        else:
            g = "bs_other"
        groups.setdefault(g, []).append(j)
    return groups


def main():
    data = {t: load_trip(t) for t in TRIPS}
    names = data[TRIPS[0]][2]
    ff_col = names.index("face_found")
    groups = feature_groups(names)

    results = {a: {} for a in ARCHS}      # arch -> trip -> dict of metrics
    smooth_preds = {a: {} for a in ARCHS}  # for confusion later
    pipes = {a: {} for a in ARCHS}

    for held in TRIPS:
        Xte, yte, _ = data[held]
        tr = [t for t in TRIPS if t != held]
        Xtr = np.vstack([data[t][0] for t in tr])
        ytr = np.concatenate([data[t][1] for t in tr])
        keep = Xtr[:, ff_col] == 1  # drop face-not-found rows from TRAIN only
        Xtr, ytr = Xtr[keep], ytr[keep]

        for arch in ARCHS:
            pipe = make_pipe(arch)
            pipe.fit(Xtr, ytr)
            raw = pipe.predict(Xte)
            sm = majority_smooth(raw)
            a_r, f_r, c_r = metrics(yte, raw)
            a_s, f_s, c_s = metrics(yte, sm)
            results[arch][held] = dict(acc_raw=a_r, mf1_raw=f_r, comp_raw=c_r,
                                       acc_sm=a_s, mf1_sm=f_s, comp_sm=c_s)
            smooth_preds[arch][held] = sm
            pipes[arch][held] = pipe

    # ---- report ----
    for arch in ARCHS:
        mean_cr = np.mean([results[arch][t]["comp_raw"] for t in TRIPS])
        mean_cs = np.mean([results[arch][t]["comp_sm"] for t in TRIPS])
        print(f"\n=== MLP hidden={arch} ===")
        print(f"{'trip':<12}{'acc_raw':>9}{'mf1_raw':>9}{'comp_raw':>10}"
              f"{'acc_sm':>9}{'mf1_sm':>9}{'comp_sm':>10}")
        for t in TRIPS:
            r = results[arch][t]
            print(f"{t:<12}{r['acc_raw']:>9.4f}{r['mf1_raw']:>9.4f}{r['comp_raw']:>10.2f}"
                  f"{r['acc_sm']:>9.4f}{r['mf1_sm']:>9.4f}{r['comp_sm']:>10.2f}")
        print(f"{'MEAN':<12}{'':>9}{'':>9}{mean_cr:>10.2f}{'':>9}{'':>9}{mean_cs:>10.2f}")

    best = max(ARCHS, key=lambda a: np.mean([results[a][t]["comp_sm"] for t in TRIPS]))
    print(f"\nBest config by mean smoothed composite: hidden_layer_sizes={best}")

    # pooled smoothed confusion for best config
    y_all = np.concatenate([data[t][1] for t in TRIPS])
    p_all = np.concatenate([smooth_preds[best][t] for t in TRIPS])
    cm = confusion_matrix(y_all, p_all, labels=CLASSES)
    print("\nPooled SMOOTHED confusion (rows=true, cols=pred), labels:", CLASSES)
    print(cm)
    off = [(cm[i, j] + cm[j, i], CLASSES[i], CLASSES[j])
           for i in range(len(CLASSES)) for j in range(i + 1, len(CLASSES))]
    off.sort(reverse=True)
    top = off[0]
    # also report dominant direction
    i, j = CLASSES.index(top[1]), CLASSES.index(top[2])
    print(f"Most-confused pair (smoothed, symmetric): {top[1]} <-> {top[2]} "
          f"({top[0]} frames; {top[1]}->{top[2]}: {cm[i, j]}, {top[2]}->{top[1]}: {cm[j, i]})")

    # ---- feature-group importance: first-layer weight mass + group permutation ----
    print("\nFeature-group importance (best config):")
    w_imp = {g: 0.0 for g in groups}
    for t in TRIPS:
        W = np.abs(pipes[best][t].named_steps["mlp"].coefs_[0])  # (63, hidden)
        per_feat = W.mean(axis=1)
        for g, idxs in groups.items():
            w_imp[g] += per_feat[idxs].mean() / len(TRIPS)

    rng = np.random.RandomState(0)
    p_imp = {g: 0.0 for g in groups}
    for t in TRIPS:
        Xte, yte, _ = data[t]
        base = metrics(yte, pipes[best][t].predict(Xte))[2]
        for g, idxs in groups.items():
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idxs] = Xp[perm][:, idxs]
            drop = base - metrics(yte, pipes[best][t].predict(Xp))[2]
            p_imp[g] += drop / len(TRIPS)

    print(f"{'group':<18}{'mean|W1|':>10}{'perm-drop(comp)':>17}")
    for g in sorted(groups, key=lambda g: -p_imp[g]):
        print(f"{g:<18}{w_imp[g]:>10.4f}{p_imp[g]:>17.2f}")

    # machine-readable summary for the caller
    print("\n#RESULTS_JSON")
    import json
    out = {
        "best": list(best),
        "per_arch": {str(a): {t: results[a][t] for t in TRIPS} for a in ARCHS},
        "confusion_pair": [top[1], top[2], int(top[0]), int(cm[i, j]), int(cm[j, i])],
        "perm_importance": {g: round(p_imp[g], 3) for g in groups},
        "weight_importance": {g: round(w_imp[g], 4) for g in groups},
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
