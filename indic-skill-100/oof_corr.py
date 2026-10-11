import csv
import json
from pathlib import Path

import numpy as np

FEAT = Path("features")
ALPHA = 1000

names = json.load(open(FEAT / "video_names_k400.json"))
X = np.load(FEAT / "video_feats_k400.npy").astype(np.float64)
idx = {n: i for i, n in enumerate(names)}

rows = []
with open("scores.csv", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        if r["score"].strip():
            s = Path(r["filename"].strip()).stem
            if s in idx:
                v = float(r["score"])
                rows.append((s, v / 100 if v > 1 else v))
stems = [s for s, _ in rows]
y = np.array([v for _, v in rows])
Xs = np.stack([X[idx[s]] for s in stems])
n = len(y)
T = np.log(np.clip(y, 0.05, 0.95) / (1 - np.clip(y, 0.05, 0.95)))


def fit(Xtr, Ttr, a):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    tm = Ttr.mean()
    al = np.linalg.solve(Z @ Z.T + a * np.eye(len(Z)), Ttr - tm)
    w = Z.T @ al
    return w / sd, tm - (mu / sd) @ w


corrs, poor_means, good_means = [], [], []
for rep in range(10):
    perm = np.random.RandomState(rep).permutation(n)
    oof = np.zeros(n)
    for te in np.array_split(perm, 5):
        tr = np.setdiff1d(np.arange(n), te)
        W, b = fit(Xs[tr], T[tr], ALPHA)
        oof[te] = 1 / (1 + np.exp(-(Xs[te] @ W + b)))
    corrs.append(np.corrcoef(oof, y)[0, 1])
    poor_means.append(oof[y <= 0.4].mean())
    good_means.append(oof[y >= 0.8].mean())

print(f"{n} clips, 10 x 5-fold, alpha={ALPHA}")
print(f"out-of-fold correlation with human ratings: {np.mean(corrs):.2f}  (in-sample was 0.85)")
print(f"mean model score for human-poor clips (<=0.4): {np.mean(poor_means):.2f}")
print(f"mean model score for human-good clips (>=0.8): {np.mean(good_means):.2f}")