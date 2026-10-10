import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = Path("features")

vnames = json.load(open(FEAT / "video_names.json"))
vfeats = np.load(FEAT / "video_feats.npy")
anames = json.load(open(FEAT / "audio_names.json"))
afeats = np.load(FEAT / "audio_feats.npy")

# keep only clips that have BOTH video and audio features
vidx = {n: i for i, n in enumerate(vnames)}
aidx = {n: i for i, n in enumerate(anames)}
common = [n for n in anames if n in vidx]
V = np.stack([vfeats[vidx[n]] for n in common])
A = np.stack([afeats[aidx[n]] for n in common])
N = len(common)
import sys
if "--shuffle" in sys.argv:
    A = A[np.random.RandomState(123).permutation(N)]
print(f"Paired video-audio clips: {N}")


class Proj(nn.Module):
    def __init__(self, d_in, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce(zv, za, temp=0.07):
    """Symmetric InfoNCE: video i should match audio i, and vice versa."""
    logits = zv @ za.T / temp
    t = torch.arange(len(zv))
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2


@torch.no_grad()
def retrieval(zv, za):
    """Does video i retrieve audio i (and audio i retrieve video i)?"""
    sim = zv @ za.T
    t = torch.arange(len(zv))[:, None]
    out = {}
    for tag, s in (("v2a", sim), ("a2v", sim.T)):
        rank = (s.argsort(dim=1, descending=True) == t).float().argmax(dim=1)
        out[tag + "_R1"] = (rank == 0).float().mean().item()
        out[tag + "_R5"] = (rank < 5).float().mean().item()
    return out


def run(seed, epochs=100):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_test = max(5, int(0.2 * N))
    te, tr = perm[:n_test], perm[n_test:]

    # normalise features using training statistics only
    def norm(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        return torch.tensor((X - mu) / sd, dtype=torch.float32)

    Vn, An = norm(V), norm(A)
    pv, pa = Proj(Vn.shape[1]), Proj(An.shape[1])
    opt = torch.optim.AdamW(list(pv.parameters()) + list(pa.parameters()),
                            lr=1e-3, weight_decay=1e-2)

    pv.train(); pa.train()
    for _ in range(epochs):
        loss = info_nce(pv(Vn[tr]), pa(An[tr]))
        opt.zero_grad(); loss.backward(); opt.step()

    pv.eval(); pa.eval()
    with torch.no_grad():
        res = retrieval(pv(Vn[te]), pa(An[te]))
    res["final_train_loss"] = loss.item()
    res["n_train"], res["n_test"] = len(tr), len(te)
    return res


results = [run(seed) for seed in range(5)]
n_test = results[0]["n_test"]
print(f"\nHeld-out test clips per split: {n_test}")
print(f"Chance level: R@1 = {1 / n_test:.2f}, R@5 = {min(1.0, 5 / n_test):.2f}\n")
for key in ("v2a_R1", "v2a_R5", "a2v_R1", "a2v_R5"):
    vals = [r[key] for r in results]
    print(f"{key}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

json.dump(results, open(FEAT / "contrastive_results.json", "w"), indent=2)