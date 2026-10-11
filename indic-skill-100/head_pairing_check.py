import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import VideoMAEImageProcessor

import gradcam_aqa as g

FEAT = Path("features")
sd = torch.load(FEAT / "aqa_head.pt", map_location="cpu")
W = sd["weight"].numpy().reshape(-1).astype(np.float64)
b = float(sd["bias"].numpy().reshape(-1)[0])

human = {}
with open("scores.csv", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        if r["score"].strip():
            v = float(r["score"])
            human[Path(r["filename"].strip()).stem] = v / 100 if v > 1 else v


def feat_scores(npy, js):
    names = json.load(open(FEAT / js))
    X = np.load(FEAT / npy).astype(np.float64)
    return dict(zip(names, 1 / (1 + np.exp(-(X @ W + b)))))


fs = {"base": feat_scores("video_feats.npy", "video_names.json"),
      "k400": feat_scores("video_feats_k400.npy", "video_names_k400.json")}
stems = [s for s in sorted(human) if all(s in d for d in fs.values())]
h = np.array([human[s] for s in stems])

print("In-sample correlation of aqa_head.pt with human ratings:")
for tag, d in fs.items():
    p = np.array([d[s] for s in stems])
    print(f"  {tag:5s} features: {np.corrcoef(p, h)[0, 1]:.2f}")

proc = VideoMAEImageProcessor.from_pretrained(g.MODEL)
model = g.AQAModel().eval()
pick = stems[::15][:10]
demo, saved = [], []
for s in pick:
    frames_raw, _ = g.load_frames(g.VID / f"{s}.mp4")
    frames = [g.center_crop(f) for f in frames_raw]
    with torch.no_grad():
        demo.append(float(model(proc(frames, return_tensors="pt")["pixel_values"])))
    saved.append(fs["k400"][s])
demo, saved = np.array(demo), np.array(saved)
print(f"\nDemo path vs saved-feature path on {len(pick)} clips ({g.MODEL}):")
print(f"  correlation {np.corrcoef(demo, saved)[0, 1]:.3f}, mean abs diff {np.abs(demo - saved).mean():.3f}")