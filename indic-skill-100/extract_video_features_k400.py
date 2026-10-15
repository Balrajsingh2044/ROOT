r"""Extract the Kinetics-finetuned VideoMAE features that oof_corr.py needs.

Same method as extract_video_features.py (16 evenly spaced frames, mean-pooled last hidden state),
but with the backbone the AQA model actually uses.

Run from the indic-skill-100 folder:   python extract_video_features_k400.py
Needs:   processed\video_15fps\*.mp4   (your 152 clips)
Writes:  features\video_feats_k400.npy and features\video_names_k400.json
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

VID = Path("processed/video_15fps")
OUT = Path("features")
MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"   # same as gradcam_aqa.py
NUM_FRAMES = 16

if not VID.is_dir():
    raise SystemExit(f"Cannot find {VID.resolve()}. Run this from the indic-skill-100 folder "
                     "and make sure processed\\video_15fps exists there.")
OUT.mkdir(exist_ok=True)


def sample_frames(path, n=NUM_FRAMES):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, n).astype(int)
    need, kept, i = set(idxs.tolist()), {}, 0
    last = int(idxs.max())
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in need:
            kept[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    frames = [kept[j] for j in idxs if j in kept]
    if not frames:
        return None
    while len(frames) < n:
        frames.append(frames[-1])
    return frames


processor = VideoMAEImageProcessor.from_pretrained(MODEL)
model = VideoMAEModel.from_pretrained(MODEL).eval()

names, feats = [], []
files = sorted(VID.glob("*.mp4"))
for k, p in enumerate(files, 1):
    frames = sample_frames(p)
    if frames is None:
        print(f"[{k}/{len(files)}] skip (unreadable): {p.name}")
        continue
    with torch.no_grad():
        out = model(**processor(frames, return_tensors="pt"))
    feats.append(out.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    names.append(p.stem)
    print(f"[{k}/{len(files)}] ok: {p.stem}", flush=True)

np.save(OUT / "video_feats_k400.npy", np.stack(feats))
json.dump(names, open(OUT / "video_names_k400.json", "w"))
print("Saved", len(names), "features, dim", feats[0].shape[0])
