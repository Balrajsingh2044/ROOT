import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

VID = Path("processed/video_15fps")
OUT = Path("features")
OUT.mkdir(exist_ok=True)

MODEL = "MCG-NJU/videomae-base"
NUM_FRAMES = 16   # VideoMAE-base takes 16 frames


def sample_frames(path, n=NUM_FRAMES):
    """Pick n evenly spaced frames without loading the whole video into memory."""
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
    while len(frames) < n:          # pad very short clips
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
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    feats.append(out.last_hidden_state.mean(dim=1).squeeze(0).numpy())  # 768-d
    names.append(p.stem)
    print(f"[{k}/{len(files)}] ok: {p.stem}")

np.save(OUT / "video_feats.npy", np.stack(feats))
json.dump(names, open(OUT / "video_names.json", "w"))
print("Saved", len(names), "video features, dim", feats[0].shape[0])