import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

AUD = Path("processed/audio")
OUT = Path("features")
OUT.mkdir(exist_ok=True)

MODEL = "facebook/wav2vec2-base"
MAX_SEC = 30   # cap length to keep CPU time reasonable

fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
model = Wav2Vec2Model.from_pretrained(MODEL).eval()

names, feats = [], []
files = sorted(AUD.glob("*.wav"))
for k, w in enumerate(files, 1):
    wav, sr = sf.read(str(w))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav[: 16000 * MAX_SEC].astype(np.float32)
    inputs = fe(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = model(**inputs).last_hidden_state
    feats.append(h.mean(dim=1).squeeze(0).numpy())  # 768-d
    names.append(w.stem)
    print(f"[{k}/{len(files)}] ok: {w.stem}")

np.save(OUT / "audio_feats.npy", np.stack(feats))
json.dump(names, open(OUT / "audio_names.json", "w"))
print("Saved", len(names), "audio features")