# TruWage: multimodal skill assessment and fair-wage prototype

Informal workers in India often have no certificates. TruWage takes a short smartphone video of a
task, checks the video is genuine, scores the technique, suggests a fair hourly wage range and
produces a verifiable skill passport.

This is a working prototype. The table below says exactly what is real and what is not.

## Feature status

| Feature | Status | What it does today |
|---|---|---|
| Technique scoring (AQA) with Grad-CAM heatmap | Implemented | VideoMAE backbone with a head trained on 152 clips rated by humans. Returns a 0-1 score and a heatmap of where the technique lost points. |
| Video integrity check | Implemented (heuristic) | Flags frozen videos and looped (repeated) clips from frame statistics. Does not detect AI-generated or face-swap video. |
| Fair-wage range | Implemented (synthetic data) | Small neural network with Monte-Carlo dropout gives a wage range and a confidence label. Trained on synthetic data, so it is a demonstration of the method, not a market estimate. |
| Skill card | Implemented (keyword-based) | Maps the chosen trade (or words in a transcript) to a skill card. NSQF levels are illustrative placeholders. |
| Digital Skill Passport | Implemented (tamper-evident) | HTML passport with a SHA-256 hash of the video and results, plus a QR code. Not a digital signature. |
| Voice transcription | Separate prototype | `AI Project/robust_transcriber.py` (Whisper). Not connected to the API. |
| Graph-network wage model | Separate prototype | `AI Project/gnn_wage.py`, `gnn_bayesian.py` (synthetic data). Not connected to the API. |
| Dispute resolution | Separate prototype | `AI Project/dispute_resolution.py` compares two synthetic images with CLIP. Not a verified feature. |
| Contrastive video-audio alignment | Preliminary experiment | `indic-skill-100/contrastive.py`. Results are only slightly above chance and do not feed the score. |
| Counterfactual explanations | Planned | Today the explanation is the Grad-CAM heatmap. |
| Trained deepfake / AI-video detector | Planned | |

## Run it (Windows PowerShell)

```powershell
cd truwage-mvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

In a second terminal:

```powershell
cd truwage-mvp
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_demo.py
```

The first assessment downloads the VideoMAE weights, so it needs internet and takes longer.

## Test it

```powershell
cd truwage-mvp
python -m pytest -q
```

To make looped and frozen test clips from a real video:

```powershell
python tools\make_tamper_clips.py path\to\genuine.mp4 demo_clips
```

## Layout

- `truwage-mvp/app/` FastAPI backend (integrity, wage, skill card, passport)
- `truwage-mvp/streamlit_demo.py` demo UI
- `indic-skill-100/` pilot dataset metadata, AQA model and experiments
- `AI Project/` separate prototypes (transcription, graph wage model, dispute resolution, Gradio dashboard)

## Limitations

- Wage figures come from a model trained on synthetic data.
- The dataset is a 152-clip pilot (90 stock clips, 62 YouTube Shorts), not verified footage of Indian informal workers.
- The integrity check is a heuristic for frozen and looped video. On tests with synthetic clips it caught every frozen clip and every loop except steady-camera loops under extreme compression. Tune the thresholds in `integrity_checker.py` on your own clips.
- NSQF levels are placeholders. Check them against the official NSDC Qualification Packs.

## Team

Shreyansh Khatri (24BCE0541), Ayansh Agarwal (24BDS0166), Balraj Singh (24BCE0531)

Data sources and licences: see `DATA_LICENSES.md`.
