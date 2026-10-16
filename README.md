# TruWage: multimodal skill assessment and fair-wage prototype

Informal workers in India often have no certificates. TruWage takes a short smartphone video of a
task (and an optional voice note), checks that the video is genuine, scores the technique, suggests
a fair hourly wage range and produces a verifiable skill passport.

This is a working prototype. The tables below say exactly what is real, what is a prototype and
what is still planned.

## Feature status

| # | Feature in the project report | Status | What exists today |
|---|---|---|---|
| 1 | Action quality assessment (AQA) | Implemented | VideoMAE (Kinetics-finetuned) backbone with a linear head trained on 152 human-rated clips. Returns a 0-1 technique score and a Grad-CAM heatmap. Not trained on SOPs. |
| 2 | Graph-network fair-wage engine | Separate prototype | `AI Project/gnn_wage.py` is a graph convolutional network (torch_geometric) trained on synthetic worker data. The live app uses a simpler neural network (see 8). |
| 3 | Voice-to-skill mapping | Partial | The optional voice note is transcribed with Whisper and the words are matched to the skill card. Keyword-based, not zero-shot. No skill graph. NSQF levels are illustrative. |
| 4 | Evidence-based dispute resolution | Separate prototype | `AI Project/dispute_resolution.py` compares synthetic images with CLIP. The comparison logic needs redesign. Not part of the app. |
| 5 | Contrastive video-audio alignment | Preliminary experiment | `indic-skill-100/contrastive.py`. Results are slightly above chance and do not feed the score. |
| 6 | Explainability layer | Partial | Grad-CAM heatmap in the app and poor-frame timestamps saved under `explain/`. No counterfactual feedback. Hindi text-to-speech exists only as a scripted example. |
| 7 | Video manipulation detection | Partial (different method) | A frame-statistics heuristic that flags frozen and looped videos. It is not a 3D CNN and does not detect AI-generated video. |
| 8 | Bayesian wage discovery | Implemented (synthetic data) | A NumPy neural network with Monte-Carlo dropout returns a wage range and a confidence label. `AI Project/gnn_bayesian.py` is a Bayesian graph version. Trained on synthetic data, so it demonstrates the method and is not a market estimate. |
| - | Digital Skill Passport | Implemented | HTML passport with a SHA-256 hash of the video and results, plus a QR code. Tamper-evident, not a digital signature. |

## How the app works

`/api/assess` runs these steps in order:

1. Upload (video, optional audio, trade, complexity, demand, worker name)
2. Integrity check: frozen or looped video is rejected with a reason
3. AQA technique score and Grad-CAM heatmap
4. Optional voice note: Whisper transcript, checked against the selected trade
5. Wage range with a confidence label (from the technique score, complexity and demand)
6. Skill card from the selected trade and the words in the transcript
7. Passport with verification hash and QR code

The worker must choose the trade. The system does not detect the trade from the video.

## Run it (Windows PowerShell)

Python 3.9 or newer (3.11 recommended).

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

The first assessment downloads the VideoMAE weights, and the first voice note downloads Whisper
Medium (about 1.5 GB), so allow time and an internet connection. Set `TRUWAGE_WHISPER_MODEL`
(for example `openai/whisper-small`) to use a smaller speech model.

Optional environment variables: `TRUWAGE_API` (backend URL for the UI) and `TRUWAGE_CORS_ORIGINS`.

## Test it

```powershell
cd truwage-mvp
python -m pytest -q
```

The 29 tests cover the wage model, integrity check, skill mapping, passport, audio decoding and
the API. They replace the AQA and Whisper models with stand-ins, so they need no model downloads.

## Demo and validation tools

```powershell
# Make looped and frozen test clips (H.264, so they play in the browser) from a real clip
python tools\make_tamper_clips.py demo_clips\genuine.mp4 demo_clips

# Run the integrity check on a whole folder; writes integrity_report.csv
python tools\check_folder.py C:\path\to\indic-skill-100\processed\video_15fps
```

Validation of the scoring features (needs `indic-skill-100\processed\video_15fps`, which is not in
the repository):

```powershell
cd indic-skill-100
python extract_video_features_k400.py
python oof_corr.py
python contrastive.py
python contrastive.py --shuffle
```

Separate prototype scripts (run from `AI Project`): `gnn_wage.py`, `gnn_bayesian.py`,
`robust_transcriber.py`, `dispute_resolution.py`, `test_noisy_audio.py`.

## Measured so far

- Integrity check on synthetic clips (240 variants, three compression levels): no genuine clip was
  rejected and every frozen clip was caught. The only missed loops were steady-camera loops under
  extreme compression. Thresholds in `app/integrity_checker.py` should be tuned on real footage.
- Wage model: on its own synthetic data, predictions track the generating formula (mean error about
  25 rupees per hour against 15 of noise). Wages rise with skill, complexity and demand.
- Contrastive alignment (5 random splits, 50 training and 12 test pairs): video-to-audio recall@5
  about 0.53 against a chance level of about 0.42. Training loss is very low, which suggests
  overfitting. Treat it as preliminary.
- To be generated by running the tools above: integrity results on the 152 real clips and the
  cross-validated AQA correlation (`oof_corr.py`).

## Limitations

- Wage figures come from a model trained on synthetic data.
- The dataset is a 152-clip pilot (90 stock clips, 62 YouTube Shorts), not verified footage of
  Indian informal workers. Some metadata rows are still unlabelled.
- The AQA head is trained on a small set with human ratings and is not trained on SOPs.
- The integrity check covers frozen and looped video only.
- Voice transcription uses plain Whisper. Accuracy drops on noisy or heavily accented speech, and
  Whisper can produce text for silent audio.
- NSQF levels are placeholders. Check them against the official NSDC Qualification Packs.

## Repository layout

- `truwage-mvp/app/` backend (integrity, wage, skill card, passport, transcription)
- `truwage-mvp/streamlit_demo.py` demo UI
- `truwage-mvp/tools/` demo and validation tools
- `truwage-mvp/tests/` automated tests
- `indic-skill-100/` pilot dataset metadata, AQA model and experiments
- `AI Project/` separate prototypes (graph wage models, transcription, dispute resolution, Gradio dashboard)

Data sources and licences: see `DATA_LICENSES.md`.

## Team TruWage

Shreyansh Khatri (24BCE0541), Ayansh Agarwal (24BDS0166), Balraj Singh (24BCE0531)
