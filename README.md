# Truwage: Multimodal AQA & Fair-Wage Engine
A software-based AI system that verifies informal worker skills via short smartphone videos, generates a Digital Skill Passport, and recommends fair market wages.

## Setup & Run
1. `pip install -r truwage-mvp/requirements.txt`
2. Backend: `cd truwage-mvp && uvicorn app.main:app --reload --port 8000`
3. Frontend: `cd "AI Project" && streamlit run streamlit_demo.py`

## Feature Status
| Feature | Status | Notes |
|---|---|---|
| Spatiotemporal AQA + Grad-CAM | ✅ Implemented | VideoMAE backbone with custom head |
| Integrity Check (Deepfake/Loop) | ✅ Implemented | Heuristics + R3D-18 feature classifier |
| Bayesian Fair Wage Engine | 🟡 Prototype | MC-Dropout on synthetic trade graph |
| Zero-Shot Voice-to-NSQF Mapping | 🟡 Prototype | Keyword-based fallback for robustness |
| Contrastive Video-Audio Alignment | 🟡 Prototype | Offline evaluation shows preliminary alignment |
| Counterfactual Explanations | ✅ Implemented | Temporal segment replacement scoring |
| Digital Skill Passport | ✅ Implemented | SHA-256 hashed HTML + QR code |
| Dispute Resolution |  Prototype | CLIP zero-shot before/after comparison |