"""TruWage demo UI.

Run the backend first:   uvicorn app.main:app --port 8000
Then:                    streamlit run streamlit_demo.py
"""
import base64
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API = os.environ.get("TRUWAGE_API", "http://127.0.0.1:8000")
TRADES = ["plumbing", "electrical", "carpentry", "welding", "tailoring", "painting", "masonry", "mechanic"]

st.set_page_config(page_title="TruWage", page_icon="\U0001F6E0", layout="centered")
st.title("TruWage: skill assessment and fair-wage estimate")
st.caption("Prototype. The wage model is trained on synthetic data, NSQF levels are illustrative, "
           "and the integrity check looks for frozen or looped video only.")

with st.sidebar:
    api_url = st.text_input("Backend URL", DEFAULT_API).rstrip("/")
    if st.button("Check backend"):
        try:
            health = requests.get(f"{api_url}/health", timeout=5).json()
            st.success(f"Backend is up. AQA model loaded: {health.get('aqa_model_loaded')}")
        except requests.RequestException as exc:
            st.error(f"Cannot reach the backend: {exc}")

col1, col2 = st.columns(2)
with col1:
    worker_name = st.text_input("Worker name (optional)")
    trade = st.selectbox("Trade shown in the video", TRADES, index=None, placeholder="Choose the trade (required)")
with col2:
    complexity = st.slider("Task complexity", 1, 5, 3)
    demand = st.selectbox("Local demand", ["low", "medium", "high"], index=1)

video = st.file_uploader("Work video (15-30 seconds)", type=["mp4", "mov", "avi", "mkv", "webm"])
if video is not None:
    st.video(video)

if video is not None and trade is None:
    st.info("Choose the trade shown in the video. The skill card and passport use this selection.")

if video is not None and st.button("Assess skill and estimate wage", type="primary", disabled=trade is None):
    with st.spinner("Checking integrity, scoring technique, estimating wage... (the first run loads the model)"):
        try:
            resp = requests.post(
                f"{api_url}/api/assess",
                files={"file": (video.name, video.getvalue(), video.type or "video/mp4")},
                data={"task_type": trade, "complexity": complexity, "demand": demand, "worker_name": worker_name},
                timeout=300,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {api_url}. Is it running? ({exc})")
            st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Backend error {resp.status_code}: {detail}")
        st.stop()

    data = resp.json()

    if data.get("deepfake_status") != "PASS":
        st.error("Integrity check failed")
        st.write(data.get("integrity_message") or "The video did not pass the integrity check.")
        st.info("No score or wage is produced for videos that fail this check.")
        st.stop()

    st.success("Assessment complete")
    m1, m2, m3 = st.columns(3)
    m1.metric("Technique score", f"{data['score']:.2f} / 1.00")
    m2.metric("Integrity check", "PASS")
    m3.metric("Fair wage range", data["wage_range_text"])
    st.caption(f"Wage model confidence: {data.get('wage_confidence', 'n/a')} "
               "(95% Monte-Carlo-dropout interval, synthetic training data)")

    heatmap = data.get("heatmap") or ""
    if heatmap.startswith("data:image"):
        st.subheader("Where the technique lost points")
        st.image(base64.b64decode(heatmap.split(",", 1)[1]),
                 caption="Grad-CAM overlay: red marks the regions that lowered the score",
                 use_container_width=True)

    st.subheader("Skill card")
    st.write(f"**Trade:** {data.get('trade')}  \n**Level:** {data.get('nsqf_level')}")
    if data.get("key_tasks"):
        st.write("**Key tasks:** " + ", ".join(data["key_tasks"]))
    st.caption("Mapped from the selected trade (keyword-based prototype). Levels are illustrative.")

    if data.get("passport_url"):
        st.subheader("Digital Skill Passport")
        try:
            page = requests.get(f"{api_url}{data['passport_url']}", timeout=15)
            page.raise_for_status()
            components.html(page.text, height=640, scrolling=True)
            st.download_button("Download passport (HTML)", page.text, file_name="truwage_passport.html", mime="text/html")
            st.caption(f"Verification hash: {data.get('verification_hash')}. "
                       "Tamper-evident record, not a digitally signed credential.")
        except requests.RequestException as exc:
            st.warning(f"Passport could not be loaded: {exc}")
