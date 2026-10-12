import gradio as gr
import requests
import os
import warnings
from robust_transcriber import robust_transcribe
from gtts import gTTS
import time
import tempfile
import base64
import tempfile

warnings.filterwarnings("ignore")


BACKEND_URL = "http://127.0.0.1:8000/api/assess"
LANG_CODES = {"hindi": "hi", "tamil": "ta"}

def data_uri_to_file(uri):
    """Convert 'data:image/jpeg;base64,...' into a temp file Gradio can display."""
    if not uri or not uri.startswith("data:image"):
        return None
    raw = base64.b64decode(uri.split(",", 1)[1])
    path = os.path.join(tempfile.gettempdir(), f"heatmap_{int(time.time()*1000)}.jpg")
    with open(path, "wb") as f:
        f.write(raw)
    return path

def run_full_assessment(video_file, audio_file, language="hindi"):
    if video_file is None:
        raise gr.Error("Please upload a task video first!")

    # 1. Call backend
    try:
        with open(video_file, "rb") as f:
            files = {"file": (os.path.basename(video_file), f)}
            data = {"task_type": "plumbing", "complexity": 3}
            response = requests.post(BACKEND_URL, files=files, data=data, timeout=180)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        raise gr.Error("❌ Cannot connect to backend. Is the FastAPI server running on port 8000?")
    except requests.exceptions.HTTPError:
        detail = response.json().get("detail", response.text) if response.content else ""
        raise gr.Error(f"❌ Backend error: {detail}")
    except Exception as e:
        raise gr.Error(f"❌ Backend processing failed: {e}")

    # 2. Integrity result
    passed = result.get("deepfake_status") == "PASS"
    emoji = "✅" if passed else "❌"
    deepfake_status = f"### {emoji} **Integrity Check: {result.get('deepfake_status', 'UNKNOWN')}**"

    # FAIL: stop here, show a clear message
    if not passed:
        return (
            deepfake_status + "\n\nThis video appears looped or manipulated. Please upload a genuine task video.",
            0,
            '<p style="color:gray;">No heatmap (integrity check failed).</p>',
            "### 💰 **Estimated Wage Range**\n# N/A",
            None,
        )

    # 3. Score as percentage
    raw_score = result.get("score", 0) or 0
    score_pct = round(raw_score * 100, 1) if raw_score <= 1 else round(raw_score, 1)

    # 4. Wage text
    wage_text = result.get("wage_range_text", "N/A")
    wage_md = f"### 💰 **Estimated Wage Range**\n# {wage_text}"

    # 5. Heatmap: accept URL or base64
    hm = result.get("heatmap")
    if isinstance(hm, str) and hm.startswith(("data:image", "http")):
        heatmap_html = f'<img src="{hm}" style="max-width:100%; height:auto; border-radius:8px;">'
    else:
        heatmap_html = '<p style="color:gray;">No heatmap available.</p>'

    # 6. Optional transcription (never blocks the result)
    transcript_note = ""
    if audio_file:
        try:
            text, _english, conf = robust_transcribe(audio_file, language=language)
            transcript_note = f"{text}"
            print(f"Transcript ({conf}): {text}")
        except Exception as e:
            print(f"Transcription error: {e}")

    # 7. Spoken feedback built from the real results
    wage_spoken = wage_text.replace("₹", "").replace("-", "से")
    if language == "hindi":
        feedback_text = (
            f"आपका तकनीक स्कोर {int(score_pct)} प्रतिशत है। "
            f"उचित मजदूरी {wage_spoken} रुपये है।"
        )
    else:
        feedback_text = f"உங்கள் திறன் மதிப்பெண் {int(score_pct)} சதவீதம். மதிப்பிடப்பட்ட ஊதியம் {wage_spoken} ரூபாய்."

    audio_filename = None
    try:
        out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        out.close()
        gTTS(text=feedback_text, lang=LANG_CODES.get(language, "hi")).save(out.name)
        if os.path.getsize(out.name) > 0:
            audio_filename = out.name
    except Exception as e:
        print(f"TTS error (needs internet for gTTS): {e}")

    return deepfake_status, score_pct, heatmap_html, wage_md, audio_filename

# ==========================================
# BUILD THE GRADIO INTERFACE (gr.Blocks)
# ==========================================
with gr.Blocks(title="Skill Assessment MVP - Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ AI Skill Assessment & Fair Wage Dashboard")
    gr.Markdown("Upload a task video and a voice note to receive an integrity check, technique score, and fair wage prediction.")
    
    with gr.Row():
        # LEFT COLUMN: INPUTS
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Inputs")
            video_input = gr.Video(label="Upload Task Video", height=300)
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath",label="Worker Voice Note (optional)")
            lang_input = gr.Dropdown(["hindi", "tamil"], value="hindi", label="Language")
            submit_btn = gr.Button("🚀 Run Full AI Assessment", variant="primary", size="lg")
            
        # RIGHT COLUMN: OUTPUTS (DASHBOARD)
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Assessment Dashboard")
            
            # 1. Deepfake Status
            deepfake_out = gr.Markdown(label="Integrity Status", value="*Waiting for assessment...*")
            
            # 2. Technique Score & Heatmap
            with gr.Row():
                score_out = gr.Number(label="Technique Score (%)", interactive=False)
                heatmap_out = gr.HTML(label="Technique Heatmap (Grad-CAM)")

            wage_out = gr.Markdown(value="*Waiting...*")   # label removed, since Markdown ignores it anyway
            
            # 4. Vernacular Audio Feedback
            gr.Markdown("### 🔊 Vernacular Audio Feedback")
            tts_out = gr.Audio(label="AI Feedback", type="filepath", interactive=False)

    # Link button to function
    submit_btn.click(
        fn=run_full_assessment,
        inputs=[video_input, audio_input, lang_input],
        outputs=[deepfake_out, score_out, heatmap_out, wage_out, tts_out],
    )

if __name__ == "__main__":
    demo.launch()