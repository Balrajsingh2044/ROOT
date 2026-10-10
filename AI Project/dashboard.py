import gradio as gr
import requests
import os
import warnings
from robust_transcriber import robust_transcribe
from gtts import gTTS
import time

warnings.filterwarnings("ignore")

# Shreyansh's Backend URL
BACKEND_URL = "http://127.0.0.1:8000/api/assess"
# ==========================================
def run_full_assessment(video_file, audio_file):
    if video_file is None:
        raise gr.Error("Please upload a task video first!")
        
    # ==========================================
    # 1. CALL SHREYANSH'S BACKEND (Video Analysis & Wage)
    # ==========================================
    try:
        with open(video_file, "rb") as f:
            files = {"file": f}
            data = {"task_type": "plumbing", "complexity": 3}
            response = requests.post(BACKEND_URL, files=files, data=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
    except requests.exceptions.ConnectionError:
        raise gr.Error("❌ Cannot connect to backend! Is Shreyansh's FastAPI server running on port 8000?")
    except Exception as e:
        raise gr.Error(f"❌ Backend processing failed: {str(e)}")

    # ==========================================
    # 2. EXTRACT BACKEND RESULTS (WITH DEFENSIVE CHECKS)
    # ==========================================
    status_emoji = "✅" if result.get('deepfake_status') == 'PASS' else "❌"
    deepfake_status = f"### {status_emoji} **Integrity Check: {result.get('deepfake_status', 'UNKNOWN')}**"
    
    score = result.get('score', 0)
    
    # CRITICAL FIX: If heatmap is a URL, set it to None to prevent Gradio from crashing
    heatmap_value = result.get('heatmap')
    if heatmap_value and (heatmap_value.startswith('http://') or heatmap_value.startswith('https://')):
        heatmap_value = None  # Don't try to download external URLs
    
    wage_range_text = f"### 💰 **Fair Wage Range**\n# {result.get('wage_range_text', 'N/A')}"

    # ==========================================
    # 3. PROCESS AUDIO LOCALLY (Your Day 5 Robust System)
    # ==========================================
    audio_filename = None
    if audio_file:
        hindi_text, english_text, confidence = robust_transcribe(audio_file, language="hindi")
        feedback_text = f"ट्रान्सक्रिप्शन: {hindi_text} | Confidence: {confidence}"
        
        try:
            # CRITICAL FIX: Use a unique timestamped filename to prevent Gradio file-locking crashes
            unique_id = int(time.time() * 1000)
            audio_filename = f"feedback_hindi_{unique_id}.mp3"
            
            tts = gTTS(text=feedback_text, lang='hi')
            tts.save(audio_filename)
            
            # Double-check the file was actually created and isn't empty
            import os
            if not os.path.exists(audio_filename) or os.path.getsize(audio_filename) == 0:
                print("⚠️ TTS file was empty or failed to save.")
                audio_filename = None
                
        except Exception as e:
            print(f"TTS Generation Error: {e}")
            audio_filename = None
    else:
        feedback_text = "कोई ऑडियो नहीं मिला।"

    # Return exactly 5 variables to match the Gradio outputs
    return deepfake_status, score, heatmap_value, wage_range_text, audio_filename

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
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Worker Voice Note (Hindi/Tamil)")
            submit_btn = gr.Button("🚀 Run Full AI Assessment", variant="primary", size="lg")
            
        # RIGHT COLUMN: OUTPUTS (DASHBOARD)
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Assessment Dashboard")
            
            # 1. Deepfake Status
            deepfake_out = gr.Markdown(label="Integrity Status", value="*Waiting for assessment...*")
            
            # 2. Technique Score & Heatmap
            with gr.Row():
                score_out = gr.Number(label="Technique Score (%)", interactive=False)
                heatmap_out = gr.Image(label="Technique Heatmap (Grad-CAM)", height=200)
                
            # 3. Fair Wage Range
            wage_out = gr.Markdown(label="Fair Wage Range (95% Confidence)", value="*Waiting...*")
            
            # 4. Vernacular Audio Feedback
            gr.Markdown("### 🔊 Vernacular Audio Feedback")
            tts_out = gr.Audio(label="AI Feedback (Hindi)", type="filepath", interactive=False)

    # Link button to function
    submit_btn.click(
        fn=run_full_assessment,
        inputs=[video_input, audio_input],
        outputs=[deepfake_out, score_out, heatmap_out, wage_out, tts_out]
    )

if __name__ == "__main__":
    demo.launch()