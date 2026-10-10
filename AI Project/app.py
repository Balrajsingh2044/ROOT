import gradio as gr
import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

print("🔄 Loading Whisper model for the UI... (Please wait 1-2 minutes)")

# 1. Setup device and model (Same as your test script)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-medium"

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("✅ Model loaded and ready in UI!")
except Exception as e:
    print(f"⚠️ Model load warning: {e}")
    pipe = None

# 2. The function that runs when you click the button
def transcribe_audio(audio_file_path):
    if audio_file_path is None:
        return "⚠️ Please upload or record an audio file first."
    if pipe is None:
        return "❌ Model failed to load. Check the VS Code terminal for errors."
    
    try:
        print(f"🎙️ Processing: {os.path.basename(audio_file_path)}")
        # Force Hindi transcription (change "hindi" to "tamil" if testing Tamil audio)
        result = pipe(audio_file_path, generate_kwargs={"language": "hindi"})
        return f"✅ Transcription Successful:\n\n{result['text']}"
    except Exception as e:
        return f"❌ Error during transcription: {e}"

# 3. Build the Gradio Interface
with gr.Blocks(title="Skill Assessment MVP", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ Skill Assessment MVP - Voice Note Analyzer")
    gr.Markdown("Upload or record a short voice note (Hindi/Tamil) to get an AI transcription.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Input")
            # Allows both file upload AND direct microphone recording!
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record Voice Note")
            submit_btn = gr.Button("Transcribe & Analyze", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Output")
            text_output = gr.Textbox(label="Transcription Result", lines=6, interactive=False)
            
    # Connect the button to the function
    submit_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=text_output)

if __name__ == "__main__":
    # Launch the UI
    demo.launch()