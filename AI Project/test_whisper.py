import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

print("🔄 Loading Whisper Medium model... (This may take 1-2 minutes)")

# 1. Detect device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 2. Use the official, public OpenAI Whisper Medium model (excellent Hindi/Tamil support)
model_id = "openai/whisper-medium"

try:
    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create the transcription pipeline
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
    print("✅ Model loaded successfully!")

    # 3. Test on your sample audio file
    audio_file = "sample_hindi.wav" 
    
    if os.path.exists(audio_file):
        print(f"🎙️ Transcribing '{audio_file}'...")
        
        # We explicitly tell the model to expect Hindi (use "ta" for Tamil)
        generate_kwargs = {"language": "hindi"}
        
        result = pipe(audio_file, generate_kwargs=generate_kwargs)
        
        print("\n📝 --- TRANSCRIPTION RESULT ---")
        print(result["text"])
        print("------------------------------\n")
    else:
        print(f"⚠️ Could not find '{audio_file}'.")
        print("👉 Please run: python download_sample.py first!")

except Exception as e:
    print(f"❌ Error loading model: {e}")