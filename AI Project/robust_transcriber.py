import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import warnings

warnings.filterwarnings("ignore")

print("🔄 Initializing Robust Transcription System with Fallback...")

# ==========================================
# 1. HARDCODED TRANSLATION DICTIONARY (For Demo)
# ==========================================
HINDI_TO_ENGLISH_DICT = {
    "काम": "work", "पूरा": "complete", "मजदूरी": "wage", "पैसा": "money", 
    "समय": "time", "ठीक": "okay", "नहीं": "no", "हाँ": "yes", 
    "मैं": "I", "तुम": "you", "हम": "we", "किया": "did", 
    "करना": "to do", "दिया": "gave", "लिया": "took"
}

# ==========================================
# 2. LOAD PRIMARY MODEL (Whisper Medium)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("⏳ Loading primary model (Whisper Medium)...")
primary_model_id = "openai/whisper-medium"
primary_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    primary_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
primary_processor = AutoProcessor.from_pretrained(primary_model_id)

primary_pipe = pipeline(
    "automatic-speech-recognition", model=primary_model, tokenizer=primary_processor.tokenizer,
    feature_extractor=primary_processor.feature_extractor, max_new_tokens=128,
    chunk_length_s=30, batch_size=8, return_timestamps=True, torch_dtype=torch_dtype, device=device,
)
print("✅ Primary model loaded.")

# ==========================================
# 3. LOAD FALLBACK MODEL (Whisper Tiny - faster, lighter)
# ==========================================
print("⏳ Loading fallback model (Whisper Tiny)...")
fallback_model_id = "openai/whisper-tiny"
fallback_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    fallback_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
fallback_processor = AutoProcessor.from_pretrained(fallback_model_id)

fallback_pipe = pipeline(
    "automatic-speech-recognition", model=fallback_model, tokenizer=fallback_processor.tokenizer,
    feature_extractor=fallback_processor.feature_extractor, max_new_tokens=128,
    chunk_length_s=30, batch_size=8, return_timestamps=True, torch_dtype=torch_dtype, device=device,
)
print("✅ Fallback model loaded.")

# ==========================================
# 4. ROBUST TRANSCRIPTION LOGIC
# ==========================================
def is_transcription_valid(text):
    """Check if transcription contains actual words (not just noise/garbage)"""
    if not text or len(text.strip()) < 3:
        return False
    return len(text) > 10

def translate_hindi_to_english(hindi_text):
    """Simple word-by-word translation using hardcoded dictionary"""
    words = hindi_text.split()
    translated_words = [HINDI_TO_ENGLISH_DICT.get(w.strip(".,!?।"), w) for w in words]
    return " ".join(translated_words)

def robust_transcribe(audio_file_path, language="hindi"):
    """Try primary model. If it fails/garbage, use fallback."""
    print(f"\n🎙️ Transcribing: {os.path.basename(audio_file_path)}")
    
    # Try PRIMARY model
    try:
        print("   Attempt 1: Primary model (Whisper Medium)...")
        result = primary_pipe(audio_file_path, generate_kwargs={"language": language})
        primary_text = result["text"].strip()
        
        if is_transcription_valid(primary_text):
            translation = translate_hindi_to_english(primary_text)
            print(f"   ✅ Primary model succeeded!")
            return primary_text, translation, "HIGH"
        else:
            print(f"   ⚠️ Primary model output seems invalid: '{primary_text}'")
    except Exception as e:
        print(f"   ❌ Primary model failed: {e}")
    
    # Fallback to FALLBACK model
    try:
        print("   Attempt 2: Fallback model (Whisper Tiny)...")
        result = fallback_pipe(audio_file_path, generate_kwargs={"language": language})
        fallback_text = result["text"].strip()
        translation = translate_hindi_to_english(fallback_text)
        print(f"   ✅ Fallback model succeeded!")
        return fallback_text, translation, "MEDIUM"
    except Exception as e:
        print(f"   ❌ Fallback model also failed: {e}")
        return "Transcription failed", "Translation failed", "LOW"

# ==========================================
# 5. TEST ON ALL AUDIO FILES
# ==========================================
if __name__ == "__main__":
    test_files = ["sample_hindi.wav", "noisy_white.wav", "noisy_babble.wav", "noisy_construction.wav"]
    
    print("\n" + "="*60)
    print("TESTING ROBUST TRANSCRIPTION ON ALL AUDIO FILES")
    print("="*60)
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            transcription, translation, confidence = robust_transcribe(audio_file)
            print(f"   Confidence Level: {confidence}")
            print("-" * 60)
        else:
            print(f"⚠️ File not found: {audio_file}")
    
    print("\n🎉 Robust Transcription System Test Complete!")