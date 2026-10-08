import numpy as np
import soundfile as sf
from scipy import signal
import os
import warnings

warnings.filterwarnings("ignore")

print("🔄 Creating Noisy Audio Test Samples...")

# 1. Load your original clean audio (from Day 1)
clean_audio_path = "sample_hindi.wav"

if not os.path.exists(clean_audio_path):
    print(f"⚠️ {clean_audio_path} not found. Creating a synthetic test tone instead.")
    # Create a synthetic 3-second audio at 16kHz
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
else:
    clean_audio, sample_rate = sf.read(clean_audio_path)
    if len(clean_audio.shape) > 1:  # Convert stereo to mono
        clean_audio = np.mean(clean_audio, axis=1)

print(f"✅ Clean audio loaded: {len(clean_audio)/sample_rate:.2f} seconds at {sample_rate}Hz")

# 2. Add different types of noise
def add_white_noise(audio, noise_level=0.1):
    """Add random white noise"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def add_babble_noise(audio, noise_level=0.2):
    """Simulate background chatter (pink noise approximation)"""
    noise = np.random.randn(len(audio))
    # Apply a low-pass filter to make it sound like crowd noise
    b, a = signal.butter(4, 800 / (sample_rate / 2), btype='low')
    noise = signal.filtfilt(b, a, noise) * noise_level
    return audio + noise

def add_construction_noise(audio, noise_level=0.3):
    """Simulate construction/machinery noise (low-frequency rumble)"""
    noise = np.random.randn(len(audio))
    b, a = signal.butter(4, 200 / (sample_rate / 2), btype='low')
    noise = signal.filtfilt(b, a, noise) * noise_level
    return audio + noise

# 3. Save noisy versions
noisy_samples = {
    "noisy_white.wav": add_white_noise(clean_audio, noise_level=0.1),
    "noisy_babble.wav": add_babble_noise(clean_audio, noise_level=0.2),
    "noisy_construction.wav": add_construction_noise(clean_audio, noise_level=0.3)
}

for filename, noisy_audio in noisy_samples.items():
    # Normalize to prevent clipping
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio)) * 0.9
    sf.write(filename, noisy_audio, sample_rate)
    print(f"✅ Created: {filename}")

print("\n🎧 Next Step: Test these noisy files with your Whisper model!")
print("   Run: python test_whisper_noisy.py")