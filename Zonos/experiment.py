import os
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import csv

# To suppress inductor error
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Configuration
SPEAKER_FILE = r"C:\Users\QiXuan\Downloads\Youtube\audio_wav_only\PritamOnGSTHike.wav"
OUTPUT_DIR = r"C:\Users\QiXuan\Downloads\Audio Deepfakes"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILE = r"C:\Users\QiXuan\Downloads\Audio Deepfakes\audio_list.csv"

# Load the Zonos model
try:
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
    print("Model loaded on GPU")
except RuntimeError as e:
    print(f"GPU loading failed: {e}. Switching to CPU.")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")

# Load the speaker audio file
try:
    wav, sampling_rate = torchaudio.load(SPEAKER_FILE)
    duration = wav.shape[1] / sampling_rate
    print(f"Speaker audio duration: {duration:.2f} seconds")
except Exception as e:
    print(f"Error loading {SPEAKER_FILE}: {e}")
    exit(1)

# Prompt for a single high-quality segment
MIN_DURATION = 5
MAX_DURATION = 30
print(f"Audio duration is {duration:.2f} seconds.")
print("Tip: Choose a single segment with clear, natural Singlish speech (e.g., with 'lah', 'sia', minimal noise).")
start_time = float(input(f"Enter start time (0 to {duration:.2f}): ").strip())
if start_time < 0 or start_time > duration:
    print(f"Error: Start time must be between 0 and {duration:.2f} seconds.")
    exit(1)
end_time = float(input(f"Enter end time ({start_time:.2f} to {duration:.2f}): ").strip())
if end_time <= start_time or end_time > duration:
    print(f"Error: End time must be > {start_time:.2f} and <= {duration:.2f} seconds.")
    exit(1)
trim_duration = end_time - start_time
if trim_duration < MIN_DURATION or trim_duration > MAX_DURATION:
    print(f"Error: Segment duration ({trim_duration:.2f} seconds) must be 5-30 seconds.")
    exit(1)

start_sample = int(start_time * sampling_rate)
end_sample = int(end_time * sampling_rate)
trimmed_wav = wav[:, start_sample:end_sample]
trimmed_duration = trimmed_wav.shape[1] / sampling_rate
print(f"Trimmed audio duration: {trimmed_duration:.2f} seconds")
if trimmed_duration < 5 or trimmed_duration > 30:
    print("Warning: Zonos recommends 5-30 seconds for optimal voice cloning.")

# Create speaker embedding
try:
    embedding = model.make_speaker_embedding(trimmed_wav, sampling_rate)
    speaker_name = os.path.splitext(os.path.basename(SPEAKER_FILE))[0]
    print(f"Loaded speaker: {speaker_name}")
except Exception as e:
    print(f"Error creating speaker embedding: {e}")
    exit(1)

# Set language code
print("\nLanguage code: Zonos uses this for phoneme conversion (default 'en-gb' to reduce American accent).")
language = input("Enter language code (default 'en-gb'): ").strip() or "en-gb"

# Loop to generate 5 deepfakes
NUM_DEEPFAKES = 5
for i in range(1, NUM_DEEPFAKES + 1):
    print(f"\n=== Generating Deepfake {i}/{NUM_DEEPFAKES} ===")
    print("Tip: Use phonetic Singlish (e.g., 'Dis iz shiok lah, vote for me sia!') matching the segment's style.")
    TEXT_PROMPT = input(f"Enter text for deepfake {i}: ").strip()
    if not TEXT_PROMPT:
        print("Error: Text prompt cannot be empty.")
        exit(1)

    print("\nEmotion presets: 1=Happiness, 2=Sadness, 3=Disgust, 4=Fear, 5=Surprise, 6=Anger, 7=Other, 8=Neutral")
    emotion_choice = input("Enter preset (1-8, default 8): ").strip() or "8"
    emotion_vector = {"1": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      "2": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      "3": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      "4": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      "5": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      "6": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      "7": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      "8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}.get(emotion_choice, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    emotion_tensor = torch.tensor(emotion_vector, dtype=torch.float32).to(model.device)

    print("\nAdjust tone and style (optimized for Singlish):")
    pitch_std = float(input("Pitch variation (200-250, default 120): ").strip() or "120")  # Higher for expressiveness
    speaking_rate = float(input("Speaking rate (20-25, default 12): ").strip() or "12")  # Faster for Singlish pace
    fmax = float(input("Fmax (20000-22000, default 22000): ").strip() or "22000")  # Warmer tone
    cfg_scale = float(input("CFG scale (4-5, default 4.5): ").strip() or "4.5")  # Stronger speaker adherence
    seed = int(input("Seed (default 42): ").strip() or "42")

    if not (0 <= pitch_std <= 300 and 0 <= speaking_rate <= 30 and 0 <= fmax <= 24000 and 1 <= cfg_scale <= 5):
        print("Error: Invalid parameters.")
        exit(1)

    print(f"Generating deepfake {i}/{NUM_DEEPFAKES}...")
    cond_dict = make_cond_dict(text=TEXT_PROMPT, speaker=embedding, language=language,
                               speaking_rate=speaking_rate, pitch_std=pitch_std, fmax=fmax,
                               emotion=emotion_tensor)
    conditioning = model.prepare_conditioning(cond_dict)

    try:
        torch.manual_seed(seed)
        codes = model.generate(conditioning, cfg_scale=cfg_scale)
        wavs = model.autoencoder.decode(codes).cpu()
        output_filename = f"{speaker_name}_deepfake_{i}_start_{int(start_time)}s_end_{int(end_time)}s_emotion{emotion_choice}_pitch_{int(pitch_std)}_rate_{int(speaking_rate)}_fmax_{int(fmax)}_cfg_{cfg_scale}_seed_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        print(f"Saved: {output_path}")

        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([output_filename, "TTS", "Zonos", os.path.basename(SPEAKER_FILE), "spoof"])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Error: {e}")
        if "CUDA out of memory" in str(e):
            print("Try shorter text or lower cfg_scale.")
        exit(1)

print("\nAll deepfakes generated!")