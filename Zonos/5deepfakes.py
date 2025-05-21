import os
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import csv
import torch._dynamo
torch._dynamo.config.suppress_errors = True

SPEAKER_FILE = r"C:\Users\QiXuan\Downloads\Youtube\audio_wav_only\GayMengSeng.mp3.wav"
OUTPUT_DIR = r"C:\Users\QiXuan\Downloads\Audio Deepfakes"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILE = r"C:\Users\QiXuan\Downloads\Audio Deepfakes\audio_list.csv"  # Path to the CSV file


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

# Prompt for trimming
MIN_DURATION = 5
MAX_DURATION = 30
print(f"Audio duration is {duration:.2f} seconds.")
start_time = float(input(f"Enter the start time (in seconds) for the segment (0 to {duration:.2f}): ").strip())
if start_time < 0 or start_time > duration:
    print(f"Error: Start time must be between 0 and {duration:.2f} seconds.")
    exit(1)

end_time = float(input(f"Enter the end time (in seconds) for the segment ({start_time:.2f} to {duration:.2f}): ").strip())
if end_time <= start_time or end_time > duration:
    print(f"Error: End time must be greater than start time ({start_time:.2f}) and less than or equal to {duration:.2f} seconds.")
    exit(1)

trim_duration = end_time - start_time
if trim_duration < MIN_DURATION or trim_duration > MAX_DURATION:
    print(f"Error: Trimmed duration ({trim_duration:.2f} seconds) must be between {MIN_DURATION} and {MAX_DURATION} seconds.")
    exit(1)

start_sample = int(start_time * sampling_rate)
end_sample = int(end_time * sampling_rate)
trimmed_wav = wav[:, start_sample:end_sample]

trimmed_duration = trimmed_wav.shape[1] / sampling_rate
print(f"Trimmed audio duration: {trimmed_duration:.2f} seconds")
if trimmed_duration < 5 or trimmed_duration > 30:
    print("Warning: Zonos recommends 5-30 seconds for optimal voice cloning. Results may vary.")

# Create speaker embedding
try:
    embedding = model.make_speaker_embedding(trimmed_wav, sampling_rate)
    speaker_name = os.path.splitext(os.path.basename(SPEAKER_FILE))[0]
    print(f"Loaded speaker: {speaker_name}")
except Exception as e:
    print(f"Error creating speaker embedding: {e}")
    exit(1)

# Set language code to en-gb or whatever language he speaks
print("\nLanguage code: Zonos uses this for phoneme conversion (default 'en-gb' and there is no Singaporean english fyi).")
print("Examples: 'en-us' (American English), 'en-gb' (British English).")
language = input("Enter language code (default 'en-gb'): ").strip() or "en-gb"

# Loop to generate 5 deepfakes
NUM_DEEPFAKES = 5
for i in range(1, NUM_DEEPFAKES + 1):
    print(f"\n=== Generating Deepfake {i}/{NUM_DEEPFAKES} ===")

    TEXT_PROMPT = input(f"Enter the text for deepfake {i} (e.g., 'Can you please stop winning the election?'): ").strip()
    if not TEXT_PROMPT:
        print("Error: Text prompt cannot be empty.")
        exit(1)

    # Prompt for emotion vector
    print(f"\nEmotion conditioning for deepfake {i}: Specify an 8D vector for emotions [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral].")
    print("Presets: 1=Happiness, 2=Sadness, 3=Disgust, 4=Fear, 5=Surprise, 6=Anger, 7=Other, 8=Neutral")
    emotion_choice = input(f"Enter preset number (1-8) or 'custom' to enter your own vector (default 8 for Neutral): ").strip() or "8"

    emotion_presets = {
        "1": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Happiness
        "2": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Sadness
        "3": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Disgust
        "4": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Fear
        "5": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Surprise
        "6": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Anger
        "7": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Other
        "8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Neutral
    }

    if emotion_choice.lower() == "custom":
        print("Enter the 8D emotion vector as 8 numbers (0.0 to 1.0) separated by spaces:")
        print("Format: Happiness Sadness Disgust Fear Surprise Anger Other Neutral")
        emotion_input = input("Example: 0.8 0.1 0.0 0.0 0.1 0.0 0.0 0.0 (mostly Happy): ").strip()
        emotion_vector = [float(x) for x in emotion_input.split()]
        if len(emotion_vector) != 8 or any(x < 0.0 or x > 1.0 for x in emotion_vector):
            print("Error: Emotion vector must be 8 numbers between 0.0 and 1.0.")
            exit(1)
    else:
        if emotion_choice not in emotion_presets:
            print("Error: Invalid preset number. Choose 1-8 or 'custom'.")
            exit(1)
        emotion_vector = emotion_presets[emotion_choice]

    emotion_tensor = torch.tensor(emotion_vector, dtype=torch.float32).to(model.device)

    # Prompt for pitch and speaking rate
    print(f"\nAdjust the following to control the tone and style of deepfake {i}:")
    print("Pitch variation (0 to 300): Higher values = more variation")
    pitch_std = float(input("Enter pitch variation (default 120): ").strip() or "120")
    if pitch_std < 0 or pitch_std > 300:
        print("Error: Pitch variation must be between 0 and 300.")
        exit(1)

    print("Speaking rate (0 to 30): Higher is Faster")
    speaking_rate = float(input("Enter speaking rate (default 12): ").strip() or "12")
    if speaking_rate < 0 or speaking_rate > 30:
        print("Error: Speaking rate must be between 0 and 30.")
        exit(1)

    print("Fmax (Hz, 0 to 24000): Maximum frequency for the generated audio")
    fmax = float(input("Enter fmax (default 24000): ").strip() or "24000")
    if fmax < 0 or fmax > 24000:
        print("Error: Fmax must be between 0 and 24000.")
        exit(1)

    print("CFG scale (1 to 5): Higher values = closer to the speaker's voice")
    cfg_scale = float(input("Enter CFG scale (default 4): ").strip() or "4")
    if cfg_scale < 1 or cfg_scale > 5:
        print("Error: CFG scale must be between 1 and 5.")
        exit(1)

    print("Random seed: Controls generation randomness")
    seed = int(input("Enter random seed (default 42): ").strip() or "42")

    # Generate deepfake audio
    print(f"\nGenerating deepfake {i}/{NUM_DEEPFAKES} for {speaker_name} with emotion vector {emotion_vector}, pitch_std={pitch_std}, speaking_rate={speaking_rate}...")

    cond_dict = make_cond_dict(
        text=TEXT_PROMPT,
        speaker=embedding,
        language=language,
        speaking_rate=speaking_rate,
        pitch_std=pitch_std,
        fmax=fmax,
        emotion=emotion_tensor,
    )
    conditioning = model.prepare_conditioning(cond_dict)

    try:
        torch.manual_seed(seed)
        codes = model.generate(conditioning, cfg_scale=cfg_scale)
        wavs = model.autoencoder.decode(codes).cpu()

        emotion_label = f"emotion_{emotion_choice}" if emotion_choice in emotion_presets else "emotion_custom"
        output_filename = f"{speaker_name}_deepfake_{i}_start_{int(start_time)}s_end_{int(end_time)}s_{emotion_label}_pitch_{int(pitch_std)}_rate_{int(speaking_rate)}_fmax_{int(fmax)}_cfg_{cfg_scale}_seed_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        print(f"Saved deepfake audio: {output_path}")

        # Update CSV file with new deepfake entry
        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                output_filename,
                "TTS",
                "Zonos",
                os.path.basename(SPEAKER_FILE),
                "spoof"
            ])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Error generating deepfake audio {i}: {e}")
        if "CUDA out of memory" in str(e):
            print("Try shorter text or switch to CPU (edit script: device='cpu').")
        exit(1)

print("\nAll 5 deepfake audios generated successfully!")