import os
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import torch._dynamo
torch._dynamo.config.suppress_errors = True

SPEAKER_FILE = r"C:\Users\QiXuan\Downloads\Youtube\audio_wav_only\AnthonyAlbaneseElection.wav"  
OUTPUT_DIR = r"C:\Users\QiXuan\Downloads\Audio Deepfake Model\TTS\Zonos\Sample Deepfake audio"  # Directory to save the output audio
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Zonos model
try:
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
    print("Model loaded on GPU")
except RuntimeError as e:
    print(f"GPU loading failed: {e}. Switching to CPU.")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")

# Load speaker audio file
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

# Prompt for start and end points
start_time = float(input(f"Enter the start time (in seconds) for the segment (0 to {duration:.2f}): ").strip())
if start_time < 0 or start_time > duration:
    print(f"Error: Start time must be between 0 and {duration:.2f} seconds.")
    exit(1)

end_time = float(input(f"Enter the end time (in seconds) for the segment ({start_time:.2f} to {duration:.2f}): ").strip())
if end_time <= start_time or end_time > duration:
    print(f"Error: End time must be greater than start time ({start_time:.2f}) and less than or equal to {duration:.2f} seconds.")
    exit(1)

# Calculate duration and validate
trim_duration = end_time - start_time
if trim_duration < MIN_DURATION or trim_duration > MAX_DURATION:
    print(f"Error: Trimmed duration ({trim_duration:.2f} seconds) must be between {MIN_DURATION} and {MAX_DURATION} seconds.")
    exit(1)

# Trim the audio
start_sample = int(start_time * sampling_rate)
end_sample = int(end_time * sampling_rate)
trimmed_wav = wav[:, start_sample:end_sample]

# Verify trimmed duration
trimmed_duration = trimmed_wav.shape[1] / sampling_rate
print(f"Trimmed audio duration: {trimmed_duration:.2f} seconds")
if trimmed_duration < 5 or trimmed_duration > 30:
    print("Warning: Zonos recommends 5-30 seconds for optimal voice cloning. Results may vary.")

# Create speaker embedding from trimmed audio
try:
    embedding = model.make_speaker_embedding(trimmed_wav, sampling_rate)
    speaker_name = os.path.splitext(os.path.basename(SPEAKER_FILE))[0]
    print(f"Loaded speaker: {speaker_name}")
except Exception as e:
    print(f"Error creating speaker embedding: {e}")
    exit(1)

# Prompt for text to synthesize
TEXT_PROMPT = input("Enter the text you want to synthesize: ").strip()
if not TEXT_PROMPT:
    print("Error: Text prompt cannot be empty.")
    exit(1)

# Prompt for language code
print("\nLanguage code: Zonos uses this for phoneme conversion (default 'en-gb').")
print("Examples: 'en-us' (American English), 'es-es' (Spanish). Note: Voice cloning quality may vary with non-English languages.")
language = input("Enter language code (default 'en-gb'): ").strip() or "en-gb"

# Prompt for pitch and speaking rate
print("\nAdjust the following to control the tone and style of the voice:")
print("Pitch variation (0 to 300): Higher values (180-250) = more dynamic/engaging, Lower values (50-100) = more monotone")
pitch_std = float(input("Enter pitch variation (default 150): ").strip() or "150")
if pitch_std < 0 or pitch_std > 300:
    print("Error: Pitch variation must be between 0 and 300.")
    exit(1)

print("Speaking rate (0 to 30): Faster (20-30) = energetic/urgent, Slower (5-10) = calm/deliberate")
speaking_rate = float(input("Enter speaking rate (default 12): ").strip() or "12")
if speaking_rate < 0 or speaking_rate > 30:
    print("Error: Speaking rate must be between 0 and 30.")
    exit(1)

# Prompt for fmax
print("Fmax (Hz, 0 to 24000): Maximum frequency for the generated audio. Higher values = brighter sound, Lower values = muffled sound")
fmax = float(input("Enter fmax (default 24000): ").strip() or "24000")
if fmax < 0 or fmax > 24000:
    print("Error: Fmax must be between 0 and 24000.")
    exit(1)

# Prompt for CFG scale
print("CFG scale (1 to 5): Classifier-free guidance scale. Higher values = closer to the speaker's voice, but may reduce quality if too high")
cfg_scale = float(input("Enter CFG scale (default 4): ").strip() or "4")
if cfg_scale < 1 or cfg_scale > 5:
    print("Error: CFG scale must be between 1 and 5.")
    exit(1)

# Prompt for random seed
print("Random seed: Controls generation randomness. Use the same seed for reproducibility, or change for variation")
seed = int(input("Enter random seed (default 42): ").strip() or "42")

# Prompt for emotion vector
print("\nEmotion conditioning: Specify an 8D vector for emotions [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral].")
print("You can select a preset emotion or enter a custom vector.")
print("Presets: 1=Happiness, 2=Sadness, 3=Disgust, 4=Fear, 5=Surprise, 6=Anger, 7=Other, 8=Neutral")
emotion_choice = input("Enter preset number (1-8) or 'custom' to enter your own vector (default 8 for Neutral): ").strip() or "8"

# Define preset emotion vectors (1.0 for the chosen emotion, 0.0 for others)
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

# Convert emotion vector to tensor
emotion_tensor = torch.tensor(emotion_vector, dtype=torch.float32).to(model.device)

# Generate deepfake audio
print(f"\nGenerating deepfake audio for {speaker_name} with emotion vector {emotion_vector}, pitch_std={pitch_std}, speaking_rate={speaking_rate}...")

# Prepare conditioning with emotion
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

# Generate audio
try:
    torch.manual_seed(seed)
    codes = model.generate(conditioning, cfg_scale=cfg_scale)
    wavs = model.autoencoder.decode(codes).cpu()

    # Save the output
    emotion_label = f"emotion_{emotion_choice}" if emotion_choice in emotion_presets else "emotion_custom"
    output_filename = f"{speaker_name}_deepfake_start_{int(start_time)}s_end_{int(end_time)}s_{emotion_label}_pitch_{int(pitch_std)}_rate_{int(speaking_rate)}_fmax_{int(fmax)}_cfg_{cfg_scale}_seed_{seed}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
    print(f"Saved deepfake audio: {output_path}")
except RuntimeError as e:
    print(f"Error generating deepfake audio: {e}")
    if "CUDA out of memory" in str(e):
        print("Try shorter text or switch to CPU (edit script: device='cpu').")
    exit(1)

print("Deepfake audio generation complete!")

##Forgot to work on updating CSV files here but you can do it the same way as the 5deepfakes.py file