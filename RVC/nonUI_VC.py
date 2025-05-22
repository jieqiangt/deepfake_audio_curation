import os
import csv
from rvc_python.infer import RVCInference
import soundfile as sf
import numpy as np
import torch
from fairseq.data.dictionary import Dictionary
from pydub import AudioSegment
torch.serialization.add_safe_globals([Dictionary])

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA not available, falling back to CPU")

source_audio = r"C:\Users\QiXuan\Downloads\Youtube\audio_wav_only\MichealThngIntro.mp3.wav"
model_path = r"C:\Users\QiXuan\Downloads\Audio Deepfake Model\RVC\actual_models\model\joebiden.pth"
file_index = r"C:\Users\QiXuan\Downloads\Audio Deepfake Model\RVC\actual_models\index\joebiden.index"
output_audio = r"C:\Users\QiXuan\Downloads\Audio Deepfakes\MichaelThng_toBiden.wav"
csv_file = r"C:\Users\QiXuan\Downloads\Audio Deepfakes\audio_list.csv"
trimmed_audio = "trimmed_source.wav"

# Prompt user for trimming
print(f"Source audio: {source_audio}")
trim = input("Would you like to trim the audio? (yes/no): ").lower()
if trim == "yes":
    audio = AudioSegment.from_wav(source_audio)
    duration_ms = len(audio)
    duration_s = duration_ms / 1000
    print(f"Audio duration: {duration_s:.2f} seconds")
    while True:
        try:
            start_time = float(input(f"Enter start time (seconds, 0 to {duration_s}): "))
            end_time = float(input(f"Enter end time (seconds, {start_time} to {duration_s}): "))
            if 0 <= start_time < end_time <= duration_s:
                break
            else:
                print("Invalid times. Ensure 0 <= start < end <= duration.")
        except ValueError:
            print("Please enter valid numbers.")
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    trimmed = audio[start_ms:end_ms]
    trimmed.export(trimmed_audio, format="wav")
    print(f"Trimmed audio saved to {trimmed_audio}")
    input_audio = trimmed_audio
else:
    input_audio = source_audio

rvc = RVCInference(model_path=model_path, device=device, version="v2")

# Change parameters as needed (tinker around)
result = rvc.vc.vc_single(
    sid=0,
    input_audio_path=input_audio,
    f0_up_key=0,
    f0_file=None,
    f0_method="harvest",
    file_index=file_index,
    file_index2="",
    index_rate=0.6,
    filter_radius=0,
    resample_sr=0,
    rms_mix_rate=0.2,
    protect=0.15
)
if isinstance(result, tuple) and len(result) == 2:
    info, audio_opt = result
    if audio_opt[0] is None:
        print(f"Conversion failed: {info}")
        exit(1)
    audio_data = audio_opt[1]
else:
    audio_data = result

# Save the audio file
tgt_sr = rvc.vc.tgt_sr
print("Output audio shape:", audio_data.shape, "max:", np.abs(audio_data).max())
sf.write(output_audio, audio_data, tgt_sr)
print(f"Voice conversion complete! Output saved as {output_audio}")

filename_only = os.path.basename(output_audio)
source_audio_only = os.path.basename(source_audio)
metadata = {
    "filename": filename_only,
    "deepfake_method": "VC",
    "model_used": "RVC",
    "original_audio":source_audio_only,
    "label": "spoof",
}

# Add to CSV
with open(csv_file, mode='a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metadata.keys())
    if os.stat(csv_file).st_size == 0:
        writer.writeheader()
    writer.writerow(metadata)
print(f"Metadata appended to {csv_file}")

if os.name == "nt":
    os.system(f'start "" "{output_audio}"')

if trim == "yes" and os.path.exists(trimmed_audio):
    os.remove(trimmed_audio)
    print(f"Cleaned up temporary file: {trimmed_audio}")