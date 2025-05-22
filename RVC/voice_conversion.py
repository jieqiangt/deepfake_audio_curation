import os
from rvc_python.infer import RVCInference
import soundfile as sf
import numpy as np
import torch
from fairseq.data.dictionary import Dictionary
torch.serialization.add_safe_globals([Dictionary])

source_audio = r"C:\Users\QiXuan\Downloads\Youtube\audio_wav_only\TheresaMayBrexit.wav"
model_path = r"C:\Users\QiXuan\Downloads\HarryStyles.pth"
file_index = r"C:\Users\QiXuan\Downloads\HarryStyles.index"
output_audio = "TheresaMay_toHarry.wav"

# Initialize RVC
rvc = RVCInference(model_path=model_path, device="cpu", version="v2")

# Perform conversion
result = rvc.vc.vc_single(
    sid=0,  # Usually 0 for single-speaker models; check docs if different
    input_audio_path=source_audio,
    f0_up_key=-2,
    f0_file=None,
    f0_method="pm",
    file_index=file_index,
    file_index2="",
    index_rate=0.1,  # Full index influence
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33
)
if isinstance(result, tuple) and len(result) == 2:
    info, audio_opt = result
    if audio_opt[0] is None:
        print(f"Conversion failed: {info}")
        exit(1)
    audio_data = audio_opt[1]  # Extract audio from (tgt_sr, audio)
else:
    audio_data = result  # Direct audio array

print("Output audio shape:", audio_data.shape, "max:", np.abs(audio_data).max())
sf.write(output_audio, audio_data, rvc.vc.tgt_sr)
print(f"Voice conversion complete! Output saved to {output_audio}")

# Play output (Windows)
if os.name == "nt":
    os.system(f"start {output_audio}")