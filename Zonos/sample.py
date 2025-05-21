import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

cond_dict = make_cond_dict(text="Can you please stop talking to me! I'm so frustrated at what kind of works you have done!", speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)
print("Starting generation...")
codes = model.generate(conditioning, max_new_tokens=1000, disable_torch_compile=True)
print("Generation complete!")
wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
print("Autosaved as sample.wav")