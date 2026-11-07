"""Writes a reference fixture from the official loader and HF processor.

    python -I tools/py_fixture.py <model dir> <audio.wav> <out dir>

out/wave.f32      16 kHz mono input with the 0.25 s lead-in (what the processor sees)
out/feats.f32     processor features [T, 128]
out/fixture.json  T, valid frames, reference token ids, transcript, and token timestamps from the
                  processor (seconds from the start of wave.f32, lead-in included)
"""
import importlib.util
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

d, audio, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("official_bark", os.path.join(d, "bark_loader.py"))
bark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bark)

wave, sr = sf.read(audio, dtype="float32", always_2d=True)
wave = wave.mean(1)
if sr != 16000:
    import soxr
    wave = soxr.resample(wave, sr, 16000)
w = np.concatenate([np.zeros(4000, np.float32), wave.astype(np.float32)])
w.tofile(os.path.join(out, "wave.f32"))

from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
inp = proc([w], sampling_rate=16000, return_tensors="pt", padding=True)
f = inp["input_features"][0].float().numpy()
f.tofile(os.path.join(out, "feats.f32"))

model = bark.load_model(os.path.join(d, "model.bark"))
with torch.inference_mode():
    gen = model.generate(**inp)
seq = gen.sequences[0].tolist()
text = proc.batch_decode([seq], skip_special_tokens=True)[0].strip()
_, stamps = proc.decode(gen.sequences, durations=gen.durations, skip_special_tokens=True)
json.dump({"frames": f.shape[0], "valid": int(inp["attention_mask"].sum()), "ids": seq, "text": text,
           "tokens": stamps[0]}, open(os.path.join(out, "fixture.json"), "w"))
print(f.shape, text)
