"""Export Underdog Bark 3.0 (ParakeetForTDT in a .bark file) to ONNX and check it.

  python -I bark_export.py <bark model dir> <audio.wav> <out dir>

Writes encoder.onnx (features [1,T,128] + mask -> projected frames [1,T',640] + length),
decoder.onnx (one LSTM step), joint.onnx (token + duration logits), vocab.json, and
compares an ONNX Runtime greedy TDT decode with the original model.generate transcript.
"""
import importlib.util, json, os, sys, time
sys.path += [p for p in [os.environ.get("BARK_PYLIB")] if p]  # optional site dir for onnx
import numpy as np, soundfile as sf, torch

model_dir, audio, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("official_bark", os.path.join(model_dir, "bark_loader.py"))
bark = importlib.util.module_from_spec(spec); spec.loader.exec_module(bark)
t0 = time.time()
model = bark.load_model(os.path.join(model_dir, "model.bark"))
print(f"loaded in {time.time() - t0:.1f}s")
cfg = model.config
V, BLANK, DUR = cfg.vocab_size, cfg.blank_token_id, list(cfg.durations)

wave, sr = sf.read(audio, dtype="float32", always_2d=True)
wave = wave.mean(1)
ref = bark.transcribe(model, wave, sr)
print("REF :", ref)

from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
if sr != 16000:
    import soxr; wave = soxr.resample(wave, sr, 16000)
w = np.concatenate([np.zeros(4000, np.float32), wave])
inp = proc([w], sampling_rate=16000, return_tensors="pt", padding=True)
feats, mask = inp["input_features"].float(), inp["attention_mask"].long()
print("features", tuple(feats.shape))


class Enc(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, features, mask):
        o = s.m.get_audio_features(input_features=features, attention_mask=mask, output_attention_mask=True)
        return o.pooler_output, o.attention_mask.sum(-1)


class Dec(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, token, h, c):
        y, (h2, c2) = s.d.lstm(s.d.embedding(token), (h, c))
        return s.d.decoder_projector(y)[:, 0], h2, c2


class Joint(torch.nn.Module):
    def __init__(s, j): super().__init__(); s.j = j
    def forward(s, enc, dec):
        return s.j.head(s.j.activation(enc + dec))


L, H = cfg.num_decoder_layers, cfg.decoder_hidden_size
with torch.inference_mode():
    t0 = time.time()
    torch.onnx.export(Enc(model), (feats, mask), os.path.join(out, "encoder.onnx"), dynamo=False, opset_version=17,
                      input_names=["features", "mask"], output_names=["frames", "length"],
                      dynamic_axes={"features": {1: "T"}, "mask": {1: "T"}, "frames": {1: "F"}})
    print(f"encoder exported in {time.time() - t0:.1f}s")
    tok, h0 = torch.tensor([[BLANK]]), torch.zeros(L, 1, H)
    torch.onnx.export(Dec(model.decoder), (tok, h0, h0), os.path.join(out, "decoder.onnx"), dynamo=False, opset_version=17,
                      input_names=["token", "h", "c"], output_names=["out", "h2", "c2"])
    torch.onnx.export(Joint(model.joint), (torch.zeros(1, H), torch.zeros(1, H)), os.path.join(out, "joint.onnx"),
                      dynamo=False, opset_version=17, input_names=["enc", "dec"], output_names=["logits"])

tk = proc.tokenizer
vocab = [tk.convert_ids_to_tokens(i) for i in range(V - 1)]
json.dump({"vocab": vocab, "blank": BLANK, "durations": DUR, "layers": L, "hidden": H}, open(os.path.join(out, "vocab.json"), "w"))


def decode(enc_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    e = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    d = ort.InferenceSession(os.path.join(out, "decoder.onnx"), providers=["CPUExecutionProvider"])
    j = ort.InferenceSession(os.path.join(out, "joint.onnx"), providers=["CPUExecutionProvider"])
    t0 = time.time()
    frames, n = e.run(None, {"features": feats.numpy(), "mask": mask.numpy()})
    te = time.time() - t0
    n = int(n[0]); h = np.zeros((L, 1, H), np.float32); c = h.copy()
    dout, h, c = d.run(None, {"token": np.array([[BLANK]], np.int64), "h": h, "c": c})
    t, ids = 0, []
    while t < n:
        lg = j.run(None, {"enc": frames[:, t], "dec": dout})[0][0]
        k, dur = int(lg[:V].argmax()), DUR[int(lg[V:].argmax())]
        if k == BLANK:
            dur = max(dur, 1)
        else:
            ids.append(k)
            dout, h, c = d.run(None, {"token": np.array([[k]], np.int64), "h": h, "c": c})
        t += dur
    return tk.decode(ids, skip_special_tokens=True).strip(), te


hyp, te = decode(os.path.join(out, "encoder.onnx"))
print(f"ONNX: {hyp}\n  encoder {te:.2f}s, match={hyp == ref}")

# int8 weight-only, per output channel: Bark's encoder rows hold only {0, +-a, +-b}, so this is close to exact
from onnxruntime.quantization import quantize_dynamic, QuantType
q = os.path.join(out, "encoder.int8.onnx")
quantize_dynamic(os.path.join(out, "encoder.onnx"), q, weight_type=QuantType.QInt8, per_channel=True,
                 op_types_to_quantize=["MatMul", "Gemm"], use_external_data_format=False)
hq, tq = decode(q)
print(f"INT8: {hq}\n  encoder {tq:.2f}s, match={hq == ref}")
for f in sorted(os.listdir(out)):
    print(f"{f:28s} {os.path.getsize(os.path.join(out, f)) / 1e6:9.1f} MB")
