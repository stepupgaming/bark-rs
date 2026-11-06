# bark-rs

Runs [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) speech to text in Rust, on ONNX Runtime, straight from the 163 MB `model.bark` file. There's no Python at run time and no second copy of the weights on disk. The target is Android (aarch64); desktop is where it was built and checked.

```rust
let mut bark = bark::Bark::load("model.bark", "graphs", 0)?; // 0 threads = ORT default
let t = bark.transcribe(&samples_16k_mono)?;
println!("{}", t.text);
```

## How it works

- **`src/format.rs`** reads the `.bark` file (format `bark-levels-v3`): `levels2` (per-row levels, ANS coded), `ansint5`/`ansint6` (scaled integers, ANS coded) and `raw` tensors, xz where flagged. All 723 tensors unpack bit for bit the same as the official `bark_loader.py` (sha256 per tensor, `examples/hashes.rs` against `tools/py_hashes.py`), in about 1.3 s.
- **`src/mel.rs`** is the log-mel frontend of transformers' `ParakeetFeatureExtractor`: 128 Slaney mels, preemphasis, centered 512-point STFT, per-bin normalization. Max difference from the processor is on the order of float rounding (`examples/mel_check.rs`).
- **`graphs/`** holds the three ONNX graphs (encoder, one LSTM decoder step, joint) without weights, 2.3 MB in all. `manifest.json` says which `model.bark` tensor feeds each initializer and how (as stored, transposed, LSTM gate order, packed). `src/model.rs` builds every initializer from `model.bark` at load time and hands it to ORT.
- **Packed encoder.** Each row of a `levels2` weight holds only 0 and ±K levels (K is 2, or 3 for four tensors). So W = Σ lev_j · T_j, with each T_j ternary. Every encoder MatMul and 1×1 Conv becomes K `MatMulNBits` 2-bit ternary matrices plus a Sum. That's 604M weights at 4 bits each, and the result is bit-exact with fp32 weights.
- **Decoding** is greedy TDT, like `model.generate`: blank, then tokens plus duration skips, on a 0.25 s silent lead-in.

## Numbers

Windows desktop CPU, ORT 1.31 (pyke `ort` 2.0.0-rc.15), 67.2 s English clip:

| graphs | peak RAM | transcribe | load | transcript |
|---|---|---|---|---|
| fp32 (`make_graphs.py ... fp32`) | 5.55 GB | 2.00 s | 2.6 s | same as official |
| **exact 2-bit (default, `graphs/`)** | **1.43 GB** | 3.53 s | 2.2 s | same as official |
| q8 (`make_graphs.py ... q8`) | 1.98 GB | 2.24 s | 1.5 s | same on this clip |

fp32 needs two copies of 2.4 GB, because ORT copies external initializers when it creates the session. The packed graphs avoid that. q8 picks an 8-bit scale per row so that every level lands close to an integer step (weight error about 1e-4). It is faster on desktop, but it isn't exact.

On a phone, the 2-bit kernel speed hasn't been measured yet. That and `accuracy_level` 4 (int8 activations) are the next things to test on a real device.

## Checks

```bash
cargo run --release --example transcribe -- model.bark graphs audio.wav
cargo run --release --example transcribe -- model.bark graphs wave.f32 fixture.json   # exits 1 on a mismatch
```

`tools/py_fixture.py` writes `wave.f32`, `feats.f32` and `fixture.json` (ids and text) from the official model.

## Rebuilding the graphs

The graphs are generated, but they're committed so the crate works without Python. To regenerate them:

1. `tools/bark_export.py <model dir> <audio.wav> <onnx dir>` exports full fp32 ONNX from the official loader (torch + transformers) and checks an ORT decode against `model.generate`.
2. `tools/make_graphs.py <model dir> <onnx dir> graphs [exact|q8|fp32]` matches every initializer to a `model.bark` tensor by sha256, makes it external, packs the encoder, and writes the manifest.

Set `BARK_PYLIB` to an extra site directory if `onnx`/`onnxruntime` live outside your venv. `tools/nbits_probe.py` is the single-matrix MatMulNBits test that picked the packing.

## Limits

- 16 kHz mono in. Resampling is the caller's job.
- One pass over the whole clip. The model card says long-recording segmentation hasn't been qualified. 67 s works; longer audio should be cut into chunks at silences.
- English only (the model).
- Android: pyke ships a static ORT for `aarch64-linux-android`. Building and testing on a phone is the next step.

## License

Model weights and derived graphs: CC-BY-4.0, see [NOTICE](NOTICE). Code: not yet chosen (private repo).
