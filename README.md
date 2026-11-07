# bark-rs

Runs [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) speech to text in Rust, on ONNX Runtime, straight from the 163 MB `model.bark` file. There's no Python at run time and no second copy of the weights on disk. The target is Android (aarch64); desktop is where it was built and checked.

```rust
let mut bark = bark::Bark::load("model.bark", "graphs", 0)?; // 0 threads = ORT default
let t = bark.transcribe(&samples_16k_mono)?;
println!("{}", t.text);
```

## How it works

- **`src/format.rs`** reads the `.bark` file (format `bark-levels-v3`): `levels2` (per-row levels, ANS coded), `ansint5`/`ansint6` (scaled integers, ANS coded) and `raw` tensors, xz where flagged. All 723 tensors unpack bit for bit the same as the official `bark_loader.py` (sha256 per tensor, `examples/hashes.rs` against `tools/py_hashes.py`), in about 1.3 s.
- **`src/mel.rs`** is the log-mel frontend of transformers' `ParakeetFeatureExtractor`: 128 Slaney mels, preemphasis, centered 512-point STFT, per-bin normalization. Against the processor on the 67 s clip: max abs difference 2.7e-4, mean 5e-7 (`examples/mel_check.rs`).
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

With `graphs/` (exact), the desktop transcript is identical to the official model on 13 clips: the 67 s fixture plus twelve 60 s cuts from a podcast, a talk and a game stream.

Galaxy S25 Ultra (SM-S938U, Android 17, 11 GB RAM), same 67.2 s clip, cool phone, screen on:

| graphs | peak RAM | transcribe | load | transcript |
|---|---|---|---|---|
| **exact 2-bit** | **1.42 GB** | 14.9 s | 2.9 s | same as official; also on 9 of the 60 s cuts (the run was stopped there) |
| q8 | 1.97 GB | 13.5 s | 3.6 s | same on this clip |
| exact, `accuracy_level` 4 | 1.48 GB | 8.5 s | 4.9 s | same on this clip |
| q8, `accuracy_level` 4 | 1.99 GB | 4.6 s | 3.5 s | same on this clip |

`accuracy_level` 4 quantizes the activations to int8, so ARM's int8 dot-product units do the work. On the phone, q8 with level 4 is 3.2x faster than exact (about 14x real time). It is not exact. On the desktop, across the 13 clips (2,636 words), it gives the identical transcript on 10. The other 3 each differ by one word, all fillers the speaker half says: an extra "Yeah", "have tried" heard as "I've tried", and a dropped "Uh". That's a 0.11% word difference, so it's the pick for the app. x86 and ARM int8 kernels round differently, so this is a close proxy for the phone, not the phone itself. A longer phone run heats the phone up and slows it down: a later exact run took 23.7 s for 60 s of audio. Keep phone tests short.

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
- Android: tested as a command-line binary over adb (see below), not yet inside an app.

## Android

pyke ships a static ORT for `aarch64-linux-android`, so the binary needs no `.so` (32 MB unstripped for the example). With the NDK (29 here):

```bash
BIN="$LOCALAPPDATA/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/windows-x86_64/bin"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$BIN/aarch64-linux-android28-clang.cmd"
export CC_aarch64_linux_android="$BIN/aarch64-linux-android28-clang.cmd" AR_aarch64_linux_android="$BIN/llvm-ar.exe"
cargo build --release --target aarch64-linux-android --example transcribe
```

To try it on a phone: `adb push` the binary, `model.bark`, `graphs/`, a 16 kHz WAV and the NDK's `sysroot/usr/lib/aarch64-linux-android/libc++_shared.so` to `/data/local/tmp/bark`. Run `chmod 755 transcribe` after every push, because a push resets the mode. Then run `LD_LIBRARY_PATH=. ./transcribe model.bark graphs audio.wav` in `adb shell`. Set `MSYS_NO_PATHCONV=1` in Git Bash, or it rewrites the device paths.

In an app, `libc++_shared.so` goes in `jniLibs/arm64-v8a`. For a Tauri 2 app (Catalyzer, for example), add this crate to `src-tauri` and expose `transcribe` as a command; no JNI glue is needed.

## License

Model weights and derived graphs: CC-BY-4.0, see [NOTICE](NOTICE). Code: MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
