# bark-rs

Runs [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) speech to text in Rust, on ONNX Runtime, straight from the 163 MB `model.bark` file. There's no Python at run time and no second copy of the weights on disk. The target is Android (aarch64); desktop is where it was built and checked.

```bash
cargo add bark-asr
```

The crate includes `graphs/` and `graphs-int8/`; ship one of them with your app next to `model.bark`.

```rust
let mut bark = bark_asr::Bark::load("model.bark", "graphs", 0)?; // 0 threads = ORT default
let t = bark.transcribe_pcm(&interleaved, channels, sample_rate)?; // or transcribe(&mono_16k)
println!("{}", t.text);
for w in &t.words {
    println!("{:.2} {:.2} {}", w.start, w.end, w.text); // seconds
}
```

Any length, any sample rate, any channel count. Every word and token comes with a start and end time.

## How it works

- **`src/format.rs`** reads the `.bark` file (format `bark-levels-v3`): `levels2` (per-row levels, ANS coded), `ansint5`/`ansint6` (scaled integers, ANS coded) and `raw` tensors, xz where flagged. All 723 tensors unpack bit for bit the same as the official `bark_loader.py` (sha256 per tensor, `examples/hashes.rs` against `tools/py_hashes.py`), in about 1.3 s.
- **`src/mel.rs`** is the log-mel frontend of transformers' `ParakeetFeatureExtractor`: 128 Slaney mels, preemphasis, centered 512-point STFT, per-bin normalization. Against the processor on the 67 s clip: max abs difference 2.7e-4, mean 5e-7 (`examples/mel_check.rs`).
- **`graphs/`** holds the three ONNX graphs (encoder, one LSTM decoder step, joint) without weights, 2.3 MB in all. `manifest.json` says which `model.bark` tensor feeds each initializer and how (as stored, transposed, LSTM gate order, packed). `src/model.rs` builds every initializer from `model.bark` at load time and hands it to ORT.
- **Packed encoder.** Each row of a `levels2` weight holds only 0 and ±K levels (K is 2, or 3 for four tensors). So W = Σ lev_j · T_j, with each T_j ternary. Every encoder MatMul and 1×1 Conv becomes K `MatMulNBits` 2-bit ternary matrices plus a Sum. That's 604M weights at 4 bits each, and the result is bit-exact with fp32 weights.
- **Decoding** is greedy TDT, like `model.generate`: blank, then tokens plus duration skips, on a 0.25 s silent lead-in.
- **Times** follow transformers' `ParakeetProcessor` (NeMo's rule): a token starts on the encoder frame it was emitted on (80 ms frames) and ends after its predicted TDT duration. Punctuation takes the end of the token before it. Words run from their first token's start to their last token's end, and punctuation joins the word before it. Times count from the start of your audio, without the lead-in. On all 14 fixtures, every token time matches the official processor exactly.
- **Other rates** (`src/audio.rs`): channels are averaged and the audio is resampled to 16 kHz with rubato's FFT resampler (pure Rust). Conway Research's own runtime (`bark_coreml.py`) uses soxr. Against soxr the output has no delay and agrees to better than -52 dB up to 7.3 kHz. It differs only from 7.6 to 8 kHz, where soxr rolls off sooner. On the 12 clips resampled to 44.1 kHz, 9 transcripts are identical to the official model and 3 differ by one or two words (4 of 2,384, 0.17%).
- **Long audio** is cut into pieces of at most 30 s (`set_max_piece`) the way `bark_coreml.py` does it: 10 ms energy smoothed over 200 ms, each cut in the middle of the quietest frame between a third of the limit and the limit. The cut points are identical to a port of the authors' Python. Each piece gets its own lead-in and the times are shifted back.

## Numbers

Windows desktop CPU, ORT 1.31 (pyke `ort` 2.0.0-rc.15), 67.2 s English clip:

| graphs | peak RAM | transcribe | load | transcript |
|---|---|---|---|---|
| fp32 (`make_graphs.py ... fp32`) | 5.55 GB | 2.00 s | 2.6 s | same as official |
| **exact 2-bit (default, `graphs/`)** | **1.43 GB** | 3.53 s | 2.2 s | same as official |
| q8 (`make_graphs.py ... q8`) | 1.98 GB | 2.24 s | 1.5 s | same on this clip |
| **int8 (`graphs-int8/`)** | 1.85 GB | 1.1 s | 1.5 s | 0.11% words differ on 13 clips |

fp32 needs two copies of 2.4 GB, because ORT copies external initializers when it creates the session. The packed graphs avoid that. q8 picks an 8-bit scale per row so that every level lands close to an integer step (weight error about 1e-4). It is faster on desktop, but it isn't exact.

With `graphs/` (exact), the desktop transcript is identical to the official model on 13 clips: the 67 s fixture plus twelve 60 s cuts from a podcast, a talk and a game stream.

Galaxy S25 Ultra (SM-S938U, Android 17, 11 GB RAM), same 67.2 s clip, cool phone, screen on:

| graphs | peak RAM | transcribe | load | transcript |
|---|---|---|---|---|
| **exact 2-bit** | **1.42 GB** | 14.9 s | 2.9 s | same as official; also on 9 of the 60 s cuts (the run was stopped there) |
| q8 | 1.97 GB | 13.5 s | 3.6 s | same on this clip |
| exact, `accuracy_level` 4 | 1.48 GB | 8.5 s | 4.9 s | same on this clip |
| q8, `accuracy_level` 4 | 1.99 GB | 4.6 s | 3.5 s | same on this clip |

`accuracy_level` 4 quantizes the activations to int8, so ARM's int8 dot-product units do the work. On the phone, q8 with level 4 is 3.2x faster than exact (about 14x real time). It is not exact. On the desktop, across the 13 clips (2,636 words), it gives the identical transcript on 10. The other 3 each differ by one word, all fillers the speaker half says: an extra "Yeah", "have tried" heard as "I've tried", and a dropped "Uh". That's a 0.11% word difference, so it's the pick for the app. Those graphs are committed as `graphs-int8/`. x86 and ARM int8 kernels round differently, so this is a close proxy for the phone, not the phone itself. A longer phone run heats the phone up and slows it down: a later exact run took 23.7 s for 60 s of audio. Keep phone tests short.

### Long audio

A 276 s narration with its exact script (824 words), desktop CPU:

| pieces | exact: time | exact: peak private | int8: time | int8: peak private | words off the script |
|---|---|---|---|---|---|
| 10 s | 40.9 s | 1.21 GB | | | 4 |
| 15 s | 32.9 s | 1.39 GB | 4.7 s | 1.96 GB | 5 (int8: 6) |
| **30 s (default)** | 20.6 s | 1.60 GB | 4.5 s | 2.17 GB | 3 (int8: 5) |
| 60 s | 15.1 s | 2.15 GB | 4.8 s | 2.73 GB | 4 (int8: 6) |
| one pass | 15.8 s | 4.39 GB | | | 5 (official model, one pass: 5) |

Piece length doesn't change accuracy on clean speech; the few misses are the same words in every row. It changes memory, and for the exact graphs speed: their encoder costs about 0.85 s per call whatever the length, while the int8 graphs have no fixed cost. On two 240 s spontaneous recordings, 30 s pieces differ from one pass by 2.8 to 3.2% of words, mostly "uh" and "um" that one side keeps. Without a human transcript there's no telling which side is right. Turning off ORT's CPU arena would save about 0.8 GB at 60 s pieces but runs 50% slower, so the crate keeps it on.

## Checks

```bash
cargo run --release --example transcribe -- model.bark graphs audio.wav --words      # any rate; --pieces shows the cuts
cargo run --release --example transcribe -- model.bark graphs wave.f32 fixture.json   # exits 1 on a mismatch
cargo run --release --example resample_check -- audio.wav wave.f32                    # our 16 kHz input vs soxr's
cargo test --release
```

`tools/py_fixture.py` writes `wave.f32`, `feats.f32` and `fixture.json` (ids, text and token times) from the official model. It takes any WAV and resamples with soxr like the authors' runtime. With a fixture, `transcribe` reads the clip in one pass like the reference unless `--max-piece` says otherwise.

## Rebuilding the graphs

The graphs are generated, but they're committed so the crate works without Python. To regenerate them:

1. `tools/bark_export.py <model dir> <audio.wav> <onnx dir>` exports full fp32 ONNX from the official loader (torch + transformers) and checks an ORT decode against `model.generate`.
2. `tools/make_graphs.py <model dir> <onnx dir> graphs [exact|q8|int8|fp32]` matches every initializer to a `model.bark` tensor by sha256, makes it external, packs the encoder, and writes the manifest.

Set `BARK_PYLIB` to an extra site directory if `onnx`/`onnxruntime` live outside your venv. `tools/nbits_probe.py` is the single-matrix MatMulNBits test that picked the packing.

## Limits

- Long audio is cut at pauses, as in the authors' runtime. The model card says long-recording segmentation hasn't been qualified, and the numbers above are from three recordings.
- Resampled input is close to the official runtime's, not identical (see above). 16 kHz input is exact.
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

Model weights and derived graphs: CC-BY-4.0, see [NOTICE](NOTICE). Code: MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). The crate ships both, so its license field is `(MIT OR Apache-2.0) AND CC-BY-4.0`.
