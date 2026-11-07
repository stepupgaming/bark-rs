//! Compares this crate's 16 kHz input conversion against a reference, such as the `wave.f32`
//! that `tools/py_fixture.py` writes (downmix + `soxr.resample`, after the 0.25 s lead-in).
//!
//!     cargo run --release --example resample_check -- audio.wav wave.f32 [out.f32]
//!
//! Prints the signal-to-difference ratio in dB and the largest sample difference.

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        bail!("usage: resample_check <audio.wav> <reference wave.f32> [out.f32]");
    }
    let mut r = hound::WavReader::open(&a[0])?;
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|x| x.map(|x| x as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    let t0 = std::time::Instant::now();
    let ours = bark_asr::audio::to_model_input(&s, spec.channels as usize, spec.sample_rate)?;
    let secs = t0.elapsed().as_secs_f64();
    if let Some(out) = a.get(2) {
        std::fs::write(out, ours.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let want: Vec<f32> = std::fs::read(&a[1])?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .skip(bark_asr::model::LEAD_IN)
        .collect();
    let n = ours.len().min(want.len());
    let (mut sig, mut diff, mut max) = (0f64, 0f64, 0f32);
    for i in 0..n {
        let d = ours[i] - want[i];
        sig += (want[i] as f64).powi(2);
        diff += (d as f64).powi(2);
        max = max.max(d.abs());
    }
    println!(
        "{} Hz x{} -> 16 kHz in {:.3}s: {} samples (reference {}), {:.1} dB, max diff {:.2e}",
        spec.sample_rate,
        spec.channels,
        secs,
        ours.len(),
        want.len(),
        10.0 * (sig / diff.max(1e-30)).log10(),
        max
    );
    Ok(())
}
