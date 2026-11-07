//! Transcribes 16 kHz audio and, given a fixture, checks it against the official model.
//!
//!     cargo run --release --example transcribe -- model.bark <graphs dir> <audio.wav|wave.f32> [fixture.json]
//!
//! A `.f32` file is raw little-endian samples with the 0.25 s lead-in, as `tools/py_fixture.py` writes them.

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        bail!("usage: transcribe <model.bark> <graphs dir> <audio.wav|wave.f32> [fixture.json]");
    }
    let t0 = std::time::Instant::now();
    let mut bark = bark::Bark::load(&a[0], &a[1], 0)?;
    eprintln!("loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let wave = if a[2].ends_with(".f32") { raw_f32(&a[2])? } else { wav(&a[2])? };
    let t0 = std::time::Instant::now();
    let out = bark.transcribe(&wave)?;
    let secs = t0.elapsed().as_secs_f64();
    println!("{}", out.text);
    eprintln!("{:.1}s of audio in {secs:.2}s", wave.len() as f64 / 16000.0);
    if let Some((ws, private)) = peak() {
        eprintln!("peak memory: working set {:.2} GB, private {:.2} GB", ws as f64 / 1e9, private as f64 / 1e9);
    }
    if let Some(fx) = a.get(3) {
        let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(fx)?)?;
        let want = fx["text"].as_str().unwrap_or("");
        let same = out.text == want;
        eprintln!("matches the official model: {same}");
        if !same {
            eprintln!("want: {want}");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn wav(path: &str) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    if spec.sample_rate != 16000 {
        bail!("{} Hz; resample to 16 kHz first", spec.sample_rate);
    }
    let ch = spec.channels as usize;
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|x| x.map(|x| x as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    Ok(s.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect())
}

/// `wave.f32` from tools/py_fixture.py already starts with the lead-in; `transcribe` adds its own.
fn raw_f32(path: &str) -> Result<Vec<f32>> {
    let v: Vec<f32> =
        std::fs::read(path)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(v.get(bark::model::LEAD_IN..).unwrap_or_default().to_vec())
}

/// Peak working set and peak private bytes of this process (Windows only).
#[cfg(windows)]
fn peak() -> Option<(usize, usize)> {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_faults: u32,
        peak_ws: usize,
        ws: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (ok != 0).then_some((c.peak_ws, c.peak_pagefile))
}

/// Peak resident set (VmHWM) on Linux and Android; private bytes are not tracked there, so
/// both numbers are the same.
#[cfg(not(windows))]
fn peak() -> Option<(usize, usize)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: usize = s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some((kb * 1024, kb * 1024))
}
