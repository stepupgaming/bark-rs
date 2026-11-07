//! Transcribes audio and, given a fixture, checks it against the official model.
//!
//!     cargo run --release --example transcribe -- model.bark <graphs dir> <audio.wav|wave.f32> [fixture.json] [--words] [--pieces] [--max-piece S]
//!
//! A WAV can have any rate and channel count. A `.f32` file is raw little-endian 16 kHz samples
//! with the 0.25 s lead-in, as `tools/py_fixture.py` writes them. With a fixture, the text and
//! every token time must match the official processor's (`tokens` in the fixture), or it exits 1.

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let show_words = take_flag(&mut a, "--words");
    let show_pieces = take_flag(&mut a, "--pieces");
    let max_piece = take_opt(&mut a, "--max-piece")?;
    if a.len() < 3 {
        bail!("usage: transcribe <model.bark> <graphs dir> <audio.wav|wave.f32> [fixture.json] [--words] [--pieces] [--max-piece S]");
    }
    let t0 = std::time::Instant::now();
    let mut bark = bark::Bark::load(&a[0], &a[1], 0)?;
    // the official reference reads a fixture's clip in one pass, so do the same unless told
    match max_piece {
        Some(s) => bark.set_max_piece(s),
        None if a.len() > 3 => bark.set_max_piece(f64::INFINITY),
        None => {}
    }
    eprintln!("loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let (samples, channels, rate) = if a[2].ends_with(".f32") { (raw_f32(&a[2])?, 1, 16000) } else { wav(&a[2])? };
    if show_pieces {
        let max = max_piece.unwrap_or(bark::model::MAX_PIECE).max(1.0);
        let wave = bark::audio::to_model_input(&samples, channels, rate)?;
        for r in bark::audio::segments(&wave, max, max / 3.0) {
            eprintln!("piece {:9.3} {:9.3}", r.start as f64 / 16000.0, r.end as f64 / 16000.0);
        }
    }
    let t0 = std::time::Instant::now();
    let out = bark.transcribe_pcm(&samples, channels, rate)?;
    let secs = t0.elapsed().as_secs_f64();
    println!("{}", out.text);
    if show_words {
        for w in &out.words {
            println!("{:8.2} {:8.2}  {}", w.start, w.end, w.text);
        }
    }
    let audio = samples.len() as f64 / channels as f64 / rate as f64;
    eprintln!("{audio:.1}s of audio in {secs:.2}s, {} words", out.words.len());
    if let Some((ws, private)) = peak() {
        eprintln!("peak memory: working set {:.2} GB, private {:.2} GB", ws as f64 / 1e9, private as f64 / 1e9);
    }
    if let Some(fx) = a.get(3) {
        let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(fx)?)?;
        let want = fx["text"].as_str().unwrap_or("");
        let same = out.text == want;
        eprintln!("text matches the official model: {same}");
        let mut bad = !same;
        if !same {
            eprintln!("want: {want}");
        }
        if let Some(tokens) = fx["tokens"].as_array() {
            // the processor's times count from the start of the lead-in
            let lead = bark::model::LEAD_IN as f64 / 16000.0;
            let mut off = 0usize;
            let mut worst = 0f64;
            for (w, t) in tokens.iter().zip(&out.tokens) {
                let s = (w["start"].as_f64().unwrap_or(-1.0) - lead).max(0.0);
                let e = (w["end"].as_f64().unwrap_or(-1.0) - lead).max(0.0);
                let d = (s - t.start).abs().max((e - t.end).abs());
                if d > 1e-6 {
                    off += 1;
                    worst = worst.max(d);
                }
            }
            let same_len = tokens.len() == out.tokens.len();
            eprintln!(
                "{} token times match the official processor: {}{}",
                out.tokens.len(),
                same_len && off == 0,
                if off > 0 { format!(" ({off} differ, by up to {worst:.2} s)") } else { String::new() }
            );
            if !same_len {
                eprintln!("token count differs: official {}", tokens.len());
            }
            let off = !same_len || off > 0;
            bad |= off;
        }
        if bad {
            std::process::exit(1);
        }
    }
    Ok(())
}

fn take_flag(a: &mut Vec<String>, name: &str) -> bool {
    let n = a.len();
    a.retain(|x| x != name);
    a.len() != n
}

fn take_opt(a: &mut Vec<String>, name: &str) -> Result<Option<f64>> {
    let Some(i) = a.iter().position(|x| x == name) else { return Ok(None) };
    if i + 1 >= a.len() {
        bail!("{name} needs a value");
    }
    let v = a.remove(i + 1).parse()?;
    a.remove(i);
    Ok(Some(v))
}

/// Interleaved samples, channel count and rate.
fn wav(path: &str) -> Result<(Vec<f32>, usize, u32)> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|x| x.map(|x| x as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    Ok((s, spec.channels as usize, spec.sample_rate))
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
