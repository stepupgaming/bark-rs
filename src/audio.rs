//! Audio in: downmix, resampling to 16 kHz, and cutting long audio at pauses.
//!
//! This follows the input handling of Conway Research's own `bark_coreml.py`: channels are
//! averaged, other rates are resampled to 16 kHz (`soxr` there, rubato's FFT resampler here),
//! and audio longer than one piece is cut in the middle of the quietest 200 ms span.

use std::ops::Range;

use anyhow::{Result, anyhow};
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// The model's sample rate.
pub const RATE: u32 = 16000;

/// Averages interleaved channels to mono.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect()
}

/// Resamples mono audio from `rate` to 16 kHz. A fixed-ratio FFT resampler with one large
/// block (rubato `Fft`, `process_all`, which trims the filter delay so the output lines up
/// with the input).
pub fn resample(mono: &[f32], rate: u32) -> Result<Vec<f32>> {
    if rate == RATE || mono.is_empty() {
        return Ok(mono.to_vec());
    }
    let chunk = 4096;
    let mut r = Fft::<f32>::new_custom(
        rate as usize,
        RATE as usize,
        chunk,
        1,
        1,
        rubato::WindowFunction::BlackmanHarris2,
        FixedSync::Input,
    )
    .map_err(|e| anyhow!("resampler {rate} Hz -> {RATE} Hz: {e}"))?;
    let input = InterleavedSlice::new(mono, 1, mono.len()).map_err(|e| anyhow!("{e}"))?;
    let out = r.process_all(&input, mono.len(), None).map_err(|e| anyhow!("resampling: {e}"))?;
    Ok(out.take_data())
}

/// Interleaved samples at any rate to 16 kHz mono.
pub fn to_model_input(interleaved: &[f32], channels: usize, rate: u32) -> Result<Vec<f32>> {
    resample(&downmix(interleaved, channels), rate)
}

/// Splits 16 kHz audio longer than `max_s` seconds into pieces, as `segments()` in
/// `bark_coreml.py` does: smoothed 10 ms energy (a 200 ms box), and each cut goes in the middle
/// of the quietest frame between `min_s` and `max_s` after the previous cut, so no word is
/// split. Returns sample ranges that cover the input.
pub fn segments(x: &[f32], max_s: f64, min_s: f64) -> Vec<Range<usize>> {
    let sr = RATE as f64;
    if x.len() <= (max_s * sr) as usize {
        #[allow(clippy::single_range_in_vec_init)]
        return vec![0..x.len()];
    }
    let fr = RATE as usize / 100;
    let n = x.len() / fr;
    let raw: Vec<f64> = (0..n).map(|i| x[i * fr..(i + 1) * fr].iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / fr as f64).collect();
    // np.convolve(raw, ones(20) / 20, mode="same"): e[i] = mean of raw[i - 10 ..= i + 9], zero outside
    let mut pre = vec![0f64; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + raw[i];
    }
    let e: Vec<f64> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(10);
            let hi = (i + 10).min(n);
            (pre[hi] - pre[lo]) / 20.0
        })
        .collect();
    let (mut out, mut start) = (Vec::new(), 0usize);
    while (x.len() - start) as f64 > max_s * sr {
        let lo = (start + (min_s * sr) as usize) / fr;
        let hi = ((start + (max_s * sr) as usize) / fr).min(n);
        let mut best = lo;
        for i in lo..hi {
            if e[i] < e[best] {
                best = i;
            }
        }
        let cut = best * fr + fr / 2;
        out.push(start..cut);
        start = cut;
    }
    out.push(start..x.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_cut_in_the_gaps() {
        // 100 s of tone with a 0.5 s silence starting every 7 s
        let sr = RATE as usize;
        let x: Vec<f32> = (0..100 * sr)
            .map(|i| if i % (7 * sr) < 7 * sr - sr / 2 { 0.3 * (i as f32 * 0.07).sin() } else { 0.0 })
            .collect();
        let pieces = segments(&x, 15.0, 5.0);
        assert!(pieces.len() > 6);
        assert_eq!(pieces[0].start, 0);
        assert_eq!(pieces.last().unwrap().end, x.len());
        for w in pieces.windows(2) {
            assert_eq!(w[0].end, w[1].start);
            let cut = w[0].end % (7 * sr);
            assert!(cut >= 7 * sr - sr / 2, "cut at {} s is not in a gap", w[0].end as f64 / sr as f64);
        }
        for p in &pieces {
            assert!(p.len() <= 15 * sr);
        }
    }

    #[test]
    fn short_audio_is_one_piece() {
        assert_eq!(segments(&vec![0.0; 16000 * 10], 15.0, 5.0), vec![0..160000]);
    }

    #[test]
    fn downmix_averages() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
    }
}
