//! Log-mel features, matching transformers' `ParakeetFeatureExtractor` (128 mels, 16 kHz).
//!
//! preemphasis 0.97, centered STFT (n_fft 512, hop 160, zero padding), symmetric Hann
//! window of 400 placed in the middle of the 512 frame, power spectrum, Slaney mel
//! filters (librosa `norm="slaney"`), `ln(x + 2^-24)`, then each mel bin normalized to
//! zero mean and unit (unbiased) std over the valid frames. The last frame is padding.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 512;
pub const HOP: usize = 160;
pub const WIN: usize = 400;
pub const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 1.0 / (1u32 << 24) as f32;
const EPS: f32 = 1e-5;

pub struct Mel {
    pub n_mels: usize,
    /// `[n_mels, N_FFT/2 + 1]`, row major.
    filters: Vec<f32>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

/// Features `[frames, n_mels]` row major, and how many frames are valid.
pub struct Features {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
    pub n_mels: usize,
}

impl Mel {
    pub fn new(n_mels: usize) -> Self {
        let mut window = vec![0f32; N_FFT];
        let left = (N_FFT - WIN) / 2;
        for n in 0..WIN {
            let x = 2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64;
            window[left + n] = (0.5 - 0.5 * x.cos()) as f32;
        }
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Self { n_mels, filters: slaney_filters(n_mels), window, fft }
    }

    /// `wave`: mono 16 kHz samples.
    pub fn features(&self, wave: &[f32]) -> Features {
        let n = wave.len();
        let bins = N_FFT / 2 + 1;
        // preemphasis
        let mut x = vec![0f32; n + N_FFT];
        let pad = N_FFT / 2;
        for i in 0..n {
            x[pad + i] = if i == 0 { wave[0] } else { wave[i] - PREEMPH * wave[i - 1] };
        }
        let frames = 1 + n / HOP;
        let valid = n / HOP;
        let mut data = vec![0f32; frames * self.n_mels];
        let mut buf = vec![0f32; N_FFT];
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0f32; bins];
        for t in 0..frames {
            let s = &x[t * HOP..t * HOP + N_FFT];
            for ((b, &v), &w) in buf.iter_mut().zip(s).zip(&self.window) {
                *b = v * w;
            }
            self.fft.process_with_scratch(&mut buf, &mut spec, &mut scratch).unwrap();
            for (p, c) in power.iter_mut().zip(&spec) {
                let m = (c.re * c.re + c.im * c.im).sqrt();
                *p = m * m;
            }
            let out = &mut data[t * self.n_mels..(t + 1) * self.n_mels];
            for (m, o) in out.iter_mut().enumerate() {
                let f = &self.filters[m * bins..(m + 1) * bins];
                let e: f32 = f.iter().zip(&power).map(|(a, b)| a * b).sum();
                *o = (e + LOG_GUARD).ln();
            }
        }
        // per-bin normalization over the valid frames
        for m in 0..self.n_mels {
            let mut sum = 0f64;
            for t in 0..valid {
                sum += data[t * self.n_mels + m] as f64;
            }
            let mean = (sum / valid as f64) as f32;
            let mut var = 0f64;
            for t in 0..valid {
                let d = (data[t * self.n_mels + m] - mean) as f64;
                var += d * d;
            }
            let std = ((var / (valid as f64 - 1.0)) as f32).sqrt();
            for t in 0..frames {
                let v = &mut data[t * self.n_mels + m];
                *v = if t < valid { (*v - mean) / (std + EPS) } else { 0.0 };
            }
        }
        Features { data, frames, valid, n_mels: self.n_mels }
    }
}

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    if f < 1000.0 { f / f_sp } else { 1000.0 / f_sp + (f / 1000.0).ln() / (6.4f64.ln() / 27.0) }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_mel = 1000.0 / f_sp;
    if m < min_log_mel { m * f_sp } else { 1000.0 * ((6.4f64.ln() / 27.0) * (m - min_log_mel)).exp() }
}

/// numpy `linspace(start, stop, num)`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

/// librosa `filters.mel(sr=16000, n_fft=512, n_mels, fmin=0, fmax=8000, norm="slaney")`.
fn slaney_filters(n_mels: usize) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    // numpy rfftfreq(n=N_FFT, d=1/sr)
    let val = 1.0 / (N_FFT as f64 * (1.0 / SAMPLE_RATE as f64));
    let fft_f: Vec<f64> = (0..bins).map(|k| k as f64 * val).collect();
    let mel_f: Vec<f64> = linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0), n_mels + 2)
        .into_iter()
        .map(mel_to_hz)
        .collect();
    let mut w = vec![0f32; n_mels * bins];
    for i in 0..n_mels {
        let (d0, d1) = (mel_f[i + 1] - mel_f[i], mel_f[i + 2] - mel_f[i + 1]);
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (k, &f) in fft_f.iter().enumerate() {
            let lower = -(mel_f[i] - f) / d0;
            let upper = (mel_f[i + 2] - f) / d1;
            let tri = lower.min(upper).max(0.0) as f32; // librosa stores the ramp as f32
            w[i * bins + k] = (tri as f64 * enorm) as f32;
        }
    }
    w
}
