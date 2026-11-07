//! Compares Rust features with the processor's (from tools/py_fixture.py).
//!
//!     cargo run --release --example mel_check -- <fixture dir>

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: mel_check <fixture dir>"));
    let wave = read_f32(&dir.join("wave.f32"));
    let want = read_f32(&dir.join("feats.f32"));
    let t0 = std::time::Instant::now();
    let f = bark_asr::mel::Mel::new(128).features(&wave);
    eprintln!("{} frames ({} valid) in {:.3}s", f.frames, f.valid, t0.elapsed().as_secs_f64());
    assert_eq!(f.data.len(), want.len(), "frame count differs: want {}", want.len() / 128);
    let (mut max, mut sum) = (0f32, 0f64);
    for (a, b) in f.data.iter().zip(&want) {
        let d = (a - b).abs();
        max = max.max(d);
        sum += d as f64;
    }
    println!("max abs diff {max:.3e}, mean {:.3e}", sum / want.len() as f64);
}
