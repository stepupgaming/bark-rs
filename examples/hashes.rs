//! Prints `name sha256` for every decoded tensor (f32 / int little-endian bytes).
//! Compare with `tools/py_hashes.py` to check the reader against the official loader.
//!
//!     cargo run --release --example hashes -- model.bark > rust.txt

use sha2::{Digest, Sha256};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: hashes <model.bark>");
    let t0 = std::time::Instant::now();
    let f = bark_asr::BarkFile::open(&path)?;
    let all = f.all()?;
    eprintln!("{} tensors decoded in {:.2}s", all.len(), t0.elapsed().as_secs_f64());
    for (name, _, data) in all {
        let h = Sha256::digest(data.to_le_bytes());
        let hex: String = h.iter().map(|b| format!("{b:02x}")).collect();
        println!("{name} {hex}");
    }
    Ok(())
}
