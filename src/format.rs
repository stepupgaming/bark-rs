//! Reader for the `.bark` weight file (formats `bark-levels-v2` and `bark-levels-v3`).
//!
//! Layout: 8-byte little-endian header length, a JSON header (`format`, `index`, `config`),
//! then each tensor's blobs back to back in index order. Tensor kinds:
//!
//! - `levels2`: per-row fp16 levels `[rows, K]` and one code per weight. Code 0 is zero,
//!   `1..=K` is `+lev[code-1]`, `K+1..=2K` is `-lev[code-K-1]`. Codes are ANS coded with
//!   16 column classes, each with its own symbol counts. Symbols run column by column
//!   (all rows of one column, then the next column of the same class).
//! - `ansint5` / `ansint6`: symmetric integers, ANS coded, times fp16 row scales.
//! - `raw`: fp16 (or int64 / int32) bytes.
//!
//! v3 compresses the small side blobs (levels, scales, raw tensors) with xz.
//! Decoding matches the official Python loader (`bark_loader.py`) bit for bit.

use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use constriction::stream::Decode;
use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::stack::DefaultAnsCoder;
use half::f16;
use rayon::prelude::*;
use serde_json::Value;

/// Number of column classes in a `levels2` tensor.
pub const NC: usize = 16;

#[derive(Debug, Clone)]
pub enum Kind {
    /// `k` levels per row (2 or 3 in the shipped models).
    Levels { k: usize, lev: Blob, meta: Blob, words: Blob },
    /// Symmetric `bits`-bit integers with one fp16 scale per row.
    AnsInt { bits: u32, scale: Blob, meta: Blob, words: Blob },
    Raw { dtype: RawType, data: Blob },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawType {
    F16,
    I64,
    I32,
}

/// A byte range in the file, xz compressed or not.
#[derive(Debug, Clone, Copy)]
pub struct Blob {
    pub off: usize,
    pub len: usize,
    pub xz: bool,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: Kind,
}

impl Entry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A decoded tensor. Floats come back as f32 (what the model runs on).
#[derive(Debug, Clone)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

impl Data {
    /// Little-endian bytes, as numpy's `tobytes()` gives them.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Data::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Data::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Data::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

pub struct BarkFile {
    bytes: Vec<u8>,
    pub format: String,
    /// The `ParakeetTDTConfig` dict.
    pub config: Value,
    pub entries: Vec<Entry>,
}

impl BarkFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() >= 8, "file too short");
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        ensure!(8 + n <= bytes.len(), "header runs past the end of the file");
        let head: Value = serde_json::from_slice(&bytes[8..8 + n]).context("header JSON")?;
        let format = head["format"].as_str().unwrap_or_default().to_string();
        if format != "bark-levels-v2" && format != "bark-levels-v3" {
            bail!("unsupported format {format:?}");
        }
        let config = match &head["config"] {
            Value::String(s) => serde_json::from_str(s).context("config JSON")?,
            v => v.clone(),
        };
        let index = head["index"].as_array().ok_or_else(|| anyhow!("no index"))?;
        let mut p = 8 + n;
        let mut entries = Vec::with_capacity(index.len());
        for e in index {
            let name = e["n"].as_str().ok_or_else(|| anyhow!("entry without a name"))?.to_string();
            let shape: Vec<usize> = e["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("{name}: no shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| anyhow!("{name}: bad shape")))
                .collect::<Result<_>>()?;
            let mut take = |key: &str, zkey: Option<&str>| -> Result<Blob> {
                let len = e[key].as_u64().ok_or_else(|| anyhow!("{name}: no {key}"))? as usize;
                let xz = zkey.map(|z| e[z].as_bool().unwrap_or(false)).unwrap_or(false);
                let b = Blob { off: p, len, xz };
                p += len;
                Ok(b)
            };
            let k = e["k"].as_str().unwrap_or_default();
            let kind = if k == "levels2" {
                let kk = e["K"].as_u64().ok_or_else(|| anyhow!("{name}: no K"))? as usize;
                let lev = take("lev", Some("zl"))?;
                let meta = take("meta", None)?;
                let words = take("b", None)?;
                Kind::Levels { k: kk, lev, meta, words }
            } else if let Some(bits) = k.strip_prefix("ansint") {
                let bits: u32 = bits.parse().with_context(|| format!("{name}: kind {k}"))?;
                let scale = take("s", Some("zs"))?;
                let meta = take("meta", None)?;
                let words = take("b", None)?;
                Kind::AnsInt { bits, scale, meta, words }
            } else if k == "raw" {
                let dtype = match e["dtype"].as_str().unwrap_or("") {
                    "torch.int64" => RawType::I64,
                    "torch.int32" => RawType::I32,
                    _ => RawType::F16,
                };
                let data = take("b", Some("zr"))?;
                Kind::Raw { dtype, data }
            } else {
                bail!("{name}: unknown kind {k:?}");
            };
            entries.push(Entry { name, shape, kind });
        }
        ensure!(p == bytes.len(), "index covers {p} bytes, file has {}", bytes.len());
        Ok(Self { bytes, format, config, entries })
    }

    fn blob(&self, b: Blob) -> Result<Vec<u8>> {
        let s = &self.bytes[b.off..b.off + b.len];
        if !b.xz {
            return Ok(s.to_vec());
        }
        let mut out = Vec::new();
        lzma_rust2::XzReader::new(s, false).read_to_end(&mut out).context("xz")?;
        Ok(out)
    }

    fn words(&self, b: Blob) -> Vec<u32> {
        u32s(&self.bytes[b.off..b.off + b.len])
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.name == name)
    }

    /// Per-row levels `[rows, K]` (as f32) and codes `[rows, cols]` of a `levels2` tensor.
    pub fn levels(&self, i: usize) -> Result<(Vec<f32>, Vec<u8>)> {
        let e = &self.entries[i];
        let Kind::Levels { k, lev, meta, words } = e.kind else {
            bail!("{} is not a levels tensor", e.name)
        };
        let rows = e.shape[0];
        let cols = e.numel() / rows;
        let lev = f16s(&self.blob(lev)?);
        ensure!(lev.len() == rows * k, "{}: {} levels, want {}", e.name, lev.len(), rows * k);
        let meta = self.blob(meta)?;
        let nb = (cols * 4).div_ceil(8);
        // 4-bit class per column, high nibble first (numpy unpackbits/packbits order).
        let cls: Vec<u8> = (0..cols)
            .map(|j| if j % 2 == 0 { meta[j / 2] >> 4 } else { meta[j / 2] & 15 })
            .collect();
        let counts = u32s(&meta[nb..]);
        let ns = 2 * k + 1;
        ensure!(counts.len() == NC * ns, "{}: {} counts, want {}", e.name, counts.len(), NC * ns);
        let mut dec = DefaultAnsCoder::from_compressed(self.words(words))
            .map_err(|_| anyhow!("{}: ANS words end in a zero word", e.name))?;
        let mut codes = vec![0u8; rows * cols];
        for c in 0..NC {
            let idx: Vec<usize> = (0..cols).filter(|&j| cls[j] as usize == c).collect();
            if idx.is_empty() {
                continue;
            }
            let model = categorical(&counts[c * ns..(c + 1) * ns])
                .with_context(|| format!("{} class {c}", e.name))?;
            let mut syms = dec.decode_iid_symbols(rows * idx.len(), &model);
            for &j in &idx {
                for r in 0..rows {
                    let s = syms.next().unwrap().map_err(|_| anyhow!("{}: ANS decode", e.name))?;
                    codes[r * cols + j] = s as u8;
                }
            }
        }
        Ok((lev, codes))
    }

    /// Integer symbols `q` (row major) and per-row scales of an `ansint` tensor.
    pub fn ansint(&self, i: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        let e = &self.entries[i];
        let Kind::AnsInt { bits, scale, meta, words } = e.kind else {
            bail!("{} is not an ansint tensor", e.name)
        };
        let scale = f16s(&self.blob(scale)?);
        let counts = u32s(&self.blob(meta)?);
        let model = categorical(&counts).with_context(|| e.name.clone())?;
        let mut dec = DefaultAnsCoder::from_compressed(self.words(words))
            .map_err(|_| anyhow!("{}: ANS words end in a zero word", e.name))?;
        let zero = (1i32 << (bits - 1)) - 1;
        let q = dec
            .decode_iid_symbols(e.numel(), &model)
            .map(|s| s.map(|s| s as i32 - zero))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("{}: ANS decode", e.name))?;
        ensure!(q.len() % scale.len() == 0, "{}: scales do not divide the tensor", e.name);
        Ok((q, scale))
    }

    /// Decodes one tensor to the values the model runs on.
    pub fn tensor(&self, i: usize) -> Result<Data> {
        let e = &self.entries[i];
        Ok(match e.kind {
            Kind::Levels { k, .. } => {
                let (lev, codes) = self.levels(i)?;
                let cols = e.numel() / e.shape[0];
                let mut w = vec![0f32; codes.len()];
                for (r, (row, out)) in codes.chunks(cols).zip(w.chunks_mut(cols)).enumerate() {
                    let l = &lev[r * k..(r + 1) * k];
                    for (o, &c) in out.iter_mut().zip(row) {
                        let c = c as usize;
                        *o = if c == 0 {
                            0.0
                        } else if c <= k {
                            l[c - 1]
                        } else {
                            -l[c - k - 1]
                        };
                    }
                }
                Data::F32(w)
            }
            Kind::AnsInt { .. } => {
                let (q, scale) = self.ansint(i)?;
                let per = q.len() / scale.len();
                Data::F32(q.iter().enumerate().map(|(j, &v)| v as f32 * scale[j / per]).collect())
            }
            Kind::Raw { dtype, data } => {
                let b = self.blob(data)?;
                match dtype {
                    RawType::F16 => Data::F32(f16s(&b)),
                    RawType::I64 => Data::I64(
                        b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
                    ),
                    RawType::I32 => Data::I32(
                        b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect(),
                    ),
                }
            }
        })
    }

    /// Decodes every tensor in parallel: `(name, shape, data)` in index order.
    pub fn all(&self) -> Result<Vec<(String, Vec<usize>, Data)>> {
        (0..self.entries.len())
            .into_par_iter()
            .map(|i| {
                let e = &self.entries[i];
                Ok((e.name.clone(), e.shape.clone(), self.tensor(i)?))
            })
            .collect()
    }
}

/// The Python loader's `Categorical(counts / counts.sum(), perfect=False)`.
fn categorical(counts: &[u32]) -> Result<DefaultContiguousCategoricalEntropyModel> {
    let sum: f64 = counts.iter().map(|&c| c as f64).sum();
    let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / sum).collect();
    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(&probs, None)
        .map_err(|_| anyhow!("bad symbol counts"))
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}
