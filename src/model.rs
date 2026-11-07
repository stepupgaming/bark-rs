//! Bark 3.0 on ONNX Runtime: weightless graphs, weights built from `model.bark`, greedy TDT decoding.
//!
//! The graphs directory holds `encoder.onnx`, `decoder.onnx`, `joint.onnx` (made once by
//! `tools/make_graphs.py`), `manifest.json` (which `model.bark` tensor feeds each external
//! initializer, and how) and `vocab.json`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use rayon::prelude::*;
use serde_json::Value;

use crate::format::{BarkFile, Data};
use crate::mel::Mel;

/// Silence put before the audio, as in the evaluated setting (0.25 s).
pub const LEAD_IN: usize = 4000;

/// Default longest piece read in one pass, in seconds. Against a 276 s script, 10 to 60 s pieces
/// and one whole pass all differ from the text by the same 3 to 5 words. Longer pieces need more
/// memory; the exact graphs also pay about 0.85 s per piece on desktop, the int8 graphs don't.
pub const MAX_PIECE: f64 = 30.0;

pub struct Bark {
    enc: Session,
    dec: Session,
    joint: Session,
    vocab: Vec<String>,
    blank: usize,
    durations: Vec<usize>,
    layers: usize,
    hidden: usize,
    mel: Mel,
    max_piece: f64,
    _env: ort::environment::Environment,
}

/// Seconds per encoder frame: 10 ms mel hop times 8x subsampling.
pub const FRAME: f64 = 0.08;

/// Punctuation that takes the end time of the token before it (NeMo's TDT timestamp rule, as in
/// transformers' `ParakeetProcessor`).
const PUNCT: [&str; 11] = ["?", "'", "\u{a1}", "\u{bf}", "-", ":", ",", "%", "/", ".", "!"];

#[derive(Clone, Debug)]
pub struct Transcript {
    pub text: String,
    /// Every emitted token id, in order.
    pub ids: Vec<usize>,
    /// Text tokens with times; `<unk>` and other special tokens are left out.
    pub tokens: Vec<Token>,
    pub words: Vec<Word>,
}

/// A token and when it was said, in seconds from the start of the audio passed in.
/// `start` is the encoder frame it was emitted on, `end` adds its predicted TDT duration.
#[derive(Clone, Debug)]
pub struct Token {
    pub id: usize,
    /// The SentencePiece piece, with `\u{2581}` (a word start) shown as a space.
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// A word: from its first token's start to its last token's end. Punctuation joins the word
/// before it.
#[derive(Clone, Debug)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl Bark {
    pub fn load(model: impl AsRef<Path>, graphs: impl AsRef<Path>, threads: usize) -> Result<Self> {
        let graphs = graphs.as_ref();
        let file = BarkFile::open(model)?;
        let manifest: Value = serde_json::from_slice(&std::fs::read(graphs.join("manifest.json"))?)?;
        let voc: Value = serde_json::from_slice(&std::fs::read(graphs.join("vocab.json"))?)?;
        // every external initializer, built straight from model.bark in parallel, one job per
        // source tensor (a packed weight has several initializers from one decode)
        let mut groups: HashMap<(&str, String), Vec<(String, Value)>> = HashMap::new();
        for g in ["encoder", "decoder", "joint"] {
            let entries = manifest[g].as_object().ok_or_else(|| anyhow!("manifest has no {g}"))?;
            for (n, s) in entries {
                let src = s["src"].as_str().unwrap_or("").to_string();
                groups.entry((g, src)).or_default().push((n.clone(), s.clone()));
            }
        }
        let res: Vec<_> = groups
            .into_par_iter()
            .map(|((g, src), specs)| {
                let out = build(&src, &specs, &file).with_context(|| format!("{g}: {src}"))?;
                Ok::<_, anyhow::Error>((g, out))
            })
            .collect::<Result<_>>()?;
        drop(file);
        let mut built: HashMap<&str, Vec<Init>> = HashMap::new();
        for (g, out) in res {
            built.entry(g).or_default().extend(out);
        }
        let env = ort::init().with_name("bark").build().map_err(|e| anyhow!("{e}"))?;
        let mut session = |name: &str| -> Result<Session> {
            let path = graphs.join(format!("{name}.onnx"));
            let e = |err: ort::Error<_>| anyhow!("{}: {err}", path.display());
            let mut b = Session::builder(&env)
                .map_err(|err| anyhow!("{err}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(e)?
                .with_intra_threads(threads)
                .map_err(e)?;
            for (init, shape, data) in built.remove(name).unwrap_or_default() {
                let t = match data {
                    Buf::F32(v) => Tensor::from_array((shape, v))?.into_dyn(),
                    Buf::U8(v) => Tensor::from_array((shape, v))?.into_dyn(),
                };
                b = b.with_external_initializer(init, t).map_err(e)?;
            }
            b.commit_from_file(&path).map_err(|err| anyhow!("{}: {err}", path.display()))
        };
        let enc = session("encoder")?;
        let dec = session("decoder")?;
        let joint = session("joint")?;
        let vocab = voc["vocab"]
            .as_array()
            .ok_or_else(|| anyhow!("vocab.json: no vocab"))?
            .iter()
            .map(|t| t.as_str().unwrap_or("").to_string())
            .collect();
        let num = |k: &str| voc[k].as_u64().map(|v| v as usize).ok_or_else(|| anyhow!("vocab.json: no {k}"));
        let durations = voc["durations"]
            .as_array()
            .ok_or_else(|| anyhow!("vocab.json: no durations"))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .collect();
        Ok(Self {
            enc,
            dec,
            joint,
            vocab,
            blank: num("blank")?,
            durations,
            layers: num("layers")?,
            hidden: num("hidden")?,
            mel: Mel::new(128),
            max_piece: MAX_PIECE,
            _env: env,
        })
    }

    /// Longest piece, in seconds, read in one pass; longer audio is cut at pauses into pieces
    /// between a third of this and this long. Default [`MAX_PIECE`].
    pub fn set_max_piece(&mut self, seconds: f64) {
        self.max_piece = seconds.max(1.0);
    }

    /// Any rate and channel count: downmixed, resampled to 16 kHz, then [`Bark::transcribe`].
    pub fn transcribe_pcm(&mut self, interleaved: &[f32], channels: usize, rate: u32) -> Result<Transcript> {
        let wave = crate::audio::to_model_input(interleaved, channels, rate)?;
        self.transcribe(&wave)
    }

    /// `wave`: mono 16 kHz samples, any length. Audio longer than the piece limit is cut at
    /// pauses ([`crate::audio::segments`]); each piece gets its own 0.25 s lead-in, and times
    /// are shifted back onto the whole recording.
    pub fn transcribe(&mut self, wave: &[f32]) -> Result<Transcript> {
        let rate = crate::audio::RATE as f64;
        let lead = LEAD_IN as f64 / rate;
        let pieces = crate::audio::segments(wave, self.max_piece, self.max_piece / 3.0);
        let mut all = Transcript { text: String::new(), ids: Vec::new(), tokens: Vec::new(), words: Vec::new() };
        let mut texts = Vec::new();
        let mut decoded = Vec::with_capacity(pieces.len());
        for r in &pieces {
            decoded.push(self.decode(&wave[r.clone()])?);
        }
        for (r, raw) in pieces.into_iter().zip(decoded) {
            let offset = r.start as f64 / rate;
            let ids: Vec<usize> = raw.iter().map(|t| t.0).collect();
            texts.push(self.detokenize(&ids));
            all.ids.extend(&ids);
            for (id, frame, dur) in raw {
                if self.special(id) {
                    continue;
                }
                let piece = &self.vocab[id];
                let mut start = (frame as f64 * FRAME - lead).max(0.0) + offset;
                let mut end = ((frame + dur) as f64 * FRAME - lead).max(0.0) + offset;
                if PUNCT.contains(&piece.as_str())
                    && let Some(prev) = all.tokens.last()
                {
                    start = prev.end;
                    end = prev.end;
                }
                all.tokens.push(Token { id, text: piece.replace('\u{2581}', " "), start, end });
            }
        }
        all.text = texts.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
        all.words = words(&all.tokens, &self.vocab);
        Ok(all)
    }

    /// One pass over a piece: (token id, encoder frame, predicted duration) per emitted token,
    /// frames counted from the start of the lead-in.
    fn decode(&mut self, wave: &[f32]) -> Result<Vec<(usize, usize, usize)>> {
        let mut w = vec![0f32; LEAD_IN];
        w.extend_from_slice(wave);
        let f = self.mel.features(&w);
        let mask: Vec<i64> = (0..f.frames).map(|t| (t < f.valid) as i64).collect();
        let out = self.enc.run(ort::inputs![
            Tensor::from_array(([1usize, f.frames, f.n_mels], f.data))?,
            Tensor::from_array(([1usize, f.frames], mask))?
        ])?;
        let (shape, frames) = out[0].try_extract_tensor::<f32>()?;
        let dim = shape[2] as usize;
        let frames = frames.to_vec();
        let n = out[1].try_extract_tensor::<i64>()?.1[0] as usize;
        drop(out);
        ensure!(dim == self.hidden, "encoder frames are {dim} wide, decoder is {}", self.hidden);

        let v = self.blank + 1;
        let mut h = vec![0f32; self.layers * self.hidden];
        let mut c = h.clone();
        let mut dout = self.step(self.blank, &mut h, &mut c)?;
        let (mut t, mut ids) = (0usize, Vec::<(usize, usize, usize)>::new());
        while t < n {
            let enc = &frames[t * dim..(t + 1) * dim];
            let out = self.joint.run(ort::inputs![
                Tensor::from_array(([1usize, dim], enc.to_vec()))?,
                Tensor::from_array(([1usize, dim], dout.clone()))?
            ])?;
            let lg = out[0].try_extract_tensor::<f32>()?.1;
            let k = argmax(&lg[..v]);
            let mut dur = self.durations[argmax(&lg[v..])];
            drop(out);
            if k == self.blank {
                dur = dur.max(1);
            } else {
                ids.push((k, t, dur));
                dout = self.step(k, &mut h, &mut c)?;
            }
            t += dur;
        }
        Ok(ids)
    }

    /// `<unk>`, `<pad>`, blank and anything past the vocabulary.
    fn special(&self, id: usize) -> bool {
        id == 0 || id == 2 || id >= self.vocab.len()
    }

    /// One decoder step: projected output, and the new LSTM state in place.
    fn step(&mut self, token: usize, h: &mut Vec<f32>, c: &mut Vec<f32>) -> Result<Vec<f32>> {
        let s = [self.layers, 1, self.hidden];
        let out = self.dec.run(ort::inputs![
            Tensor::from_array(([1usize, 1], vec![token as i64]))?,
            Tensor::from_array((s, h.clone()))?,
            Tensor::from_array((s, c.clone()))?
        ])?;
        *h = out[1].try_extract_tensor::<f32>()?.1.to_vec();
        *c = out[2].try_extract_tensor::<f32>()?.1.to_vec();
        Ok(out[0].try_extract_tensor::<f32>()?.1.to_vec())
    }

    /// SentencePiece-style pieces: `▁` starts a word. Drops `<unk>`, `<pad>` and blank.
    pub fn detokenize(&self, ids: &[usize]) -> String {
        let mut s = String::new();
        for &i in ids {
            if self.special(i) {
                continue;
            }
            s.push_str(&self.vocab[i].replace('\u{2581}', " "));
        }
        s.trim().to_string()
    }
}

/// NeMo's word grouping for SentencePiece: a piece that starts with `\u{2581}` starts a word,
/// unless it is punctuation, which joins the word before it.
fn words(tokens: &[Token], vocab: &[String]) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    for t in tokens {
        let piece = vocab[t.id].as_str();
        let bare = piece.trim_start_matches('\u{2581}');
        let starts = piece.starts_with('\u{2581}') && !PUNCT.contains(&bare);
        match out.last_mut() {
            Some(w) if !starts => {
                w.text.push_str(bare);
                w.end = t.end;
            }
            _ => out.push(Word { text: bare.to_string(), start: t.start, end: t.end }),
        }
    }
    out
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

fn f32s(file: &BarkFile, key: &str) -> Result<Vec<f32>> {
    let i = file.find(key).ok_or_else(|| anyhow!("model.bark has no {key}"))?;
    match file.tensor(i)? {
        Data::F32(v) => Ok(v),
        _ => bail!("{key} is not a float tensor"),
    }
}

/// Torch LSTM gate order (i, f, g, o) to ONNX (i, o, f, c).
fn gates(a: &[f32]) -> Vec<f32> {
    let q = a.len() / 4;
    [0, 3, 1, 2].iter().flat_map(|&g| a[g * q..(g + 1) * q].iter().copied()).collect()
}

enum Buf {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

impl Buf {
    fn len(&self) -> usize {
        match self {
            Buf::F32(v) => v.len(),
            Buf::U8(v) => v.len(),
        }
    }
}

type Init = (String, Vec<usize>, Buf);

/// MatMulNBits block size, as in `tools/make_graphs.py`.
const BS: usize = 128;

/// Every external initializer that comes from one `model.bark` tensor, by manifest recipe.
fn build(src: &str, specs: &[(String, Value)], file: &BarkFile) -> Result<Vec<Init>> {
    let packed = specs.iter().any(|(_, s)| {
        matches!(s["op"].as_str(), Some("tern" | "tern_s" | "q8" | "q8_s"))
    });
    // packed weights come from the stored codes, never through fp32
    let levels = if packed {
        let i = file.find(src).ok_or_else(|| anyhow!("model.bark has no {src}"))?;
        Some(file.levels(i)?)
    } else {
        None
    };
    let mut q8_scale: Option<Vec<f32>> = None;
    let mut out = Vec::with_capacity(specs.len());
    for (init, spec) in specs {
        let shape: Vec<usize> =
            spec["shape"].as_array().into_iter().flatten().map(|d| d.as_u64().unwrap_or(0) as usize).collect();
        let op = spec["op"].as_str().unwrap_or("");
        let data = if let Some((lev, codes)) = &levels {
            let rows = shape[0];
            let cols = spec["cols"].as_u64().ok_or_else(|| anyhow!("{init}: no cols"))? as usize;
            let k = lev.len() / rows;
            let term = spec["term"].as_u64().unwrap_or(0) as usize;
            match op {
                "tern" => Buf::U8(pack_ternary(codes, rows, cols, k, term)),
                "tern_s" => Buf::F32(repeat_rows(&(0..rows).map(|r| lev[r * k + term]).collect::<Vec<_>>(), cols)),
                "q8" | "q8_s" => {
                    let s = q8_scale.get_or_insert_with(|| q8_scales(lev, k));
                    if op == "q8" {
                        Buf::U8(pack_q8(codes, lev, s, rows, cols, k))
                    } else {
                        Buf::F32(repeat_rows(s, cols))
                    }
                }
                _ => bail!("{init}: op {op:?} on a packed weight"),
            }
        } else {
            let src = f32s(file, src)?;
            Buf::F32(match op {
                "id" => src,
                "t" => {
                    // stored [rows, rest], wanted [rest, rows]
                    let rest = shape[..shape.len() - 1].iter().product::<usize>();
                    let rows = src.len() / rest;
                    let mut o = vec![0f32; src.len()];
                    for r in 0..rows {
                        for j in 0..rest {
                            o[j * rows + r] = src[r * rest + j];
                        }
                    }
                    o
                }
                "lstm" => gates(&src),
                "lstm_b" => {
                    let mut o = gates(&src);
                    o.extend(gates(&f32s(file, spec["src2"].as_str().unwrap_or(""))?));
                    o
                }
                _ => bail!("unknown op {op:?}"),
            })
        };
        ensure!(data.len() == shape.iter().product::<usize>(), "{init}: size {} does not fit {shape:?}", data.len());
        out.push((init.clone(), shape, data));
    }
    Ok(out)
}

/// Per-row scales repeated for every block: `[rows, ceil(cols / BS)]`.
fn repeat_rows(s: &[f32], cols: usize) -> Vec<f32> {
    let nblk = cols.div_ceil(BS);
    s.iter().flat_map(|&v| std::iter::repeat_n(v, nblk)).collect()
}

/// Ternary matrix of level `term`: +1 where the code is `term + 1`, -1 where it is
/// `k + term + 1`, else 0. Stored as `t + 2` (zero point 2), four per byte, low bits first,
/// each row padded to whole blocks.
fn pack_ternary(codes: &[u8], rows: usize, cols: usize, k: usize, term: usize) -> Vec<u8> {
    let per_row = cols.div_ceil(BS) * BS / 4;
    let (pos, neg) = ((term + 1) as u8, (k + term + 1) as u8);
    let mut out = vec![0b1010_1010u8; rows * per_row];
    out.par_chunks_mut(per_row).enumerate().for_each(|(r, row)| {
        for (c, &code) in codes[r * cols..(r + 1) * cols].iter().enumerate() {
            let q = if code == pos { 3 } else if code == neg { 1 } else { 2 };
            let (byte, shift) = (c / 4, 2 * (c % 4));
            row[byte] = (row[byte] & !(3 << shift)) | (q << shift);
        }
    });
    out
}

/// Per-row 8-bit scale `max level / n`, n in 1..=127 chosen so every level lands closest to
/// a multiple of it (the largest level is exact).
fn q8_scales(lev: &[f32], k: usize) -> Vec<f32> {
    lev.chunks(k)
        .map(|l| {
            let top = l.iter().fold(0f32, |a, &b| a.max(b.abs()));
            if top == 0.0 {
                return 1.0;
            }
            let mut best = (f32::INFINITY, top / 127.0);
            for n in (1..=127).rev() {
                let s = top / n as f32;
                let err = l.iter().map(|&v| (v - (v / s).round() * s).abs()).fold(0f32, f32::max);
                if err < best.0 {
                    best = (err, s);
                }
            }
            best.1
        })
        .collect()
}

/// 8-bit codes `round(w / s) + 128`, one per byte, rows padded to whole blocks.
fn pack_q8(codes: &[u8], lev: &[f32], s: &[f32], rows: usize, cols: usize, k: usize) -> Vec<u8> {
    let per_row = cols.div_ceil(BS) * BS;
    let mut out = vec![128u8; rows * per_row];
    out.par_chunks_mut(per_row).enumerate().for_each(|(r, row)| {
        let q: Vec<u8> = (0..=2 * k)
            .map(|c| {
                let w = match c {
                    0 => 0.0,
                    c if c <= k => lev[r * k + c - 1],
                    c => -lev[r * k + c - k - 1],
                };
                ((w / s[r]).round().clamp(-127.0, 127.0) as i32 + 128) as u8
            })
            .collect();
        for (o, &code) in row.iter_mut().zip(&codes[r * cols..(r + 1) * cols]) {
            *o = q[code as usize];
        }
    });
    out
}
