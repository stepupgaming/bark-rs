//! Underdog Bark 3.0 speech to text in Rust, straight from the 163 MB `model.bark` file.

pub mod audio;
pub mod format;
pub mod mel;
pub mod model;

pub use format::BarkFile;
pub use model::{Bark, Token, Transcript, Word};
