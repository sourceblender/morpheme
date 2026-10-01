//! `splinter` — a Rust tokenizer library compatible with Hugging Face
//! `tokenizers`.
//!
//! A [`Tokenizer`] is a pipeline:
//!
//! ```text
//! input ─▶ added tokens ─▶ Normalizer ─▶ PreTokenizer ─▶ Model
//!       ─▶ truncation ─▶ PostProcessor ─▶ padding ─▶ Encoding
//! ```
//!
//! Tokenizers load from and save to the Hugging Face `tokenizer.json`
//! format, and produce the same ids, tokens and offsets as the
//! reference implementation.
//!
//! ```no_run
//! use splinter::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//! let encoding = tokenizer.encode("Hello, world!", true)?;
//! println!("{:?}", encoding.tokens());
//! let text = tokenizer.decode(encoding.ids(), true)?;
//! # Ok::<(), splinter::Error>(())
//! ```
//!
//! See `docs/architecture.md` for the design.

#![deny(missing_docs)]

/// Library version, mirrored from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A `(start, end)` byte or char range.
pub type Offsets = (usize, usize);

pub mod added_vocabulary;
pub mod decoders;
pub mod encoding;
pub mod error;
pub mod models;
pub mod normalized_string;
pub mod normalizers;
pub mod pattern;
pub mod pre_tokenized_string;
pub mod pre_tokenizers;
pub mod processors;
pub mod tokenizer;
pub mod trainers;
pub mod traits;

pub use added_vocabulary::{AddedToken, AddedVocabulary};
pub use decoders::DecoderWrapper;
pub use encoding::{Encoding, PaddingDirection, TruncationDirection};
pub use error::{Error, Result};
pub use models::ModelWrapper;
pub use normalized_string::{NormalizedString, OffsetRange, SplitDelimiterBehavior};
pub use normalizers::NormalizerWrapper;
pub use pre_tokenized_string::{OffsetType, PreTokenizedString, Split};
pub use pre_tokenizers::PreTokenizerWrapper;
pub use processors::PostProcessorWrapper;
pub use tokenizer::{
    DecodeStream, EncodeInput, InputSequence, PaddingParams, PaddingStrategy, Tokenizer,
    TruncationParams, TruncationStrategy,
};
pub use traits::{Decoder, Model, Normalizer, PostProcessor, PreTokenizer, Trainer};

/// A token produced by a [`Model`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Vocabulary id.
    pub id: u32,
    /// Token string.
    pub value: String,
    /// Byte offsets into the text the model was given.
    pub offsets: Offsets,
}

impl Token {
    /// Build a token.
    pub fn new(id: u32, value: String, offsets: Offsets) -> Self {
        Self { id, value, offsets }
    }
}
