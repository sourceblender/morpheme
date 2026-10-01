//! `morpheme` — fast, pure-Rust subword tokenization compatible with
//! Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers).
//!
//! morpheme loads, runs, trains and saves tokenizers in the Hugging Face
//! `tokenizer.json` format (BPE, WordPiece, WordLevel and Unigram) and
//! produces the same ids, tokens, offsets and decoded text as the
//! reference implementation.
//!
//! # Quick start
//!
//! Load a `tokenizer.json` (from disk, or from the Hub with the `hub`
//! feature) and encode:
//!
//! ```no_run
//! use morpheme::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//! let encoding = tokenizer.encode("Hello, world!", true)?;
//! println!("{:?} {:?}", encoding.tokens(), encoding.ids());
//! let text = tokenizer.decode(encoding.ids(), true)?;
//! # Ok::<(), morpheme::Error>(())
//! ```
//!
//! Or build one in code, train it, and save it:
//!
//! ```
//! use morpheme::models::WordPiece;
//! use morpheme::normalizers::BertNormalizer;
//! use morpheme::pre_tokenizers::BertPreTokenizer;
//! use morpheme::trainers::WordPieceTrainer;
//! use morpheme::{AddedToken, Tokenizer};
//!
//! let mut tokenizer = Tokenizer::new(WordPiece::default())
//!     .with_normalizer(BertNormalizer::default())
//!     .with_pre_tokenizer(BertPreTokenizer);
//! let trainer = WordPieceTrainer::builder()
//!     .vocab_size(100)
//!     .special_tokens(vec![AddedToken::new("[UNK]", true)])
//!     .show_progress(false)
//!     .build()?;
//! tokenizer.train(trainer, ["Hello world!", "hello again, World"].into_iter())?;
//!
//! let encoding = tokenizer.encode("HELLO, world", false)?;
//! assert_eq!(encoding.tokens(), ["hello", ",", "world"]);
//! assert_eq!(encoding.offsets()[1], (5, 6)); // byte offsets into the input
//!
//! let json = tokenizer.to_json(false)?; // a valid Hugging Face tokenizer.json
//! let reloaded = Tokenizer::from_json(&json)?;
//! assert_eq!(reloaded.encode("HELLO, world", false)?, encoding);
//! # Ok::<(), morpheme::Error>(())
//! ```
//!
//! # The pipeline
//!
//! ```text
//! input ─▶ added tokens ─▶ Normalizer ─▶ PreTokenizer ─▶ Model
//!       ─▶ truncation ─▶ PostProcessor ─▶ padding ─▶ Encoding
//!
//! ids ─▶ (skip special tokens) ─▶ Decoder ─▶ text
//! ```
//!
//! | Stage | Trait | Built-in components |
//! | --- | --- | --- |
//! | Normalize text | [`Normalizer`] | [`normalizers`] |
//! | Split into words | [`PreTokenizer`] | [`pre_tokenizers`] |
//! | Words → tokens | [`Model`] | [`models`] |
//! | Add special tokens, merge pairs | [`PostProcessor`] | [`processors`] |
//! | Tokens → text | [`Decoder`] | [`decoders`] |
//! | Learn a model | [`Trainer`] | [`trainers`] |
//!
//! Each module has a wrapper enum ([`NormalizerWrapper`], [`ModelWrapper`],
//! …) that a [`Tokenizer`] stores and that (de)serializes with the
//! Hugging Face `"type"` tag. Offsets always point into the original input,
//! even after normalization, thanks to [`NormalizedString`]'s alignment
//! tracking: [`Tokenizer::encode`] returns byte offsets and
//! [`Tokenizer::encode_char_offsets`] returns char offsets (what the Python
//! library returns).
//!
//! # Cargo features
//!
//! | Feature | Default | Enables |
//! | --- | --- | --- |
//! | `progressbar` | yes | Trainer progress bars on stderr (`show_progress`) |
//! | `hub` | no | [`Tokenizer::from_pretrained`]: download from the Hugging Face Hub into the cache shared with Python |
//!
//! # Compatibility
//!
//! Behavior is verified against 11 real tokenizers (BERT, GPT-2, RoBERTa,
//! Llama, T5, XLM-R, …) recorded from `tokenizers` 0.23. Known, deliberate
//! differences are listed in the repository's `docs/interop.md`.

#![deny(missing_docs)]
#![warn(missing_debug_implementations, unreachable_pub)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
mod progress;
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
#[cfg(all(feature = "hub", not(target_arch = "wasm32")))]
#[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
pub use tokenizer::FromPretrainedParameters;
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
