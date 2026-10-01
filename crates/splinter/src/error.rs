//! Crate-wide error type.

use thiserror::Error;

/// Errors produced by `splinter`.
#[derive(Debug, Error)]
pub enum Error {
    /// A token was not present in the vocabulary.
    #[error("token not in vocabulary: {0:?}")]
    UnknownToken(String),

    /// An id was outside the valid range of the vocabulary.
    #[error("id {id} out of range (vocab size {vocab_size})")]
    UnknownId {
        /// The requested id.
        id: u32,
        /// The actual vocab size.
        vocab_size: usize,
    },

    /// The configured normalizer failed.
    #[error("normalizer failed: {0}")]
    Normalizer(String),

    /// The configured pre-tokenizer failed.
    #[error("pre-tokenizer failed: {0}")]
    PreTokenizer(String),

    /// The configured model failed.
    #[error("model failed: {0}")]
    Model(String),

    /// The configured post-processor failed.
    #[error("post-processor failed: {0}")]
    PostProcessor(String),

    /// The configured decoder failed.
    #[error("decoder failed: {0}")]
    Decoder(String),

    /// JSON (de)serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for `splinter`.
pub type Result<T> = std::result::Result<T, Error>;
