//! Crate-wide error type.

use thiserror::Error;

/// Errors produced by `morpheme`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A token was not present in the vocabulary and the model has no
    /// way to represent it (no `unk_token`, no byte fallback).
    #[error("token not in vocabulary: {0:?}")]
    UnknownToken(String),

    /// An id was outside the valid range of the vocabulary.
    #[error("id {0} is not in the vocabulary")]
    UnknownId(u32),

    /// A component's configuration is invalid or unsupported.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A regular expression failed to compile or to run.
    #[error("regex error: {0}")]
    Regex(String),

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

    /// Truncation could not be applied with the given parameters.
    #[error("truncation failed: {0}")]
    Truncation(String),

    /// Padding could not be applied with the given parameters.
    #[error("padding failed: {0}")]
    Padding(String),

    /// A trainer failed.
    #[error("training failed: {0}")]
    Training(String),

    /// Downloading from the Hugging Face Hub failed (feature `hub`).
    #[error("hub error: {0}")]
    Hub(String),

    /// JSON (de)serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<fancy_regex::Error> for Error {
    fn from(e: fancy_regex::Error) -> Self {
        Error::Regex(e.to_string())
    }
}

/// Result alias for `morpheme`.
pub type Result<T> = std::result::Result<T, Error>;
