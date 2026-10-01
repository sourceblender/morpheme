//! Subword models. v0.1 ships BPE, WordPiece, and Unigram.

use std::any::Any;

use crate::error::Result;
use crate::pre_tokenizer::PreToken;

/// Anything that maps a pre-token (a slice of the input buffer) to a
/// sequence of token strings.
pub trait Model: Send + Sync + Any {
    /// Tokenize a single pre-token.
    fn tokenize(&self, pre_token: PreToken<'_>) -> Result<Vec<String>>;
}

pub mod bpe;
pub mod unigram;
pub mod wordpiece;
pub use bpe::Bpe;
pub use unigram::Unigram;
pub use wordpiece::WordPiece;
