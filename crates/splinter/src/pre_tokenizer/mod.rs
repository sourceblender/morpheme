//! Pre-tokenizers — split normalized text into pre-tokens.
//!
//! v0.1 ships `Whitespace`, `BertPreTokenizer`, and `ByteLevel`.

pub mod bert;
pub mod byte_level;
pub mod metaspace;
pub mod whitespace;

use crate::error::Result;

/// A pre-token. Holds either a borrowed view into the original input
/// or an owned string for pre-tokenizers that transform the bytes
/// (e.g. `ByteLevel`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreToken<'a> {
    /// The pre-token text. Either a sub-slice of the original buffer
    /// or an owned string — see [`PreToken::borrowed`] / [`PreToken::owned`].
    pub text: PreTokenText<'a>,
    /// Byte offsets into the original buffer.
    pub span: (usize, usize),
}

/// Borrowed or owned text inside a [`PreToken`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreTokenText<'a> {
    /// A sub-slice of the original buffer.
    Borrowed(&'a str),
    /// A string owned by the pre-token (used when the pre-tokenizer
    /// transforms the input — e.g. `ByteLevel`'s byte→char mapping).
    Owned(String),
}

impl<'a> PreTokenText<'a> {
    /// Borrow the underlying string regardless of variant.
    pub fn as_str(&self) -> &str {
        match self {
            PreTokenText::Borrowed(s) => s,
            PreTokenText::Owned(s) => s.as_str(),
        }
    }
}

impl<'a> PreToken<'a> {
    /// Build a borrowed pre-token.
    pub fn borrowed(text: &'a str, span: (usize, usize)) -> Self {
        Self {
            text: PreTokenText::Borrowed(text),
            span,
        }
    }

    /// Build an owned pre-token. `span` is in the *original* input's
    /// byte coordinates; `text` is the mapped/owned string.
    pub fn owned(text: String, span: (usize, usize)) -> Self {
        Self {
            text: PreTokenText::Owned(text),
            span,
        }
    }
}

/// Anything that splits a string into pre-tokens.
pub trait PreTokenizer: Send + Sync + std::fmt::Debug {
    /// Split `text` into pre-tokens.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<PreToken<'a>>>;
}

pub use bert::BertPreTokenizer;
pub use byte_level::{ByteLevel, ByteLevelAddChar};
pub use metaspace::MetaspacePreTokenizer;
pub use whitespace::Whitespace;
