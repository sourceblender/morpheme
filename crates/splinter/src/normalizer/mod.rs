//! Normalizers — text → canonicalized text.
//!
//! v0.1 ships `IdentityNormalizer`, `BertNormalizer`, `Lowercase`,
//! `Nfd`/`Nfkc`, `StripAccents`, and `Replace`.
//! The trait shape mirrors `huggingface/tokenizers` so the rest of the
//! pipeline doesn't have to change when more normalizers land.

pub mod bert;
pub mod lowercase;
pub mod replace;
pub mod strip_accents;
pub mod unicode;

use crate::error::Result;

/// Anything that maps an input string to a normalized string.
pub trait Normalizer: Send + Sync + std::fmt::Debug {
    /// Normalize a string slice. May return a borrowed view when no
    /// mutation is needed.
    fn normalize<'a>(&self, text: &'a str) -> Result<std::borrow::Cow<'a, str>>;
}

/// A no-op normalizer. Pass-through.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityNormalizer;

impl Normalizer for IdentityNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Result<std::borrow::Cow<'a, str>> {
        Ok(std::borrow::Cow::Borrowed(text))
    }
}

pub use bert::{BertNormalizer, BertNormalizerOpts};
pub use lowercase::Lowercase;
pub use replace::Replace;
pub use strip_accents::StripAccents;
pub use unicode::{Nfd, Nfkc};
