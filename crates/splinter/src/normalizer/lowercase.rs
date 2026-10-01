//! `Lowercase` — ASCII lowercase fold.
//!
//! Uses `char::to_lowercase` so non-ASCII characters get their full
//! Unicode case fold (e.g. `İ` → `i\u{307}`), matching the HF
//! `tokenizers` `Lowercase` behaviour.

use std::borrow::Cow;

use super::Normalizer;
use crate::error::Result;

/// ASCII / Unicode case-folding normalizer.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lowercase;

impl Normalizer for Lowercase {
    fn normalize<'a>(&self, text: &'a str) -> Result<Cow<'a, str>> {
        // Fast path: no uppercase chars.
        if !text.chars().any(|c| c.is_uppercase()) {
            return Ok(Cow::Borrowed(text));
        }
        Ok(Cow::Owned(text.to_lowercase()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii() {
        assert_eq!(Lowercase.normalize("HELLO").unwrap(), "hello");
    }

    #[test]
    fn passthrough_when_already_lower() {
        let r = Lowercase.normalize("hello").unwrap();
        assert!(matches!(r, Cow::Borrowed(_)));
    }
}
