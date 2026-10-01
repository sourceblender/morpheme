//! `Nfd` / `Nfkc` — Unicode normalization forms.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

use super::Normalizer;
use crate::error::Result;

/// Canonical decomposition (`NFD`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Nfd;

impl Normalizer for Nfd {
    fn normalize<'a>(&self, text: &'a str) -> Result<Cow<'a, str>> {
        // Cheap check: compare decomposed length to source.
        let decomposed: String = text.nfd().collect();
        if decomposed == text {
            Ok(Cow::Borrowed(text))
        } else {
            Ok(Cow::Owned(decomposed))
        }
    }
}

/// Compatibility decomposition followed by canonical composition (`NFKC`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Nfkc;

impl Normalizer for Nfkc {
    fn normalize<'a>(&self, text: &'a str) -> Result<Cow<'a, str>> {
        let out: String = text.nfkc().collect();
        if out == text {
            Ok(Cow::Borrowed(text))
        } else {
            Ok(Cow::Owned(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfd_decomposes() {
        let out = Nfd.normalize("é").unwrap();
        assert_eq!(out.chars().count(), 2);
    }

    #[test]
    fn nfkc_collapses() {
        // U+FB01 (ﬁ) → "fi" under NFKC.
        let out = Nfkc.normalize("\u{FB01}ne").unwrap();
        assert_eq!(out, "fine");
    }
}
