//! `StripAccents` — NFD-decompose and drop combining marks.
//!
//! This is the building block that `BertNormalizer` uses internally;
//! exposed standalone for users who want it without the rest of the
//! BERT pipeline.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

use super::Normalizer;
use crate::error::Result;

/// NFD-decompose and drop combining marks.
#[derive(Debug, Default, Clone, Copy)]
pub struct StripAccents;

impl Normalizer for StripAccents {
    fn normalize<'a>(&self, text: &'a str) -> Result<Cow<'a, str>> {
        // We need to drop combining marks (Unicode Mn category). Without
        // the `unicode-general-category` crate we approximate by
        // checking known Mn blocks.
        fn is_mn(c: char) -> bool {
            let u = c as u32;
            matches!(u,
                0x0300..=0x036F
                | 0x0483..=0x0489
                | 0x0591..=0x05BD
                | 0x05BF
                | 0x05C1..=0x05C2
                | 0x05C4..=0x05C5
                | 0x05C7
                | 0x0610..=0x061A
                | 0x064B..=0x065F
                | 0x0670
                | 0x06D6..=0x06DC
                | 0x06DF..=0x06E4
                | 0x06E7..=0x06E8
                | 0x06EA..=0x06ED
            )
        }

        let mut out = String::with_capacity(text.len());
        let mut changed = false;
        for ch in text.nfd() {
            if is_mn(ch) {
                changed = true;
                continue;
            }
            out.push(ch);
        }
        if changed {
            Ok(Cow::Owned(out))
        } else {
            Ok(Cow::Borrowed(text))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_combining_acute() {
        // "é" decomposed is e + combining acute.
        let out = StripAccents.normalize("é").unwrap();
        assert_eq!(out, "e");
    }

    #[test]
    fn passthrough_when_no_accents() {
        let r = StripAccents.normalize("hello").unwrap();
        assert!(matches!(r, Cow::Borrowed(_)));
    }
}
