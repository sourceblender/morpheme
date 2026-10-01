//! `Whitespace` — split on runs of ASCII whitespace.

use crate::error::Result;
use crate::pre_tokenizer::{PreToken, PreTokenizer};

/// Split on runs of ASCII whitespace.
///
/// Non-ASCII whitespace (e.g. U+00A0 NBSP) is treated as a non-separator
/// for now — matches the simplest `tokenizers` configuration.
#[derive(Debug, Default, Clone, Copy)]
pub struct Whitespace;

impl PreTokenizer for Whitespace {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<PreToken<'a>>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut start = None;

        for (i, &b) in bytes.iter().enumerate() {
            if b.is_ascii_whitespace() {
                if let Some(s) = start.take() {
                    out.push(PreToken::borrowed(&text[s..i], (s, i)));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            out.push(PreToken::borrowed(&text[s..], (s, text.len())));
        }
        Ok(out)
    }
}
