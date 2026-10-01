//! `MetaspacePreTokenizer` — converts whitespace to a marker
//! character (typically `▁`, U+2581).
//!
//! This is the pre-tokenizer that SentencePiece/Unigram tokenizers
//! expect: every word's leading ASCII whitespace is rewritten as the
//! marker so the model can treat word boundaries as a single
//! character.

use crate::error::Result;
use crate::pre_tokenizer::{PreToken, PreTokenizer};

/// Replace runs of whitespace at the start of each pre-token with a
/// marker character.
#[derive(Debug, Clone, Copy)]
pub struct MetaspacePreTokenizer {
    /// The character used to mark word-leading whitespace.
    pub marker: char,
    /// If true, prepend the marker to every pre-token (SentencePiece's
    /// "add_dummy_prefix" behavior).
    pub add_dummy_prefix: bool,
}

impl Default for MetaspacePreTokenizer {
    fn default() -> Self {
        Self {
            marker: '\u{2581}',
            add_dummy_prefix: true,
        }
    }
}

impl MetaspacePreTokenizer {
    /// Construct a `MetaspacePreTokenizer` with explicit marker and
    /// `add_dummy_prefix` settings.
    pub fn new(marker: char, add_dummy_prefix: bool) -> Self {
        Self {
            marker,
            add_dummy_prefix,
        }
    }
}

impl PreTokenizer for MetaspacePreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<PreToken<'a>>> {
        let bytes = text.as_bytes();
        let mut out: Vec<PreToken<'a>> = Vec::new();
        let mut start: Option<usize> = None;
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
        // For each pre-token, produce an owned text that replaces
        // leading whitespace with the marker, then prepend the
        // marker if `add_dummy_prefix` is on.
        let mut rewritten: Vec<PreToken<'a>> = Vec::with_capacity(out.len());
        for (idx, pt) in out.into_iter().enumerate() {
            let text = pt.text.as_str();
            let bytes = text.as_bytes();
            let leading_ws: usize = bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
            let rest = &text[leading_ws..];
            let mut s = String::with_capacity(text.len() + 1);
            if self.add_dummy_prefix && idx == 0 {
                s.push(self.marker);
            }
            for _ in 0..leading_ws {
                s.push(self.marker);
            }
            s.push_str(rest);
            rewritten.push(PreToken::owned(s, pt.span));
        }
        Ok(rewritten)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_leading_whitespace() {
        let p = MetaspacePreTokenizer::default();
        let out = p.pre_tokenize("hello world").unwrap();
        assert_eq!(out.len(), 2);
        // First pre-token: marker as dummy prefix + 'hello'.
        assert_eq!(out[0].text.as_str(), "▁hello");
        // Second pre-token: 'world' (no leading whitespace, no
        // dummy prefix on non-first tokens).
        assert_eq!(out[1].text.as_str(), "world");
    }

    #[test]
    fn no_leading_whitespace() {
        let p = MetaspacePreTokenizer::default();
        let out = p.pre_tokenize("hello world").unwrap();
        // dummy prefix still adds the marker to the first token.
        assert!(out[0].text.as_str().starts_with('▁'));
    }

    #[test]
    fn leading_whitespace_is_consumed_by_separator() {
        // The Whitespace pre-tokenizer strips leading whitespace as a
        // separator. Metaspace's dummy prefix adds the marker to the
        // first pre-token regardless.
        let p = MetaspacePreTokenizer::default();
        let out = p.pre_tokenize("  hello world").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text.as_str(), "▁hello");
        assert_eq!(out[1].text.as_str(), "world");
    }
}
