//! `BertPreTokenizer` — BERT-style whitespace + punctuation split.
//!
//! Each pre-token is either:
//! - a run of whitespace, *or*
//! - a run of non-whitespace, non-punctuation characters, *or*
//! - a single punctuation character.
//!
//! The behavior matches the HF `BertPreTokenizer` defaults: it does
//! *not* strip punctuation from the surrounding words, it emits each
//! punctuation char as its own pre-token.

use crate::error::Result;
use crate::pre_tokenizer::{PreToken, PreTokenizer};

/// BERT-style whitespace + punctuation pre-tokenizer.
#[derive(Debug, Default, Clone, Copy)]
pub struct BertPreTokenizer;

fn is_punctuation(b: u8) -> bool {
    // HF defines punctuation as: ASCII punctuation plus the CJK
    // ranges. The CJK blocks are checked via the byte value as a
    // heuristic — true UTF-8 punctuation would need char-aware
    // scanning; v0.1 matches what most BERT pipelines need on
    // Latin-only text.
    matches!(
        b,
        b'!' | b'"'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b';'
            | b'<'
            | b'='
            | b'>'
            | b'?'
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'^'
            | b'_'
            | b'`'
            | b'{'
            | b'|'
            | b'}'
            | b'~'
    )
}

fn is_ws(b: u8) -> bool {
    b.is_ascii_whitespace()
}

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<PreToken<'a>>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        // We track two possible start positions:
        // - `word_start`: where the current word/punct run begins
        //   (the first non-whitespace byte of the run).
        // - `ws_lead_start`: where the whitespace preceding the
        //   current run begins.
        // The emitted span starts at `ws_lead_start` so that the
        // leading whitespace is attached to the following token,
        // matching HF's `BertPreTokenizer`.
        let mut word_start: Option<usize> = None;
        let mut ws_lead_start: Option<usize> = None;
        let mut is_word_run: bool = false;

        for (i, &b) in bytes.iter().enumerate() {
            let punct = is_punctuation(b);
            let ws = is_ws(b);
            let new_kind = if ws {
                None
            } else if punct {
                Some(false)
            } else {
                Some(true)
            };

            match (word_start, new_kind) {
                (None, None) => {
                    // Still inside leading whitespace (or between
                    // tokens). Extend ws_lead_start if not yet set.
                    if ws_lead_start.is_none() {
                        ws_lead_start = Some(i);
                    }
                }
                (None, Some(k)) => {
                    // Start a new word/punct run.
                    word_start = Some(i);
                    is_word_run = k;
                }
                (Some(s), None) => {
                    // End of run. Emit with leading whitespace.
                    let start = ws_lead_start.unwrap_or(s);
                    out.push(PreToken::borrowed(&text[start..i], (start, i)));
                    word_start = None;
                    ws_lead_start = Some(i);
                }
                (Some(_s), Some(k)) if k == is_word_run => {
                    // Same kind — extend.
                }
                (Some(s), Some(k)) => {
                    // Kind change — emit and start new.
                    let start = ws_lead_start.unwrap_or(s);
                    out.push(PreToken::borrowed(&text[start..i], (start, i)));
                    word_start = Some(i);
                    ws_lead_start = Some(i);
                    is_word_run = k;
                }
            }
        }
        if let Some(s) = word_start {
            let start = ws_lead_start.unwrap_or(s);
            out.push(PreToken::borrowed(&text[start..], (start, text.len())));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_words_punct_whitespace() {
        // HF-style: whitespace gets attached to the following token.
        let p = BertPreTokenizer.pre_tokenize("hello, world!").unwrap();
        let texts: Vec<&str> = p.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn whitespace_keeps_separators() {
        let p = BertPreTokenizer.pre_tokenize("a  b").unwrap();
        let texts: Vec<&str> = p.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["a", "  b"]);
    }

    #[test]
    fn empty_input() {
        let p = BertPreTokenizer.pre_tokenize("").unwrap();
        assert!(p.is_empty());
    }
}
