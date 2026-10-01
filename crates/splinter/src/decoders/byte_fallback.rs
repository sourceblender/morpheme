//! Decode `<0xNN>` byte tokens (SentencePiece / Llama byte fallback).

use crate::error::Result;
use crate::traits::Decoder;

/// Turns runs of `<0xNN>` tokens back into the UTF-8 text they encode.
/// Invalid byte sequences become one `�` per byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteFallback;

crate::pre_tokenizers::impl_unit_serde!(ByteFallback);

impl ByteFallback {
    /// Build a `ByteFallback` decoder.
    pub fn new() -> Self {
        Self
    }
}

fn parse_byte(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

fn flush(bytes: &mut Vec<u8>, out: &mut Vec<String>) {
    if bytes.is_empty() {
        return;
    }
    match String::from_utf8(std::mem::take(bytes)) {
        Ok(s) => out.push(s),
        Err(e) => out.extend(std::iter::repeat_n("�".to_string(), e.as_bytes().len())),
    }
}

impl Decoder for ByteFallback {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(tokens.len());
        let mut bytes = Vec::new();
        for token in tokens {
            match parse_byte(&token) {
                Some(b) => bytes.push(b),
                None => {
                    flush(&mut bytes, &mut out);
                    out.push(token);
                }
            }
        }
        flush(&mut bytes, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tokens: &[&str]) -> Vec<String> {
        ByteFallback
            .decode_chain(tokens.iter().map(|s| s.to_string()).collect())
            .unwrap()
    }

    #[test]
    fn decodes_bytes() {
        assert_eq!(run(&["Hey", "friend!"]), vec!["Hey", "friend!"]);
        assert_eq!(run(&["<0x61>"]), vec!["a"]);
        assert_eq!(run(&["<0xE5>"]), vec!["�"]);
        assert_eq!(run(&["<0xE5>", "<0x8f>"]), vec!["�", "�"]);
        assert_eq!(run(&["<0xE5>", "<0x8f>", "<0xab>"]), vec!["叫"]);
        assert_eq!(run(&["<0xE5>", "<0x8f>", "<0xab>", "a"]), vec!["叫", "a"]);
        assert_eq!(run(&["<0xE5>", "<0x8f>", "a"]), vec!["�", "�", "a"]);
    }
}
