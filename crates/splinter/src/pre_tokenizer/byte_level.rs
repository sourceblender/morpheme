//! `ByteLevel` — GPT-2-style byte-to-unicode pre-tokenizer.
//!
//! Each input byte is mapped through a deterministic byte→char table
//! (defined in the GPT-2 paper and reproduced in HF `tokenizers`):
//! printable ASCII bytes map to themselves, every other byte maps to
//! a unique Unicode char in the U+0100–U+017F range. After mapping,
//! the result is split on whitespace using the same boundaries as
//! `Whitespace`.
//!
//! The reverse map (char → byte) is also exposed for the
//! [`ByteLevelDecoder`](crate::decoder::ByteLevelDecoder).

use crate::error::Result;
use crate::pre_tokenizer::{PreToken, PreTokenizer};

/// Pre-computed byte → char table. See module docs.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in b'\xA1'..=b'\xAC' {
        bs.push(b as u32);
    }
    for b in b'\xAE'..=b'\xFF' {
        bs.push(b as u32);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (i, &c) in cs.iter().enumerate() {
        table[bs[i] as usize] = char::from_u32(c).unwrap();
    }
    table
}

/// Inverse of [`bytes_to_unicode`]. Returns `None` for chars that
/// don't appear in the mapping. Indexed by full Unicode codepoint
/// (so both ASCII-pass-through and mapped-range chars live in the
/// same table without collisions).
pub fn unicode_to_bytes() -> [Option<u8>; 512] {
    let fwd = bytes_to_unicode();
    let mut rev = [None; 512];
    for (b, &c) in fwd.iter().enumerate() {
        let cu = c as u32;
        rev[cu as usize] = Some(b as u8);
    }
    rev
}

/// Controls whether the GPT-2 mapped space (U+0120) is rewritten to an
/// ASCII space before splitting. GPT-2 default is `Add`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ByteLevelAddChar {
    /// Map U+0120 → ' ' before splitting. GPT-2 default.
    #[default]
    Add,
    /// Leave the mapped chars in place.
    NoAdd,
}

/// GPT-2-style byte-to-unicode pre-tokenizer.
#[derive(Debug, Clone, Copy)]
pub struct ByteLevel {
    /// Prepend a leading space to the input before mapping.
    pub add_prefix_space: bool,
    /// Rewrite the mapped space (U+0120) back to an ASCII space.
    pub add_char: ByteLevelAddChar,
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            add_char: ByteLevelAddChar::Add,
        }
    }
}

impl ByteLevel {
    /// Construct a `ByteLevel` with explicit `add_prefix_space` and
    /// `add_char` settings.
    pub fn new(add_prefix_space: bool, add_char: ByteLevelAddChar) -> Self {
        Self {
            add_prefix_space,
            add_char,
        }
    }

    /// Map a single byte through the byte-to-unicode table.
    pub fn map_byte(b: u8) -> char {
        bytes_to_unicode()[b as usize]
    }

    /// Inverse: turn a mapped char back into its byte, if any.
    pub fn unmap_char(c: char) -> Option<u8> {
        let u = c as u32;
        if (u as usize) < 512 {
            unicode_to_bytes()[u as usize]
        } else {
            None
        }
    }
}

impl PreTokenizer for ByteLevel {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<PreToken<'a>>> {
        // Prepend a leading space if requested.
        let mut mapped = String::with_capacity(text.len() * 2);
        if self.add_prefix_space && !text.starts_with(' ') {
            mapped.push(' ');
        }
        let table = bytes_to_unicode();
        for &b in text.as_bytes() {
            mapped.push(table[b as usize]);
        }

        // Optionally collapse mapped space back to ASCII space.
        let mapped: String = if matches!(self.add_char, ByteLevelAddChar::Add) {
            mapped.replace('\u{0120}', " ")
        } else {
            mapped
        };

        // Walk through the mapped string, splitting on whitespace and
        // producing owned pre-tokens. The span we record is in the
        // *original* input's byte coordinates.
        let bytes = mapped.as_bytes();
        let mut out = Vec::new();
        let mut start: Option<usize> = None;

        // The leading-space prefix in `mapped` is an extra character
        // before position 0 of the input. Track an offset.
        let prefix_len = if self.add_prefix_space && !text.starts_with(' ') {
            1
        } else {
            0
        };

        for (i, &b) in bytes.iter().enumerate() {
            if b.is_ascii_whitespace() {
                if let Some(s) = start.take() {
                    let mapped_text: String = mapped[s..i].to_owned();
                    let real_start = s.saturating_sub(prefix_len);
                    let real_end = i.saturating_sub(prefix_len);
                    out.push(PreToken::owned(mapped_text, (real_start, real_end)));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            let mapped_text: String = mapped[s..].to_owned();
            let real_start = s.saturating_sub(prefix_len);
            let real_end = mapped.len().saturating_sub(prefix_len);
            out.push(PreToken::owned(mapped_text, (real_start, real_end)));
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_ascii_maps_to_self() {
        assert_eq!(ByteLevel::map_byte(b'a'), 'a');
        assert_eq!(ByteLevel::map_byte(b'Z'), 'Z');
    }

    #[test]
    fn space_maps_to_gpt2_char() {
        assert_eq!(ByteLevel::map_byte(b' '), '\u{0120}');
    }

    #[test]
    fn unmaps_round_trip() {
        for b in 0u8..=255 {
            let c = ByteLevel::map_byte(b);
            assert_eq!(ByteLevel::unmap_char(c), Some(b), "byte {b} round trip");
        }
    }

    #[test]
    fn pre_tokenize_simple() {
        let p = ByteLevel::default();
        let out = p.pre_tokenize("hello world").unwrap();
        let texts: Vec<&str> = out.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "world"]);
    }

    #[test]
    fn pre_tokenize_no_prefix_space() {
        let p = ByteLevel::new(false, ByteLevelAddChar::Add);
        let out = p.pre_tokenize("hello world").unwrap();
        let texts: Vec<&str> = out.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "world"]);
    }
}
