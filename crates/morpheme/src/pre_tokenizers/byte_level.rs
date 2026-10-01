//! GPT-2 byte-level pre-tokenizer, decoder and post-processor.
//!
//! Every byte of the input is mapped to a printable Unicode char (so a
//! space becomes `Ġ`, a newline `Ċ`), which lets a BPE vocabulary of 256
//! base symbols represent any text without unknown tokens.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::encoding::Encoding;
use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::normalizers::byte_level::bytes_to_chars;
use crate::pattern::SysRegex;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::{Decoder, PostProcessor, PreTokenizer};

/// The GPT-2 pre-tokenization pattern.
pub const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn gpt2_regex() -> &'static SysRegex {
    static RE: OnceLock<SysRegex> = OnceLock::new();
    RE.get_or_init(|| SysRegex::new(GPT2_PATTERN).expect("GPT-2 pattern is valid"))
}

/// The GPT-2 byte → char table, indexed by byte: printable bytes map to
/// themselves, the rest are shifted past 255 so that every byte gets a
/// visible, unambiguous char.
pub fn bytes_char() -> &'static [char; 256] {
    bytes_to_chars()
}

/// The GPT-2 char → byte table (inverse of [`bytes_char`]).
pub fn char_bytes() -> &'static HashMap<char, u8> {
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| {
        bytes_char()
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect()
    })
}

/// Highest code point in the byte-level alphabet (`U+0143`).
const MAX_BYTE_LEVEL_CHAR: usize = 256 + 67;

/// Inverse of [`bytes_char`] as a dense array indexed by code point, so
/// decoding does not hash.
fn char_to_byte(c: char) -> Option<u8> {
    static TABLE: OnceLock<[Option<u8>; MAX_BYTE_LEVEL_CHAR + 1]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [None; MAX_BYTE_LEVEL_CHAR + 1];
        for (b, c) in bytes_char().iter().enumerate() {
            table[*c as usize] = Some(b as u8);
        }
        table
    });
    table.get(c as usize).copied().flatten()
}

fn default_true() -> bool {
    true
}

/// Byte-level pre-tokenizer (also usable as decoder and post-processor).
///
/// # Example
///
/// ```
/// use morpheme::{OffsetType, PreTokenizedString, PreTokenizer};
///
/// fn split(pt: &impl PreTokenizer, text: &str) -> Vec<String> {
///     let mut s = PreTokenizedString::from(text);
///     pt.pre_tokenize(&mut s).unwrap();
///     s.get_splits(OffsetType::Byte).into_iter().map(|(w, _, _)| w.to_owned()).collect()
/// }
///
/// use morpheme::pre_tokenizers::ByteLevel;
///
/// // GPT-2 style: regex split, then bytes mapped to printable chars; a
/// // leading space becomes part of the next word as `Ġ`.
/// let gpt2 = ByteLevel::new(false, true, true);
/// assert_eq!(split(&gpt2, "Hello world"), ["Hello", "Ġworld"]);
///
/// // RoBERTa style adds a space in front of the input.
/// assert_eq!(split(&ByteLevel::default(), "Hello world"), ["ĠHello", "Ġworld"]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ByteLevel {
    /// Prepend a space to the input if it doesn't start with one, so the
    /// first word is treated like any other.
    #[serde(default = "default_true")]
    pub add_prefix_space: bool,
    /// As a post-processor: shrink offsets so they exclude the leading /
    /// trailing whitespace carried by tokens like `Ġhello`.
    #[serde(default = "default_true")]
    pub trim_offsets: bool,
    /// Split with the GPT-2 regex before mapping bytes.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl ByteLevel {
    /// Build a `ByteLevel`.
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    /// The 256 chars of the byte-level alphabet.
    pub fn alphabet() -> HashSet<char> {
        bytes_char().iter().copied().collect()
    }

    /// Set `add_prefix_space`.
    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    /// Set `trim_offsets`.
    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    /// Set `use_regex`.
    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                normalized.split(gpt2_regex(), SplitDelimiterBehavior::Isolated)
            } else {
                Ok(vec![normalized])
            }
        })?;
        let table = bytes_char();
        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            let mut dest: Vec<(char, isize)> = Vec::with_capacity(s.len());
            let mut buf = [0u8; 4];
            for c in s.chars() {
                for (i, b) in c.encode_utf8(&mut buf).bytes().enumerate() {
                    dest.push((table[b as usize], isize::from(i > 0)));
                }
            }
            normalized.transform(dest, 0);
            Ok(())
        })
    }
}

impl Decoder for ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let mut bytes: Vec<u8> = Vec::new();
        for token in &tokens {
            let mapped: Option<Vec<u8>> = token.chars().map(char_to_byte).collect();
            match mapped {
                Some(b) => bytes.extend(b),
                // Not a byte-level token (e.g. an added token): keep as is.
                None => bytes.extend_from_slice(token.as_bytes()),
            }
        }
        Ok(vec![String::from_utf8_lossy(&bytes).into_owned()])
    }
}

impl PostProcessor for ByteLevel {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if self.trim_offsets {
            for encoding in encodings.iter_mut() {
                process_offsets(encoding, self.add_prefix_space);
                for o in encoding.overflowing_mut() {
                    process_offsets(o, self.add_prefix_space);
                }
            }
        }
        for (i, encoding) in encodings.iter_mut().enumerate() {
            encoding.set_sequence_id(i);
        }
        Ok(encodings)
    }
}

/// Trim byte-level whitespace (`Ġ` and real whitespace) from the
/// offsets of every token. A single leading `Ġ` on the first token is
/// kept when `add_prefix_space` is on, since it was added by the
/// pre-tokenizer and maps to no original char.
pub fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    /// `bytes_char()[b' ']`: the byte-level char for a space.
    const SPACE: char = 'Ġ';
    let is_space = |c: &char| *c == SPACE || c.is_whitespace();
    let tokens: Vec<String> = encoding.tokens().to_vec();
    for (i, (token, offsets)) in tokens
        .iter()
        .zip(encoding.offsets_mut().iter_mut())
        .enumerate()
    {
        let mut leading = token.chars().take_while(is_space).count();
        let trailing = token.chars().rev().take_while(is_space).count();
        if leading == 0 && trailing == 0 {
            continue;
        }
        if leading > 0 {
            let is_first = i == 0 || offsets.0 == 0;
            if is_first && add_prefix_space && leading == 1 {
                leading = 0;
            }
            offsets.0 = (offsets.0 + leading).min(offsets.1);
        }
        if trailing > 0 && offsets.1 >= trailing {
            offsets.1 = (offsets.1 - trailing).max(offsets.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Token;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn table_is_a_bijection() {
        assert_eq!(bytes_char().len(), 256);
        assert_eq!(char_bytes().len(), 256);
        assert_eq!(bytes_char()[b' ' as usize], 'Ġ');
        assert_eq!(bytes_char()[b'\n' as usize], 'Ċ');
        assert_eq!(bytes_char()[b'a' as usize], 'a');
        assert_eq!(bytes_char().iter().copied().max(), Some('Ń'));
        assert_eq!('Ń' as usize, MAX_BYTE_LEVEL_CHAR);
        for (b, c) in bytes_char().iter().enumerate() {
            assert_eq!(char_to_byte(*c), Some(b as u8));
            assert_eq!(char_bytes()[c], b as u8);
        }
        assert_eq!(char_to_byte('€'), None);
        assert_eq!(char_to_byte('\u{144}'), None);
    }

    #[test]
    fn pre_tokenization() {
        let bl = ByteLevel::default().add_prefix_space(false);
        assert_eq!(
            splits(&bl, "Hello my friend, how is your day going?"),
            vec![
                ("Hello", (0, 5)),
                ("Ġmy", (5, 8)),
                ("Ġfriend", (8, 15)),
                (",", (15, 16)),
                ("Ġhow", (16, 20)),
                ("Ġis", (20, 23)),
                ("Ġyour", (23, 28)),
                ("Ġday", (28, 32)),
                ("Ġgoing", (32, 38)),
                ("?", (38, 39)),
            ]
        );
    }

    #[test]
    fn no_regex_maps_whole_string() {
        let bl = ByteLevel::default().use_regex(false);
        assert_eq!(
            splits(&bl, "Hello my friend"),
            vec![("ĠHelloĠmyĠfriend", (0, 15))]
        );
    }

    #[test]
    fn multibyte_offsets_point_into_original() {
        let bl = ByteLevel::default().add_prefix_space(false);
        // "é" is two bytes → two byte-level chars, both mapping to 0..2.
        assert_eq!(
            splits(&bl, "héllo wörld"),
            vec![("hÃ©llo", (0, 6)), ("ĠwÃ¶rld", (6, 13))]
        );
    }

    #[test]
    fn decode_roundtrips_non_ascii() {
        let bl = ByteLevel::default().add_prefix_space(false);
        let input = "héllo 😀 wörld\n";
        let mut pts = PreTokenizedString::from(input);
        bl.pre_tokenize(&mut pts).unwrap();
        let tokens: Vec<String> = pts
            .get_splits(crate::OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_owned())
            .collect();
        assert_eq!(bl.decode(tokens).unwrap(), input);
    }

    #[test]
    fn decode_keeps_non_byte_level_tokens() {
        let bl = ByteLevel::default();
        let out = bl
            .decode(vec![
                "Hello".into(),
                "<|endoftext|>".into(),
                "Ġthere".into(),
            ])
            .unwrap();
        assert_eq!(out, "Hello<|endoftext|> there");
    }

    #[test]
    fn trims_offsets() {
        let mut enc = Encoding::from_tokens(
            vec![
                Token::new(0, "Ġ".into(), (0, 1)),
                Token::new(1, "Ġhello".into(), (0, 6)),
                Token::new(2, "ĠĠthere".into(), (6, 13)),
                Token::new(3, "Ġ".into(), (13, 14)),
            ],
            0,
        );
        process_offsets(&mut enc, true);
        assert_eq!(enc.offsets(), &[(0, 0), (0, 6), (8, 13), (14, 14)]);
    }
}
