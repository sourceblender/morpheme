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

fn build_bytes_char() -> HashMap<u8, char> {
    // Printable bytes map to themselves; the rest are shifted past 255 so
    // that every byte gets a visible, unambiguous char.
    let mut printable: Vec<u8> = Vec::with_capacity(256);
    printable.extend(b'!'..=b'~');
    printable.extend(0xA1u8..=0xAC);
    printable.extend(0xAEu8..=0xFF);
    let mut map: HashMap<u8, char> = printable.iter().map(|&b| (b, char::from(b))).collect();
    let mut next = 256u32;
    for b in 0..=255u8 {
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(b) {
            e.insert(char::from_u32(next).expect("256..=323 are valid chars"));
            next += 1;
        }
    }
    map
}

/// The GPT-2 byte → char table.
pub fn bytes_char() -> &'static HashMap<u8, char> {
    static MAP: OnceLock<HashMap<u8, char>> = OnceLock::new();
    MAP.get_or_init(build_bytes_char)
}

/// The GPT-2 char → byte table (inverse of [`bytes_char`]).
pub fn char_bytes() -> &'static HashMap<char, u8> {
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| bytes_char().iter().map(|(b, c)| (*c, *b)).collect())
}

fn default_true() -> bool {
    true
}

/// Byte-level pre-tokenizer (also usable as decoder and post-processor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
        bytes_char().values().copied().collect()
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
            for c in s.chars() {
                let mut buf = [0u8; 4];
                for (i, b) in c.encode_utf8(&mut buf).bytes().enumerate() {
                    dest.push((table[&b], isize::from(i > 0)));
                }
            }
            normalized.transform(dest, 0);
            Ok(())
        })
    }
}

impl Decoder for ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let table = char_bytes();
        let mut bytes: Vec<u8> = Vec::new();
        for token in &tokens {
            let mapped: Option<Vec<u8>> = token.chars().map(|c| table.get(&c).copied()).collect();
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
    let space = bytes_char()[&b' '];
    let is_space = |c: &char| *c == space || c.is_whitespace();
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
        assert_eq!(bytes_char()[&b' '], 'Ġ');
        assert_eq!(bytes_char()[&b'\n'], 'Ċ');
        assert_eq!(bytes_char()[&b'a'], 'a');
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
