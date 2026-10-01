//! SentencePiece's precompiled character map normalizer (T5, ALBERT,
//! XLM-RoBERTa, …).
//!
//! The `precompiled_charsmap` blob is the normalization rule set that
//! SentencePiece compiles into its model files:
//!
//! ```text
//! u32 LE  trie_size (bytes)
//! [u32 LE; trie_size / 4]   darts-clone double-array trie units
//! [u8]                       pool of NUL-terminated replacement strings
//! ```
//!
//! The trie maps UTF-8 input sequences to offsets in the pool.

use std::cmp::Ordering;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use unicode_segmentation::UnicodeSegmentation;

use crate::error::{Error, Result};
use crate::normalized_string::NormalizedString;
use crate::traits::Normalizer;

/// A darts-clone double-array trie.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct DoubleArray {
    units: Vec<u32>,
}

impl DoubleArray {
    fn has_leaf(unit: u32) -> bool {
        (unit >> 8) & 1 == 1
    }

    fn value(unit: u32) -> u32 {
        unit & ((1 << 31) - 1)
    }

    fn label(unit: u32) -> u32 {
        unit & ((1 << 31) | 0xFF)
    }

    fn offset(unit: u32) -> usize {
        ((unit >> 10) << ((unit & (1 << 9)) >> 6)) as usize
    }

    /// Values of every key that is a prefix of `key`, shortest first.
    /// Malformed tries end the search instead of panicking.
    fn common_prefix_search(&self, key: &[u8]) -> Vec<u32> {
        let mut results = Vec::new();
        let Some(&root) = self.units.first() else {
            return results;
        };
        let mut pos = Self::offset(root);
        for &c in key {
            if c == 0 {
                break;
            }
            pos ^= c as usize;
            let Some(&unit) = self.units.get(pos) else {
                return results;
            };
            if Self::label(unit) != c as u32 {
                return results;
            }
            pos ^= Self::offset(unit);
            if Self::has_leaf(unit) {
                match self.units.get(pos) {
                    Some(&leaf) => results.push(Self::value(leaf)),
                    None => return results,
                }
            }
        }
        results
    }
}

/// SentencePiece precompiled normalization rules.
///
/// # Example
///
/// ```
/// use splinter::normalizers::Precompiled;
///
/// // Precompiled charsmaps come from SentencePiece models (T5, ALBERT,
/// // XLM-R, …) and are normally loaded from a `tokenizer.json`. Malformed
/// // data is an error, never a panic.
/// assert!(Precompiled::from_bytes(&[1, 2, 3]).is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Precompiled {
    precompiled_charsmap: Vec<u8>,
    normalized: Vec<u8>,
    trie: DoubleArray,
}

impl Precompiled {
    /// Parse a `precompiled_charsmap` blob.
    pub fn from_bytes(precompiled_charsmap: &[u8]) -> Result<Self> {
        let bad = |m: &str| Error::Config(format!("invalid precompiled_charsmap: {m}"));
        let header: [u8; 4] = precompiled_charsmap
            .get(..4)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| bad("too short"))?;
        let trie_size = u32::from_le_bytes(header) as usize;
        let trie_end = 4usize
            .checked_add(trie_size)
            .filter(|&e| e <= precompiled_charsmap.len())
            .ok_or_else(|| bad("trie size exceeds data"))?;
        let units = precompiled_charsmap[4..trie_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let normalized = precompiled_charsmap[trie_end..].to_vec();
        std::str::from_utf8(&normalized).map_err(|_| bad("replacement pool is not UTF-8"))?;
        Ok(Self {
            precompiled_charsmap: precompiled_charsmap.to_vec(),
            normalized,
            trie: DoubleArray { units },
        })
    }

    /// The raw charsmap blob.
    pub fn precompiled_charsmap(&self) -> &[u8] {
        &self.precompiled_charsmap
    }

    /// Replacement for `chunk`, if any rule matches a prefix of it
    /// (the shortest matching rule wins, as in Hugging Face's
    /// implementation).
    pub fn transform(&self, chunk: &str) -> Option<&str> {
        let start = *self.trie.common_prefix_search(chunk.as_bytes()).first()? as usize;
        let rest = self.normalized.get(start..)?;
        let len = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        std::str::from_utf8(&rest[..len]).ok()
    }

    /// Normalize a plain string (no alignment tracking).
    pub fn normalize_str(&self, s: &str) -> String {
        let mut ns = NormalizedString::from(s);
        // Infallible: `normalize` never returns an error.
        let _ = self.normalize(&mut ns);
        ns.get().to_owned()
    }
}

/// Append the transformations replacing `old` (several chars) by `new`.
fn push_replacement(dest: &mut Vec<(char, isize)>, old: &str, new: &str) {
    let diff = new.chars().count() as isize - old.chars().count() as isize;
    dest.extend(new.chars().map(|c| (c, 0)));
    match diff.cmp(&0) {
        Ordering::Greater => {
            for (_, change) in dest.iter_mut().rev().take(diff as usize) {
                *change = 1;
            }
        }
        Ordering::Less => {
            if let Some((_, change)) = dest.last_mut() {
                *change += diff;
            }
        }
        Ordering::Equal => {}
    }
}

impl Normalizer for Precompiled {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut dest = Vec::with_capacity(normalized.len());
        let mut modified = false;
        for grapheme in normalized.get().graphemes(true) {
            if grapheme.len() < 6 {
                if let Some(norm) = self.transform(grapheme) {
                    modified = true;
                    push_replacement(&mut dest, grapheme, norm);
                    continue;
                }
            }
            for (i, c) in grapheme.char_indices() {
                let part = &grapheme[i..i + c.len_utf8()];
                match self.transform(part) {
                    Some(norm) => {
                        modified = true;
                        push_replacement(&mut dest, part, norm);
                    }
                    None => dest.push((c, 0)),
                }
            }
        }
        if modified {
            normalized.transform(dest, 0);
        }
        Ok(())
    }
}

// ---- serde: {"precompiled_charsmap": "<base64>"} ----

impl Serialize for Precompiled {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Precompiled", 1)?;
        s.serialize_field(
            "precompiled_charsmap",
            &base64_encode(&self.precompiled_charsmap),
        )?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Precompiled {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        struct Config {
            precompiled_charsmap: String,
        }
        let c = Config::deserialize(deserializer)?;
        let bytes = base64_decode(&c.precompiled_charsmap).map_err(D::Error::custom)?;
        Precompiled::from_bytes(&bytes).map_err(D::Error::custom)
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Decode standard base64 (padding optional, whitespace not allowed).
fn base64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid base64 character {:?}", c as char)),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if bits >= 6 {
        return Err("truncated base64 input".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;

    #[test]
    fn base64_roundtrip() {
        for data in [
            &b""[..],
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
            &[0, 255, 128],
        ] {
            let enc = base64_encode(data);
            assert_eq!(base64_decode(&enc).unwrap(), data);
        }
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert!(base64_decode("Zm9v!").is_err());
        assert!(base64_decode("Z").is_err());
    }

    #[test]
    fn expansion_followed_by_removal() {
        let mut dest = vec![];
        let mut n = NormalizedString::from("™\x1eg");
        push_replacement(&mut dest, "™", "TM");
        push_replacement(&mut dest, "\x1e", "");
        dest.push(('g', 0));
        n.transform(dest, 0);
        assert_eq!(n.get(), "TMg");
    }

    #[test]
    fn malformed_blobs_error_or_noop_instead_of_panicking() {
        assert!(Precompiled::from_bytes(&[]).is_err());
        assert!(Precompiled::from_bytes(&[255, 255, 255, 255]).is_err());
        // A trie whose units point out of bounds must not panic.
        let mut blob = 8u32.to_le_bytes().to_vec();
        blob.extend(0xFFFF_FFFFu32.to_le_bytes());
        blob.extend(0xFFFF_FFFFu32.to_le_bytes());
        let p = Precompiled::from_bytes(&blob).unwrap();
        assert_eq!(p.normalize_str("abc"), "abc");
    }

    fn fixture_charsmap(name: &str) -> Option<Precompiled> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/hf")
            .join(format!("{name}.json"));
        let raw = std::fs::read_to_string(&path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let n = &v["normalizer"];
        let n = if n["type"] == "Sequence" {
            n["normalizers"]
                .as_array()?
                .iter()
                .find(|x| x["type"] == "Precompiled")?
                .clone()
        } else {
            n.clone()
        };
        Some(serde_json::from_value(n).expect("valid Precompiled config"))
    }

    // Expected outputs from Python `tokenizers` 0.23.2 using the
    // google-t5/t5-small normalizer.
    #[test]
    fn t5_charsmap_matches_hf() {
        let Some(p) = fixture_charsmap("t5-small") else {
            eprintln!("skipping: fixtures missing (scripts/fetch-hf-fixtures.sh)");
            return;
        };
        for (input, expected) in T5_CASES {
            assert_eq!(p.normalize_str(input), *expected, "input {input:?}");
        }
        // Re-serialization is byte-identical.
        let raw = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hf/t5-small.json"),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            serde_json::to_value(&p).unwrap()["precompiled_charsmap"],
            v["normalizer"]["precompiled_charsmap"]
        );
    }

    #[test]
    fn t5_charsmap_offsets() {
        let Some(p) = fixture_charsmap("t5-small") else {
            return;
        };
        let mut ns = NormalizedString::from("ﬁx");
        p.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "fix");
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(0..2)),
            Some("ﬁ")
        );
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(2..3)),
            Some("x")
        );
    }

    const T5_CASES: &[(&str, &str)] = &[
        ("Hello, World!", "Hello, World!"),
        ("  leading and trailing  ", "  leading and trailing  "),
        ("Café naïve ﬁ ½ ①", "Café naïve fi 1⁄2 1"),
        ("你好，世界！", "你好,世界!"),
        ("日本語のテキストとカタカナ", "日本語のテキストとカタカナ"),
        ("한국어", "한국어"),
        ("Россия Р", "Россия Р"),
        ("مرحبا", "مرحبا"),
        ("Emoji: 😀👍🏽🇺🇸 ❤️", "Emoji: 😀👍🏽🇺🇸 ❤️"),
        ("zero\u{200b}width and\u{a0}nbsp", "zero width and nbsp"),
        ("control\u{0}char\u{7}bell", "control\u{0}charbell"),
        (
            "tabs\u{9}and\u{a}newlines\u{d}\u{a}mixed",
            "tabs and newlines mixed",
        ),
        ("e\u{301} combining", "é combining"),
        ("™\u{1e}g", "TMg"),
        ("ＡＢＣ１２３", "ABC123"),
        ("ﬀ ﬃ", "ff ffi"),
        ("…", "..."),
        ("‘quotes’ “double”", "‘quotes’ “double”"),
        ("multiple   spaces", "multiple   spaces"),
    ];
}
