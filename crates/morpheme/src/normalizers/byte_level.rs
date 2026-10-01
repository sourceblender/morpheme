//! Byte-level normalizer: map every UTF-8 byte to its GPT-2
//! "byte-level" printable char.

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::pre_tokenizers::impl_unit_serde;
use crate::traits::Normalizer;

/// The GPT-2 byte → char table: printable Latin-1 bytes map to
/// themselves, every other byte to `U+0100 + n`.
pub(crate) fn bytes_to_chars() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = ['\0'; 256];
        let mut n = 0u32;
        for b in 0..=255u8 {
            let printable = matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
            table[b as usize] = if printable {
                char::from(b)
            } else {
                let c = char::from_u32(256 + n).expect("valid codepoint");
                n += 1;
                c
            };
        }
        table
    })
}

/// Map every byte of the text to its GPT-2 byte-level char (each input
/// char becomes one char per UTF-8 byte).
///
/// # Example
///
/// ```
/// use morpheme::{NormalizedString, Normalizer};
///
/// fn normalize(n: &impl Normalizer, text: &str) -> String {
///     let mut s = NormalizedString::from(text);
///     n.normalize(&mut s).unwrap();
///     s.get().to_owned()
/// }
///
/// use morpheme::normalizers::ByteLevel;
///
/// // Every byte maps to a printable char (GPT-2's byte-to-unicode table).
/// assert_eq!(normalize(&ByteLevel::new(), "hé!"), "hÃ©!");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteLevel;

impl_unit_serde!(ByteLevel);

impl ByteLevel {
    /// Build a byte-level normalizer.
    pub fn new() -> Self {
        Self
    }

    /// The 256 chars of the byte-level alphabet.
    pub fn alphabet() -> HashSet<char> {
        bytes_to_chars().iter().copied().collect()
    }
}

impl Normalizer for ByteLevel {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if normalized.is_empty() {
            return Ok(());
        }
        let table = bytes_to_chars();
        let mut dest = Vec::with_capacity(normalized.len());
        let mut buf = [0u8; 4];
        for c in normalized.get().chars() {
            for (i, b) in c.encode_utf8(&mut buf).bytes().enumerate() {
                dest.push((table[b as usize], isize::from(i > 0)));
            }
        }
        normalized.transform(dest, 0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;

    // Expected outputs from Python `tokenizers` 0.23.2.
    #[test]
    fn matches_hf() {
        let mut ns = NormalizedString::from("Hello world");
        ByteLevel.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "HelloĠworld");

        let mut ns = NormalizedString::from("é😀");
        ByteLevel.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "Ã©ðŁĺĢ");
        // "Ã©" (normalized 0..4) is "é" (original 0..2).
        assert_eq!(
            ns.convert_offsets(OffsetRange::Normalized(0..4)),
            Some(0..2)
        );
    }

    #[test]
    fn alphabet_has_256_distinct_chars() {
        assert_eq!(ByteLevel::alphabet().len(), 256);
        assert_eq!(bytes_to_chars()[b' ' as usize], 'Ġ');
        assert_eq!(bytes_to_chars()[b'\n' as usize], 'Ċ');
    }
}
