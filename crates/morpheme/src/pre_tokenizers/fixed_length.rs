//! Split into fixed-size chunks of chars.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::normalized_string::OffsetRange;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

fn default_length() -> usize {
    5
}

/// Splits every piece into chunks of `length` chars (the last chunk may
/// be shorter).
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
/// use morpheme::pre_tokenizers::FixedLength;
///
/// assert_eq!(split(&FixedLength::new(3), "abcdefgh"), ["abc", "def", "gh"]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FixedLength {
    /// Chunk size in chars.
    #[serde(default = "default_length")]
    pub length: usize,
}

impl Default for FixedLength {
    fn default() -> Self {
        Self::new(default_length())
    }
}

impl FixedLength {
    /// Chunks of `length` chars.
    pub fn new(length: usize) -> Self {
        Self { length }
    }
}

impl PreTokenizer for FixedLength {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        if self.length == 0 {
            return Err(Error::Config("FixedLength length must be > 0".into()));
        }
        pretokenized.split(|_, normalized| {
            let positions: Vec<(usize, char)> = normalized.get().char_indices().collect();
            positions
                .chunks(self.length)
                .map(|chunk| {
                    let start = chunk[0].0;
                    let (last, c) = chunk[chunk.len() - 1];
                    normalized
                        .slice(OffsetRange::Normalized(start..last + c.len_utf8()))
                        .ok_or_else(|| Error::PreTokenizer("FixedLength: bad slice".into()))
                })
                .collect::<Result<Vec<_>>>()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn chunks() {
        assert_eq!(
            splits(&FixedLength::new(2), "abcde"),
            vec![("ab", (0, 2)), ("cd", (2, 4)), ("e", (4, 5))]
        );
        assert_eq!(
            splits(&FixedLength::new(2), "héllo"),
            vec![("hé", (0, 3)), ("ll", (3, 5)), ("o", (5, 6))]
        );
    }
}
