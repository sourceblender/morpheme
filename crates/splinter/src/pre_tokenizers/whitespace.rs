//! Whitespace pre-tokenizers.

use std::sync::OnceLock;

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pattern::{Invert, SysRegex};
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Splits into runs of word chars and runs of other non-space chars:
/// the regex `\w+|[^\w\s]+`. `"Hey man!"` → `["Hey", "man", "!"]`.
///
/// # Example
///
/// ```
/// use splinter::{OffsetType, PreTokenizedString, PreTokenizer};
///
/// fn split(pt: &impl PreTokenizer, text: &str) -> Vec<String> {
///     let mut s = PreTokenizedString::from(text);
///     pt.pre_tokenize(&mut s).unwrap();
///     s.get_splits(OffsetType::Byte).into_iter().map(|(w, _, _)| w.to_owned()).collect()
/// }
///
/// use splinter::pre_tokenizers::Whitespace;
///
/// // Words and runs of punctuation (`\w+|[^\w\s]+`).
/// assert_eq!(split(&Whitespace, "Hello, world!!"), ["Hello", ",", "world", "!!"]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Whitespace;

super::impl_unit_serde!(Whitespace);

fn word_regex() -> &'static SysRegex {
    static RE: OnceLock<SysRegex> = OnceLock::new();
    RE.get_or_init(|| SysRegex::new(r"\w+|[^\w\s]+").expect("valid regex"))
}

impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(Invert(word_regex()), SplitDelimiterBehavior::Removed))
    }
}

/// Splits on whitespace only. `"Hey man!"` → `["Hey", "man!"]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhitespaceSplit;

super::impl_unit_serde!(WhitespaceSplit);

impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn whitespace() {
        assert_eq!(
            splits(&Whitespace, "Hey man!"),
            vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))]
        );
        assert_eq!(
            splits(&Whitespace, "How are you doing?"),
            vec![
                ("How", (0, 3)),
                ("are", (4, 7)),
                ("you", (8, 11)),
                ("doing", (12, 17)),
                ("?", (17, 18)),
            ]
        );
        assert!(splits(&Whitespace, "\n").is_empty());
    }

    #[test]
    fn whitespace_split() {
        assert_eq!(
            splits(&WhitespaceSplit, "Hey, man, Good?"),
            vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))]
        );
    }
}
