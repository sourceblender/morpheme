//! Split numbers out of the text.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Isolates numeric chars: each digit on its own
/// (`individual_digits: true`) or runs of digits together.
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
/// use splinter::pre_tokenizers::Digits;
///
/// assert_eq!(split(&Digits::new(true), "abc123"), ["abc", "1", "2", "3"]);
/// assert_eq!(split(&Digits::new(false), "abc123"), ["abc", "123"]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Digits {
    /// Split every digit into its own piece.
    #[serde(default)]
    pub individual_digits: bool,
}

impl Digits {
    /// Build a `Digits` pre-tokenizer.
    pub fn new(individual_digits: bool) -> Self {
        Self { individual_digits }
    }
}

impl PreTokenizer for Digits {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let behavior = if self.individual_digits {
            SplitDelimiterBehavior::Isolated
        } else {
            SplitDelimiterBehavior::Contiguous
        };
        pretokenized.split(|_, s| s.split(char::is_numeric, behavior))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn numbers() {
        assert_eq!(
            splits(&Digits::new(false), "Hey 123 friend!"),
            vec![("Hey ", (0, 4)), ("123", (4, 7)), (" friend!", (7, 15))]
        );
    }

    #[test]
    fn individual_digits() {
        assert_eq!(
            splits(&Digits::new(true), "Hey 123 friend!"),
            vec![
                ("Hey ", (0, 4)),
                ("1", (4, 5)),
                ("2", (5, 6)),
                ("3", (6, 7)),
                (" friend!", (7, 15))
            ]
        );
    }
}
