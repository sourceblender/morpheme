//! Split on a single delimiter char.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Splits on `delimiter`, removing it.
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
/// use splinter::pre_tokenizers::CharDelimiterSplit;
///
/// assert_eq!(split(&CharDelimiterSplit::new('-'), "state-of-the-art"), ["state", "of", "the", "art"]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CharDelimiterSplit {
    /// The delimiter char.
    pub delimiter: char,
}

impl CharDelimiterSplit {
    /// Split on `delimiter`.
    pub fn new(delimiter: char) -> Self {
        Self { delimiter }
    }
}

impl PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(self.delimiter, SplitDelimiterBehavior::Removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn splits_on_delimiter() {
        assert_eq!(
            splits(&CharDelimiterSplit::new('-'), "a-b--c"),
            vec![("a", (0, 1)), ("b", (2, 3)), ("c", (5, 6))]
        );
    }
}
