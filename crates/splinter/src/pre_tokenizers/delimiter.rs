//! Split on a single delimiter char.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Splits on `delimiter`, removing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
