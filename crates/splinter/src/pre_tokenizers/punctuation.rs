//! Split on punctuation.

use serde::{Deserialize, Serialize};

use super::bert::is_bert_punc;
use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

fn default_behavior() -> SplitDelimiterBehavior {
    SplitDelimiterBehavior::Isolated
}

/// Splits on punctuation chars (ASCII punctuation plus Unicode `P*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Punctuation {
    /// What to do with the punctuation (default `Isolated`).
    #[serde(default = "default_behavior")]
    pub behavior: SplitDelimiterBehavior,
}

impl Default for Punctuation {
    fn default() -> Self {
        Self::new(default_behavior())
    }
}

impl Punctuation {
    /// Build a `Punctuation` pre-tokenizer.
    pub fn new(behavior: SplitDelimiterBehavior) -> Self {
        Self { behavior }
    }
}

impl PreTokenizer for Punctuation {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(is_bert_punc, self.behavior))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn basic() {
        assert_eq!(
            splits(&Punctuation::default(), "Hey friend!     How are you?!?"),
            vec![
                ("Hey friend", (0, 10)),
                ("!", (10, 11)),
                ("     How are you", (11, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }

    #[test]
    fn default_behavior_when_missing() {
        let p: Punctuation = serde_json::from_str("{}").unwrap();
        assert_eq!(p, Punctuation::default());
    }
}
