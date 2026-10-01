//! Run several pre-tokenizers in order.

use serde::{Deserialize, Serialize};

use super::PreTokenizerWrapper;
use crate::error::Result;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Applies each pre-tokenizer in turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pretokenizers: Vec<PreTokenizerWrapper>,
}

impl Sequence {
    /// Build a sequence.
    pub fn new(pretokenizers: Vec<PreTokenizerWrapper>) -> Self {
        Self { pretokenizers }
    }

    /// The pre-tokenizers, in order.
    pub fn pretokenizers(&self) -> &[PreTokenizerWrapper] {
        &self.pretokenizers
    }

    /// Mutable access to the pre-tokenizers.
    pub fn pretokenizers_mut(&mut self) -> &mut Vec<PreTokenizerWrapper> {
        &mut self.pretokenizers
    }
}

impl PreTokenizer for Sequence {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        for p in &self.pretokenizers {
            p.pre_tokenize(pretokenized)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;
    use crate::pre_tokenizers::{Punctuation, WhitespaceSplit};

    #[test]
    fn basic() {
        let seq = Sequence::new(vec![WhitespaceSplit.into(), Punctuation::default().into()]);
        assert_eq!(
            splits(&seq, "Hey friend!     How are you?!?"),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }
}
