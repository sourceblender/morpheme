//! A chain of post-processors.

use serde::{Deserialize, Serialize};

use super::PostProcessorWrapper;
use crate::encoding::Encoding;
use crate::error::Result;
use crate::traits::PostProcessor;

/// Applies several post-processors in order (e.g. byte-level offset
/// trimming followed by a template).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    processors: Vec<PostProcessorWrapper>,
}

impl Sequence {
    /// Chain `processors`.
    pub fn new(processors: Vec<PostProcessorWrapper>) -> Self {
        Self { processors }
    }

    /// The chained processors.
    pub fn processors(&self) -> &[PostProcessorWrapper] {
        &self.processors
    }

    /// Mutable access to the chained processors.
    pub fn processors_mut(&mut self) -> &mut Vec<PostProcessorWrapper> {
        &mut self.processors
    }
}

impl PostProcessor for Sequence {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processors
            .iter()
            .map(|p| p.added_tokens(is_pair))
            .sum()
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        for p in &self.processors {
            encodings = p.process_encodings(encodings, add_special_tokens)?;
        }
        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::BertProcessing;
    use crate::Token;

    #[test]
    fn serde_round_trip() {
        let seq = PostProcessorWrapper::from(Sequence::new(vec![BertProcessing::default().into()]));
        let s = serde_json::to_string(&seq).unwrap();
        assert_eq!(
            s,
            r#"{"type":"Sequence","processors":[{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}]}"#
        );
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(&s).unwrap(),
            seq
        );
    }

    #[test]
    fn chains_and_sums_added_tokens() {
        let seq = Sequence::new(vec![BertProcessing::default().into()]);
        assert_eq!(seq.added_tokens(false), 2);
        assert_eq!(seq.added_tokens(true), 3);
        let twice = Sequence::new(vec![
            BertProcessing::default().into(),
            BertProcessing::default().into(),
        ]);
        assert_eq!(twice.added_tokens(true), 6);
        let e = Encoding::from_tokens(vec![Token::new(5, "x".into(), (0, 1))], 0);
        let out = seq.process(e, None, true).unwrap();
        assert_eq!(out.ids(), &[101, 5, 102]);
        assert_eq!(out.special_tokens_mask(), &[1, 0, 1]);
    }
}
