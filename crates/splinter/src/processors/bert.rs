//! BERT post-processor: `[CLS] A [SEP]` and `[CLS] A [SEP] B [SEP]`.

use serde::{Deserialize, Serialize};

use super::{mark_sequence, special_encoding};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::traits::PostProcessor;

/// Adds `[CLS]`/`[SEP]` around one or two sequences. The second sequence
/// and its `[SEP]` get type id 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BertProcessing {
    /// Separator token and its id.
    pub sep: (String, u32),
    /// Classifier token and its id.
    pub cls: (String, u32),
}

impl Default for BertProcessing {
    fn default() -> Self {
        Self {
            sep: ("[SEP]".into(), 102),
            cls: ("[CLS]".into(), 101),
        }
    }
}

impl BertProcessing {
    /// Build with explicit `(token, id)` pairs.
    pub fn new(sep: (String, u32), cls: (String, u32)) -> Self {
        Self { sep, cls }
    }

    fn special(&self, (token, id): &(String, u32), type_id: u32) -> Encoding {
        special_encoding(vec![*id], vec![token.clone()], type_id)
    }
}

impl PostProcessor for BertProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            3
        } else {
            2
        }
    }

    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if !add_special_tokens {
            return Ok(encodings);
        }
        if encodings.is_empty() || encodings.len() > 2 {
            return Err(Error::PostProcessor(format!(
                "BertProcessing expects 1 or 2 encodings, got {}",
                encodings.len()
            )));
        }
        Ok(encodings
            .into_iter()
            .enumerate()
            .map(|(i, mut encoding)| {
                mark_sequence(&mut encoding, i);
                if i == 0 {
                    Encoding::merge(
                        [
                            self.special(&self.cls, 0),
                            encoding,
                            self.special(&self.sep, 0),
                        ],
                        false,
                    )
                } else {
                    Encoding::merge([encoding, self.special(&self.sep, 1)], false)
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Token;

    fn enc(tokens: &[(u32, &str, (usize, usize))]) -> Encoding {
        Encoding::from_tokens(
            tokens
                .iter()
                .map(|(id, v, o)| Token::new(*id, (*v).into(), *o))
                .collect(),
            0,
        )
    }

    #[test]
    fn single_and_pair() {
        let p = BertProcessing::default();
        assert_eq!(p.added_tokens(false), 2);
        assert_eq!(p.added_tokens(true), 3);

        let a = enc(&[(12, "Hello", (0, 5)), (14, "there", (6, 11))]);
        let b = enc(&[(15, "pair", (0, 4))]);

        let single = p.process(a.clone(), None, true).unwrap();
        assert_eq!(single.ids(), &[101, 12, 14, 102]);
        assert_eq!(single.type_ids(), &[0, 0, 0, 0]);
        assert_eq!(single.tokens(), &["[CLS]", "Hello", "there", "[SEP]"]);
        assert_eq!(single.offsets(), &[(0, 0), (0, 5), (6, 11), (0, 0)]);
        assert_eq!(single.special_tokens_mask(), &[1, 0, 0, 1]);
        assert_eq!(single.attention_mask(), &[1, 1, 1, 1]);
        assert_eq!(single.token_to_sequence(2), Some(0));
        assert_eq!(single.token_to_sequence(3), None);

        let pair = p.process(a.clone(), Some(b.clone()), true).unwrap();
        assert_eq!(pair.ids(), &[101, 12, 14, 102, 15, 102]);
        assert_eq!(pair.type_ids(), &[0, 0, 0, 0, 1, 1]);
        assert_eq!(pair.special_tokens_mask(), &[1, 0, 0, 1, 0, 1]);
        assert_eq!(
            pair.offsets(),
            &[(0, 0), (0, 5), (6, 11), (0, 0), (0, 4), (0, 0)]
        );
        assert_eq!(pair.token_to_sequence(4), Some(1));
        assert_eq!(pair.token_to_sequence(5), None);

        let raw = p.process(a, Some(b), false).unwrap();
        assert_eq!(raw.ids(), &[12, 14, 15]);
        assert_eq!(raw.type_ids(), &[0, 0, 1]);
        assert_eq!(raw.special_tokens_mask(), &[0, 0, 0]);
    }

    #[test]
    fn overflowing_parts_get_specials_too() {
        let p = BertProcessing::default();
        let mut a = enc(&[(1, "a", (0, 1)), (2, "b", (1, 2)), (3, "c", (2, 3))]);
        a.truncate(2, 0, crate::TruncationDirection::Right);
        let out = p.process(a, None, true).unwrap();
        assert_eq!(out.ids(), &[101, 1, 2, 102]);
        assert_eq!(out.overflowing().len(), 1);
        let o = &out.overflowing()[0];
        assert_eq!(o.ids(), &[101, 3, 102]);
        assert_eq!(o.special_tokens_mask(), &[1, 0, 1]);
        assert_eq!(o.token_to_sequence(1), Some(0));
        assert_eq!(o.token_to_sequence(2), None);
    }
}
