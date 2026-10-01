//! RoBERTa post-processor: `<s> A </s>` and `<s> A </s></s> B </s>`.

use serde::{Deserialize, Serialize};

use super::{mark_sequence, special_encoding};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::pre_tokenizers::byte_level::process_offsets;
use crate::traits::PostProcessor;

fn default_true() -> bool {
    true
}

/// Adds `<s>`/`</s>` around one or two sequences. Every token gets type
/// id 0 (RoBERTa has no segment embeddings). With `trim_offsets`, the
/// leading/trailing whitespace that byte-level tokens carry (`Ġ`) is
/// removed from their offsets.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use morpheme::models::WordLevel;
/// use morpheme::pre_tokenizers::WhitespaceSplit;
/// use morpheme::processors::RobertaProcessing;
/// use morpheme::Tokenizer;
///
/// let vocab: HashMap<String, u32> =
///     [("<unk>", 3), ("hello", 7), ("world", 8)].map(|(t, i)| (t.to_string(), i)).into();
/// let tokenizer = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("<unk>").build()?)
///     .with_pre_tokenizer(WhitespaceSplit)
///     .with_post_processor(RobertaProcessing::new(("</s>", 2), ("<s>", 0)));
///
/// let pair = tokenizer.encode(("hello", "world"), true)?;
/// assert_eq!(pair.tokens(), ["<s>", "hello", "</s>", "</s>", "world", "</s>"]);
/// assert_eq!(pair.type_ids(), [0, 0, 0, 0, 0, 0]);
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RobertaProcessing {
    /// Separator token and its id.
    pub sep: (String, u32),
    /// Classifier (start) token and its id.
    pub cls: (String, u32),
    /// Trim whitespace from token offsets.
    #[serde(default = "default_true")]
    pub trim_offsets: bool,
    /// Whether the pre-tokenizer added a prefix space (affects trimming
    /// of the first token).
    #[serde(default = "default_true")]
    pub add_prefix_space: bool,
}

impl Default for RobertaProcessing {
    fn default() -> Self {
        Self {
            sep: ("</s>".into(), 2),
            cls: ("<s>".into(), 0),
            trim_offsets: true,
            add_prefix_space: true,
        }
    }
}

impl RobertaProcessing {
    /// Build with explicit `(token, id)` pairs and default flags.
    pub fn new(sep: (impl Into<String>, u32), cls: (impl Into<String>, u32)) -> Self {
        Self {
            sep: (sep.0.into(), sep.1),
            cls: (cls.0.into(), cls.1),
            ..Default::default()
        }
    }

    /// Set `trim_offsets`.
    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    /// Set `add_prefix_space`.
    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    fn special(&self, (token, id): &(String, u32)) -> Encoding {
        special_encoding(vec![*id], vec![token.clone()], 0)
    }
}

impl PostProcessor for RobertaProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair { 4 } else { 2 }
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if self.trim_offsets {
            for e in &mut encodings {
                process_offsets(e, self.add_prefix_space);
                for o in e.overflowing_mut() {
                    process_offsets(o, self.add_prefix_space);
                }
            }
        }
        for e in &mut encodings {
            e.set_type_ids(vec![0; e.len()]);
        }
        if !add_special_tokens {
            return Ok(encodings);
        }
        if encodings.is_empty() || encodings.len() > 2 {
            return Err(Error::PostProcessor(format!(
                "RobertaProcessing expects 1 or 2 encodings, got {}",
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
                        [self.special(&self.cls), encoding, self.special(&self.sep)],
                        false,
                    )
                } else {
                    // Overflowing parts of the second sequence are all type 0
                    // too (the top-level one was reset above).
                    for o in encoding.overflowing_mut() {
                        o.set_type_ids(vec![0; o.len()]);
                    }
                    Encoding::merge(
                        [self.special(&self.sep), encoding, self.special(&self.sep)],
                        false,
                    )
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
    fn single_and_pair_layout() {
        let p = RobertaProcessing::default();
        assert_eq!(p.added_tokens(false), 2);
        assert_eq!(p.added_tokens(true), 4);

        let a = enc(&[(12, "Hello", (0, 5)), (14, "there", (6, 11))]);
        let b = enc(&[(15, "pair", (0, 4))]);

        let single = p.process(a.clone(), None, true).unwrap();
        assert_eq!(single.ids(), &[0, 12, 14, 2]);
        assert_eq!(single.type_ids(), &[0, 0, 0, 0]);
        assert_eq!(single.tokens(), &["<s>", "Hello", "there", "</s>"]);
        assert_eq!(single.special_tokens_mask(), &[1, 0, 0, 1]);
        assert_eq!(single.token_to_sequence(0), None);
        assert_eq!(single.token_to_sequence(1), Some(0));

        let pair = p.process(a.clone(), Some(b.clone()), true).unwrap();
        assert_eq!(pair.ids(), &[0, 12, 14, 2, 2, 15, 2]);
        assert_eq!(pair.type_ids(), &[0; 7]);
        assert_eq!(
            pair.tokens(),
            &["<s>", "Hello", "there", "</s>", "</s>", "pair", "</s>"]
        );
        assert_eq!(pair.special_tokens_mask(), &[1, 0, 0, 1, 1, 0, 1]);
        assert_eq!(
            pair.offsets(),
            &[(0, 0), (0, 5), (6, 11), (0, 0), (0, 0), (0, 4), (0, 0)]
        );
        assert_eq!(pair.word_ids()[0], None);
        assert_eq!(pair.token_to_sequence(4), None);
        assert_eq!(pair.token_to_sequence(5), Some(1));

        let raw = p.process(a, Some(b), false).unwrap();
        assert_eq!(raw.ids(), &[12, 14, 15]);
        assert_eq!(raw.type_ids(), &[0, 0, 0]);
    }

    #[test]
    fn trims_byte_level_offsets() {
        // Ground truth (Python tokenizers 0.23.2, roberta-base,
        // add_special_tokens=True): "Hello world" ->
        // offsets [(0,0),(0,5),(6,11),(0,0)] — the `Ġworld` token's
        // leading space is trimmed from (5,11) to (6,11).
        let p = RobertaProcessing::default().add_prefix_space(false);
        let a = enc(&[(31414, "Hello", (0, 5)), (232, "Ġworld", (5, 11))]);
        let out = p.process(a, None, true).unwrap();
        assert_eq!(out.offsets(), &[(0, 0), (0, 5), (6, 11), (0, 0)]);

        let untrimmed = RobertaProcessing::default().trim_offsets(false);
        let a = enc(&[(31414, "Hello", (0, 5)), (232, "Ġworld", (5, 11))]);
        let out = untrimmed.process(a, None, true).unwrap();
        assert_eq!(out.offsets(), &[(0, 0), (0, 5), (5, 11), (0, 0)]);
    }
}
