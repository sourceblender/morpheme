//! Post-processors — combine tokenized encodings into final model
//! inputs.
//!
//! v0.1 supports two HF-compatible types:
//!
//! - [`TemplatePostProcessor`] — a list of pieces where each piece is
//!   either a literal `[CLS]`-style token type or a `$A`/`$B`/`$0`
//!   reference. This is the most general form; `RobertaPostProcessor`
//!   is just a `TemplatePostProcessor` with specific pieces.
//! - [`RobertaPostProcessor`] — the RoBERTa-style `<s> $A </s> $B </s>`
//!   pattern with type ids `[0, ..., 0, 2, ..., 1, 2]`.
//!
//! Both run after the model has produced ids / tokens / offsets. They
//! insert special tokens at the front and back, optionally merge two
//! encodings for sentence-pair tasks, and assign per-token type ids.

// TemplatePiece, TemplateEntry, etc. are public so users can build
// templates, but their fields don't need docstrings — the trait
// documents the public API.
#![allow(missing_docs)]

use crate::encoding::Encoding;
use crate::error::{Error as SplinterError, Result};

/// One piece of a [`TemplatePostProcessor`].
#[derive(Debug, Clone)]
pub enum TemplatePiece {
    /// A literal token type (e.g. `"[CLS]"`). Matched against vocab
    /// entries by exact string equality.
    TokenType(String),
    /// Reference to the first sentence's tokens (`$A`).
    SequenceA,
    /// Reference to the second sentence's tokens (`$B`). Only valid
    /// when a pair encoding is supplied.
    SequenceB,
    /// Reference to the type-id sequence for sentence A (`$0`).
    TypeIdA,
    /// Reference to the type-id sequence for sentence B (`$1`).
    TypeIdB,
}

/// One row of a template — the piece plus its `type_id`.
#[derive(Debug, Clone)]
pub struct TemplateEntry {
    pub piece: TemplatePiece,
    pub type_id: u32,
}

/// A `TemplatePostProcessor` template — the order of entries is the
/// order they appear in the output.
#[derive(Debug, Clone)]
pub struct Template {
    entries: Vec<TemplateEntry>,
}

impl Template {
    /// Build a template from an ordered list of `(piece, type_id)`
    /// entries.
    pub fn new(entries: Vec<TemplateEntry>) -> Self {
        Self { entries }
    }
}

/// Post-processor combining two encodings (or a single one) into a
/// final model input. Public so downstream callers can pattern-match.
#[derive(Debug, Clone)]
pub enum PostProcessorKind {
    /// `RobertaPostProcessor` style: `<s> $A </s> $B </s>` with type
    /// ids `[0, ..., 0, 2, ..., 1, 2]`.
    Roberta,
    /// User-supplied template.
    Template(Template),
}

/// Anything that takes one (or two) encodings and produces a final
/// model input.
pub trait PostProcessor: Send + Sync + std::fmt::Debug {
    /// Apply the post-processor to a single encoding (single-sentence
    /// tasks).
    fn apply(&self, encoding: Encoding) -> Result<Encoding>;

    /// Apply the post-processor to a pair of encodings (sentence-pair
    /// tasks).
    fn apply_pair(&self, encoding: Encoding, pair_encoding: Encoding) -> Result<Encoding>;
}

/// RoBERTa-style post-processor: `<s> $A </s> $B </s>` with type ids
/// `[0, ..., 0, 2, ..., 1, 2]`.
#[derive(Debug, Clone, Default)]
pub struct RobertaPostProcessor;

impl RobertaPostProcessor {
    const CLS_ID: u32 = 0;
    const SEP_ID: u32 = 2;
    const CLS_STR: &'static str = "<s>";
    const SEP_STR: &'static str = "</s>";
}

impl PostProcessor for RobertaPostProcessor {
    fn apply(&self, encoding: Encoding) -> Result<Encoding> {
        let mut ids = vec![Self::CLS_ID];
        let mut tokens = vec![Self::CLS_STR.to_string()];
        let mut offsets = vec![(0, 0)];
        let mut type_ids = vec![0];
        ids.extend(encoding.ids.iter().copied());
        tokens.extend(encoding.tokens.iter().cloned());
        offsets.extend(encoding.offsets.iter().copied());
        type_ids.extend(encoding.type_ids.iter().copied());
        ids.push(Self::SEP_ID);
        tokens.push(Self::SEP_STR.to_string());
        offsets.push((0, 0));
        type_ids.push(2);
        Ok(Encoding {
            ids,
            tokens,
            offsets,
            type_ids,
        })
    }

    fn apply_pair(&self, encoding: Encoding, pair_encoding: Encoding) -> Result<Encoding> {
        // <s> $A </s> $B </s>
        // type ids: 0, ..., 0, 2, ..., 1, 2
        let mut ids = vec![Self::CLS_ID];
        let mut tokens = vec![Self::CLS_STR.to_string()];
        let mut offsets = vec![(0, 0)];
        let mut type_ids = vec![0];
        ids.extend(encoding.ids.iter().copied());
        tokens.extend(encoding.tokens.iter().cloned());
        offsets.extend(encoding.offsets.iter().copied());
        type_ids.extend(encoding.type_ids.iter().copied());
        ids.push(Self::SEP_ID);
        tokens.push(Self::SEP_STR.to_string());
        offsets.push((0, 0));
        type_ids.push(2);
        ids.extend(pair_encoding.ids.iter().copied());
        tokens.extend(pair_encoding.tokens.iter().cloned());
        offsets.extend(pair_encoding.offsets.iter().copied());
        type_ids.extend(pair_encoding.type_ids.iter().map(|_| 1));
        ids.push(Self::SEP_ID);
        tokens.push(Self::SEP_STR.to_string());
        offsets.push((0, 0));
        type_ids.push(2);
        Ok(Encoding {
            ids,
            tokens,
            offsets,
            type_ids,
        })
    }
}

/// Template-based post-processor (BERT-style).
#[derive(Debug, Clone)]
pub struct TemplatePostProcessor {
    /// Template for single-sentence input.
    single: Template,
    /// Template for sentence-pair input. If absent, single is used
    /// for both.
    pair: Option<Template>,
}

impl TemplatePostProcessor {
    /// Build a `TemplatePostProcessor` from a single template. Pair
    /// inputs use the same template by default.
    pub fn new(single: Template) -> Self {
        Self { single, pair: None }
    }

    /// Build with distinct single / pair templates.
    pub fn with_pair(single: Template, pair: Template) -> Self {
        Self {
            single,
            pair: Some(pair),
        }
    }

    fn render(&self, template: &Template, encoding: Encoding, type_id_a: u32) -> Result<Encoding> {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut type_ids = Vec::new();
        for entry in &template.entries {
            match &entry.piece {
                TemplatePiece::TokenType(s) => {
                    ids.push(0); // v0.1: lookup_special placeholder
                    tokens.push(s.clone());
                    offsets.push((0, 0));
                    type_ids.push(entry.type_id);
                }
                TemplatePiece::SequenceA => {
                    ids.extend(encoding.ids.iter().copied());
                    tokens.extend(encoding.tokens.iter().cloned());
                    offsets.extend(encoding.offsets.iter().copied());
                    type_ids.extend(std::iter::repeat(entry.type_id).take(encoding.ids.len()));
                    let _ = type_id_a;
                }
                TemplatePiece::SequenceB => {
                    return Err(SplinterError::Model(
                        "TemplatePostProcessor::apply called with \
                         `$B` but no pair encoding supplied"
                            .into(),
                    ));
                }
                TemplatePiece::TypeIdA | TemplatePiece::TypeIdB => {
                    return Err(SplinterError::Model(
                        "type-id pieces (`$0`, `$1`) are not yet \
                         supported"
                            .into(),
                    ));
                }
            }
        }
        Ok(Encoding {
            ids,
            tokens,
            offsets,
            type_ids,
        })
    }

    fn render_pair(
        &self,
        template: &Template,
        encoding: Encoding,
        pair: Encoding,
    ) -> Result<Encoding> {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut type_ids = Vec::new();
        for entry in &template.entries {
            match &entry.piece {
                TemplatePiece::TokenType(s) => {
                    ids.push(0);
                    tokens.push(s.clone());
                    offsets.push((0, 0));
                    type_ids.push(entry.type_id);
                }
                TemplatePiece::SequenceA => {
                    ids.extend(encoding.ids.iter().copied());
                    tokens.extend(encoding.tokens.iter().cloned());
                    offsets.extend(encoding.offsets.iter().copied());
                    type_ids.extend(encoding.type_ids.iter().copied());
                }
                TemplatePiece::SequenceB => {
                    ids.extend(pair.ids.iter().copied());
                    tokens.extend(pair.tokens.iter().cloned());
                    offsets.extend(pair.offsets.iter().copied());
                    eprintln!(
                        "SequenceB: entry.type_id={}, pair.ids.len()={}",
                        entry.type_id,
                        pair.ids.len()
                    );
                    type_ids.extend(std::iter::repeat(entry.type_id).take(pair.ids.len()));
                }
                TemplatePiece::TypeIdA | TemplatePiece::TypeIdB => {
                    return Err(SplinterError::Model(
                        "type-id pieces (`$0`, `$1`) are not yet \
                         supported"
                            .into(),
                    ));
                }
            }
        }
        Ok(Encoding {
            ids,
            tokens,
            offsets,
            type_ids,
        })
    }
}

impl PostProcessor for TemplatePostProcessor {
    fn apply(&self, encoding: Encoding) -> Result<Encoding> {
        self.render(&self.single, encoding, 0)
    }

    fn apply_pair(&self, encoding: Encoding, pair_encoding: Encoding) -> Result<Encoding> {
        let template = self.pair.as_ref().unwrap_or(&self.single);
        self.render_pair(template, encoding, pair_encoding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoding(ids: Vec<u32>, tokens: Vec<&str>) -> Encoding {
        let n = tokens.len();
        Encoding {
            ids,
            tokens: tokens.into_iter().map(String::from).collect(),
            offsets: vec![(0, 0); n],
            type_ids: vec![0; n],
        }
    }

    #[test]
    fn roberta_single_sentence() {
        let p = RobertaPostProcessor;
        let enc = make_encoding(vec![10, 20, 30], vec!["the", "quick", "fox"]);
        let out = p.apply(enc).unwrap();
        assert_eq!(out.tokens, vec!["<s>", "the", "quick", "fox", "</s>"]);
        assert_eq!(out.ids, vec![0, 10, 20, 30, 2]);
        assert_eq!(out.type_ids, vec![0, 0, 0, 0, 2]);
    }

    #[test]
    fn roberta_pair() {
        let p = RobertaPostProcessor;
        let a = make_encoding(vec![10, 20], vec!["a", "b"]);
        let b = make_encoding(vec![30, 40], vec!["c", "d"]);
        let out = p.apply_pair(a, b).unwrap();
        assert_eq!(out.tokens, vec!["<s>", "a", "b", "</s>", "c", "d", "</s>"]);
        assert_eq!(out.ids, vec![0, 10, 20, 2, 30, 40, 2]);
        assert_eq!(out.type_ids, vec![0, 0, 0, 2, 1, 1, 2]);
    }

    #[test]
    fn template_single() {
        let template = Template::new(vec![
            TemplateEntry {
                piece: TemplatePiece::TokenType("[CLS]".into()),
                type_id: 0,
            },
            TemplateEntry {
                piece: TemplatePiece::SequenceA,
                type_id: 0,
            },
            TemplateEntry {
                piece: TemplatePiece::TokenType("[SEP]".into()),
                type_id: 0,
            },
        ]);
        let p = TemplatePostProcessor::new(template);
        let enc = make_encoding(vec![10, 20], vec!["the", "quick"]);
        let out = p.apply(enc).unwrap();
        assert_eq!(out.tokens, vec!["[CLS]", "the", "quick", "[SEP]"]);
        assert_eq!(out.ids.len(), 4);
        assert_eq!(out.type_ids.len(), 4);
    }
}
