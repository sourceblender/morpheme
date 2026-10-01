//! Post-processors: add special tokens and combine sequence pairs.
//!
//! - [`BertProcessing`] — `[CLS] A [SEP] B [SEP]`.
//! - [`RobertaProcessing`] — `<s> A </s></s> B </s>`, optionally
//!   trimming whitespace from byte-level offsets.
//! - [`TemplateProcessing`] — any layout, described by a template such as
//!   `"[CLS] $A [SEP] $B:1 [SEP]:1"`.
//! - [`Sequence`] — several post-processors applied in order.
//! - [`ByteLevel`] — offset trimming for byte-level BPE (lives in
//!   [`crate::pre_tokenizers::byte_level`]).

mod bert;
mod roberta;
mod sequence;
pub mod template;

use serde::{Deserialize, Serialize};

use crate::encoding::Encoding;
use crate::error::Result;
use crate::pre_tokenizers::ByteLevel;
use crate::traits::PostProcessor;

pub use bert::BertProcessing;
pub use roberta::RobertaProcessing;
pub use sequence::Sequence;
pub use template::{
    Piece, SpecialToken, Template, TemplateProcessing, TemplateProcessingBuilder, Tokens,
};

/// Any built-in post-processor. Serializes as `tokenizer.json`'s
/// `"post_processor"` field, tagged by `"type"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum PostProcessorWrapper {
    /// See [`BertProcessing`].
    BertProcessing(BertProcessing),
    /// See [`RobertaProcessing`].
    RobertaProcessing(RobertaProcessing),
    /// See [`ByteLevel`].
    ByteLevel(ByteLevel),
    /// See [`TemplateProcessing`].
    TemplateProcessing(TemplateProcessing),
    /// See [`Sequence`].
    Sequence(Sequence),
}

impl PostProcessorWrapper {
    pub(crate) fn rebind_token_ids(&mut self, lookup: &impl Fn(&str) -> Result<u32>) -> Result<()> {
        match self {
            Self::BertProcessing(p) => {
                p.sep.1 = lookup(&p.sep.0)?;
                p.cls.1 = lookup(&p.cls.0)?;
            }
            Self::RobertaProcessing(p) => {
                p.sep.1 = lookup(&p.sep.0)?;
                p.cls.1 = lookup(&p.cls.0)?;
            }
            Self::ByteLevel(_) => {}
            Self::TemplateProcessing(p) => p.rebind_token_ids(lookup)?,
            Self::Sequence(p) => {
                for processor in p.processors_mut() {
                    processor.rebind_token_ids(lookup)?;
                }
            }
        }
        Ok(())
    }
}

impl PostProcessor for PostProcessorWrapper {
    fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            Self::BertProcessing(p) => p.added_tokens(is_pair),
            Self::RobertaProcessing(p) => p.added_tokens(is_pair),
            Self::ByteLevel(p) => p.added_tokens(is_pair),
            Self::TemplateProcessing(p) => p.added_tokens(is_pair),
            Self::Sequence(p) => p.added_tokens(is_pair),
        }
    }

    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        match self {
            Self::BertProcessing(p) => p.process_encodings(encodings, add_special_tokens),
            Self::RobertaProcessing(p) => p.process_encodings(encodings, add_special_tokens),
            Self::ByteLevel(p) => p.process_encodings(encodings, add_special_tokens),
            Self::TemplateProcessing(p) => p.process_encodings(encodings, add_special_tokens),
            Self::Sequence(p) => p.process_encodings(encodings, add_special_tokens),
        }
    }
}

impl From<BertProcessing> for PostProcessorWrapper {
    fn from(p: BertProcessing) -> Self {
        Self::BertProcessing(p)
    }
}

impl From<RobertaProcessing> for PostProcessorWrapper {
    fn from(p: RobertaProcessing) -> Self {
        Self::RobertaProcessing(p)
    }
}

impl From<ByteLevel> for PostProcessorWrapper {
    fn from(p: ByteLevel) -> Self {
        Self::ByteLevel(p)
    }
}

impl From<TemplateProcessing> for PostProcessorWrapper {
    fn from(p: TemplateProcessing) -> Self {
        Self::TemplateProcessing(p)
    }
}

impl From<Sequence> for PostProcessorWrapper {
    fn from(p: Sequence) -> Self {
        Self::Sequence(p)
    }
}

/// A single-sequence piece made only of special tokens: `(0, 0)` offsets,
/// no word ids, special-tokens mask set.
pub(crate) fn special_encoding(ids: Vec<u32>, tokens: Vec<String>, type_id: u32) -> Encoding {
    let n = ids.len();
    Encoding::new(
        ids,
        vec![type_id; n],
        tokens,
        vec![None; n],
        vec![(0, 0); n],
        vec![1; n],
        vec![1; n],
        Vec::new(),
    )
}

/// Mark `encoding` (and its overflowing parts) as sequence `id`, so the
/// merged result records which tokens belong to which input sequence.
pub(crate) fn mark_sequence(encoding: &mut Encoding, id: usize) {
    encoding.set_sequence_id(id);
    for o in encoding.overflowing_mut() {
        o.set_sequence_id(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_serde_is_tagged() {
        let bert = PostProcessorWrapper::from(BertProcessing::default());
        let s = serde_json::to_string(&bert).unwrap();
        assert_eq!(
            s,
            r#"{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}"#
        );
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(&s).unwrap(),
            bert
        );

        let roberta = PostProcessorWrapper::from(RobertaProcessing::default());
        let s = serde_json::to_string(&roberta).unwrap();
        assert_eq!(
            s,
            r#"{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}"#
        );
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(&s).unwrap(),
            roberta
        );
    }

    #[test]
    fn missing_or_unknown_type_is_an_error() {
        assert!(
            serde_json::from_str::<PostProcessorWrapper>(
                r#"{"sep":["[SEP]",102],"cls":["[CLS]",101]}"#
            )
            .is_err()
        );
        assert!(serde_json::from_str::<PostProcessorWrapper>(r#"{"type":"Nope"}"#).is_err());
        assert!(
            serde_json::from_str::<PostProcessorWrapper>(
                r#"{"type":"BertProcessing","sep":["[SEP]",102]}"#
            )
            .is_err()
        );
    }

    /// Every `post_processor` in the downloaded HF fixtures
    /// (`scripts/fetch-hf-fixtures.sh`) parses, and non-ByteLevel ones
    /// re-serialize to the identical JSON value.
    #[test]
    fn real_fixture_configs_round_trip() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hf");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!(
                "skipping: {} missing (run scripts/fetch-hf-fixtures.sh)",
                dir.display()
            );
            return;
        };
        let mut checked = 0;
        for entry in entries {
            let path = entry.unwrap().path();
            let raw = std::fs::read_to_string(&path).unwrap();
            let file: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let pp = &file["post_processor"];
            if pp.is_null() {
                continue;
            }
            let parsed: PostProcessorWrapper = serde_json::from_value(pp.clone())
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            // ByteLevel lives in `pre_tokenizers`; older files omit fields
            // (e.g. gpt2 has no `use_regex`) that re-serialization adds, as
            // HF does. Parsing is what matters here.
            if pp["type"] == "ByteLevel" {
                continue;
            }
            assert_eq!(
                &serde_json::to_value(&parsed).unwrap(),
                pp,
                "{} did not round-trip",
                path.display()
            );
            checked += 1;
        }
        eprintln!("checked {checked} post-processor configs");
    }
}
