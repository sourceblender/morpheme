//! Splinter's own JSON tokenizer format (v1.0).
//!
//! Supports two model variants:
//!
//! - `bpe` — vocab + ordered merge list + end-of-word suffix.
//! - `wordpiece` — vocab + continuing-subword prefix + unk_token +
//!   max_input_chars_per_word. No merges.
//!
//! The exact set of fields required depends on `model.type`.
//!
//! ## BPE schema
//!
//! ```json
//! {
//!   "version": "1.0",
//!   "model": {
//!     "type": "bpe",
//!     "dropout": null,
//!     "unk_token": null,
//!     "end_of_word_suffix": "</w>",
//!     "fuse_unk": false,
//!     "byte_fallback": false
//!   },
//!   "vocab": ["<unk>", "a", "b", ...],
//!   "merges": [["a", "a</w>"], ...]
//! }
//! ```
//!
//! ## WordPiece schema
//!
//! ```json
//! {
//!   "version": "1.0",
//!   "model": {
//!     "type": "wordpiece",
//!     "unk_token": "<unk>",
//!     "continuing_subword_prefix": "##",
//!     "max_input_chars_per_word": 100,
//!     "end_of_word_suffix": "</w>"
//!   },
//!   "vocab": ["<unk>", "h", "##e", "##l", "##o", "hello", ...],
//!   "merges": []
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Error as SplinterError, Result};
use crate::model::bpe::Bpe;
use crate::model::unigram::Unigram;
use crate::model::wordpiece::WordPiece;
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;

/// On-disk schema version.
pub const SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TokenizerFile {
    pub version: String,
    pub model: ModelConfig,
    pub vocab: Vec<String>,
    #[serde(default)]
    pub merges: Vec<[String; 2]>,
    #[serde(default)]
    pub log_probs: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ModelConfig {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub dropout: Option<f32>,
    #[serde(default)]
    pub unk_token: Option<String>,
    #[serde(default)]
    pub end_of_word_suffix: Option<String>,
    #[serde(default)]
    pub continuing_subword_suffix: Option<String>,
    #[serde(default)]
    pub continuing_subword_prefix: Option<String>,
    #[serde(default)]
    pub max_input_chars_per_word: Option<usize>,
    #[serde(default)]
    pub fuse_unk: bool,
    #[serde(default)]
    pub byte_fallback: bool,
    #[serde(default)]
    pub whitespace_marker: Option<String>,
    #[serde(default)]
    pub min_score: Option<f64>,
}

/// Parse a tokenizer from a JSON string.
pub fn from_str(s: &str) -> Result<Tokenizer> {
    let file: TokenizerFile = serde_json::from_str(s)?;
    if file.version != SCHEMA_VERSION {
        return Err(SplinterError::Model(format!(
            "unsupported schema version {:?} (expected {:?})",
            file.version, SCHEMA_VERSION
        )));
    }
    let vocab = Vocab::from_tokens(file.vocab)
        .map_err(|e| SplinterError::Model(format!("vocab construction failed: {e}")))?;
    let log_probs = file.log_probs;

    match file.model.kind.as_str() {
        "bpe" => {
            if file.model.dropout.is_some() {
                return Err(SplinterError::Model(
                    "dropout is not implemented in splinter v0.1".into(),
                ));
            }
            if file.model.byte_fallback {
                return Err(SplinterError::Model(
                    "byte_fallback is not implemented in splinter v0.1".into(),
                ));
            }
            if file.model.continuing_subword_suffix.is_some() {
                return Err(SplinterError::Model(
                    "continuing_subword_suffix is not implemented in splinter v0.1".into(),
                ));
            }
            let eow = file
                .model
                .end_of_word_suffix
                .unwrap_or_else(|| "</w>".to_string());
            let merges: Vec<(String, String)> =
                file.merges.into_iter().map(|[a, b]| (a, b)).collect();
            let bpe = Bpe::new(vocab, merges, eow);
            Ok(Tokenizer::new(bpe))
        }
        "wordpiece" => {
            let unk = file
                .model
                .unk_token
                .ok_or_else(|| SplinterError::Model("wordpiece model requires unk_token".into()))?;
            let prefix = file.model.continuing_subword_prefix.unwrap_or_else(|| "##".into());
            let max_chars = file.model.max_input_chars_per_word.unwrap_or(100);
            let wp = WordPiece::new(vocab, prefix, unk, max_chars);
            Ok(Tokenizer::wordpiece(wp))
        }
        "unigram" => {
            let unk = file
                .model
                .unk_token
                .ok_or_else(|| SplinterError::Model("unigram model requires unk_token".into()))?;
            let ws = file
                .model
                .whitespace_marker
                .unwrap_or_else(|| "\u{2581}".to_string());
            let min_score = file.model.min_score.unwrap_or(-10.0);
            if log_probs.len() != vocab.len() {
                return Err(SplinterError::Model(format!(
                    "unigram log_probs length {} does not match vocab length {}",
                    log_probs.len(),
                    vocab.len()
                )));
            }
            let m = Unigram::new(vocab, log_probs, ws, unk, min_score);
            Ok(Tokenizer::unigram(m))
        }
        other => Err(SplinterError::Model(format!(
            "unsupported model type {other:?} (only \"bpe\", \"wordpiece\", and \"unigram\" are supported in v1.0)"
        ))),
    }
}

/// Serialize a tokenizer to a compact JSON string.
pub fn to_string(t: &Tokenizer) -> Result<String> {
    let file = to_file(t);
    Ok(serde_json::to_string(&file)?)
}

/// Serialize a tokenizer to a pretty JSON string.
pub fn to_string_pretty(t: &Tokenizer) -> Result<String> {
    let file = to_file(t);
    Ok(serde_json::to_string_pretty(&file)?)
}

fn to_file(t: &Tokenizer) -> TokenizerFile {
    let vocab = t.vocab();
    let vocab_list: Vec<String> = vocab.iter().map(|(_, s)| s.to_owned()).collect();

    if let Some(bpe) = t.bpe_model() {
        let mut merges: Vec<[String; 2]> = Vec::with_capacity(bpe.num_merges());
        for pair in bpe.merges_iter() {
            merges.push([pair.0.clone(), pair.1.clone()]);
        }
        TokenizerFile {
            version: SCHEMA_VERSION.into(),
            model: ModelConfig {
                kind: "bpe".into(),
                dropout: None,
                unk_token: None,
                end_of_word_suffix: Some(bpe.end_of_word_suffix().to_owned()),
                continuing_subword_suffix: None,
                continuing_subword_prefix: None,
                max_input_chars_per_word: None,
                fuse_unk: false,
                byte_fallback: false,
                whitespace_marker: None,
                min_score: None,
            },
            vocab: vocab_list,
            merges,
            log_probs: Vec::new(),
        }
    } else if let Some(wp) = t.wordpiece_model() {
        TokenizerFile {
            version: SCHEMA_VERSION.into(),
            model: ModelConfig {
                kind: "wordpiece".into(),
                dropout: None,
                unk_token: Some(wp.unk_token.clone()),
                end_of_word_suffix: None,
                continuing_subword_suffix: None,
                continuing_subword_prefix: Some(wp.continuing_subword_prefix.clone()),
                max_input_chars_per_word: Some(wp.max_input_chars_per_word),
                fuse_unk: false,
                byte_fallback: false,
                whitespace_marker: None,
                min_score: None,
            },
            vocab: vocab_list,
            merges: Vec::new(),
            log_probs: Vec::new(),
        }
    } else if let Some(ug) = t.unigram_model() {
        TokenizerFile {
            version: SCHEMA_VERSION.into(),
            model: ModelConfig {
                kind: "unigram".into(),
                dropout: None,
                unk_token: Some(ug.unk_token.clone()),
                end_of_word_suffix: None,
                continuing_subword_suffix: None,
                continuing_subword_prefix: None,
                max_input_chars_per_word: None,
                fuse_unk: false,
                byte_fallback: false,
                whitespace_marker: Some(ug.whitespace_marker.clone()),
                min_score: Some(ug.min_score),
            },
            vocab: vocab_list,
            merges: Vec::new(),
            log_probs: ug.log_probs.clone(),
        }
    } else {
        unreachable!("Tokenizer always has a model")
    }
}
