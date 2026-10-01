//! Models: BPE, WordPiece, WordLevel and Unigram.

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;
use crate::traits::Model;
use crate::Token;

pub use bpe::{Bpe, BpeBuilder};
pub use unigram::Unigram;
pub use wordlevel::{WordLevel, WordLevelBuilder};
pub use wordpiece::{WordPiece, WordPieceBuilder};

/// Any built-in model. This is what a [`crate::Tokenizer`] holds and
/// what `tokenizer.json`'s `"model"` field (de)serializes to.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ModelWrapper {
    /// Byte-Pair Encoding.
    Bpe(Bpe),
    /// WordPiece (BERT).
    WordPiece(WordPiece),
    /// Whole-word lookup.
    WordLevel(WordLevel),
    /// Unigram language model (SentencePiece).
    Unigram(Unigram),
}

impl<'de> Deserialize<'de> for ModelWrapper {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        let ty = value
            .get("type")
            .and_then(|t| t.as_str())
            .map(str::to_owned)
            .or_else(|| infer_model_type(&value).map(str::to_owned))
            .ok_or_else(|| D::Error::custom("cannot determine the model type"))?;
        let model = match ty.as_str() {
            "BPE" => ModelWrapper::Bpe(serde_json::from_value(value).map_err(D::Error::custom)?),
            "WordPiece" => {
                ModelWrapper::WordPiece(serde_json::from_value(value).map_err(D::Error::custom)?)
            }
            "WordLevel" => {
                ModelWrapper::WordLevel(serde_json::from_value(value).map_err(D::Error::custom)?)
            }
            "Unigram" => {
                ModelWrapper::Unigram(serde_json::from_value(value).map_err(D::Error::custom)?)
            }
            other => {
                return Err(D::Error::custom(format!(
                "unsupported model type {other:?} (expected BPE, WordPiece, WordLevel or Unigram)"
            )))
            }
        };
        Ok(model)
    }
}

/// Files written by older versions of `tokenizers` omit the model's
/// `"type"`. Infer it from the fields present.
fn infer_model_type(value: &serde_json::Value) -> Option<&'static str> {
    let obj = value.as_object()?;
    if obj.contains_key("merges") {
        Some("BPE")
    } else if obj.contains_key("max_input_chars_per_word")
        || obj.contains_key("continuing_subword_prefix")
    {
        Some("WordPiece")
    } else if obj.contains_key("unk_id") || obj.get("vocab").is_some_and(|v| v.is_array()) {
        Some("Unigram")
    } else if obj.contains_key("vocab") {
        Some("WordLevel")
    } else {
        None
    }
}

macro_rules! dispatch {
    ($self:ident, $m:ident => $e:expr) => {
        match $self {
            ModelWrapper::Bpe($m) => $e,
            ModelWrapper::WordPiece($m) => $e,
            ModelWrapper::WordLevel($m) => $e,
            ModelWrapper::Unigram($m) => $e,
        }
    };
}

impl Model for ModelWrapper {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        dispatch!(self, m => m.tokenize(sequence))
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        dispatch!(self, m => m.token_to_id(token))
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        dispatch!(self, m => m.id_to_token(id))
    }
    fn get_vocab(&self) -> HashMap<String, u32> {
        dispatch!(self, m => m.get_vocab())
    }
    fn get_vocab_size(&self) -> usize {
        dispatch!(self, m => m.get_vocab_size())
    }
}

impl From<Bpe> for ModelWrapper {
    fn from(m: Bpe) -> Self {
        ModelWrapper::Bpe(m)
    }
}

impl From<WordPiece> for ModelWrapper {
    fn from(m: WordPiece) -> Self {
        ModelWrapper::WordPiece(m)
    }
}

impl From<WordLevel> for ModelWrapper {
    fn from(m: WordLevel) -> Self {
        ModelWrapper::WordLevel(m)
    }
}

impl From<Unigram> for ModelWrapper {
    fn from(m: Unigram) -> Self {
        ModelWrapper::Unigram(m)
    }
}
