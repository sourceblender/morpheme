//! WordLevel: each pre-token maps to exactly one vocabulary entry.

use std::collections::HashMap;

use rustc_hash::FxHashMap;
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Token;
use crate::error::{Error, Result};
use crate::models::bpe::OrderedVocab;
use crate::traits::Model;

/// `token -> id`.
pub type Vocab = HashMap<String, u32>;

/// Builder for [`WordLevel`].
#[derive(Debug, Clone)]
pub struct WordLevelBuilder {
    vocab: Vocab,
    unk_token: String,
}

impl Default for WordLevelBuilder {
    fn default() -> Self {
        Self {
            vocab: Vocab::new(),
            unk_token: "<unk>".into(),
        }
    }
}

impl WordLevelBuilder {
    /// A builder with defaults (`unk_token = "<unk>"`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vocabulary.
    #[must_use]
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.vocab = vocab;
        self
    }

    /// Token for words missing from the vocabulary.
    #[must_use]
    pub fn unk_token(mut self, token: impl Into<String>) -> Self {
        self.unk_token = token.into();
        self
    }

    /// Build the model.
    pub fn build(self) -> Result<WordLevel> {
        let vocab_r = self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Ok(WordLevel {
            vocab: self.vocab,
            vocab_r,
            unk_token: self.unk_token,
        })
    }
}

/// WordLevel model.
#[derive(Clone, PartialEq, Eq)]
pub struct WordLevel {
    vocab: Vocab,
    vocab_r: FxHashMap<u32, String>,
    /// Token for words missing from the vocabulary.
    pub unk_token: String,
}

impl std::fmt::Debug for WordLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WordLevel")
            .field("unk_token", &self.unk_token)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for WordLevel {
    fn default() -> Self {
        WordLevelBuilder::default()
            .build()
            .expect("default WordLevel is valid")
    }
}

impl WordLevel {
    /// Start building a model.
    pub fn builder() -> WordLevelBuilder {
        WordLevelBuilder::new()
    }

    pub(crate) fn set_vocab(&mut self, vocab: Vocab) {
        self.vocab_r = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        self.vocab = vocab;
    }
}

impl Model for WordLevel {
    fn tokenize(&self, token: &str) -> Result<Vec<Token>> {
        if let Some(&id) = self.vocab.get(token) {
            return Ok(vec![Token::new(id, token.to_owned(), (0, token.len()))]);
        }
        match self.vocab.get(&self.unk_token) {
            Some(&id) => Ok(vec![Token::new(
                id,
                self.unk_token.clone(),
                (0, token.len()),
            )]),
            None => Err(Error::Model(format!(
                "WordLevel unk token {:?} is not in the vocabulary",
                self.unk_token
            ))),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Serialize for WordLevel {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("WordLevel", 3)?;
        s.serialize_field("type", "WordLevel")?;
        s.serialize_field("vocab", &OrderedVocab(&self.vocab_r))?;
        s.serialize_field("unk_token", &self.unk_token)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for WordLevel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_map(WordLevelVisitor)
    }
}

struct WordLevelVisitor;

impl<'de> Visitor<'de> for WordLevelVisitor {
    type Value = WordLevel;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a WordLevel model")
    }

    fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> std::result::Result<WordLevel, V::Error> {
        let mut builder = WordLevelBuilder::new();
        let (mut has_vocab, mut has_unk) = (false, false);
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    let ty: String = map.next_value()?;
                    if ty != "WordLevel" {
                        return Err(de::Error::invalid_value(
                            de::Unexpected::Str(&ty),
                            &"WordLevel",
                        ));
                    }
                }
                "vocab" => {
                    builder = builder.vocab(map.next_value()?);
                    has_vocab = true;
                }
                "unk_token" => {
                    builder = builder.unk_token(map.next_value::<String>()?);
                    has_unk = true;
                }
                _ => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        if !has_vocab {
            return Err(de::Error::missing_field("vocab"));
        }
        if !has_unk {
            return Err(de::Error::missing_field("unk_token"));
        }
        builder.build().map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_and_unk() {
        let m = WordLevel::builder()
            .vocab(
                [("<unk>".to_string(), 0), ("a".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .build()
            .unwrap();
        assert_eq!(
            m.tokenize("a").unwrap(),
            vec![Token::new(1, "a".into(), (0, 1))]
        );
        assert_eq!(
            m.tokenize("zz").unwrap(),
            vec![Token::new(0, "<unk>".into(), (0, 2))]
        );
        let no_unk = WordLevel::builder()
            .vocab([("a".to_string(), 0)].into_iter().collect())
            .build()
            .unwrap();
        assert!(no_unk.tokenize("b").is_err());
    }

    #[test]
    fn serde_matches_hf_shape() {
        let m = WordLevel::builder()
            .vocab(
                [("<unk>".to_string(), 0), ("a".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .build()
            .unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(
            json,
            r#"{"type":"WordLevel","vocab":{"<unk>":0,"a":1},"unk_token":"<unk>"}"#
        );
        assert_eq!(serde_json::from_str::<WordLevel>(&json).unwrap(), m);
    }
}
