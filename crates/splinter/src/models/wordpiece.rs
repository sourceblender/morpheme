//! WordPiece (BERT): greedy longest-match-first subword lookup.

use std::collections::HashMap;

use rustc_hash::FxHashMap;
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Token;
use crate::error::{Error, Result};
use crate::models::bpe::{Bpe, OrderedVocab, reverse_vocab};
use crate::traits::Model;

/// `token -> id`.
pub type Vocab = HashMap<String, u32>;

/// Builder for [`WordPiece`].
#[derive(Debug, Clone)]
pub struct WordPieceBuilder {
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl Default for WordPieceBuilder {
    fn default() -> Self {
        Self {
            vocab: Vocab::new(),
            unk_token: "[UNK]".into(),
            continuing_subword_prefix: "##".into(),
            max_input_chars_per_word: 100,
        }
    }
}

impl WordPieceBuilder {
    /// A builder with BERT defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vocabulary.
    #[must_use]
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.vocab = vocab;
        self
    }

    /// Token for words that cannot be tokenized (default `[UNK]`).
    #[must_use]
    pub fn unk_token(mut self, token: impl Into<String>) -> Self {
        self.unk_token = token.into();
        self
    }

    /// Prefix of non-initial subwords (default `##`).
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.continuing_subword_prefix = prefix.into();
        self
    }

    /// Words longer than this many chars become `unk_token` (default
    /// 100).
    #[must_use]
    pub fn max_input_chars_per_word(mut self, n: usize) -> Self {
        self.max_input_chars_per_word = n;
        self
    }

    /// Build the model.
    pub fn build(self) -> Result<WordPiece> {
        let vocab_r = reverse_vocab(&self.vocab);
        Ok(WordPiece {
            vocab: self.vocab,
            vocab_r,
            unk_token: self.unk_token,
            continuing_subword_prefix: self.continuing_subword_prefix,
            max_input_chars_per_word: self.max_input_chars_per_word,
        })
    }
}

/// WordPiece model.
#[derive(Clone, PartialEq, Eq)]
pub struct WordPiece {
    vocab: Vocab,
    vocab_r: FxHashMap<u32, String>,
    /// Token emitted for words that cannot be tokenized.
    pub unk_token: String,
    /// Prefix of non-initial subwords.
    pub continuing_subword_prefix: String,
    /// Longer words become `unk_token`.
    pub max_input_chars_per_word: usize,
}

impl std::fmt::Debug for WordPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WordPiece")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        WordPieceBuilder::default()
            .build()
            .expect("default WordPiece is valid")
    }
}

impl WordPiece {
    /// Start building a model.
    pub fn builder() -> WordPieceBuilder {
        WordPieceBuilder::new()
    }

    /// Build a WordPiece model from a trained BPE model's vocabulary
    /// (used by the WordPiece trainer).
    pub fn from_bpe(bpe: &Bpe) -> Self {
        let mut wp = WordPiece::builder()
            .vocab(bpe.get_vocab())
            .build()
            .expect("WordPiece build is infallible");
        if let Some(unk) = bpe.get_unk_token() {
            wp.unk_token = unk.to_owned();
        }
        if let Some(prefix) = bpe.get_continuing_subword_prefix() {
            wp.continuing_subword_prefix = prefix.to_owned();
        }
        wp
    }

    pub(crate) fn set_vocab(&mut self, vocab: Vocab) {
        self.vocab_r = reverse_vocab(&vocab);
        self.vocab = vocab;
    }

    fn unk(&self, len: usize) -> Result<Vec<Token>> {
        let id = *self.vocab.get(&self.unk_token).ok_or_else(|| {
            Error::Model(format!(
                "WordPiece unk token {:?} is not in the vocabulary",
                self.unk_token
            ))
        })?;
        Ok(vec![Token::new(id, self.unk_token.clone(), (0, len))])
    }
}

impl Model for WordPiece {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.chars().count() > self.max_input_chars_per_word {
            return self.unk(sequence.len());
        }
        let mut tokens = Vec::new();
        let mut start = 0;
        let mut candidate = String::with_capacity(sequence.len() + 8);
        while start < sequence.len() {
            let mut end = sequence.len();
            let mut found = None;
            while start < end {
                candidate.clear();
                if start > 0 {
                    candidate.push_str(&self.continuing_subword_prefix);
                }
                candidate.push_str(&sequence[start..end]);
                if let Some(&id) = self.vocab.get(&candidate) {
                    found = Some(Token::new(id, candidate.clone(), (start, end)));
                    break;
                }
                end -= sequence[start..end]
                    .chars()
                    .next_back()
                    .map_or(1, char::len_utf8);
            }
            match found {
                Some(t) => {
                    tokens.push(t);
                    start = end;
                }
                None => return self.unk(sequence.len()),
            }
        }
        Ok(tokens)
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

impl Serialize for WordPiece {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("WordPiece", 5)?;
        s.serialize_field("type", "WordPiece")?;
        s.serialize_field("unk_token", &self.unk_token)?;
        s.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        s.serialize_field("max_input_chars_per_word", &self.max_input_chars_per_word)?;
        s.serialize_field("vocab", &OrderedVocab(&self.vocab))?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for WordPiece {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_map(WordPieceVisitor)
    }
}

struct WordPieceVisitor;

impl<'de> Visitor<'de> for WordPieceVisitor {
    type Value = WordPiece;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a WordPiece model")
    }

    fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> std::result::Result<WordPiece, V::Error> {
        let mut builder = WordPieceBuilder::new();
        let (mut unk, mut prefix, mut max, mut vocab) = (false, false, false, false);
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    let ty: String = map.next_value()?;
                    if ty != "WordPiece" {
                        return Err(de::Error::invalid_value(
                            de::Unexpected::Str(&ty),
                            &"WordPiece",
                        ));
                    }
                }
                "unk_token" => {
                    builder = builder.unk_token(map.next_value::<String>()?);
                    unk = true;
                }
                "continuing_subword_prefix" => {
                    builder = builder.continuing_subword_prefix(map.next_value::<String>()?);
                    prefix = true;
                }
                "max_input_chars_per_word" => {
                    builder = builder.max_input_chars_per_word(map.next_value()?);
                    max = true;
                }
                "vocab" => {
                    builder = builder.vocab(map.next_value()?);
                    vocab = true;
                }
                _ => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        for (present, name) in [
            (unk, "unk_token"),
            (prefix, "continuing_subword_prefix"),
            (max, "max_input_chars_per_word"),
            (vocab, "vocab"),
        ] {
            if !present {
                return Err(de::Error::missing_field(name));
            }
        }
        builder.build().map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> WordPiece {
        let vocab: Vocab = [
            ("[UNK]", 0),
            ("un", 1),
            ("##aff", 2),
            ("##able", 3),
            ("aff", 4),
            ("é", 5),
            ("##é", 6),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        WordPiece::builder().vocab(vocab).build().unwrap()
    }

    fn values(t: &[Token]) -> Vec<(&str, (usize, usize))> {
        t.iter().map(|t| (t.value.as_str(), t.offsets)).collect()
    }

    #[test]
    fn greedy_longest_match() {
        let t = model().tokenize("unaffable").unwrap();
        assert_eq!(
            values(&t),
            vec![("un", (0, 2)), ("##aff", (2, 5)), ("##able", (5, 9))]
        );
        let t = model().tokenize("éé").unwrap();
        assert_eq!(values(&t), vec![("é", (0, 2)), ("##é", (2, 4))]);
    }

    #[test]
    fn unmatched_piece_makes_whole_word_unk() {
        let t = model().tokenize("unaffx").unwrap();
        assert_eq!(values(&t), vec![("[UNK]", (0, 6))]);
    }

    #[test]
    fn long_word_is_unk() {
        let m = WordPiece::builder()
            .vocab(model().get_vocab())
            .max_input_chars_per_word(3)
            .build()
            .unwrap();
        assert_eq!(
            values(&m.tokenize("unaff").unwrap()),
            vec![("[UNK]", (0, 5))]
        );
    }

    #[test]
    fn missing_unk_is_error() {
        let m = WordPiece::builder()
            .vocab([("a".to_string(), 0)].into_iter().collect())
            .build()
            .unwrap();
        assert!(m.tokenize("b").is_err());
    }

    #[test]
    fn serde_matches_hf_shape() {
        let m = WordPiece::builder()
            .vocab(
                [("[UNK]".to_string(), 0), ("a".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .build()
            .unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(
            json,
            r###"{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[UNK]":0,"a":1}}"###
        );
        let back: WordPiece = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        // Legacy files omit "type".
        let legacy = r###"{"unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[UNK]":0}}"###;
        assert!(serde_json::from_str::<WordPiece>(legacy).is_ok());
    }
}
