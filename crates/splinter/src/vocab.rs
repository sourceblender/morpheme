//! Bidirectional `token <-> id` lookup.

use rustc_hash::FxHashMap;

use crate::error::{Error as SplinterError, Result};

/// A vocabulary: ordered list of tokens with O(1) `token -> id` lookup.
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    entries: Vec<String>,
    index: FxHashMap<String, u32>,
}

impl Vocab {
    /// Build a vocabulary from a list of tokens. Order is significant —
    /// it defines the id of each token. Duplicate tokens are an error.
    pub fn from_tokens<I, T>(tokens: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let mut entries: Vec<String> = Vec::new();
        let mut index: FxHashMap<String, u32> = FxHashMap::default();
        for t in tokens {
            let s: String = t.into();
            if index.contains_key(&s) {
                return Err(SplinterError::UnknownToken(s));
            }
            let id = entries.len() as u32;
            index.insert(s.clone(), id);
            entries.push(s);
        }
        Ok(Self { entries, index })
    }

    /// Number of entries in the vocabulary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the vocabulary has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a token's id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.index.get(token).copied()
    }

    /// Look up an id's token.
    pub fn id_to_token(&self, id: u32) -> Result<&str> {
        self.entries
            .get(id as usize)
            .map(String::as_str)
            .ok_or(SplinterError::UnknownId {
                id,
                vocab_size: self.entries.len(),
            })
    }

    /// Iterate `(id, token)` pairs in id order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u32, t.as_str()))
    }
}
