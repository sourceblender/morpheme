//! The `Encoding` produced by [`crate::Tokenizer::encode`].

use serde::{Deserialize, Serialize};

/// A span of the original text mapped to token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    /// Token ids, in model order.
    pub ids: Vec<u32>,
    /// The token strings, parallel to `ids`.
    pub tokens: Vec<String>,
    /// Byte offsets `(start, end)` into the original input for each id.
    pub offsets: Vec<(usize, usize)>,
    /// Segment type ids (0 for sentence A, 1 for sentence B). For
    /// single-sentence encodings, all entries are 0. Used by
    /// BERT-style models with segment embeddings.
    #[serde(default)]
    pub type_ids: Vec<u32>,
}

impl Encoding {
    /// Construct an `Encoding` with empty `type_ids` (the common
    /// single-sentence case).
    pub fn new(ids: Vec<u32>, tokens: Vec<String>, offsets: Vec<(usize, usize)>) -> Self {
        let type_ids = vec![0; ids.len()];
        Self {
            ids,
            tokens,
            offsets,
            type_ids,
        }
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if the encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}
