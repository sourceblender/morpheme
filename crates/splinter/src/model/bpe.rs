//! Byte-Pair Encoding model.
//!
//! Holds a vocabulary and an *ordered* list of merge pairs. Pair `i`
//! has priority `i` — lower index = higher priority. Merging proceeds
//! by repeatedly replacing the highest-priority adjacent pair in a
//! pre-token until no priority pair remains.
//!
//! v0.1 supports two HF compatibility flags:
//!
//! - `byte_fallback` — for OOV characters, fall back to the
//!   byte-to-unicode mapping used by GPT-2 / RoBERTa. Without this,
//!   OOV characters produce an error.
//! - `dropout` — stored for round-tripping with HF JSON. Not applied
//!   at inference in v0.1.

use std::collections::HashMap;

use crate::error::{Error as SplinterError, Result};
use crate::model::Model;
use crate::pre_tokenizer::PreToken;
use crate::vocab::Vocab;

/// Byte-Pair Encoding model.
#[derive(Debug, Clone)]
pub struct Bpe {
    /// Vocabulary mapping `token -> id`. Each pre-token symbol after
    /// the merge loop is looked up here.
    pub vocab: Vocab,
    /// Merge table, ordered by priority (index 0 = highest).
    merges: Vec<(String, String)>,
    /// `(a, b) -> priority` lookup. `a` and `b` are looked up in
    /// `vocab` first; their ids form the key.
    merge_ranks: HashMap<(u32, u32), usize>,
    /// End-of-word marker (GPT-2 uses `</w>`).
    end_of_word_suffix: String,
    /// BPE training dropout probability. Stored for round-tripping
    /// with HF JSON. Not applied at inference in v0.1.
    pub dropout: Option<f32>,
    /// Byte-fallback table — `byte -> mapped char`. When a character
    /// in the input is missing from the vocab, the underlying UTF-8
    /// bytes are emitted as their mapped Unicode codepoints, which
    /// are guaranteed to be in the vocab. Used by GPT-2 / RoBERTa.
    byte_fallback: Option<[char; 256]>,
}

impl Bpe {
    /// Construct a BPE model from a vocabulary and an ordered list of
    /// merge pairs.
    ///
    /// # Panics
    /// Panics if a merge pair's symbols are not in the vocabulary.
    /// This is a programmer error in the input data, not a runtime
    /// condition.
    pub fn new(
        vocab: Vocab,
        merges: Vec<(String, String)>,
        end_of_word_suffix: impl Into<String>,
    ) -> Self {
        Self::builder(vocab, merges, end_of_word_suffix).build()
    }

    /// Start building a BPE model with optional dropout and
    /// byte-fallback support.
    pub fn builder(
        vocab: Vocab,
        merges: Vec<(String, String)>,
        end_of_word_suffix: impl Into<String>,
    ) -> BpeBuilder {
        BpeBuilder {
            vocab,
            merges,
            end_of_word_suffix: end_of_word_suffix.into(),
            dropout: None,
            byte_fallback: None,
        }
    }

    /// Number of merge pairs.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// The end-of-word suffix (GPT-2 uses `</w>`).
    pub fn end_of_word_suffix(&self) -> &str {
        &self.end_of_word_suffix
    }

    /// Iterate merge pairs in *insertion order* (i.e. the order they
    /// were supplied to [`Bpe::new`]). The index of each pair is the
    /// priority — lower index = higher priority.
    pub fn merges_iter(&self) -> impl Iterator<Item = &(String, String)> {
        self.merges.iter()
    }

    /// Compute the byte-to-unicode table (used by GPT-2 / RoBERTa).
    pub fn byte_fallback_table() -> [char; 256] {
        byte_level_table()
    }

    /// Apply BPE to a single pre-token string.
    fn apply(&self, word: &str) -> Result<Vec<String>> {
        if word.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Split into characters, append the end-of-word suffix to
        //    *every* character (matches GPT-2 / RoBERTa convention).
        //    BPE merges then strip the suffix from merged intermediates
        //    and only the final merged token carries `</w>`.
        let mut symbols: Vec<String> = Vec::with_capacity(word.len());
        let chars: Vec<char> = word.chars().collect();
        for c in chars.iter() {
            let mut s = String::new();
            s.push(*c);
            s.push_str(&self.end_of_word_suffix);
            symbols.push(s);
        }

        // 2. Map every symbol to its vocab id. If `byte_fallback` is
        //    enabled, OOV characters are replaced by their byte-level
        //    representations (one or more chars per OOV symbol).
        let mut ids: Vec<u32> = Vec::with_capacity(symbols.len());
        for (idx, s) in symbols.iter().enumerate() {
            let is_last = idx + 1 == symbols.len();
            if let Some(id) = self.vocab.token_to_id(s) {
                ids.push(id);
                continue;
            }
            if let Some(table) = self.byte_fallback {
                let bytes = s.as_bytes();
                for (b_idx, &byte) in bytes.iter().enumerate() {
                    let mapped_str = table[byte as usize].to_string();
                    let is_last_byte = b_idx + 1 == bytes.len();
                    // Only the last byte of the last symbol gets the
                    // `</w>` suffix — that's the only position where
                    // the suffix makes sense.
                    let try_with_suffix = is_last && is_last_byte;
                    let lookup = if try_with_suffix {
                        let mut with_suffix = mapped_str.clone();
                        with_suffix.push_str(&self.end_of_word_suffix);
                        match self.vocab.token_to_id(&with_suffix) {
                            Some(id) => id,
                            None => self
                                .vocab
                                .token_to_id(&mapped_str)
                                .ok_or_else(|| SplinterError::UnknownToken(mapped_str.clone()))?,
                        }
                    } else {
                        self.vocab
                            .token_to_id(&mapped_str)
                            .ok_or_else(|| SplinterError::UnknownToken(mapped_str.clone()))?
                    };
                    ids.push(lookup);
                }
            } else {
                return Err(SplinterError::UnknownToken(s.clone()));
            }
        }

        // 3. Merge loop: scan for the lowest-rank adjacent pair, merge
        //    it, repeat until no rank exists.
        while ids.len() >= 2 {
            let mut best_rank: Option<usize> = None;
            let mut best_idx: Option<usize> = None;
            for i in 0..ids.len() - 1 {
                let key = (ids[i], ids[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    match best_rank {
                        None => {
                            best_rank = Some(rank);
                            best_idx = Some(i);
                        }
                        Some(current) if rank < current => {
                            best_rank = Some(rank);
                            best_idx = Some(i);
                        }
                        _ => {}
                    }
                }
            }

            let Some(idx) = best_idx else { break };
            let merged_id = {
                let a = self.vocab.id_to_token(ids[idx])?;
                let b = self.vocab.id_to_token(ids[idx + 1])?;
                // Strip the trailing `</w>` from `a` before
                // concatenating. `b` retains its `</w>` so the merged
                // token also ends with `</w>` (only the final
                // symbol's suffix survives).
                let a_stripped = a.strip_suffix(&self.end_of_word_suffix).unwrap_or(a);
                let merged = format!("{a_stripped}{b}");
                self.vocab
                    .token_to_id(&merged)
                    .ok_or(SplinterError::UnknownToken(merged))?
            };

            ids[idx] = merged_id;
            ids.remove(idx + 1);
        }

        // 4. Convert ids back to token strings.
        let tokens = ids
            .into_iter()
            .map(|id| self.vocab.id_to_token(id).map(str::to_owned))
            .collect::<Result<Vec<String>>>()?;
        Ok(tokens)
    }
}

impl Model for Bpe {
    fn tokenize(&self, pre_token: PreToken<'_>) -> Result<Vec<String>> {
        self.apply(pre_token.text.as_str())
    }
}

/// Builder for [`Bpe`].
pub struct BpeBuilder {
    vocab: Vocab,
    merges: Vec<(String, String)>,
    end_of_word_suffix: String,
    dropout: Option<f32>,
    byte_fallback: Option<[char; 256]>,
}

impl BpeBuilder {
    /// Set BPE training dropout probability.
    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout = Some(p);
        self
    }

    /// Enable byte-fallback for OOV characters. Pass the table
    /// produced by [`Bpe::byte_fallback_table`] (or your own).
    pub fn byte_fallback(mut self, table: [char; 256]) -> Self {
        self.byte_fallback = Some(table);
        self
    }

    /// Build the [`Bpe`].
    pub fn build(self) -> Bpe {
        let mut merge_ranks = HashMap::with_capacity(self.merges.len());
        for (rank, (a, b)) in self.merges.iter().enumerate() {
            let id_a = self
                .vocab
                .token_to_id(a)
                .unwrap_or_else(|| panic!("merge symbol {a:?} missing from vocab"));
            let id_b = self
                .vocab
                .token_to_id(b)
                .unwrap_or_else(|| panic!("merge symbol {b:?} missing from vocab"));
            merge_ranks.insert((id_a, id_b), rank);
        }
        Bpe {
            vocab: self.vocab,
            merges: self.merges,
            merge_ranks,
            end_of_word_suffix: self.end_of_word_suffix,
            dropout: self.dropout,
            byte_fallback: self.byte_fallback,
        }
    }
}

/// Compute the byte-to-unicode table used by GPT-2 / RoBERTa byte
/// fallback. Each byte value `b` (0..=255) maps to a unique Unicode
/// character; printable ASCII bytes map to themselves.
fn byte_level_table() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in b'\xA1'..=b'\xAC' {
        bs.push(b as u32);
    }
    for b in b'\xAE'..=b'\xFF' {
        bs.push(b as u32);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (i, &c) in cs.iter().enumerate() {
        table[bs[i] as usize] = char::from_u32(c).unwrap();
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> Bpe {
        let tokens: Vec<String> = vec![
            "<unk>".into(),
            "a</w>".into(),
            "b</w>".into(),
            "c</w>".into(),
            "ab</w>".into(),
        ];
        let vocab = Vocab::from_tokens(tokens).unwrap();
        let merges = vec![("a</w>".to_string(), "b</w>".to_string())];
        Bpe::new(vocab, merges, "</w>")
    }

    #[test]
    fn merges_bigrams() {
        let m = build();
        let toks = m.apply("ab").unwrap();
        // 'a' + 'b' merges to 'ab</w>'.
        assert_eq!(toks, vec!["ab</w>"]);
    }

    #[test]
    fn unknown_char_without_fallback_errors() {
        let m = build();
        // 'z' is not in vocab and byte_fallback is disabled.
        assert!(m.apply("z").is_err());
    }

    #[test]
    fn unknown_char_with_fallback_resolves() {
        let table = byte_level_table();
        let mapped = table[b'z' as usize];
        let mapped_str = mapped.to_string();
        // Vocab includes the byte-fallback char for byte 'z' (with
        // and without `</w>`).
        let tokens: Vec<String> = vec![
            "<unk>".into(),
            mapped_str.clone(),
            format!("{mapped_str}</w>"),
        ];
        let vocab = Vocab::from_tokens(tokens).unwrap();
        let m = Bpe::builder(vocab, vec![], "</w>")
            .byte_fallback(table)
            .build();
        let toks = m.apply("z").unwrap();
        assert_eq!(toks, vec![format!("{mapped_str}</w>")]);
    }
}
