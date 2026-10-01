use std::collections::HashMap;
use std::sync::RwLock;

use rustc_hash::FxHashMap;

use super::serialization::reverse_vocab;
use super::word::{MergeMap, Word};
use crate::Token;
use crate::error::{Error, Result};
use crate::traits::Model;

/// `token -> id`.
pub type Vocab = HashMap<String, u32>;
/// Ordered merges, highest priority first.
pub type Merges = Vec<(String, String)>;

const DEFAULT_CACHE_CAPACITY: usize = 10_000;
/// Words longer than this (in bytes) are not cached.
const MAX_CACHED_WORD_LEN: usize = 256;

/// Memoizes merged words. Cloning a model gives it a fresh cache.
///
/// Sharded so that parallel batch encoding doesn't serialize on a single
/// lock.
#[derive(Debug, Default)]
struct WordCache {
    capacity: usize,
    shards: Vec<RwLock<FxHashMap<String, Word>>>,
}

const CACHE_SHARDS: usize = 64;

impl WordCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            shards: (0..CACHE_SHARDS)
                .map(|_| RwLock::new(FxHashMap::default()))
                .collect(),
        }
    }

    fn shard(&self, key: &str) -> &RwLock<FxHashMap<String, Word>> {
        use std::hash::{BuildHasher, BuildHasherDefault};
        let h = BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(key);
        &self.shards[(h as usize) % self.shards.len()]
    }

    fn get(&self, key: &str) -> Option<Word> {
        self.shard(key).try_read().ok()?.get(key).cloned()
    }

    fn insert(&self, key: &str, word: &Word) {
        if key.len() >= MAX_CACHED_WORD_LEN {
            return;
        }
        let per_shard = self.capacity.div_ceil(self.shards.len());
        if let Ok(mut map) = self.shard(key).try_write() {
            if map.len() < per_shard {
                map.insert(key.to_owned(), word.clone());
            }
        }
    }

    fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut map) = shard.write() {
                map.clear();
            }
        }
    }
}

/// Builder for [`Bpe`].
#[derive(Debug, Clone)]
pub struct BpeBuilder {
    vocab: Vocab,
    merges: Merges,
    cache_capacity: usize,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
}

impl Default for BpeBuilder {
    fn default() -> Self {
        Self {
            vocab: Vocab::new(),
            merges: Vec::new(),
            cache_capacity: DEFAULT_CACHE_CAPACITY,
            dropout: None,
            unk_token: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback: false,
            ignore_merges: false,
        }
    }
}

impl BpeBuilder {
    /// A builder with HF defaults (empty vocab, no merges).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vocabulary and the ordered merges.
    #[must_use]
    pub fn vocab_and_merges(mut self, vocab: Vocab, merges: Merges) -> Self {
        self.vocab = vocab;
        self.merges = merges;
        self
    }

    /// Word cache capacity (`0` disables the cache).
    #[must_use]
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = capacity;
        self
    }

    /// BPE-dropout probability in `[0, 1]`.
    #[must_use]
    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout = Some(p);
        self
    }

    /// Token used for chars missing from the vocabulary.
    #[must_use]
    pub fn unk_token(mut self, token: impl Into<String>) -> Self {
        self.unk_token = Some(token.into());
        self
    }

    /// Prefix carried by every non-initial subword (e.g. `##`).
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.continuing_subword_prefix = Some(prefix.into());
        self
    }

    /// Suffix carried by the last subword of a word (e.g. `</w>`).
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.end_of_word_suffix = Some(suffix.into());
        self
    }

    /// Fuse consecutive unknown chars into a single unk token.
    #[must_use]
    pub fn fuse_unk(mut self, v: bool) -> Self {
        self.fuse_unk = v;
        self
    }

    /// Represent unknown chars as `<0xNN>` byte tokens when available.
    #[must_use]
    pub fn byte_fallback(mut self, v: bool) -> Self {
        self.byte_fallback = v;
        self
    }

    /// Emit a whole word directly if it is in the vocabulary.
    #[must_use]
    pub fn ignore_merges(mut self, v: bool) -> Self {
        self.ignore_merges = v;
        self
    }

    /// Validate and build.
    pub fn build(self) -> Result<Bpe> {
        if let Some(p) = self.dropout {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::Config(format!(
                    "BPE dropout must be between 0 and 1, got {p}"
                )));
            }
        }
        let vocab_r: FxHashMap<u32, String> = reverse_vocab(&self.vocab);
        let prefix = self.continuing_subword_prefix.as_deref().unwrap_or("");

        let mut merges = MergeMap::default();
        for (rank, (a, b)) in self.merges.iter().enumerate() {
            let missing =
                |t: &str| Error::Config(format!("merge token {t:?} is not in the vocabulary"));
            let a_id = *self.vocab.get(a).ok_or_else(|| missing(a))?;
            let b_id = *self.vocab.get(b).ok_or_else(|| missing(b))?;
            let b_rest = b.strip_prefix(prefix).unwrap_or(b);
            let merged = format!("{a}{b_rest}");
            let new_id = *self.vocab.get(&merged).ok_or_else(|| missing(&merged))?;
            merges.insert((a_id, b_id), (rank as u32, new_id));
        }

        Ok(Bpe {
            vocab: self.vocab,
            vocab_r,
            merges,
            merge_list: self.merges,
            cache: (self.cache_capacity > 0).then(|| WordCache::new(self.cache_capacity)),
            dropout: self.dropout,
            unk_token: self.unk_token,
            continuing_subword_prefix: self.continuing_subword_prefix,
            end_of_word_suffix: self.end_of_word_suffix,
            fuse_unk: self.fuse_unk,
            byte_fallback: self.byte_fallback,
            ignore_merges: self.ignore_merges,
        })
    }
}

/// Byte-Pair Encoding model.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use splinter::models::Bpe;
/// use splinter::Model;
///
/// let vocab: HashMap<String, u32> =
///     [("a", 0), ("b", 1), ("c", 2), ("ab", 3), ("abc", 4)].map(|(t, i)| (t.to_string(), i)).into();
/// let merges = vec![("a".into(), "b".into()), ("ab".into(), "c".into())];
/// let bpe = Bpe::builder().vocab_and_merges(vocab, merges).build()?;
///
/// let tokens: Vec<String> = bpe.tokenize("abcab")?.into_iter().map(|t| t.value).collect();
/// assert_eq!(tokens, ["abc", "ab"]);
/// # Ok::<(), splinter::Error>(())
/// ```
pub struct Bpe {
    pub(crate) vocab: Vocab,
    pub(crate) vocab_r: FxHashMap<u32, String>,
    pub(crate) merges: MergeMap,
    /// The merges as given (or trained), in priority order. Kept verbatim
    /// for serialization: rebuilding them from ids is lossy when several
    /// tokens share an id.
    pub(crate) merge_list: Merges,
    cache: Option<WordCache>,
    /// BPE-dropout probability (`None` = deterministic).
    pub(crate) dropout: Option<f32>,
    /// Token for unknown chars.
    pub(crate) unk_token: Option<String>,
    /// Prefix of non-initial subwords.
    pub(crate) continuing_subword_prefix: Option<String>,
    /// Suffix of word-final subwords.
    pub(crate) end_of_word_suffix: Option<String>,
    /// Fuse consecutive unknown chars.
    pub(crate) fuse_unk: bool,
    /// `<0xNN>` byte fallback for unknown chars.
    pub(crate) byte_fallback: bool,
    /// Emit in-vocabulary words without merging.
    pub(crate) ignore_merges: bool,
}

impl std::fmt::Debug for Bpe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bpe")
            .field("dropout", &self.dropout)
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("end_of_word_suffix", &self.end_of_word_suffix)
            .field("fuse_unk", &self.fuse_unk)
            .field("byte_fallback", &self.byte_fallback)
            .field("ignore_merges", &self.ignore_merges)
            .field("vocab", &self.vocab.len())
            .field("merges", &self.merges.len())
            .finish()
    }
}

impl PartialEq for Bpe {
    fn eq(&self, other: &Self) -> bool {
        self.vocab == other.vocab
            && self.merges == other.merges
            && self.dropout == other.dropout
            && self.unk_token == other.unk_token
            && self.continuing_subword_prefix == other.continuing_subword_prefix
            && self.end_of_word_suffix == other.end_of_word_suffix
            && self.fuse_unk == other.fuse_unk
            && self.byte_fallback == other.byte_fallback
            && self.ignore_merges == other.ignore_merges
    }
}

impl Clone for Bpe {
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            vocab_r: self.vocab_r.clone(),
            merges: self.merges.clone(),
            merge_list: self.merge_list.clone(),
            cache: self.cache.as_ref().map(|c| WordCache::new(c.capacity)),
            dropout: self.dropout,
            unk_token: self.unk_token.clone(),
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            fuse_unk: self.fuse_unk,
            byte_fallback: self.byte_fallback,
            ignore_merges: self.ignore_merges,
        }
    }
}

impl Default for Bpe {
    fn default() -> Self {
        BpeBuilder::default().build().expect("default BPE is valid")
    }
}

impl Bpe {
    /// Start building a model.
    pub fn builder() -> BpeBuilder {
        BpeBuilder::new()
    }

    /// Build from a vocabulary and merges with default options.
    pub fn new(vocab: Vocab, merges: Merges) -> Result<Self> {
        Self::builder().vocab_and_merges(vocab, merges).build()
    }

    /// The unknown token, if any.
    pub fn unk_token(&self) -> Option<&str> {
        self.unk_token.as_deref()
    }

    /// The continuing-subword prefix, if any.
    pub fn continuing_subword_prefix(&self) -> Option<&str> {
        self.continuing_subword_prefix.as_deref()
    }

    /// The end-of-word suffix, if any.
    pub fn end_of_word_suffix(&self) -> Option<&str> {
        self.end_of_word_suffix.as_deref()
    }

    /// BPE-dropout probability (`None`: deterministic).
    pub fn dropout(&self) -> Option<f32> {
        self.dropout
    }

    /// Whether consecutive unknown chars are fused into one unknown token.
    pub fn fuse_unk(&self) -> bool {
        self.fuse_unk
    }

    /// Whether unknown chars fall back to `<0xNN>` byte tokens.
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }

    /// Whether words already in the vocabulary skip merging.
    pub fn ignore_merges(&self) -> bool {
        self.ignore_merges
    }

    /// Merges ordered by priority, as token strings.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merge_list
    }

    /// Merges ordered by priority, rebuilt from the merge map.
    fn merges_from_map(&self) -> Merges {
        let mut ranked: Vec<(&super::Pair, u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, *rank))
            .collect();
        ranked.sort_unstable_by_key(|(_, rank)| *rank);
        ranked
            .into_iter()
            .map(|(&(a, b), _)| {
                (
                    self.vocab_r.get(&a).cloned().unwrap_or_default(),
                    self.vocab_r.get(&b).cloned().unwrap_or_default(),
                )
            })
            .collect()
    }

    /// Drop all cached words.
    pub fn clear_cache(&self) {
        if let Some(c) = &self.cache {
            c.clear();
        }
    }

    /// Replace the vocabulary/merges and options from a trainer.
    pub(crate) fn set_trained(
        &mut self,
        vocab: Vocab,
        merges: MergeMap,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
    ) {
        self.vocab_r = reverse_vocab(&vocab);
        self.vocab = vocab;
        self.merges = merges;
        // Trained vocabularies have unique ids, so this is exact.
        self.merge_list = self.merges_from_map();
        self.continuing_subword_prefix = continuing_subword_prefix;
        self.end_of_word_suffix = end_of_word_suffix;
        self.clear_cache();
    }

    fn unk_id(&self, unk: &str) -> Result<u32> {
        self.vocab
            .get(unk)
            .copied()
            .ok_or_else(|| Error::Model(format!("unk token {unk:?} is not in the vocabulary")))
    }

    fn merge_word(&self, w: &str) -> Result<Word> {
        let mut word = Word::with_capacity(w.len());
        // Pending unknown run: (unk id, byte length).
        let mut unk: Option<(u32, usize)> = None;
        let mut chars = w.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let is_first = i == 0;
            let is_last = chars.peek().is_none();
            let byte_len = c.len_utf8();
            let piece = &w[i..i + byte_len];

            let mut symbol = String::with_capacity(byte_len + 8);
            if !is_first {
                if let Some(p) = &self.continuing_subword_prefix {
                    symbol.push_str(p);
                }
            }
            symbol.push_str(piece);
            if is_last {
                if let Some(s) = &self.end_of_word_suffix {
                    symbol.push_str(s);
                }
            }

            if let Some(&id) = self.vocab.get(&symbol) {
                if let Some((unk_id, unk_len)) = unk.take() {
                    word.add(unk_id, unk_len);
                }
                word.add(id, byte_len);
                continue;
            }

            if self.byte_fallback {
                let byte_ids: Option<Vec<u32>> = symbol
                    .bytes()
                    .map(|b| self.vocab.get(&format!("<0x{b:02X}>")).copied())
                    .collect();
                if let Some(ids) = byte_ids {
                    for id in ids {
                        word.add(id, 1);
                    }
                    continue;
                }
            }

            if let Some(unk_token) = &self.unk_token {
                unk = match (unk, self.fuse_unk) {
                    (Some((id, len)), true) => Some((id, len + byte_len)),
                    (Some((id, len)), false) => {
                        word.add(id, len);
                        Some((self.unk_id(unk_token)?, byte_len))
                    }
                    (None, _) => Some((self.unk_id(unk_token)?, byte_len)),
                };
            }
        }
        if let Some((id, len)) = unk {
            word.add(id, len);
        }
        word.merge_all(&self.merges, self.dropout);
        Ok(word)
    }

    fn word_to_tokens(&self, word: &Word) -> Result<Vec<Token>> {
        word.ids()
            .zip(word.offsets())
            .map(|(id, offsets)| {
                let value = self.vocab_r.get(&id).cloned().ok_or(Error::UnknownId(id))?;
                Ok(Token::new(id, value, offsets))
            })
            .collect()
    }
}

impl Model for Bpe {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(Vec::new());
        }
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(sequence) {
                return Ok(vec![Token::new(
                    id,
                    sequence.to_owned(),
                    (0, sequence.len()),
                )]);
            }
        }
        let deterministic = matches!(self.dropout, None | Some(0.0));
        if deterministic {
            if let Some(cache) = &self.cache {
                if let Some(word) = cache.get(sequence) {
                    return self.word_to_tokens(&word);
                }
                let word = self.merge_word(sequence)?;
                cache.insert(sequence, &word);
                return self.word_to_tokens(&word);
            }
        }
        let word = self.merge_word(sequence)?;
        self.word_to_tokens(&word)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(entries: &[(&str, u32)]) -> Vocab {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn values(tokens: &[Token]) -> Vec<(&str, u32, (usize, usize))> {
        tokens
            .iter()
            .map(|t| (t.value.as_str(), t.id, t.offsets))
            .collect()
    }

    #[test]
    fn unk_and_fuse_unk() {
        let v = vocab(&[("<unk>", 0), ("a", 1), ("b", 2)]);
        let bpe = Bpe::builder()
            .vocab_and_merges(v.clone(), vec![])
            .unk_token("<unk>")
            .build()
            .unwrap();
        let t = bpe.tokenize("cca").unwrap();
        assert_eq!(
            values(&t),
            vec![("<unk>", 0, (0, 1)), ("<unk>", 0, (1, 2)), ("a", 1, (2, 3))]
        );
        let fused = Bpe::builder()
            .vocab_and_merges(v, vec![])
            .unk_token("<unk>")
            .fuse_unk(true)
            .build()
            .unwrap();
        let t = fused.tokenize("ccaé").unwrap();
        assert_eq!(
            values(&t),
            vec![("<unk>", 0, (0, 2)), ("a", 1, (2, 3)), ("<unk>", 0, (3, 5))]
        );
    }

    #[test]
    fn missing_unk_and_no_fallback_drops_chars() {
        // HF behavior: without unk_token, unknown chars are silently
        // skipped.
        let bpe = Bpe::new(vocab(&[("a", 0)]), vec![]).unwrap();
        let t = bpe.tokenize("xa").unwrap();
        assert_eq!(values(&t), vec![("a", 0, (0, 1))]);
    }

    #[test]
    fn byte_fallback_uses_hex_tokens() {
        let v = vocab(&[("<unk>", 0), ("<0xC3>", 1), ("<0xA9>", 2), ("a", 3)]);
        let bpe = Bpe::builder()
            .vocab_and_merges(v, vec![])
            .unk_token("<unk>")
            .byte_fallback(true)
            .build()
            .unwrap();
        let t = bpe.tokenize("aé").unwrap();
        assert_eq!(
            values(&t),
            vec![
                ("a", 3, (0, 1)),
                ("<0xC3>", 1, (1, 2)),
                ("<0xA9>", 2, (2, 3))
            ]
        );
        // A char whose bytes are not all present falls back to unk.
        let t = bpe.tokenize("ü").unwrap();
        assert_eq!(values(&t), vec![("<unk>", 0, (0, 2))]);
    }

    #[test]
    fn merges_with_prefix_and_suffix() {
        let v = vocab(&[
            ("a", 0),
            ("##b", 1),
            ("##c</w>", 2),
            ("ab", 3),
            ("ab##c</w>", 4),
            ("abc</w>", 5),
        ]);
        let merges = vec![
            ("a".to_string(), "##b".to_string()),
            ("ab".to_string(), "##c</w>".to_string()),
        ];
        let bpe = Bpe::builder()
            .vocab_and_merges(v, merges)
            .continuing_subword_prefix("##")
            .end_of_word_suffix("</w>")
            .build()
            .unwrap();
        let t = bpe.tokenize("abc").unwrap();
        assert_eq!(values(&t), vec![("abc</w>", 5, (0, 3))]);
    }

    #[test]
    fn bad_merge_is_an_error_not_a_panic() {
        let err = Bpe::new(vocab(&[("a", 0)]), vec![("a".into(), "z".into())]);
        assert!(err.is_err());
        let err = Bpe::new(vocab(&[("a", 0), ("b", 1)]), vec![("a".into(), "b".into())]);
        assert!(err.is_err(), "merged token missing");
        assert!(Bpe::builder().dropout(1.5).build().is_err());
    }

    #[test]
    fn ignore_merges_short_circuits() {
        let v = vocab(&[("a", 0), ("b", 1), ("ab", 2), ("aab", 3)]);
        let bpe = Bpe::builder()
            .vocab_and_merges(v, vec![("a".into(), "b".into())])
            .ignore_merges(true)
            .build()
            .unwrap();
        assert_eq!(
            values(&bpe.tokenize("aab").unwrap()),
            vec![("aab", 3, (0, 3))]
        );
    }

    #[test]
    fn cache_returns_same_result() {
        let v = vocab(&[("a", 0), ("b", 1), ("ab", 2)]);
        let bpe = Bpe::new(v, vec![("a".into(), "b".into())]).unwrap();
        let first = bpe.tokenize("abab").unwrap();
        let second = bpe.tokenize("abab").unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
    }
}
