//! The [`Unigram`] model.

use std::collections::HashMap;
use std::sync::RwLock;

use rustc_hash::FxHashMap;
use serde::de::Error as _;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::lattice::Lattice;
use super::trie::Trie;
use crate::Token;
use crate::error::{Error, Result};
use crate::traits::Model;

/// Score penalty (below the lowest piece score) for unknown chars.
pub(crate) const UNK_PENALTY: f64 = 10.0;

/// Sentences at least this long (bytes) are not cached.
const CACHE_MAX_LENGTH: usize = 256;
/// Maximum number of cached sentences.
const CACHE_CAPACITY: usize = 10_000;
/// Number of cache shards (same as the BPE word cache).
const CACHE_SHARDS: usize = 64;

/// Memoizes encoded sentences. Cloning a model gives it a fresh cache.
///
/// Sharded so that parallel batch encoding doesn't serialize on a single
/// lock: lookups and inserts are non-blocking (`try_read` / `try_write`)
/// and simply miss when the shard is busy.
#[derive(Debug)]
struct SentenceCache {
    shards: Vec<RwLock<FxHashMap<String, Vec<String>>>>,
}

impl SentenceCache {
    fn new() -> Self {
        Self {
            shards: (0..CACHE_SHARDS)
                .map(|_| RwLock::new(FxHashMap::default()))
                .collect(),
        }
    }

    fn shard(&self, key: &str) -> &RwLock<FxHashMap<String, Vec<String>>> {
        use std::hash::{BuildHasher, BuildHasherDefault};
        let h = BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(key);
        &self.shards[(h as usize) % self.shards.len()]
    }

    fn get(&self, key: &str) -> Option<Vec<String>> {
        self.shard(key).try_read().ok()?.get(key).cloned()
    }

    fn insert(&self, key: &str, pieces: &[String]) {
        if key.len() >= CACHE_MAX_LENGTH {
            return;
        }
        let per_shard = CACHE_CAPACITY.div_ceil(self.shards.len());
        if let Ok(mut map) = self.shard(key).try_write() {
            if map.len() < per_shard {
                map.insert(key.to_owned(), pieces.to_vec());
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

/// A Unigram language model (SentencePiece): every piece has a score
/// (log probability) and a sentence is split into the pieces maximizing
/// the total score (Viterbi).
///
/// # Example
///
/// ```
/// use morpheme::models::Unigram;
/// use morpheme::Model;
///
/// let pieces = vec![
///     ("<unk>".to_string(), 0.0),
///     ("▁he".to_string(), -2.0),
///     ("llo".to_string(), -2.0),
///     ("▁".to_string(), -4.0),
///     ("h".to_string(), -5.0),
///     ("e".to_string(), -5.0),
///     ("l".to_string(), -5.0),
///     ("o".to_string(), -5.0),
/// ];
/// let unigram = Unigram::new(pieces, Some(0), false)?;
///
/// // Viterbi picks the most likely segmentation.
/// let tokens: Vec<String> = unigram.tokenize("▁hello")?.into_iter().map(|t| t.value).collect();
/// assert_eq!(tokens, ["▁he", "llo"]);
/// // Unknown chars get the unk id instead of failing.
/// assert_eq!(unigram.tokenize("▁z")?.last().unwrap().id, 0);
/// # Ok::<(), morpheme::Error>(())
/// ```
pub struct Unigram {
    vocab: Vec<(String, f64)>,
    token_to_ids: FxHashMap<String, u32>,
    trie: Trie,
    /// Lowest score in the vocabulary (NaN scores ignored).
    pub(crate) min_score: f64,
    unk_id: Option<usize>,
    pub(crate) bos_id: usize,
    pub(crate) eos_id: usize,
    fuse_unk: bool,
    is_optimized: bool,
    byte_fallback: bool,
    cache: SentenceCache,
}

impl Clone for Unigram {
    /// The clone starts with an empty cache.
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            token_to_ids: self.token_to_ids.clone(),
            trie: self.trie.clone(),
            min_score: self.min_score,
            unk_id: self.unk_id,
            bos_id: self.bos_id,
            eos_id: self.eos_id,
            fuse_unk: self.fuse_unk,
            is_optimized: self.is_optimized,
            byte_fallback: self.byte_fallback,
            cache: SentenceCache::new(),
        }
    }
}

impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id
            && self.byte_fallback == other.byte_fallback
            && self.vocab.len() == other.vocab.len()
            && self
                .vocab
                .iter()
                .zip(&other.vocab)
                .all(|((a, sa), (b, sb))| a == b && (sa == sb || (sa.is_nan() && sb.is_nan())))
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

impl Default for Unigram {
    fn default() -> Self {
        Self::new(vec![("<unk>".to_string(), 0.0)], Some(0), false)
            .expect("the default vocabulary is valid")
    }
}

impl Unigram {
    /// Build a model from `(piece, score)` pairs. The id of a piece is its
    /// index. `unk_id` must point inside the vocabulary; with
    /// `byte_fallback`, unknown text is emitted as `<0xNN>` byte pieces
    /// when those exist in the vocabulary.
    pub fn new(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self> {
        if let Some(unk_id) = unk_id {
            if vocab.is_empty() {
                return Err(Error::Config(
                    "Unigram: the vocabulary is empty but at least <unk> is needed".into(),
                ));
            }
            if unk_id >= vocab.len() {
                return Err(Error::Config(format!(
                    "Unigram: unk_id {unk_id} is not in the vocabulary (size {})",
                    vocab.len()
                )));
            }
        }
        let n = vocab.len();
        let mut token_to_ids = FxHashMap::default();
        token_to_ids.reserve(n);
        let mut trie = Trie::new();
        let mut min_score = f64::INFINITY;
        for (id, (token, score)) in vocab.iter().enumerate() {
            let id = u32::try_from(id)
                .map_err(|_| Error::Config("Unigram: vocabulary is too large".into()))?;
            token_to_ids.insert(token.clone(), id);
            trie.insert(token.as_bytes(), id);
            if *score < min_score {
                min_score = *score;
            }
        }
        Ok(Self {
            vocab,
            token_to_ids,
            trie,
            min_score,
            unk_id,
            bos_id: n + 1,
            eos_id: n + 2,
            fuse_unk: true,
            is_optimized: true,
            byte_fallback,
            cache: SentenceCache::new(),
        })
    }

    /// Whether unknown text falls back to `<0xNN>` byte pieces.
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }

    /// Id of the unknown piece, if any.
    pub fn unk_id(&self) -> Option<usize> {
        self.unk_id
    }

    /// The `(piece, score)` pairs, in id order.
    pub fn pieces(&self) -> &[(String, f64)] {
        &self.vocab
    }

    /// Number of pieces.
    pub(crate) fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Drop all cached encodings.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    #[cfg(test)]
    pub(crate) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
        self.clear_cache();
    }

    #[cfg(test)]
    pub(crate) fn set_optimized(&mut self, is_optimized: bool) {
        self.is_optimized = is_optimized;
        self.clear_cache();
    }

    /// Fill `lattice` with every vocabulary piece matching at every char
    /// position, adding an unknown node where no single-char piece
    /// exists.
    pub(crate) fn populate_nodes(&self, lattice: &mut Lattice<'_>) {
        let unk_score = self.min_score - UNK_PENALTY;
        let sentence = lattice.sentence();
        let bytes = sentence.as_bytes();
        let len = sentence.len();
        let mut pos = 0;
        let mut found: Vec<(usize, u32)> = Vec::new();
        while pos < len {
            let char_len = utf8_len(bytes[pos]);
            found.clear();
            self.trie
                .common_prefix_search(&bytes[pos..], |n, id| found.push((n, id)));
            let mut has_single = false;
            for &(n, id) in &found {
                lattice.insert(pos, n, self.vocab[id as usize].1, id as usize);
                has_single |= n == char_len;
            }
            if !has_single {
                if let Some(unk) = self.unk_id {
                    lattice.insert(pos, char_len, unk_score, unk);
                }
            }
            pos += char_len;
        }
    }

    /// Split `sentence` into its best pieces. Consecutive unknown chars
    /// are fused into one piece.
    pub fn encode(&self, sentence: &str) -> Result<Vec<String>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }
        if let Some(hit) = self.cache.get(sentence) {
            return Ok(hit);
        }
        let result = if self.is_optimized {
            self.encode_optimized(sentence)?
        } else {
            self.encode_unoptimized(sentence)?
        };
        self.cache.insert(sentence, &result);
        Ok(result)
    }

    /// Viterbi without building a lattice (SentencePiece's optimized
    /// encoder).
    fn encode_optimized(&self, sentence: &str) -> Result<Vec<String>> {
        #[derive(Clone, Copy)]
        struct Best {
            id: usize,
            score: f64,
            starts_at: Option<usize>,
        }
        let bytes = sentence.as_bytes();
        let size = bytes.len();
        let unk_score = self.min_score - UNK_PENALTY;
        let mut best = vec![
            Best {
                id: 0,
                score: 0.0,
                starts_at: None,
            };
            size + 1
        ];
        let mut start = 0;
        while start < size {
            let base = best[start].score;
            let char_len = utf8_len(bytes[start]);
            let mut has_single = false;
            let vocab = &self.vocab;
            self.trie.common_prefix_search(&bytes[start..], |n, id| {
                let target = &mut best[start + n];
                let candidate = vocab[id as usize].1 + base;
                if target.starts_at.is_none() || candidate > target.score {
                    *target = Best {
                        id: id as usize,
                        score: candidate,
                        starts_at: Some(start),
                    };
                }
                has_single |= n == char_len;
            });
            if !has_single {
                let target = &mut best[start + char_len];
                let candidate = unk_score + base;
                if target.starts_at.is_none() || candidate > target.score {
                    *target = Best {
                        id: self.unk_id.ok_or_else(missing_unk)?,
                        score: candidate,
                        starts_at: Some(start),
                    };
                }
            }
            start += char_len;
        }

        let mut results: Vec<String> = Vec::new();
        let mut unk_run: Vec<&str> = Vec::new();
        let mut end = size;
        while end > 0 {
            let node = best[end];
            let start = node
                .starts_at
                .ok_or_else(|| Error::Model("Unigram: no segmentation found".into()))?;
            let piece = &sentence[start..end];
            if self.fuse_unk && Some(node.id) == self.unk_id {
                unk_run.push(piece);
            } else {
                if !unk_run.is_empty() {
                    unk_run.reverse();
                    results.push(unk_run.concat());
                    unk_run.clear();
                }
                results.push(piece.to_owned());
            }
            end = start;
        }
        if !unk_run.is_empty() {
            unk_run.reverse();
            results.push(unk_run.concat());
        }
        results.reverse();
        Ok(results)
    }

    /// Viterbi over an explicit lattice.
    fn encode_unoptimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut lattice = Lattice::new(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        let path = lattice.viterbi();
        if path.is_empty() {
            return Err(missing_unk());
        }
        let mut results = Vec::new();
        let mut unk_run = String::new();
        for i in path {
            let node = lattice.node(i);
            let piece = lattice.piece(node);
            if self.fuse_unk && Some(node.id) == self.unk_id {
                unk_run.push_str(piece);
            } else {
                if !unk_run.is_empty() {
                    results.push(std::mem::take(&mut unk_run));
                }
                results.push(piece.to_owned());
            }
        }
        if !unk_run.is_empty() {
            results.push(unk_run);
        }
        Ok(results)
    }
}

fn missing_unk() -> Error {
    Error::Model("Unigram: encountered an unknown token but `unk_id` is missing".into())
}

/// Byte length of the UTF-8 char starting with `first`.
#[inline]
fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

impl Model for Unigram {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let pieces = self.encode(sequence)?;
        let mut offset = 0;
        let mut tokens = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let len = piece.len();
            let offsets = (offset, offset + len);
            offset += len;
            if let Some(&id) = self.token_to_ids.get(&piece) {
                tokens.push(Token::new(id, piece, offsets));
                continue;
            }
            if self.byte_fallback {
                let bytes: Option<Vec<Token>> = piece
                    .bytes()
                    .map(|b| {
                        let s = format!("<0x{b:02X}>");
                        self.token_to_ids
                            .get(&s)
                            .map(|&id| Token::new(id, s, offsets))
                    })
                    .collect();
                if let Some(bytes) = bytes {
                    tokens.extend(bytes);
                    continue;
                }
            }
            let unk = self.unk_id.ok_or_else(missing_unk)? as u32;
            tokens.push(Token::new(unk, piece, offsets));
        }
        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|(t, _)| t.clone())
    }

    fn vocab(&self) -> HashMap<String, u32> {
        self.token_to_ids
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Serialize for Unigram {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Unigram", 4)?;
        s.serialize_field("type", "Unigram")?;
        s.serialize_field("unk_id", &self.unk_id)?;
        s.serialize_field("vocab", &self.vocab)?;
        s.serialize_field("byte_fallback", &self.byte_fallback)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Unigram {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Repr {
            #[serde(rename = "type", default)]
            ty: Option<String>,
            vocab: Vec<(String, f64)>,
            #[serde(default)]
            unk_id: Option<usize>,
            #[serde(default)]
            byte_fallback: bool,
        }
        let r = Repr::deserialize(deserializer)?;
        if let Some(ty) = &r.ty {
            if ty != "Unigram" {
                return Err(D::Error::custom(format!(
                    "expected model type \"Unigram\", got {ty:?}"
                )));
            }
        }
        Unigram::new(r.vocab, r.unk_id, r.byte_fallback).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(v: &[(&str, f64)]) -> Vec<(String, f64)> {
        v.iter().map(|(s, f)| (s.to_string(), *f)).collect()
    }

    #[test]
    fn populate_nodes_unk() {
        let model = Unigram::new(pieces(&[("<unk>", 0.0)]), Some(0), false).unwrap();
        let mut lattice = Lattice::new("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);
        for pos in 0..3 {
            assert_eq!(lattice.begin_nodes[pos].len(), 1);
            let node_id = lattice.begin_nodes[pos][0];
            assert_eq!(lattice.node(node_id).id, 0);
            assert_eq!(node_id, pos + 2);
        }
    }

    #[test]
    fn populate_nodes_pieces() {
        let model = Unigram::new(
            pieces(&[
                ("<unk>", 0.0),
                ("a", 0.1),
                ("b", 0.2),
                ("ab", 0.3),
                ("bc", 0.4),
            ]),
            Some(0),
            false,
        )
        .unwrap();
        let mut lattice = Lattice::new("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);
        let ids = |pos: usize| -> Vec<usize> {
            lattice.begin_nodes[pos]
                .iter()
                .map(|&i| lattice.node(i).id)
                .collect()
        };
        assert_eq!(ids(0), vec![1, 3]);
        assert_eq!(ids(1), vec![2, 4]);
        assert_eq!(ids(2), vec![0]);
    }

    #[test]
    fn encode_simple() {
        let model = Unigram::new(
            pieces(&[
                ("<unk>", 0.0),
                ("a", 0.0),
                ("b", 0.0),
                ("c", 0.0),
                ("d", 0.0),
                ("cd", 1.0),
                ("ab", 2.0),
                ("abc", 5.0),
                ("abcd", 10.0),
            ]),
            Some(0),
            false,
        )
        .unwrap();
        assert_eq!(model.encode("abcd").unwrap(), vec!["abcd"]);
        assert_eq!(
            model.encode("abcdacdxx").unwrap(),
            vec!["abcd", "a", "cd", "xx"]
        );
    }

    #[test]
    fn encode_optimized_and_unoptimized_agree() {
        let mut model = Unigram::new(
            pieces(&[
                ("<unk>", 0.0),
                ("ab", 0.0),
                ("cd", -0.1),
                ("abc", -0.2),
                ("a", -0.3),
                ("b", -0.4),
                ("c", -0.5),
                ("ABC", -0.5),
                ("abcdabcd", 20.0),
                ("q", 20.5),
                ("r", 20.5),
                ("qr", -0.5),
            ]),
            Some(0),
            false,
        )
        .unwrap();
        for optimized in [true, false] {
            model.set_optimized(optimized);
            assert_eq!(model.encode("abc").unwrap(), vec!["abc"]);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);
            model.set_fuse_unk(false);
            assert_eq!(model.encode("AB").unwrap(), vec!["A", "B"]);
            model.set_fuse_unk(true);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);
            assert_eq!(model.encode("abcd").unwrap(), vec!["ab", "cd"]);
            assert_eq!(model.encode("abcc").unwrap(), vec!["abc", "c"]);
            assert_eq!(
                model.encode("xabcabaabcdd").unwrap(),
                vec!["x", "abc", "ab", "a", "ab", "cd", "d"]
            );
            model.set_fuse_unk(false);
            assert_eq!(
                model.encode("xyz東京").unwrap(),
                vec!["x", "y", "z", "東", "京"]
            );
            model.set_fuse_unk(true);
            assert_eq!(model.encode("xyz東京").unwrap(), vec!["xyz東京"]);
            assert_eq!(model.encode("ABC").unwrap(), vec!["ABC"]);
            assert_eq!(model.encode("abABCcd").unwrap(), vec!["ab", "ABC", "cd"]);
            assert_eq!(
                model.encode("ababcdabcdcd").unwrap(),
                vec!["ab", "abcdabcd", "cd"]
            );
            assert_eq!(model.encode("abqrcd").unwrap(), vec!["ab", "q", "r", "cd"]);
        }
    }

    #[test]
    fn byte_fallback() {
        let model = Unigram::new(
            pieces(&[("<unk>", 0.0), ("<0xC3>", -0.01), ("<0xA9>", -0.03)]),
            Some(0),
            true,
        )
        .unwrap();
        let tokens = model.tokenize("é").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1, "<0xC3>".into(), (0, 2)),
                Token::new(2, "<0xA9>".into(), (0, 2)),
            ]
        );
        // '?' has no byte piece: the whole fused unknown run maps to unk.
        let tokens = model.tokenize("?é").unwrap();
        assert_eq!(tokens[0].id, 0);
    }

    #[test]
    fn no_unk_id_errors_only_when_needed() {
        let model = Unigram::new(pieces(&[("a", -0.5)]), None, false).unwrap();
        assert_eq!(model.encode("aa").unwrap(), vec!["a", "a"]);
        assert!(model.encode("ab").is_err());
    }

    #[test]
    fn validation() {
        assert!(Unigram::new(vec![], Some(0), false).is_err());
        assert!(Unigram::new(pieces(&[("a", 0.0)]), Some(1), false).is_err());
        assert!(Unigram::new(vec![], None, false).is_ok());
    }

    #[test]
    fn serialization_round_trip() {
        for (v, unk) in [
            (pieces(&[("<unk>", 0.0), ("a", -0.5)]), Some(0)),
            (pieces(&[("a", -0.5), ("<unk>", 0.0)]), Some(1)),
            (pieces(&[("a", -0.5)]), None),
        ] {
            let model = Unigram::new(v, unk, false).unwrap();
            let json = serde_json::to_string(&model).unwrap();
            let back: Unigram = serde_json::from_str(&json).unwrap();
            assert_eq!(model, back);
        }
        let model = Unigram::new(pieces(&[("<unk>", 0.0), ("a", -0.5)]), Some(0), false).unwrap();
        assert_eq!(
            serde_json::to_string(&model).unwrap(),
            r#"{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-0.5]],"byte_fallback":false}"#
        );
    }

    #[test]
    fn deserialize_legacy_and_reject_wrong_type() {
        let m: Unigram = serde_json::from_str(r#"{"unk_id":0,"vocab":[["<unk>",0.0]]}"#).unwrap();
        assert_eq!(m.vocab_size(), 1);
        assert!(serde_json::from_str::<Unigram>(r#"{"type":"BPE","vocab":[["a",0.0]]}"#).is_err());
        assert!(serde_json::from_str::<Unigram>(r#"{"unk_id":3,"vocab":[["a",0.0]]}"#).is_err());
    }

    /// Parity with real Hugging Face tokenizer files (downloaded by
    /// `scripts/fetch-hf-fixtures.sh`; skipped when absent).
    mod fixtures {
        use super::*;

        type Expected<'a> = &'a [(&'a str, &'a [(u32, &'a str, (usize, usize))])];

        fn load(name: &str) -> Option<(Unigram, serde_json::Value)> {
            let path = format!("{}/tests/data/hf/{name}.json", env!("CARGO_MANIFEST_DIR"));
            let Ok(raw) = std::fs::read_to_string(&path) else {
                eprintln!("skipping: {path} not found (run scripts/fetch-hf-fixtures.sh)");
                return None;
            };
            let file: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let model_json = file["model"].clone();
            let model: Unigram = serde_json::from_value(model_json.clone()).unwrap();
            Some((model, model_json))
        }

        fn check(name: &str, expected: Expected<'_>) {
            let Some((model, mut json)) = load(name) else {
                return;
            };
            // Re-serialization: older files omit "type"/"byte_fallback".
            let obj = json.as_object_mut().unwrap();
            obj.entry("type").or_insert("Unigram".into());
            obj.entry("byte_fallback").or_insert(false.into());
            assert_eq!(
                serde_json::to_value(&model).unwrap(),
                json,
                "{name}: re-serialization"
            );

            for (input, tokens) in expected {
                let got: Vec<(u32, String, (usize, usize))> = model
                    .tokenize(input)
                    .unwrap()
                    .into_iter()
                    .map(|t| (t.id, t.value, t.offsets))
                    .collect();
                let want: Vec<(u32, String, (usize, usize))> = tokens
                    .iter()
                    .map(|(id, v, o)| (*id, v.to_string(), *o))
                    .collect();
                assert_eq!(got, want, "{name}: {input:?}");
            }
        }

        // Expectations from Python `tokenizers` 0.23.2: `tok.model.tokenize(s)`.
        #[test]
        fn t5_small() {
            check(
                "t5-small",
                &[
                    ("▁Hello", &[(8774, "▁Hello", (0, 8))]),
                    (
                        "▁world▁test",
                        &[(296, "▁world", (0, 8)), (794, "▁test", (8, 15))],
                    ),
                    (
                        "▁𝔘nicode",
                        &[
                            (3, "▁", (0, 3)),
                            (2, "𝔘", (3, 7)),
                            (29, "n", (7, 8)),
                            (23, "i", (8, 9)),
                            (4978, "code", (9, 13)),
                        ],
                    ),
                    (
                        "▁日本語のテキスト",
                        &[(3, "▁", (0, 3)), (2, "日本語のテキスト", (3, 27))],
                    ),
                    (
                        "▁Ünïcödé▁ťéxt",
                        &[
                            (3, "▁", (0, 3)),
                            (14858, "Ü", (3, 5)),
                            (29, "n", (5, 6)),
                            (2, "ï", (6, 8)),
                            (75, "c", (8, 9)),
                            (1872, "ö", (9, 11)),
                            (3764, "dé", (11, 14)),
                            (3, "▁", (14, 17)),
                            (2, "ť", (17, 19)),
                            (154, "é", (19, 21)),
                            (226, "x", (21, 22)),
                            (17, "t", (22, 23)),
                        ],
                    ),
                    (
                        "▁supercalifragilistic",
                        &[
                            (1355, "▁super", (0, 8)),
                            (15534, "cali", (8, 12)),
                            (20791, "frag", (12, 16)),
                            (173, "il", (16, 18)),
                            (3040, "istic", (18, 23)),
                        ],
                    ),
                    ("▁a", &[(3, "▁", (0, 3)), (9, "a", (3, 4))]),
                    ("▁😀👍", &[(3, "▁", (0, 3)), (2, "😀👍", (3, 11))]),
                ],
            );
        }

        #[test]
        fn xlm_roberta_base() {
            check(
                "xlm-roberta-base",
                &[
                    ("▁Hello", &[(35378, "▁Hello", (0, 8))]),
                    (
                        "▁world▁test",
                        &[(8999, "▁world", (0, 8)), (3034, "▁test", (8, 15))],
                    ),
                    (
                        "▁𝔘nicode",
                        &[
                            (6, "▁", (0, 3)),
                            (3, "𝔘", (3, 7)),
                            (12150, "nico", (7, 11)),
                            (112, "de", (11, 13)),
                        ],
                    ),
                    (
                        "▁日本語のテキスト",
                        &[
                            (6, "▁", (0, 3)),
                            (98449, "日本語", (3, 12)),
                            (154, "の", (12, 15)),
                            (193508, "テキスト", (15, 27)),
                        ],
                    ),
                    (
                        "▁Ünïcödé▁ťéxt",
                        &[
                            (8578, "▁Ü", (0, 5)),
                            (19, "n", (5, 6)),
                            (9392, "ï", (6, 8)),
                            (238, "c", (8, 9)),
                            (21166, "öd", (9, 12)),
                            (446, "é", (12, 14)),
                            (6, "▁", (14, 17)),
                            (1981, "ť", (17, 19)),
                            (446, "é", (19, 21)),
                            (29062, "xt", (21, 23)),
                        ],
                    ),
                    (
                        "▁supercalifragilistic",
                        &[
                            (1601, "▁super", (0, 8)),
                            (52777, "cali", (8, 12)),
                            (6000, "fra", (12, 15)),
                            (8726, "gil", (15, 18)),
                            (48242, "istic", (18, 23)),
                        ],
                    ),
                    ("▁a", &[(10, "▁a", (0, 4))]),
                    ("▁😀👍", &[(21119, "▁😀", (0, 7)), (118280, "👍", (7, 11))]),
                ],
            );
        }

        #[test]
        fn albert_base_v2() {
            check(
                "albert-base-v2",
                &[
                    (
                        "▁Hello",
                        &[(13, "▁", (0, 3)), (1, "H", (3, 4)), (7523, "ello", (4, 8))],
                    ),
                    (
                        "▁world▁test",
                        &[(126, "▁world", (0, 8)), (1289, "▁test", (8, 15))],
                    ),
                    (
                        "▁𝔘nicode",
                        &[
                            (13, "▁", (0, 3)),
                            (1, "𝔘", (3, 7)),
                            (889, "ni", (7, 9)),
                            (9375, "code", (9, 13)),
                        ],
                    ),
                    (
                        "▁日本語のテキスト",
                        &[(13, "▁", (0, 3)), (1, "日本語のテキスト", (3, 27))],
                    ),
                    (
                        "▁Ünïcödé▁ťéxt",
                        &[
                            (13, "▁", (0, 3)),
                            (1, "Ü", (3, 5)),
                            (103, "n", (5, 6)),
                            (1, "ï", (6, 8)),
                            (150, "c", (8, 9)),
                            (1, "ö", (9, 11)),
                            (43, "d", (11, 12)),
                            (1, "é", (12, 14)),
                            (13, "▁", (14, 17)),
                            (1, "ťé", (17, 21)),
                            (396, "x", (21, 22)),
                            (38, "t", (22, 23)),
                        ],
                    ),
                    (
                        "▁supercalifragilistic",
                        &[
                            (1026, "▁super", (0, 8)),
                            (3430, "cal", (8, 11)),
                            (49, "i", (11, 12)),
                            (22133, "frag", (12, 16)),
                            (947, "il", (16, 18)),
                            (3771, "istic", (18, 23)),
                        ],
                    ),
                    ("▁a", &[(21, "▁a", (0, 4))]),
                    ("▁😀👍", &[(13, "▁", (0, 3)), (1, "😀👍", (3, 11))]),
                ],
            );
        }

        #[test]
        fn large_vocab_encoding_is_fast() {
            let Some((model, _)) = load("xlm-roberta-base") else {
                return;
            };
            let words: Vec<String> = (0..20_000).map(|i| format!("▁word{i}xyz")).collect();
            let start = std::time::Instant::now();
            for w in &words {
                model.tokenize(w).unwrap();
            }
            let elapsed = start.elapsed();
            // ~250k-piece vocab: the trie is built once, so this is
            // milliseconds; rebuilding it per call would take minutes.
            assert!(elapsed.as_secs() < 5, "encoding took {elapsed:?}");
        }
    }
}
