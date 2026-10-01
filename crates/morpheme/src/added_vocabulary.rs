//! Added tokens: strings (usually special tokens such as `[CLS]` or
//! `<|endoftext|>`) that are matched verbatim in the input *before* the
//! model sees it, so they are never split.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::normalized_string::{NormalizedString, OffsetRange};
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::{Model, Normalizer};
use crate::{Offsets, Token};

/// A token added on top of the model's vocabulary.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use morpheme::models::WordLevel;
/// use morpheme::pre_tokenizers::Whitespace;
/// use morpheme::{AddedToken, Tokenizer};
///
/// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("fill", 1)].map(|(t, i)| (t.to_string(), i)).into();
/// let mut tokenizer = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?)
///     .with_pre_tokenizer(Whitespace);
/// // Special tokens are matched before normalization and pre-tokenization,
/// // so they are never split; `lstrip` also swallows the space before them.
/// tokenizer.add_special_tokens(&[AddedToken::new("<mask>", true).lstrip(true)])?;
///
/// let encoding = tokenizer.encode("fill <mask>", false)?;
/// assert_eq!(encoding.tokens(), ["fill", " <mask>"]);
/// assert_eq!(tokenizer.decode(encoding.ids(), true)?, "fill"); // specials skipped
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AddedToken {
    /// The token text.
    pub content: String,
    /// Only match when not surrounded by word characters.
    #[serde(default)]
    pub single_word: bool,
    /// Swallow whitespace on the left of a match.
    #[serde(default)]
    pub lstrip: bool,
    /// Swallow whitespace on the right of a match.
    #[serde(default)]
    pub rstrip: bool,
    /// Match against the normalized text (`true`) or the raw input.
    #[serde(default = "default_true")]
    pub normalized: bool,
    /// Special tokens can be skipped when decoding.
    #[serde(default)]
    pub special: bool,
}

fn default_true() -> bool {
    true
}

impl Default for AddedToken {
    fn default() -> Self {
        Self {
            content: String::new(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: true,
            special: false,
        }
    }
}

impl AddedToken {
    /// A token with default flags. Special tokens match the raw input
    /// (`normalized: false`); others match normalized text.
    pub fn new(content: impl Into<String>, special: bool) -> Self {
        Self {
            content: content.into(),
            normalized: !special,
            special,
            ..Default::default()
        }
    }

    /// Set `single_word`.
    #[must_use]
    pub fn single_word(mut self, v: bool) -> Self {
        self.single_word = v;
        self
    }

    /// Set `lstrip`.
    #[must_use]
    pub fn lstrip(mut self, v: bool) -> Self {
        self.lstrip = v;
        self
    }

    /// Set `rstrip`.
    #[must_use]
    pub fn rstrip(mut self, v: bool) -> Self {
        self.rstrip = v;
        self
    }

    /// Set `normalized`.
    #[must_use]
    pub fn normalized(mut self, v: bool) -> Self {
        self.normalized = v;
        self
    }

    /// Set `special`.
    #[must_use]
    pub fn special(mut self, v: bool) -> Self {
        self.special = v;
        self
    }
}

/// An [`AddedToken`] with its id, as stored in `tokenizer.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddedTokenWithId {
    /// The token id.
    pub id: u32,
    /// The token.
    #[serde(flatten)]
    pub token: AddedToken,
}

/// The set of added tokens, with the matchers used to find them.
#[derive(Debug, Clone, Default)]
pub struct AddedVocabulary {
    by_content: HashMap<String, u32>,
    by_id: HashMap<u32, AddedToken>,
    special: HashSet<String>,
    /// Normalized form of `normalized` tokens, when it differs.
    normalized_cache: HashMap<u32, String>,
    raw_matcher: Option<Matcher>,
    normalized_matcher: Option<Matcher>,
    encode_special_tokens: bool,
}

#[derive(Debug, Clone)]
struct Matcher {
    automaton: AhoCorasick,
    ids: Vec<u32>,
}

impl Matcher {
    fn build(patterns: Vec<(String, u32)>) -> Result<Option<Self>> {
        if patterns.is_empty() {
            return Ok(None);
        }
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns.iter().map(|(p, _)| p))
            .map_err(|e| Error::Config(format!("added tokens matcher: {e}")))?;
        Ok(Some(Self {
            automaton,
            ids: patterns.into_iter().map(|(_, id)| id).collect(),
        }))
    }
}

fn word_regexes() -> &'static (fancy_regex::Regex, fancy_regex::Regex) {
    static RE: OnceLock<(fancy_regex::Regex, fancy_regex::Regex)> = OnceLock::new();
    RE.get_or_init(|| {
        (
            fancy_regex::Regex::new(r"^\w").expect("valid regex"),
            fancy_regex::Regex::new(r"\w$").expect("valid regex"),
        )
    })
}

fn starts_with_word(s: &str) -> bool {
    word_regexes().0.is_match(s).unwrap_or(false)
}

fn ends_with_word(s: &str) -> bool {
    word_regexes().1.is_match(s).unwrap_or(false)
}

impl AddedVocabulary {
    /// An empty added vocabulary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of added tokens.
    pub fn len(&self) -> usize {
        self.by_content.len()
    }

    /// True if there are no added tokens.
    pub fn is_empty(&self) -> bool {
        self.by_content.is_empty()
    }

    /// `content -> id` for every added token.
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.by_content
    }

    /// `id -> token` for every added token.
    pub fn added_tokens_decoder(&self) -> &HashMap<u32, AddedToken> {
        &self.by_id
    }

    /// Added tokens sorted by id.
    pub fn tokens_with_ids(&self) -> Vec<AddedTokenWithId> {
        let mut v: Vec<AddedTokenWithId> = self
            .by_id
            .iter()
            .map(|(id, token)| AddedTokenWithId {
                id: *id,
                token: token.clone(),
            })
            .collect();
        v.sort_by_key(|t| t.id);
        v
    }

    /// Id of `token` (added tokens first, then the model).
    pub fn token_to_id(&self, token: &str, model: &dyn Model) -> Option<u32> {
        self.by_content
            .get(token)
            .copied()
            .or_else(|| model.token_to_id(token))
    }

    /// Token string for an added token id (normalized form if any).
    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        self.by_id.get(&id).map(|t| {
            self.normalized_cache
                .get(&id)
                .cloned()
                .unwrap_or_else(|| t.content.clone())
        })
    }

    /// Whether `token` is a special token.
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special.contains(token)
    }

    /// When true, special tokens are *not* matched in the input and get
    /// tokenized like regular text.
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.encode_special_tokens = value;
    }

    /// See [`set_encode_special_tokens`](Self::set_encode_special_tokens).
    pub fn encode_special_tokens(&self) -> bool {
        self.encode_special_tokens
    }

    /// Add tokens, assigning ids: a token already in the model keeps the
    /// model's id; new tokens get ids after the model's vocabulary.
    /// Returns how many tokens were actually added.
    /// On error, the vocabulary and its matchers remain unchanged.
    pub fn add_tokens(
        &mut self,
        tokens: &[AddedToken],
        model: &dyn Model,
        normalizer: Option<&dyn Normalizer>,
    ) -> Result<usize> {
        let mut updated = self.clone();
        let added = updated.add_tokens_in_place(tokens, model, normalizer)?;
        *self = updated;
        Ok(added)
    }

    fn add_tokens_in_place(
        &mut self,
        tokens: &[AddedToken],
        model: &dyn Model,
        normalizer: Option<&dyn Normalizer>,
    ) -> Result<usize> {
        let model_vocab = model.vocab();
        let mut next_id = model_vocab
            .values()
            .chain(self.by_id.keys())
            .max()
            .map_or(Some(0), |id| id.checked_add(1));
        let mut added = 0;
        for token in tokens {
            if token.content.is_empty() {
                continue;
            }
            if let Some(id) = self.by_content.get(&token.content) {
                if self.by_id.get(id) == Some(token) {
                    continue;
                }
            }
            let id = match self.token_to_id(&token.content, model) {
                Some(id) => id,
                None => {
                    let id = next_id.ok_or_else(|| {
                        Error::Config("added token ids exhausted the u32 range".into())
                    })?;
                    next_id = id.checked_add(1);
                    id
                }
            };
            self.insert(id, token.clone());
            added += 1;
        }
        self.refresh(normalizer)?;
        Ok(added)
    }

    /// Add tokens with explicit ids (as loaded from `tokenizer.json`).
    pub fn add_tokens_with_ids(
        &mut self,
        tokens: &[AddedTokenWithId],
        normalizer: Option<&dyn Normalizer>,
    ) -> Result<()> {
        for t in tokens {
            if t.token.content.is_empty() {
                continue;
            }
            self.insert(t.id, t.token.clone());
        }
        self.refresh(normalizer)
    }

    fn insert(&mut self, id: u32, token: AddedToken) {
        if let Some(old_id) = self.by_content.insert(token.content.clone(), id) {
            if old_id != id {
                self.by_id.remove(&old_id);
            }
        }
        if token.special {
            self.special.insert(token.content.clone());
        } else {
            self.special.remove(&token.content);
        }
        self.by_id.insert(id, token);
    }

    /// Recompute normalized forms and matchers (call after changing the
    /// normalizer).
    pub fn refresh(&mut self, normalizer: Option<&dyn Normalizer>) -> Result<()> {
        self.normalized_cache.clear();
        let mut raw = Vec::new();
        let mut normalized = Vec::new();
        let mut ids: Vec<&u32> = self.by_id.keys().collect();
        ids.sort();
        for id in ids {
            let token = &self.by_id[id];
            if token.normalized {
                let mut pattern = token.content.clone();
                if let Some(n) = normalizer {
                    let mut s = NormalizedString::from(token.content.as_str());
                    n.normalize(&mut s)?;
                    if s.get() != token.content {
                        pattern = s.get().to_owned();
                        self.normalized_cache.insert(*id, pattern.clone());
                    }
                }
                normalized.push((pattern, *id));
            } else {
                raw.push((token.content.clone(), *id));
            }
        }
        // Empty patterns (e.g. a token that normalizes to nothing) would
        // match everywhere.
        normalized.retain(|(p, _)| !p.is_empty());
        self.raw_matcher = Matcher::build(raw)?;
        self.normalized_matcher = Matcher::build(normalized)?;
        Ok(())
    }

    fn find_matches(
        &self,
        sentence: &str,
        matcher: &Option<Matcher>,
    ) -> Vec<(Option<u32>, Offsets)> {
        if sentence.is_empty() {
            return vec![(None, (0, 0))];
        }
        let Some(matcher) = matcher else {
            return vec![(None, (0, sentence.len()))];
        };
        let mut splits = Vec::new();
        let mut cursor = 0;
        for m in matcher.automaton.find_iter(sentence) {
            let id = matcher.ids[m.pattern().as_usize()];
            let token = &self.by_id[&id];
            let (mut start, mut stop) = (m.start(), m.end());
            if self.encode_special_tokens && self.special.contains(&token.content) {
                continue;
            }
            if token.single_word {
                let start_ok = start == 0 || !ends_with_word(&sentence[..start]);
                let stop_ok = stop == sentence.len() || !starts_with_word(&sentence[stop..]);
                if !start_ok || !stop_ok {
                    continue;
                }
            }
            if token.lstrip {
                let trimmed = sentence[..start]
                    .trim_end_matches(char::is_whitespace)
                    .len();
                start = trimmed.max(cursor);
            }
            if token.rstrip {
                let rest = &sentence[stop..];
                stop += rest.len() - rest.trim_start_matches(char::is_whitespace).len();
            }
            if cursor < start {
                splits.push((None, (cursor, start)));
            }
            splits.push((Some(id), (start, stop)));
            cursor = stop;
        }
        if cursor != sentence.len() {
            splits.push((None, (cursor, sentence.len())));
        }
        splits
    }

    fn split_with_matches(
        &self,
        sentence: NormalizedString,
        matcher: &Option<Matcher>,
    ) -> Vec<(NormalizedString, Option<Vec<Token>>)> {
        self.find_matches(sentence.get(), matcher)
            .into_iter()
            .map(|(id, (s, e))| {
                let slice = sentence
                    .slice(OffsetRange::Normalized(s..e))
                    .expect("added-token matches are on char boundaries");
                match id {
                    Some(id) => {
                        let value = slice.get().to_owned();
                        let len = value.len();
                        (slice, Some(vec![Token::new(id, value, (0, len))]))
                    }
                    None => (slice, None),
                }
            })
            .collect()
    }

    /// Split `sequence` around added tokens: first raw (non-normalized)
    /// tokens on the original text, then normalize the remaining pieces
    /// and split around normalized tokens.
    pub fn extract_and_normalize(
        &self,
        normalizer: Option<&dyn Normalizer>,
        sequence: &str,
    ) -> Result<PreTokenizedString> {
        let mut pretokenized = PreTokenizedString::from(sequence);
        pretokenized.split(|_, s| Ok(self.split_with_matches(s, &self.raw_matcher)))?;
        pretokenized.split(|_, mut s| {
            if let Some(n) = normalizer {
                n.normalize(&mut s)?;
            }
            Ok(self.split_with_matches(s, &self.normalized_matcher))
        })?;
        Ok(pretokenized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mock(HashMap<String, u32>);
    impl Model for Mock {
        fn tokenize(&self, _: &str) -> Result<Vec<Token>> {
            Ok(vec![])
        }
        fn token_to_id(&self, t: &str) -> Option<u32> {
            self.0.get(t).copied()
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0
                .iter()
                .find(|(_, v)| **v == id)
                .map(|(k, _)| k.clone())
        }
        fn vocab(&self) -> HashMap<String, u32> {
            self.0.clone()
        }
        fn vocab_size(&self) -> usize {
            self.0.len()
        }
    }

    fn model() -> Mock {
        Mock(
            [("a".to_string(), 0), ("b".to_string(), 1)]
                .into_iter()
                .collect(),
        )
    }

    #[test]
    fn ids_follow_model_vocab() {
        let mut v = AddedVocabulary::new();
        let m = model();
        v.add_tokens(
            &[AddedToken::new("[CLS]", true), AddedToken::new("a", false)],
            &m,
            None,
        )
        .unwrap();
        assert_eq!(v.token_to_id("[CLS]", &m), Some(2));
        assert_eq!(v.token_to_id("a", &m), Some(0));
        assert!(v.is_special_token("[CLS]"));
    }

    #[test]
    fn extracts_tokens_with_lstrip_rstrip() {
        let mut v = AddedVocabulary::new();
        let m = model();
        v.add_tokens(
            &[AddedToken::new("<mask>", true).lstrip(true).rstrip(true)],
            &m,
            None,
        )
        .unwrap();
        let pts = v.extract_and_normalize(None, "hi <mask> there").unwrap();
        let splits = pts.get_splits(crate::OffsetType::Byte);
        let texts: Vec<_> = splits
            .iter()
            .map(|(s, o, t)| (*s, *o, t.is_some()))
            .collect();
        assert_eq!(
            texts,
            vec![
                ("hi", (0, 2), false),
                (" <mask> ", (2, 10), true),
                ("there", (10, 15), false)
            ]
        );
    }

    #[test]
    fn single_word_respects_boundaries() {
        let mut v = AddedVocabulary::new();
        let m = model();
        v.add_tokens(&[AddedToken::new("ab", false).single_word(true)], &m, None)
            .unwrap();
        let pts = v.extract_and_normalize(None, "xab ab").unwrap();
        let n_matched = pts.splits().iter().filter(|s| s.tokens.is_some()).count();
        assert_eq!(n_matched, 1);
    }
}
