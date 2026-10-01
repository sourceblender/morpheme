//! BPE trainer, producing the same vocabulary and merges as Hugging Face
//! `tokenizers`' `BpeTrainer` (Sennrich et al. 2016 with incremental
//! pair counts).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::added_vocabulary::AddedToken;
use crate::error::{Error, Result};
use crate::models::bpe::{Bpe, MergeMap, Word};
use crate::progress::Progress;
use crate::traits::Trainer;

type Pair = (u32, u32);

/// A candidate merge in the trainer queue: highest count first, then the
/// smallest pair of ids.
#[derive(Debug)]
struct Candidate {
    pair: Pair,
    count: u64,
    /// Words where this pair (newly) occurs.
    pos: FxHashSet<usize>,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Builder for [`BpeTrainer`].
#[derive(Debug, Clone)]
pub struct BpeTrainerBuilder {
    trainer: BpeTrainer,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            trainer: BpeTrainer {
                min_frequency: 0,
                vocab_size: 30_000,
                show_progress: true,
                special_tokens: Vec::new(),
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
                words: HashMap::new(),
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// A builder with HF defaults (vocab size 30 000).
    pub fn new() -> Self {
        Self::default()
    }

    /// Pairs occurring fewer times than this are never merged.
    #[must_use]
    pub fn min_frequency(mut self, n: u64) -> Self {
        self.trainer.min_frequency = n;
        self
    }

    /// Target vocabulary size (special tokens and alphabet included).
    #[must_use]
    pub fn vocab_size(mut self, n: usize) -> Self {
        self.trainer.vocab_size = n;
        self
    }

    /// Whether to report progress.
    #[must_use]
    pub fn show_progress(mut self, v: bool) -> Self {
        self.trainer.show_progress = v;
        self
    }

    /// Special tokens, placed first in the vocabulary.
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.trainer.special_tokens = tokens;
        self
    }

    /// Keep at most this many distinct chars in the alphabet (the most
    /// frequent ones).
    #[must_use]
    pub fn limit_alphabet(mut self, n: usize) -> Self {
        self.trainer.limit_alphabet = Some(n);
        self
    }

    /// Chars always included in the alphabet.
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.trainer.initial_alphabet = alphabet;
        self
    }

    /// Prefix of non-initial subwords (e.g. `##`).
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.trainer.continuing_subword_prefix = Some(prefix.into());
        self
    }

    /// Suffix of word-final subwords (e.g. `</w>`).
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.trainer.end_of_word_suffix = Some(suffix.into());
        self
    }

    /// Maximum length (in chars) of learned tokens. Unlimited by default.
    #[must_use]
    pub fn max_token_length(mut self, n: usize) -> Self {
        self.trainer.max_token_length = Some(n);
        self
    }

    /// Build the trainer.
    ///
    /// # Errors
    /// Fails if `vocab_size`, `limit_alphabet` or `max_token_length` is
    /// zero.
    pub fn build(self) -> Result<BpeTrainer> {
        let t = &self.trainer;
        if t.vocab_size == 0 {
            return Err(Error::Config("BpeTrainer: vocab_size must be > 0".into()));
        }
        if t.limit_alphabet == Some(0) {
            return Err(Error::Config(
                "BpeTrainer: limit_alphabet must be > 0".into(),
            ));
        }
        if t.max_token_length == Some(0) {
            return Err(Error::Config(
                "BpeTrainer: max_token_length must be > 0".into(),
            ));
        }
        Ok(self.trainer)
    }
}

/// Trains a [`Bpe`] model.
///
/// # Example
///
/// ```
/// use morpheme::models::Bpe;
/// use morpheme::pre_tokenizers::ByteLevel;
/// use morpheme::trainers::BpeTrainer;
/// use morpheme::{AddedToken, Tokenizer};
///
/// let mut tokenizer = Tokenizer::new(Bpe::default())
///     .with_pre_tokenizer(ByteLevel::new(false, true, true))
///     .with_decoder(ByteLevel::default());
/// let trainer = BpeTrainer::builder()
///     .vocab_size(300)
///     .initial_alphabet(ByteLevel::alphabet()) // every byte is representable
///     .special_tokens(vec![AddedToken::new("<|endoftext|>", true)])
///     .show_progress(false)
///     .build()?;
/// tokenizer.train(trainer, ["the cat sat", "the cat ran"].into_iter())?;
///
/// let ids = tokenizer.encode("the dog 🐕", false)?.ids().to_vec();
/// assert_eq!(tokenizer.decode(&ids, false)?, "the dog 🐕"); // lossless
/// assert_eq!(tokenizer.token_to_id("<|endoftext|>"), Some(0));
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct BpeTrainer {
    /// Minimum pair frequency.
    pub(crate) min_frequency: u64,
    /// Target vocabulary size.
    pub(crate) vocab_size: usize,
    /// Whether to report progress.
    pub(crate) show_progress: bool,
    /// Special tokens, placed first in the vocabulary.
    pub(crate) special_tokens: Vec<AddedToken>,
    /// Maximum alphabet size.
    pub(crate) limit_alphabet: Option<usize>,
    /// Chars always included in the alphabet.
    pub(crate) initial_alphabet: HashSet<char>,
    /// Prefix of non-initial subwords.
    pub(crate) continuing_subword_prefix: Option<String>,
    /// Suffix of word-final subwords.
    pub(crate) end_of_word_suffix: Option<String>,
    /// Maximum length (in chars) of learned tokens.
    pub(crate) max_token_length: Option<usize>,
    words: HashMap<String, u64>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        BpeTrainerBuilder::default()
            .build()
            .expect("the default configuration is valid")
    }
}

/// Vocabulary under construction.
#[derive(Default)]
struct VocabBuilder {
    to_id: FxHashMap<String, u32>,
    tokens: Vec<String>,
}

impl VocabBuilder {
    fn contains(&self, s: &str) -> bool {
        self.to_id.contains_key(s)
    }

    fn get_or_insert(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.to_id.get(s) {
            return id;
        }
        let id = self.tokens.len() as u32;
        self.tokens.push(s.to_owned());
        self.to_id.insert(s.to_owned(), id);
        id
    }

    fn len(&self) -> usize {
        self.to_id.len()
    }
}

impl BpeTrainer {
    /// Start building a trainer.
    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Number of distinct words fed so far.
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Words sorted for deterministic processing.
    fn sorted_words(word_counts: &HashMap<String, u64>) -> Vec<(&String, u64)> {
        let mut v: Vec<(&String, u64)> = word_counts.iter().map(|(w, c)| (w, *c)).collect();
        v.sort_unstable_by(|a, b| a.0.cmp(b.0));
        v
    }

    fn compute_alphabet(&self, words: &[(&String, u64)], vocab: &mut VocabBuilder) {
        let mut alphabet: FxHashMap<char, usize> = FxHashMap::default();
        for (word, count) in words {
            for c in word.chars() {
                let slot = alphabet.entry(c).or_default();
                *slot = slot.saturating_add(*count as usize);
            }
        }
        for c in &self.initial_alphabet {
            alphabet.insert(*c, usize::MAX);
        }
        let mut kept: Vec<(char, usize)> = alphabet.into_iter().collect();
        let to_remove = self
            .limit_alphabet
            .map_or(0, |limit| kept.len().saturating_sub(limit));
        if to_remove > 0 {
            // Drop the rarest chars (ties: larger code point first, for
            // determinism).
            kept.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
            kept.drain(..to_remove);
        }
        kept.sort_unstable_by_key(|(c, _)| *c as u32);
        for (c, _) in kept {
            vocab.get_or_insert(c.encode_utf8(&mut [0; 4]));
        }
    }

    fn tokenize_words(&self, words: &[(&String, u64)], vocab: &mut VocabBuilder) -> Vec<Word> {
        let mut out = Vec::with_capacity(words.len());
        for (word, _) in words {
            let mut w = Word::with_capacity(word.len());
            let mut chars = word.chars().peekable();
            let mut first = true;
            while let Some(c) = chars.next() {
                let is_last = chars.peek().is_none();
                let mut s = c.to_string();
                if vocab.contains(&s) {
                    if !first {
                        if let Some(p) = &self.continuing_subword_prefix {
                            s.insert_str(0, p);
                        }
                    }
                    if is_last {
                        if let Some(sfx) = &self.end_of_word_suffix {
                            s.push_str(sfx);
                        }
                    }
                    w.add(vocab.get_or_insert(&s), 1);
                }
                first = false;
            }
            out.push(w);
        }
        out
    }

    fn count_pairs(
        words: &[Word],
        counts: &[u64],
    ) -> (FxHashMap<Pair, i64>, FxHashMap<Pair, FxHashSet<usize>>) {
        words
            .par_iter()
            .enumerate()
            .fold(
                || (FxHashMap::default(), FxHashMap::default()),
                |(mut pc, mut wtu): (FxHashMap<Pair, i64>, FxHashMap<Pair, FxHashSet<usize>>),
                 (i, word)| {
                    let ids = word.id_vec();
                    for w in ids.windows(2) {
                        let pair = (w[0], w[1]);
                        *pc.entry(pair).or_default() += counts[i] as i64;
                        wtu.entry(pair).or_default().insert(i);
                    }
                    (pc, wtu)
                },
            )
            .reduce(
                || (FxHashMap::default(), FxHashMap::default()),
                |(mut pc, mut wtu), (pc2, wtu2)| {
                    for (k, v) in pc2 {
                        *pc.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu2 {
                        wtu.entry(k).or_default().extend(v);
                    }
                    (pc, wtu)
                },
            )
    }

    /// Train `model` from explicit word counts.
    pub(crate) fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut Bpe,
    ) -> Result<Vec<AddedToken>> {
        let max_token_length = self.max_token_length.unwrap_or(usize::MAX);
        let mut vocab = VocabBuilder::default();

        // 1. Special tokens first.
        for t in &self.special_tokens {
            vocab.get_or_insert(&t.content);
        }

        // 2. Alphabet, then 3. split words into alphabet symbols.
        let sorted = Self::sorted_words(word_counts);
        self.compute_alphabet(&sorted, &mut vocab);
        let progress = Progress::new(
            self.show_progress,
            "Tokenize words",
            Some(sorted.len() as u64),
        );
        let mut words = self.tokenize_words(&sorted, &mut vocab);
        progress.set_position(sorted.len() as u64);
        progress.finish();
        let counts: Vec<u64> = sorted.iter().map(|(_, c)| *c).collect();

        // 4. Initial pair counts.
        let progress = Progress::new(self.show_progress, "Count pairs", Some(words.len() as u64));
        let (mut pair_counts, mut where_to_update) = Self::count_pairs(&words, &counts);
        progress.set_position(words.len() as u64);
        progress.finish();
        let mut queue: BinaryHeap<Candidate> = BinaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Candidate {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        }

        // 5. Merge until the vocabulary is big enough.
        let progress = Progress::new(
            self.show_progress,
            "Compute merges",
            Some(self.vocab_size.saturating_sub(vocab.len()) as u64),
        );
        let initial_vocab = vocab.len();
        let mut merges: Vec<(Pair, u32)> = Vec::new();
        while vocab.len() < self.vocab_size {
            let Some(mut top) = queue.pop() else { break };
            let current = pair_counts.get(&top.pair).copied().unwrap_or(0);
            if top.count as i64 != current {
                top.count = current.max(0) as u64;
                queue.push(top);
                continue;
            }
            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let a = &vocab.tokens[top.pair.0 as usize];
            let b = &vocab.tokens[top.pair.1 as usize];
            let b = match &self.continuing_subword_prefix {
                Some(p) => b.strip_prefix(p.as_str()).unwrap_or(b),
                None => b,
            };
            let new_token = format!("{a}{b}");
            let new_id = vocab.get_or_insert(&new_token);
            merges.push((top.pair, new_id));
            progress.set_position((vocab.len() - initial_vocab) as u64);

            let mut positions: Vec<usize> = top.pos.iter().copied().collect();
            positions.sort_unstable();
            for i in positions {
                for (pair, change) in
                    words[i].merge(top.pair.0, top.pair.1, new_id, max_token_length)
                {
                    *pair_counts.entry(pair).or_default() += change as i64 * counts[i] as i64;
                    if change > 0 {
                        where_to_update.entry(pair).or_default().insert(i);
                    }
                }
            }
            for (pair, pos) in where_to_update.drain() {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Candidate {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            }
        }

        progress.finish();

        // Later duplicates of a pair override earlier ranks, as in HF.
        let mut merge_map = MergeMap::default();
        for (rank, (pair, new_id)) in merges.into_iter().enumerate() {
            merge_map.insert(pair, (rank as u32, new_id));
        }
        let vocab_map: HashMap<String, u32> = vocab.to_id.into_iter().collect();
        model.set_trained(
            vocab_map,
            merge_map,
            self.continuing_subword_prefix.clone(),
            self.end_of_word_suffix.clone(),
        );
        Ok(self.special_tokens.clone())
    }
}

/// Count words from a corpus in parallel.
pub(crate) fn count_words<I, S, F>(
    iterator: I,
    process: F,
    show_progress: bool,
) -> Result<HashMap<String, u64>>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let progress = Progress::new(show_progress, "Pre-processing sequences", None);
    let counts = iterator
        .par_bridge()
        .map(|sequence| -> Result<HashMap<String, u64>> {
            progress.inc(1);
            let mut map = HashMap::new();
            for word in process(sequence.as_ref())? {
                *map.entry(word).or_default() += 1;
            }
            Ok(map)
        })
        .try_reduce(HashMap::new, |mut acc, other| {
            for (k, v) in other {
                *acc.entry(k).or_default() += v;
            }
            Ok(acc)
        });
    progress.finish();
    counts
}

impl Trainer for BpeTrainer {
    type Model = Bpe;

    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn train(&self, model: &mut Bpe) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.words = count_words(iterator, process, self.show_progress)?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::traits::Model;

    /// HF `Whitespace` pre-tokenization: `\w+|[^\w\s]+`.
    pub(crate) fn whitespace_words(s: &str) -> Result<Vec<String>> {
        let re = fancy_regex::Regex::new(r"\w+|[^\w\s]+").unwrap();
        Ok(re
            .find_iter(s)
            .map(|m| m.unwrap().as_str().to_owned())
            .collect())
    }

    pub(crate) fn train(trainer: &mut BpeTrainer, corpus: &[&str]) -> Bpe {
        trainer
            .feed(corpus.iter().copied(), whitespace_words)
            .unwrap();
        let mut model = Bpe::default();
        trainer.train(&mut model).unwrap();
        model
    }

    pub(crate) type Dump = (Vec<(String, u32)>, Vec<(String, String)>);

    /// Vocab as (token, id) sorted by id, and merges as strings.
    pub(crate) fn dump(model: &Bpe) -> Dump {
        let mut v: Vec<(String, u32)> = model.vocab().into_iter().collect();
        v.sort_by_key(|(_, id)| *id);
        (v, model.merges().to_vec())
    }

    fn owned(v: &[(&str, u32)]) -> Vec<(String, u32)> {
        v.iter().map(|(s, i)| (s.to_string(), *i)).collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    // Expected values below come from Python `tokenizers` 0.23.2:
    //   tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.Whitespace()
    //   tok.train_from_iterator(corpus, trainers.BpeTrainer(...))
    //   json.loads(tok.to_str())["model"]

    #[test]
    fn parity_basic() {
        let mut t = BpeTrainer::builder()
            .vocab_size(30)
            .show_progress(false)
            .special_tokens(vec![AddedToken::new("[UNK]", true)])
            .build()
            .unwrap();
        let m = train(
            &mut t,
            &["hello hello world", "low lower lowest", "aaaa aaa aa a"],
        );
        let (vocab, merges) = dump(&m);
        assert_eq!(vocab, owned(PARITY_BASIC_VOCAB));
        assert_eq!(merges, pairs(PARITY_BASIC_MERGES));
    }

    #[test]
    fn parity_unicode_and_min_frequency() {
        let mut t = BpeTrainer::builder()
            .vocab_size(40)
            .min_frequency(2)
            .show_progress(false)
            .build()
            .unwrap();
        let m = train(
            &mut t,
            &["café café naïve 你好 你好世界", "über über alles 😀😀 😀"],
        );
        let (vocab, merges) = dump(&m);
        assert_eq!(vocab, owned(PARITY_UNICODE_VOCAB));
        assert_eq!(merges, pairs(PARITY_UNICODE_MERGES));
    }

    #[test]
    fn parity_suffix_and_max_token_length() {
        let mut t = BpeTrainer::builder()
            .vocab_size(50)
            .end_of_word_suffix("</w>")
            .max_token_length(3)
            .show_progress(false)
            .build()
            .unwrap();
        let m = train(&mut t, &["the then there these thesis", "the the the"]);
        let (vocab, merges) = dump(&m);
        let set = |v: Vec<(String, u32)>| v.into_iter().map(|(s, _)| s).collect::<HashSet<_>>();
        assert_eq!(set(vocab), set(owned(PARITY_SUFFIX_VOCAB)));
        assert_eq!(merges, pairs(PARITY_SUFFIX_MERGES));
    }

    #[test]
    fn parity_limit_alphabet_and_initial_alphabet() {
        let mut t = BpeTrainer::builder()
            .vocab_size(20)
            .limit_alphabet(4)
            .initial_alphabet(['z'].into_iter().collect())
            .show_progress(false)
            .build()
            .unwrap();
        let m = train(&mut t, &["abababab cdcdc e", "abab ab"]);
        let (vocab, merges) = dump(&m);
        assert_eq!(vocab, owned(PARITY_LIMIT_VOCAB));
        assert_eq!(merges, pairs(PARITY_LIMIT_MERGES));
    }

    #[test]
    fn trained_model_tokenizes_its_corpus() {
        let mut t = BpeTrainer::builder()
            .vocab_size(100)
            .show_progress(false)
            .build()
            .unwrap();
        let m = train(&mut t, &["the quick brown fox jumps over the lazy dog"]);
        let toks = m.tokenize("quick").unwrap();
        assert_eq!(toks.len(), 1, "{toks:?}");
        assert_eq!(toks[0].value, "quick");
    }

    include!("parity_data.rs");
}
