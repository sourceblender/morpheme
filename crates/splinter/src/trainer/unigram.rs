//! Unigram (SentencePiece) trainer — EM-based.
//!
//! The trained model is a probability distribution over subwords;
//! tokenization is Viterbi-best-path over a subword lattice. The
//! algorithm:
//!
//! 1. Build a *seed* candidate set: every unique character plus
//!    every substring (length ≤ `max_subword_len`) that appears at
//!    least `seed_min_frequency` times.
//! 2. Initialize subword probabilities from corpus frequencies
//!    (`p ∝ freq`) with a small smoothing term.
//! 3. EM loop, `num_epochs` iterations:
//!    - **E-step**: for each word, run the forward algorithm to
//!      compute the expected count of every subword in that word.
//!    - **M-step**: update `p(subword) ∝ expected count`.
//!    - **Prune**: drop any subword whose expected count falls
//!      below `min_frequency`. Recompute the partition function.
//! 4. Return the surviving vocab + per-subword log probabilities.
//!
//! v0.1 keeps this single-threaded and rebuilds the lattice per
//! E-step. Phase 4 perf work can swap in incremental updates.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use super::Corpus;
use crate::error::{Error as SplinterError, Result};
use crate::pre_tokenizer::{PreTokenizer, Whitespace};
use crate::vocab::Vocab;

/// Output of [`UnigramTrainer`].
#[derive(Debug, Clone)]
pub struct UnigramTrainedModel {
    /// Ordered vocabulary. Id `i` is the token at index `i`.
    pub vocab: Vocab,
    /// Log-probability of each subword, in the same order as
    /// `vocab`. Use [`UnigramTrainedModel::log_prob`] for safe
    /// lookup by id.
    pub log_probs: Vec<f64>,
    /// Minimum log-probability threshold used during Viterbi — any
    /// subword with a log-prob at or below this is treated as unk.
    pub min_score: f64,
    /// Continuing-subword marker. SentencePiece uses `▁` (U+2581).
    /// The model wraps each word's leading whitespace as this
    /// character.
    pub whitespace_marker: String,
    /// End-of-word marker (not consumed at decode time but kept
    /// for symmetry with BPE/WordPiece configs).
    pub end_of_word_suffix: String,
}

impl UnigramTrainedModel {
    /// Look up the log-probability of the subword at `id`.
    /// Returns `None` if `id` is out of range.
    pub fn log_prob(&self, id: u32) -> Option<f64> {
        self.log_probs.get(id as usize).copied()
    }
}

/// Configuration for [`UnigramTrainer`].
#[derive(Debug, Clone)]
pub struct UnigramTrainerOpts {
    /// Target final vocabulary size. Required.
    pub vocab_size: usize,
    /// End-of-word marker. Default `</w>`.
    pub end_of_word_suffix: String,
    /// Whitespace marker prepended to words that don't start with
    /// whitespace. Default `▁` (U+2581). The trainer converts
    /// leading ASCII spaces in the corpus to this marker.
    pub whitespace_marker: String,
    /// Maximum subword length (in characters). Default 16.
    pub max_subword_len: usize,
    /// Minimum frequency for a subword candidate to be included in
    /// the seed vocab. Default 2.
    pub seed_min_frequency: u64,
    /// Minimum expected count below which a subword is pruned
    /// during EM. Default 1.
    pub min_frequency: u64,
    /// Number of EM epochs. Default 20.
    pub num_epochs: usize,
    /// Smoothing constant added to every subword's probability.
    /// Default 1e-5.
    pub smoothing: f64,
    /// Optional alphabet — symbols pre-populated in the seed vocab
    /// (e.g. `<unk>`, byte-level bytes).
    pub alphabet: Vec<String>,
    /// If true, the trainer pre-tokenizes each line on whitespace.
    pub pre_tokenize: bool,
}

impl UnigramTrainerOpts {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(SplinterError::Model("vocab_size must be > 0".to_string()));
        }
        if self.max_subword_len == 0 {
            return Err(SplinterError::Model(
                "max_subword_len must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Builder for [`UnigramTrainer`].
pub struct UnigramTrainerBuilder {
    vocab_size: usize,
    end_of_word_suffix: String,
    whitespace_marker: String,
    max_subword_len: usize,
    seed_min_frequency: u64,
    min_frequency: u64,
    num_epochs: usize,
    smoothing: f64,
    alphabet: Vec<String>,
    pre_tokenize: bool,
}

impl UnigramTrainerBuilder {
    /// Start building a `UnigramTrainer` targeting `vocab_size`.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            end_of_word_suffix: "</w>".to_string(),
            whitespace_marker: "\u{2581}".to_string(),
            max_subword_len: 16,
            seed_min_frequency: 2,
            min_frequency: 1,
            num_epochs: 20,
            smoothing: 1e-5,
            alphabet: Vec::new(),
            pre_tokenize: true,
        }
    }

    /// Set the end-of-word suffix. Default `</w>`.
    pub fn end_of_word_suffix(mut self, s: impl Into<String>) -> Self {
        self.end_of_word_suffix = s.into();
        self
    }

    /// Set the whitespace marker. Default `▁` (U+2581).
    pub fn whitespace_marker(mut self, s: impl Into<String>) -> Self {
        self.whitespace_marker = s.into();
        self
    }

    /// Set the maximum subword length. Default 16.
    pub fn max_subword_len(mut self, n: usize) -> Self {
        self.max_subword_len = n;
        self
    }

    /// Minimum frequency for a subword candidate to be included in
    /// the seed vocab. Default 2.
    pub fn seed_min_frequency(mut self, n: u64) -> Self {
        self.seed_min_frequency = n;
        self
    }

    /// Minimum expected count below which a subword is pruned during
    /// EM. Default 1.
    pub fn min_frequency(mut self, n: u64) -> Self {
        self.min_frequency = n;
        self
    }

    /// Number of EM epochs. Default 20.
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    /// Smoothing constant. Default 1e-5.
    pub fn smoothing(mut self, s: f64) -> Self {
        self.smoothing = s;
        self
    }

    /// Add symbols to the pre-populated alphabet.
    pub fn alphabet(mut self, symbols: impl IntoIterator<Item = String>) -> Self {
        self.alphabet.extend(symbols);
        self
    }

    /// If false, each line is treated as a single word.
    pub fn pre_tokenize(mut self, yes: bool) -> Self {
        self.pre_tokenize = yes;
        self
    }

    /// Build the trainer.
    pub fn build(self) -> UnigramTrainer {
        UnigramTrainer {
            opts: UnigramTrainerOpts {
                vocab_size: self.vocab_size,
                end_of_word_suffix: self.end_of_word_suffix,
                whitespace_marker: self.whitespace_marker,
                max_subword_len: self.max_subword_len,
                seed_min_frequency: self.seed_min_frequency,
                min_frequency: self.min_frequency,
                num_epochs: self.num_epochs,
                smoothing: self.smoothing,
                alphabet: self.alphabet,
                pre_tokenize: self.pre_tokenize,
            },
        }
    }
}

/// Unigram trainer. Build with [`UnigramTrainerBuilder`].
pub struct UnigramTrainer {
    opts: UnigramTrainerOpts,
}

impl UnigramTrainer {
    /// Construct from explicit opts.
    pub fn new(opts: UnigramTrainerOpts) -> Result<Self> {
        opts.validate()?;
        Ok(Self { opts })
    }

    /// Start building a trainer.
    pub fn builder(vocab_size: usize) -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::new(vocab_size)
    }

    /// Train from a corpus.
    pub fn train<C: Corpus>(&self, corpus: C) -> Result<UnigramTrainedModel> {
        let ws = Whitespace;

        // 1. Word frequencies, with the whitespace marker prepended
        //    to words that don't begin with whitespace.
        let mut word_freq: FxHashMap<String, u64> = FxHashMap::default();
        corpus.for_each_line(|line| {
            if self.opts.pre_tokenize {
                for pt in ws
                    .pre_tokenize(line)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| p.text.as_str().to_string())
                {
                    if !pt.is_empty() {
                        let key = if pt.starts_with(' ') {
                            pt.replacen(' ', &self.opts.whitespace_marker, 1)
                        } else {
                            format!("{}{}", self.opts.whitespace_marker, pt)
                        };
                        *word_freq.entry(key).or_insert(0) += 1;
                    }
                }
            } else if !line.is_empty() {
                let key = if line.starts_with(' ') {
                    line.replacen(' ', &self.opts.whitespace_marker, 1)
                } else {
                    format!("{}{}", self.opts.whitespace_marker, line)
                };
                *word_freq.entry(key).or_insert(0) += 1;
            }
        });

        if word_freq.is_empty() {
            return Err(SplinterError::Model("corpus is empty".to_string()));
        }

        // 2. Enumerate every substring of every word (length 1..=max)
        //    as a candidate. Count occurrences weighted by word
        //    frequency.
        let mut cand_freq: HashMap<String, u64> = HashMap::new();
        for (word, &f) in &word_freq {
            let chars: Vec<char> = word.chars().collect();
            for i in 0..chars.len() {
                let max_end = (i + self.opts.max_subword_len).min(chars.len());
                let mut buf = String::new();
                for &c in &chars[i..max_end] {
                    buf.push(c);
                    *cand_freq.entry(buf.clone()).or_insert(0) += f;
                }
            }
        }

        // 3. Build the seed vocab: alphabet + every candidate with
        //    freq >= seed_min_frequency. Insertion order: alphabet
        //    first (these are exempt from pruning), then candidates
        //    sorted lexicographically.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut seed: Vec<String> = Vec::new();
        let mut pinned: Vec<bool> = Vec::new();
        for sym in &self.opts.alphabet {
            if seen.insert(sym.clone()) {
                seed.push(sym.clone());
                pinned.push(true);
            }
        }
        let mut sorted_cands: Vec<(String, u64)> = cand_freq
            .iter()
            .filter(|(_, &c)| c >= self.opts.seed_min_frequency)
            .map(|(k, &c)| (k.clone(), c))
            .collect();
        sorted_cands.sort_by(|a, b| a.0.cmp(&b.0));
        for (c, _) in sorted_cands {
            if seen.insert(c.clone()) {
                seed.push(c);
                pinned.push(false);
            }
        }
        if seed.is_empty() {
            return Err(SplinterError::Model(
                "no seed candidates — corpus too small or min_frequency too high".to_string(),
            ));
        }

        // 4. Initialize probabilities from candidate frequencies.
        let total_cand_count: u64 = cand_freq.values().sum();
        let mut probs: Vec<f64> = Vec::with_capacity(seed.len());
        for cand in &seed {
            let f = *cand_freq.get(cand).unwrap_or(&0) as f64;
            // Smoothing: every subword gets a small prior.
            probs.push(
                (f + self.opts.smoothing)
                    / (total_cand_count as f64 + self.opts.smoothing * seed.len() as f64),
            );
        }

        // 5. EM loop.
        let mut active: Vec<bool> = vec![true; seed.len()];
        let mut epoch = 0;
        while epoch < self.opts.num_epochs {
            // E-step: accumulate expected subword counts.
            let mut expected: Vec<f64> = vec![0.0; seed.len()];
            for (word, &wf) in &word_freq {
                let chars: Vec<char> = word.chars().collect();
                let forward = forward_lattice(&chars, &seed, &probs, &active);
                let backward = backward_lattice(&chars, &seed, &probs, &active);
                let total = forward.last().copied().unwrap_or(0.0);
                if total <= 0.0 {
                    continue;
                }
                // For each position j and each subword c that
                // matches chars[j..j+len], accumulate
                // expected[c] += wf * forward[j] * p(c) * backward[j+len] / total
                for (c_idx, cand) in seed.iter().enumerate() {
                    if !active[c_idx] {
                        continue;
                    }
                    let cand_chars: Vec<char> = cand.chars().collect();
                    if cand_chars.is_empty() || cand_chars.len() > chars.len() {
                        continue;
                    }
                    for j in 0..=chars.len() - cand_chars.len() {
                        if chars[j..j + cand_chars.len()] != cand_chars[..] {
                            continue;
                        }
                        expected[c_idx] +=
                            wf as f64 * forward[j] * probs[c_idx] * backward[j + cand_chars.len()]
                                / total;
                    }
                }
            }

            // M-step: re-estimate probabilities.
            let total_expected: f64 =
                expected.iter().sum::<f64>() + self.opts.smoothing * active.len() as f64;
            let mut new_probs: Vec<f64> = probs.clone();
            for i in 0..seed.len() {
                if active[i] {
                    new_probs[i] = (expected[i] + self.opts.smoothing) / total_expected;
                } else {
                    new_probs[i] = 0.0;
                }
            }

            // Prune: drop subwords whose expected count fell below
            // min_frequency, until vocab_size is reached (if we have
            // more than vocab_size active). Pinned (alphabet) entries
            // are exempt from pruning.
            let active_count = active.iter().filter(|&&a| a).count();
            if active_count > self.opts.vocab_size {
                let mut by_count: Vec<(usize, f64)> = expected
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| active[*i] && !pinned[*i])
                    .map(|(i, &c)| (i, c))
                    .collect();
                by_count.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let to_drop = active_count.saturating_sub(self.opts.vocab_size);
                for (i, _) in by_count.iter().take(to_drop) {
                    active[*i] = false;
                    new_probs[*i] = 0.0;
                }
            }
            // Also drop anyone below min_frequency even if we'd be
            // under target — keeps the model clean. Pinned entries
            // are exempt.
            for i in 0..seed.len() {
                if active[i] && !pinned[i] && expected[i] < self.opts.min_frequency as f64 {
                    active[i] = false;
                    new_probs[i] = 0.0;
                }
            }

            // Renormalize probabilities over the active set.
            let new_total: f64 = new_probs.iter().sum();
            if new_total > 0.0 {
                for p in new_probs.iter_mut() {
                    *p /= new_total;
                }
            }
            probs = new_probs;

            // Check early-exit: if active set is at target size and
            // we're not losing mass to pruning, we can stop.
            let active_count = active.iter().filter(|&&a| a).count();
            if active_count <= self.opts.vocab_size {
                // One more epoch to refine probabilities.
                if epoch + 1 >= self.opts.num_epochs {
                    break;
                }
            }
            epoch += 1;
        }

        // 6. Build final vocab list from the active set, preserving
        //    insertion order so log_probs align with vocab ids.
        let mut final_vocab: Vec<String> = Vec::new();
        let mut final_log_probs: Vec<f64> = Vec::new();
        for (i, cand) in seed.iter().enumerate() {
            if active[i] {
                final_vocab.push(cand.clone());
                final_log_probs.push(probs[i].ln());
            }
        }

        // Compute min_score as a small epsilon below the minimum
        // active log-prob. The model treats anything at-or-below this
        // as unk.
        let min_score = final_log_probs
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            - 1.0;

        let vocab = Vocab::from_tokens(final_vocab)
            .map_err(|e| SplinterError::Model(format!("vocab construction failed: {e}")))?;

        Ok(UnigramTrainedModel {
            vocab,
            log_probs: final_log_probs,
            min_score,
            whitespace_marker: self.opts.whitespace_marker.clone(),
            end_of_word_suffix: self.opts.end_of_word_suffix.clone(),
        })
    }
}

/// Forward pass on the subword lattice. `forward[i]` is the
/// (unnormalized) total probability of all segmentations of
/// `chars[0..i]`.
fn forward_lattice(chars: &[char], vocab: &[String], probs: &[f64], active: &[bool]) -> Vec<f64> {
    let n = chars.len();
    let mut forward = vec![0.0_f64; n + 1];
    forward[0] = 1.0;
    for j in 0..n {
        if forward[j] == 0.0 {
            continue;
        }
        for (c_idx, cand) in vocab.iter().enumerate() {
            if !active[c_idx] || probs[c_idx] == 0.0 {
                continue;
            }
            let cand_chars: Vec<char> = cand.chars().collect();
            if cand_chars.is_empty() || cand_chars.len() > n - j {
                continue;
            }
            if chars[j..j + cand_chars.len()] == cand_chars[..] {
                forward[j + cand_chars.len()] += forward[j] * probs[c_idx];
            }
        }
    }
    forward
}

/// Backward pass on the subword lattice. `backward[i]` is the
/// (unnormalized) total probability of all segmentations of
/// `chars[i..n]`.
fn backward_lattice(chars: &[char], vocab: &[String], probs: &[f64], active: &[bool]) -> Vec<f64> {
    let n = chars.len();
    let mut backward = vec![0.0_f64; n + 1];
    backward[n] = 1.0;
    for i in (0..n).rev() {
        for (c_idx, cand) in vocab.iter().enumerate() {
            if !active[c_idx] || probs[c_idx] == 0.0 {
                continue;
            }
            let cand_chars: Vec<char> = cand.chars().collect();
            if cand_chars.is_empty() || cand_chars.len() > n - i {
                continue;
            }
            if chars[i..i + cand_chars.len()] == cand_chars[..] {
                backward[i] += probs[c_idx] * backward[i + cand_chars.len()];
            }
        }
    }
    backward
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> &'static str {
        "aa bb cc aa bb hello hello hello world world"
    }

    #[test]
    fn train_basic_runs() {
        let trainer = UnigramTrainer::builder(20).build();
        let m = trainer.train(corpus()).unwrap();
        assert!(m.vocab.len() <= 20);
        assert!(m.vocab.len() > 1);
        assert!(!m.log_probs.is_empty());
    }

    #[test]
    fn train_empty_corpus_errors() {
        let trainer = UnigramTrainer::builder(20).build();
        assert!(trainer.train("").is_err());
    }

    #[test]
    fn train_target_vocab_size_respected() {
        let trainer = UnigramTrainer::builder(10).build();
        let m = trainer.train(corpus()).unwrap();
        assert!(m.vocab.len() <= 10);
    }

    #[test]
    fn train_alphabet_prepopulated() {
        // Alphabet symbols are added to the *seed* vocab but may
        // be pruned during EM if their expected count is low.
        // Use a large target vocab so `<unk>` survives pruning.
        let trainer = UnigramTrainer::builder(200)
            .alphabet(vec!["<unk>".to_string()])
            .build();
        let m = trainer.train(corpus()).unwrap();
        assert!(m.vocab.token_to_id("<unk>").is_some());
    }

    #[test]
    fn train_log_probs_are_finite() {
        let trainer = UnigramTrainer::builder(20).build();
        let m = trainer.train(corpus()).unwrap();
        for &p in &m.log_probs {
            assert!(p.is_finite(), "log-prob {p} is not finite");
        }
    }
}
