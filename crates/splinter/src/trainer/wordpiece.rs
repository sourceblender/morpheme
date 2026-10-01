//! WordPiece trainer — Sennrich-style "additive" WordPiece.
//!
//! The algorithm builds the vocab *up* to a target size rather than
//! trimming down, by repeatedly picking the substring whose
//! inclusion raises the corpus likelihood the most. Score for a
//! candidate `c`:
//!
//! ```text
//! score(c) = freq(c) / sum_over_occurrences(freq(left) * freq(right) * word_freq)
//! ```
//!
//! where `left`/`right` are the boundary chars at each occurrence of
//! `c` in the corpus, and `freq(...)` is the per-character total
//! frequency (which never changes during training).
//!
//! v0.1 keeps this single-threaded and rebuilds the substring
//! frequency table from scratch each iteration. That's O(n * l^2)
//! per iteration where `n` is corpus size and `l` is the average
//! word length — fine for small/medium corpora. Phase 4 perf work
//! will introduce incremental updates.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use super::Corpus;
use crate::error::{Error as SplinterError, Result};
use crate::pre_tokenizer::{PreTokenizer, Whitespace};
use crate::vocab::Vocab;

/// Output of [`WordPieceTrainer`].
#[derive(Debug, Clone)]
pub struct WordPieceTrainedModel {
    /// Ordered vocabulary. Id `i` is the token at index `i`.
    pub vocab: Vocab,
    /// Continuing-subword prefix. Every non-initial subword in a
    /// word carries this prefix. BERT uses `##`.
    pub continuing_subword_prefix: String,
    /// End-of-word marker (kept for symmetry with BPE config).
    pub end_of_word_suffix: String,
}

/// Configuration for [`WordPieceTrainer`].
#[derive(Debug, Clone)]
pub struct WordPieceTrainerOpts {
    /// Stop once the vocab reaches this size. Required.
    pub vocab_size: usize,
    /// End-of-word marker.
    pub end_of_word_suffix: String,
    /// Continuing-subword prefix.
    pub continuing_subword_prefix: String,
    /// Minimum frequency for a candidate to be eligible.
    pub min_frequency: u64,
    /// Optional alphabet — symbols pre-populated in the vocab.
    pub alphabet: Vec<String>,
    /// If true, the trainer pre-tokenizes each line on whitespace.
    pub pre_tokenize: bool,
}

impl WordPieceTrainerOpts {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(SplinterError::Model("vocab_size must be > 0".to_string()));
        }
        Ok(())
    }
}

/// Builder for [`WordPieceTrainer`].
pub struct WordPieceTrainerBuilder {
    vocab_size: usize,
    end_of_word_suffix: String,
    continuing_subword_prefix: String,
    min_frequency: u64,
    alphabet: Vec<String>,
    pre_tokenize: bool,
}

impl WordPieceTrainerBuilder {
    /// Start building a `WordPieceTrainer` targeting `vocab_size`.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            end_of_word_suffix: "</w>".to_string(),
            continuing_subword_prefix: "##".to_string(),
            min_frequency: 0,
            alphabet: Vec::new(),
            pre_tokenize: true,
        }
    }

    /// Set the continuing-subword prefix. Default `##`.
    pub fn continuing_subword_prefix(mut self, s: impl Into<String>) -> Self {
        self.continuing_subword_prefix = s.into();
        self
    }

    /// Set the end-of-word suffix. Default `</w>`.
    pub fn end_of_word_suffix(mut self, s: impl Into<String>) -> Self {
        self.end_of_word_suffix = s.into();
        self
    }

    /// Minimum frequency for a candidate to be eligible.
    pub fn min_frequency(mut self, n: u64) -> Self {
        self.min_frequency = n;
        self
    }

    /// Add symbols to the pre-populated alphabet.
    pub fn alphabet(mut self, symbols: impl IntoIterator<Item = String>) -> Self {
        self.alphabet.extend(symbols);
        self
    }

    /// If false, each line of the corpus is treated as a single word.
    pub fn pre_tokenize(mut self, yes: bool) -> Self {
        self.pre_tokenize = yes;
        self
    }

    /// Build the trainer.
    pub fn build(self) -> WordPieceTrainer {
        WordPieceTrainer {
            opts: WordPieceTrainerOpts {
                vocab_size: self.vocab_size,
                end_of_word_suffix: self.end_of_word_suffix,
                continuing_subword_prefix: self.continuing_subword_prefix,
                min_frequency: self.min_frequency,
                alphabet: self.alphabet,
                pre_tokenize: self.pre_tokenize,
            },
        }
    }
}

/// WordPiece trainer. Build with [`WordPieceTrainerBuilder`].
pub struct WordPieceTrainer {
    opts: WordPieceTrainerOpts,
}

impl WordPieceTrainer {
    /// Construct from explicit opts.
    pub fn new(opts: WordPieceTrainerOpts) -> Result<Self> {
        opts.validate()?;
        Ok(Self { opts })
    }

    /// Start building a trainer.
    pub fn builder(vocab_size: usize) -> WordPieceTrainerBuilder {
        WordPieceTrainerBuilder::new(vocab_size)
    }

    /// Train from a corpus.
    pub fn train<C: Corpus>(&self, corpus: C) -> Result<WordPieceTrainedModel> {
        let ws = Whitespace;

        // 1. Word frequencies.
        let mut freq: FxHashMap<String, u64> = FxHashMap::default();
        corpus.for_each_line(|line| {
            if self.opts.pre_tokenize {
                for pt in ws
                    .pre_tokenize(line)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| p.text.as_str().to_string())
                {
                    if !pt.is_empty() {
                        *freq.entry(pt).or_insert(0) += 1;
                    }
                }
            } else if !line.is_empty() {
                *freq.entry(line.to_string()).or_insert(0) += 1;
            }
        });

        if freq.is_empty() {
            return Err(SplinterError::Model("corpus is empty".to_string()));
        }

        // 2. Character frequency across the corpus. Each word contributes
        //    its char-frequency * word-frequency. End-of-word marker
        //    gets attached to the last character of every word.
        let mut char_freq: FxHashMap<String, u64> = FxHashMap::default();
        for (word, &f) in &freq {
            let chars: Vec<char> = word.chars().collect();
            for (i, c) in chars.iter().enumerate() {
                let mut s = String::new();
                s.push(*c);
                if i == chars.len() - 1 {
                    s.push_str(&self.opts.end_of_word_suffix);
                }
                *char_freq.entry(s).or_insert(0) += f;
            }
        }

        // 3. Initial vocab: alphabet + every char with freq >= min_frequency.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut vocab_list: Vec<String> = Vec::new();
        for sym in &self.opts.alphabet {
            if seen.insert(sym.clone()) {
                vocab_list.push(sym.clone());
            }
        }
        let mut chars_sorted: Vec<(String, u64)> = char_freq
            .iter()
            .filter(|(_, &c)| c >= self.opts.min_frequency)
            .map(|(k, &c)| (k.clone(), c))
            .collect();
        chars_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        for (c, _) in chars_sorted {
            if seen.insert(c.clone()) {
                vocab_list.push(c);
            }
        }

        // 4. Iteratively add the highest-scoring candidate.
        while vocab_list.len() < self.opts.vocab_size {
            let Some(candidate) = best_candidate(&freq, &char_freq, &vocab_list, &self.opts) else {
                break;
            };
            if !seen.insert(candidate.clone()) {
                continue;
            }
            vocab_list.push(candidate);
        }

        let vocab = Vocab::from_tokens(vocab_list)
            .map_err(|e| SplinterError::Model(format!("vocab construction failed: {e}")))?;

        Ok(WordPieceTrainedModel {
            vocab,
            continuing_subword_prefix: self.opts.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.opts.end_of_word_suffix.clone(),
        })
    }
}

/// Build the per-word symbolization: every char of the word is a
/// separate symbol, with the continuing-subword prefix prepended to
/// non-initial chars and the end-of-word suffix appended to the last
/// char.
fn word_syms(word: &str, continuing_prefix: &str, end_of_word_suffix: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut out: Vec<String> = Vec::with_capacity(chars.len());
    for (i, c) in chars.iter().enumerate() {
        let mut s = String::new();
        if i > 0 {
            s.push_str(continuing_prefix);
        }
        s.push(*c);
        if i == chars.len() - 1 {
            s.push_str(end_of_word_suffix);
        }
        out.push(s);
    }
    out
}

/// Strip the `##` prefix and `</w>` suffix from a single symbol to get
/// the raw character.
fn raw_char_from_sym(sym: &str, continuing_prefix: &str, end_of_word_suffix: &str) -> String {
    let stripped_prefix = sym.strip_prefix(continuing_prefix).unwrap_or(sym);
    let stripped_suffix = stripped_prefix
        .strip_suffix(end_of_word_suffix)
        .unwrap_or(stripped_prefix);
    stripped_suffix.to_string()
}

/// Per-occurrence info: which word and which `(sym_start, sym_end)`
/// the candidate appears at.
type Occurrences = HashMap<String, Vec<(String, usize, usize)>>;

/// Compute the substring frequency map for the corpus given the
/// current vocab. Returns `(candidate_freq, candidate_occurrences)`
/// where `occurrences[cand]` lists `(word, sym_start, sym_end)` for
/// each occurrence of the candidate.
fn substring_freqs(
    word_freq: &FxHashMap<String, u64>,
    vocab: &[String],
    end_of_word_suffix: &str,
    continuing_prefix: &str,
) -> (HashMap<String, u64>, Occurrences) {
    let mut out: HashMap<String, u64> = HashMap::new();
    let mut occ: HashMap<String, Vec<(String, usize, usize)>> = HashMap::new();
    let vocab_set: std::collections::HashSet<&str> = vocab.iter().map(String::as_str).collect();

    for (word, &f) in word_freq {
        let syms = word_syms(word, continuing_prefix, end_of_word_suffix);
        if syms.len() < 2 {
            continue;
        }
        // Enumerate all substrings of length >= 2 symbols.
        for i in 0..syms.len() - 1 {
            for j in i + 2..=syms.len() {
                let candidate: String = syms[i..j].concat();
                if vocab_set.contains(candidate.as_str()) {
                    continue;
                }
                *out.entry(candidate.clone()).or_insert(0) += f;
                occ.entry(candidate)
                    .or_default()
                    .push((word.to_string(), i, j));
            }
        }
    }

    (out, occ)
}

/// Score = freq(c) / sum_over_occurrences(freq(left) * freq(right) * word_freq).
///
/// `left`/`right` are the boundary characters at each occurrence of
/// `c` in the corpus (or `1` if at the start/end of a word). The
/// char frequencies never change during training.
fn best_candidate(
    word_freq: &FxHashMap<String, u64>,
    char_freq: &FxHashMap<String, u64>,
    vocab: &[String],
    opts: &WordPieceTrainerOpts,
) -> Option<String> {
    let (cands, occs) = substring_freqs(
        word_freq,
        vocab,
        &opts.end_of_word_suffix,
        &opts.continuing_subword_prefix,
    );
    let mut best: Option<(String, f64)> = None;
    for (cand, &freq) in &cands {
        if freq < opts.min_frequency {
            continue;
        }
        let Some(occ_list) = occs.get(cand) else {
            continue;
        };
        let mut denom: f64 = 0.0;
        for (word, sym_start, sym_end) in occ_list {
            let wf = *word_freq.get(word).unwrap_or(&0) as f64;
            let syms = word_syms(
                word,
                &opts.continuing_subword_prefix,
                &opts.end_of_word_suffix,
            );
            let n = syms.len();

            // Left boundary char (1.0 if at start of word).
            let left_char = if *sym_start == 0 {
                1.0
            } else {
                let raw = raw_char_from_sym(
                    &syms[sym_start - 1],
                    &opts.continuing_subword_prefix,
                    &opts.end_of_word_suffix,
                );
                let raw_with_suffix = if *sym_start - 1 == n - 1 {
                    format!("{}{}", raw, opts.end_of_word_suffix)
                } else {
                    raw
                };
                *char_freq.get(&raw_with_suffix).unwrap_or(&0) as f64
            };

            // Right boundary char (1.0 if at end of word).
            let right_char = if *sym_end == n {
                1.0
            } else {
                let raw = raw_char_from_sym(
                    &syms[*sym_end],
                    &opts.continuing_subword_prefix,
                    &opts.end_of_word_suffix,
                );
                let raw_with_suffix = if *sym_end == n - 1 {
                    format!("{}{}", raw, opts.end_of_word_suffix)
                } else {
                    raw
                };
                *char_freq.get(&raw_with_suffix).unwrap_or(&0) as f64
            };

            denom += left_char * right_char * wf;
        }

        if denom <= 0.0 {
            continue;
        }
        let score = (freq as f64) / denom;
        match best {
            None => best = Some((cand.clone(), score)),
            Some((_, s)) if score > s => best = Some((cand.clone(), score)),
            _ => {}
        }
    }
    best.map(|(c, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> &'static str {
        "aa bb cc aa bb hello hello hello world world"
    }

    #[test]
    fn train_includes_alphabet_and_subwords() {
        let trainer = WordPieceTrainer::builder(20).build();
        let m = trainer.train(corpus()).unwrap();
        // Initial alphabet symbols from the corpus must appear.
        assert!(m.vocab.token_to_id("a</w>").is_some());
        assert!(m.vocab.token_to_id("b</w>").is_some());
        // Common subwords should also be learned. WordPiece emits
        // `##` continuing prefixes on non-initial chars, so the
        // full-word token for `hello` is `h##e##l##l##o</w>`.
        assert!(m.vocab.token_to_id("h##e##l##l##o</w>").is_some());
        assert!(m.vocab.token_to_id("w##o##r##l##d</w>").is_some());
    }

    #[test]
    fn train_alphabet_prepopulated_first() {
        let alphabet = vec!["<unk>".to_string()];
        let trainer = WordPieceTrainer::builder(20).alphabet(alphabet).build();
        let m = trainer.train(corpus()).unwrap();
        // <unk> was added first, so its id is 0.
        assert_eq!(m.vocab.token_to_id("<unk>"), Some(0));
    }

    #[test]
    fn train_empty_corpus_errors() {
        let trainer = WordPieceTrainer::builder(20).build();
        assert!(trainer.train("").is_err());
    }

    #[test]
    fn train_min_frequency_skips_rare_subwords() {
        let trainer = WordPieceTrainer::builder(50).build();
        let m = trainer.train(corpus()).unwrap();
        assert!(m.vocab.token_to_id("z</w>").is_none());
    }

    #[test]
    fn train_target_vocab_size_respected() {
        let trainer = WordPieceTrainer::builder(15).build();
        let m = trainer.train(corpus()).unwrap();
        assert!(m.vocab.len() <= 15);
    }

    #[test]
    fn train_continuing_prefix_default() {
        let trainer = WordPieceTrainer::builder(50).build();
        let m = trainer.train(corpus()).unwrap();
        assert_eq!(m.continuing_subword_prefix, "##");
    }
}
