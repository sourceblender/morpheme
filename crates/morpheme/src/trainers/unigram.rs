//! Unigram (SentencePiece) trainer.
//!
//! 1. Seed the vocabulary with every char plus the most frequent
//!    substrings (internal nodes of the corpus suffix tree, scored by
//!    `frequency × length`).
//! 2. Run EM (with a digamma sparse prior) to re-estimate piece scores.
//! 3. Prune the pieces whose removal costs the least likelihood,
//!    shrinking by `shrinking_factor` per round, until the vocabulary is
//!    close to `vocab_size`.
//! 4. Keep every required char, add special tokens, and cut to
//!    `vocab_size`.

use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::added_vocabulary::AddedToken;
use crate::error::{Error, Result};
use crate::models::unigram::{Lattice, Unigram};
use crate::progress::Progress;
use crate::traits::Trainer;

/// A piece and its score.
type SentencePiece = (String, f64);
/// A word (pre-token) and its count in the corpus.
type Sentence = (String, u32);

/// Sentences per work unit in parallel passes. Fixed (not tied to the
/// number of threads) so floating-point sums, and therefore results, are
/// identical on every machine.
const CHUNK_SIZE: usize = 512;

/// Trains a [`Unigram`] model. Build with [`UnigramTrainer::builder`].
///
/// # Example
///
/// ```
/// use morpheme::models::Unigram;
/// use morpheme::pre_tokenizers::{Metaspace, PrependScheme};
/// use morpheme::trainers::UnigramTrainer;
/// use morpheme::{AddedToken, Tokenizer};
///
/// let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
/// let mut tokenizer = Tokenizer::new(Unigram::default())
///     .with_pre_tokenizer(metaspace.clone())
///     .with_decoder(metaspace);
/// let trainer = UnigramTrainer::builder()
///     .vocab_size(40)
///     .unk_token("<unk>")
///     .special_tokens(vec![AddedToken::new("<unk>", true)])
///     .show_progress(false)
///     .build()?;
/// tokenizer.train(trainer, ["low lower lowest", "new newer newest"].into_iter())?;
///
/// let ids = tokenizer.encode("lowest newer", false)?.ids().to_vec();
/// assert_eq!(tokenizer.decode(&ids, false)?, "lowest newer");
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct UnigramTrainer {
    /// Whether to report progress.
    pub(crate) show_progress: bool,
    /// Target vocabulary size, special tokens included.
    pub(crate) vocab_size: u32,
    /// EM iterations between pruning rounds.
    pub(crate) n_sub_iterations: u32,
    /// Fraction of pieces kept by each pruning round.
    pub(crate) shrinking_factor: f64,
    /// Special tokens, placed first in the vocabulary.
    pub(crate) special_tokens: Vec<AddedToken>,
    /// Chars always included in the vocabulary.
    pub(crate) initial_alphabet: HashSet<char>,
    /// Unknown token; added first unless it is already a special token.
    pub(crate) unk_token: Option<String>,
    /// Maximum length of a piece, in chars.
    pub(crate) max_piece_length: usize,
    /// Number of seed pieces (chars included).
    pub(crate) seed_size: usize,
    words: HashMap<String, u32>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self {
            show_progress: true,
            vocab_size: 8000,
            n_sub_iterations: 2,
            shrinking_factor: 0.75,
            special_tokens: vec![],
            initial_alphabet: HashSet::new(),
            unk_token: None,
            max_piece_length: 16,
            seed_size: 1_000_000,
            words: HashMap::new(),
        }
    }
}

/// Builder for [`UnigramTrainer`].
#[derive(Debug, Clone, Default)]
pub struct UnigramTrainerBuilder {
    trainer: UnigramTrainer,
    vocab_size: Option<usize>,
    n_sub_iterations: Option<usize>,
}

impl UnigramTrainerBuilder {
    /// A builder with HF defaults (vocab size 8000).
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether to report progress.
    #[must_use]
    pub fn show_progress(mut self, v: bool) -> Self {
        self.trainer.show_progress = v;
        self
    }
    /// Target vocabulary size, special tokens included (default 8000).
    #[must_use]
    pub fn vocab_size(mut self, v: usize) -> Self {
        self.vocab_size = Some(v);
        self
    }
    /// EM iterations per pruning round (default 2).
    #[must_use]
    pub fn n_sub_iterations(mut self, v: usize) -> Self {
        self.n_sub_iterations = Some(v);
        self
    }
    /// Fraction kept per pruning round, in `(0, 1)` (default 0.75).
    #[must_use]
    pub fn shrinking_factor(mut self, v: f64) -> Self {
        self.trainer.shrinking_factor = v;
        self
    }
    /// Special tokens, placed first in the vocabulary.
    #[must_use]
    pub fn special_tokens(mut self, v: Vec<AddedToken>) -> Self {
        self.trainer.special_tokens = v;
        self
    }
    /// Chars always included in the vocabulary.
    #[must_use]
    pub fn initial_alphabet(mut self, v: HashSet<char>) -> Self {
        self.trainer.initial_alphabet = v;
        self
    }
    /// Unknown token; added to the vocabulary first unless it is already
    /// one of the special tokens. None by default.
    #[must_use]
    pub fn unk_token(mut self, token: impl Into<String>) -> Self {
        self.trainer.unk_token = Some(token.into());
        self
    }
    /// Maximum piece length in chars (default 16).
    #[must_use]
    pub fn max_piece_length(mut self, v: usize) -> Self {
        self.trainer.max_piece_length = v;
        self
    }
    /// Number of seed pieces (default 1,000,000).
    #[must_use]
    pub fn seed_size(mut self, v: usize) -> Self {
        self.trainer.seed_size = v;
        self
    }
    /// Validate and build.
    ///
    /// # Errors
    /// Fails if `vocab_size`, `n_sub_iterations` or `max_piece_length` is
    /// zero (or too large), or `shrinking_factor` is not in `(0, 1)`.
    pub fn build(self) -> Result<UnigramTrainer> {
        let mut t = self.trainer;
        let to_u32 = |v: usize, name: &str| {
            u32::try_from(v)
                .map_err(|_| Error::Config(format!("UnigramTrainer: {name} is too large")))
        };
        if let Some(v) = self.vocab_size {
            t.vocab_size = to_u32(v, "vocab_size")?;
        }
        if let Some(v) = self.n_sub_iterations {
            t.n_sub_iterations = to_u32(v, "n_sub_iterations")?;
        }
        if t.vocab_size == 0 {
            return Err(Error::Config(
                "UnigramTrainer: vocab_size must be > 0".into(),
            ));
        }
        if !(t.shrinking_factor > 0.0 && t.shrinking_factor < 1.0) {
            return Err(Error::Config(
                "UnigramTrainer: shrinking_factor must be in (0, 1)".into(),
            ));
        }
        if t.n_sub_iterations == 0 {
            return Err(Error::Config(
                "UnigramTrainer: n_sub_iterations must be > 0".into(),
            ));
        }
        if t.max_piece_length == 0 {
            return Err(Error::Config(
                "UnigramTrainer: max_piece_length must be > 0".into(),
            ));
        }
        Ok(t)
    }
}

/// Digamma function (asymptotic expansion).
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    x -= 0.5;
    let xx = 1.0 / x;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;
    result + x.ln() + xx2 / 24.0 - 7.0 / 960.0 * xx4 + 31.0 / 8064.0 * xx4 * xx2
        - 127.0 / 30720.0 * xx4 * xx4
}

fn to_log_prob(pieces: &mut [SentencePiece]) {
    let logsum = pieces.iter().map(|(_, s)| s).sum::<f64>().ln();
    for (_, s) in pieces.iter_mut() {
        *s = s.ln() - logsum;
    }
}

/// Descending order on scores that never panics (NaN sorts last).
fn desc(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

/// Suffix array of `s` (prefix doubling, O(n log² n)).
fn suffix_array(s: &[u32]) -> Vec<usize> {
    let n = s.len();
    let mut sa: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u64> = s.iter().map(|&c| u64::from(c)).collect();
    let mut tmp = vec![0u64; n];
    let mut k = 1;
    if n <= 1 {
        return sa;
    }
    loop {
        let key = |i: usize, rank: &[u64]| -> (u64, u64) {
            let second = if i + k < n { rank[i + k] + 1 } else { 0 };
            (rank[i], second)
        };
        sa.sort_unstable_by_key(|&i| key(i, &rank));
        tmp[sa[0]] = 0;
        for w in 1..n {
            let bump = u64::from(key(sa[w - 1], &rank) != key(sa[w], &rank));
            tmp[sa[w]] = tmp[sa[w - 1]] + bump;
        }
        std::mem::swap(&mut rank, &mut tmp);
        if rank[sa[n - 1]] as usize == n - 1 {
            break;
        }
        k *= 2;
    }
    sa
}

/// Kasai's LCP: `lcp[i]` = common prefix of suffixes `sa[i-1]` and
/// `sa[i]` (`lcp[0] = 0`).
fn lcp_array(s: &[u32], sa: &[usize]) -> Vec<usize> {
    let n = s.len();
    let mut rank = vec![0usize; n];
    for (i, &p) in sa.iter().enumerate() {
        rank[p] = i;
    }
    let mut lcp = vec![0usize; n];
    let mut h = 0usize;
    for i in 0..n {
        if rank[i] > 0 {
            let j = sa[rank[i] - 1];
            while i + h < n && j + h < n && s[i + h] == s[j + h] {
                h += 1;
            }
            lcp[rank[i]] = h;
            h = h.saturating_sub(1);
        } else {
            h = 0;
        }
    }
    lcp
}

/// Internal nodes of the suffix tree of `s`: every substring occurring at
/// least twice that is followed by two different chars (or ends a
/// suffix). Returns `(start position, length, frequency)`.
fn suffix_tree_internal_nodes(s: &[u32]) -> Vec<(usize, usize, u32)> {
    let n = s.len();
    if n == 0 {
        return vec![];
    }
    let sa = suffix_array(s);
    let lcp = lcp_array(s, &sa);
    let mut out = Vec::new();
    // Stack of (left boundary, lcp value) for open lcp-intervals.
    let mut stack: Vec<(usize, usize)> = Vec::new();
    // A sentinel lcp of 0 after the last suffix closes every interval.
    for (i, &cur_lcp) in lcp.iter().chain(std::iter::once(&0)).enumerate().skip(1) {
        let mut left = i - 1;
        while let Some(&(lb, depth)) = stack.last() {
            if depth <= cur_lcp {
                break;
            }
            stack.pop();
            // Interval [lb, i) has lcp `depth` and at least 2 suffixes.
            out.push((sa[lb], depth, (i - lb) as u32));
            left = lb;
        }
        if cur_lcp > 0 && stack.last().is_none_or(|&(_, d)| d < cur_lcp) {
            stack.push((left, cur_lcp));
        }
    }
    out
}

impl UnigramTrainer {
    /// Start building a trainer.
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
    }

    fn is_valid_sentencepiece(&self, piece: &[u32]) -> bool {
        !piece.is_empty() && piece.len() <= self.max_piece_length
    }

    fn required_chars(&self, sentences: &[Sentence]) -> HashSet<String> {
        sentences
            .iter()
            .flat_map(|(s, _)| s.chars())
            .chain(self.initial_alphabet.iter().copied())
            .map(String::from)
            .collect()
    }

    fn make_seed_sentence_pieces(&self, sentences: &[Sentence]) -> Vec<SentencePiece> {
        const BOUNDARY: u32 = 0;
        let mut flat: Vec<u32> = Vec::new();
        let mut char_counts: HashMap<char, u64> = HashMap::new();
        for (s, n) in sentences {
            if s.is_empty() {
                continue;
            }
            for c in s.chars() {
                flat.push(c as u32);
                if c != '\0' {
                    *char_counts.entry(c).or_insert(0) += u64::from(*n);
                }
            }
            flat.push(BOUNDARY);
        }

        let mut chars: Vec<(u64, char)> = char_counts.into_iter().map(|(c, n)| (n, c)).collect();
        chars.sort_unstable_by_key(|&x| Reverse(x));

        let mut substrings: Vec<(u64, &[u32])> = suffix_tree_internal_nodes(&flat)
            .into_iter()
            .filter_map(|(start, len, freq)| {
                let piece = &flat[start..start + len];
                if len <= 1 || piece.contains(&BOUNDARY) || !self.is_valid_sentencepiece(piece) {
                    return None;
                }
                Some((u64::from(freq) * len as u64, piece))
            })
            .collect();
        substrings.sort_unstable_by(|a, b| b.cmp(a));

        let mut seeds: Vec<SentencePiece> = chars
            .into_iter()
            .map(|(n, c)| (c.to_string(), n as f64))
            .collect();
        for (score, piece) in substrings {
            if seeds.len() >= self.seed_size {
                break;
            }
            let s: String = piece.iter().filter_map(|&c| char::from_u32(c)).collect();
            seeds.push((s, score as f64));
        }
        to_log_prob(&mut seeds);
        seeds
    }

    fn run_e_step(&self, model: &Unigram, sentences: &[Sentence]) -> Vec<f64> {
        let size = model.len();
        let all_freq: f64 = sentences.iter().map(|(_, n)| f64::from(*n)).sum();
        #[cfg(feature = "parallel")]
        let chunks = sentences.par_chunks(CHUNK_SIZE);
        #[cfg(not(feature = "parallel"))]
        let chunks = sentences.chunks(CHUNK_SIZE);
        let partials: Vec<(f64, Vec<f64>)> = chunks
            .map(|chunk| {
                let mut expected = vec![0.0; size];
                let mut objective = 0.0;
                for (s, n) in chunk {
                    let mut lattice = Lattice::new(s, model.bos_id, model.eos_id);
                    model.populate_nodes(&mut lattice);
                    let z = lattice.populate_marginal(f64::from(*n), &mut expected);
                    objective -= z / all_freq;
                }
                (objective, expected)
            })
            .collect();
        let mut expected = vec![0.0; size];
        for (_, e) in partials {
            for (a, b) in expected.iter_mut().zip(e) {
                *a += b;
            }
        }
        expected
    }

    fn run_m_step(&self, pieces: &[SentencePiece], expected: &[f64]) -> Vec<SentencePiece> {
        const MIN_EXPECTED: f64 = 0.5;
        let mut kept: Vec<SentencePiece> = Vec::with_capacity(pieces.len());
        let mut sum = 0.0;
        for (i, ((piece, _), &freq)) in pieces.iter().zip(expected).enumerate() {
            if i == 0 {
                // Always keep the unknown piece.
                kept.push((piece.clone(), f64::NAN));
                continue;
            }
            if freq < MIN_EXPECTED {
                continue;
            }
            kept.push((piece.clone(), freq));
            sum += freq;
        }
        // Bayesian (DP) EM: digamma acts as a sparse prior.
        let logsum = digamma(sum);
        kept.into_iter()
            .map(|(s, c)| (s, digamma(c) - logsum))
            .collect()
    }

    fn prune_sentence_pieces(
        &self,
        model: &Unigram,
        pieces: &[SentencePiece],
        sentences: &[Sentence],
    ) -> Vec<SentencePiece> {
        let n_pieces = pieces.len();
        let bos_id = n_pieces + 1;
        let eos_id = n_pieces + 2;

        // How each piece would be re-segmented if it were removed: its
        // second-best segmentation.
        #[cfg(feature = "parallel")]
        let piece_iter = pieces.par_iter();
        #[cfg(not(feature = "parallel"))]
        let piece_iter = pieces.iter();
        let per_piece: Vec<(bool, Vec<usize>)> = piece_iter
            .enumerate()
            .map(|(id, (token, _))| {
                if id == 0 {
                    return (false, vec![]);
                }
                let mut lattice = Lattice::new(token, bos_id, eos_id);
                model.populate_nodes(&mut lattice);
                let nbests = lattice.nbest(2);
                if nbests.len() == 1 {
                    (true, vec![])
                } else if nbests[0].len() >= 2 {
                    (false, vec![])
                } else if nbests[0].len() == 1 {
                    let alts = nbests[1].iter().map(|&n| lattice.node(n).id).collect();
                    (true, alts)
                } else {
                    (true, vec![])
                }
            })
            .collect();
        let (always_keep, alternatives): (Vec<bool>, Vec<Vec<usize>>) =
            per_piece.into_iter().unzip();

        // Viterbi-segment the corpus: piece frequencies and, for each
        // piece, the sentences using it.
        let indexed: Vec<(usize, &Sentence)> = sentences.iter().enumerate().collect();
        #[cfg(feature = "parallel")]
        let chunks = indexed.par_chunks(CHUNK_SIZE);
        #[cfg(not(feature = "parallel"))]
        let chunks = indexed.chunks(CHUNK_SIZE);
        let partials: Vec<(f64, Vec<f64>, Vec<Vec<usize>>)> = chunks
            .map(|chunk| {
                let mut vsum = 0.0;
                let mut freq = vec![0.0; n_pieces];
                let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); n_pieces];
                for &(i, (s, count)) in chunk {
                    let mut lattice = Lattice::new(s, bos_id, eos_id);
                    model.populate_nodes(&mut lattice);
                    vsum += f64::from(*count);
                    for node in lattice.viterbi() {
                        let id = lattice.node(node).id;
                        freq[id] += f64::from(*count);
                        inverted[id].push(i);
                    }
                }
                (vsum, freq, inverted)
            })
            .collect();
        let mut vsum = 0.0;
        let mut freq = vec![0.0; n_pieces];
        let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); n_pieces];
        for (v, f, inv) in partials {
            vsum += v;
            for (a, b) in freq.iter_mut().zip(f) {
                *a += b;
            }
            for (a, b) in inverted.iter_mut().zip(inv) {
                a.extend(b);
            }
        }

        let sum: f64 = freq.iter().sum();
        let logsum = sum.ln();
        let mut candidates: Vec<(usize, f64)> = Vec::new();
        let mut new_pieces: Vec<SentencePiece> = vec![pieces[0].clone()];

        // Approximate the likelihood loss of removing each piece by
        // assuming its occurrences are replaced by its alternatives.
        for (id, (token, score)) in pieces.iter().enumerate().skip(1) {
            if freq[id] == 0.0 && !always_keep[id] {
                continue;
            }
            if alternatives[id].is_empty() {
                new_pieces.push((token.clone(), *score));
                continue;
            }
            let mut f: f64 = inverted[id]
                .iter()
                .map(|&n| f64::from(sentences[n].1))
                .sum();
            if f == 0.0 || f.is_nan() {
                continue;
            }
            f /= vsum;
            let logprob_sp = freq[id].ln() - logsum;
            // Mirrors HF tokenizers, which scales by the number of pieces
            // here (SentencePiece uses the number of alternatives).
            let logsum_alt = (sum + freq[id] * (alternatives.len() - 1) as f64).ln();
            let logprob_alt: f64 = alternatives[id]
                .iter()
                .map(|&n| (freq[n] + freq[id]).ln() - logsum_alt)
                .sum();
            let loss = f * (logprob_sp - logprob_alt);
            if !loss.is_nan() {
                candidates.push((id, loss));
            }
        }

        let desired = (self.vocab_size as usize * 11) / 10;
        let pruned_size = desired.max((n_pieces as f64 * self.shrinking_factor) as usize);
        candidates.sort_by(|a, b| desc(a.1, b.1));
        for (id, _) in candidates {
            if new_pieces.len() == pruned_size {
                break;
            }
            new_pieces.push(pieces[id].clone());
        }
        new_pieces
    }

    /// The tokens that occupy the first ids regardless of the corpus, in
    /// id order and without duplicates: the unknown token first unless it
    /// is one of the special tokens, then the special tokens.
    fn reserved_tokens(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(self.special_tokens.len() + 1);
        if let Some(unk) = &self.unk_token {
            if !self.special_tokens.iter().any(|t| &t.content == unk) {
                out.push(unk.clone());
            }
        }
        for t in &self.special_tokens {
            if !out.contains(&t.content) {
                out.push(t.content.clone());
            }
        }
        out
    }

    fn finalize(&self, model: &Unigram, required_chars: HashSet<String>) -> Result<Unigram> {
        const PENALTY_DELTA: f64 = 0.0001;
        let mut penalty = 0.0;
        let mut pieces: Vec<SentencePiece> = Vec::new();
        // Everything already in the vocabulary (or excluded from it), so
        // no token is emitted twice: the special tokens and the unknown
        // token take the first ids (even when they also occur as a
        // required char or a learned piece), and the training-time
        // unknown piece is not part of the result.
        let reserved = self.reserved_tokens();
        let mut inserted: HashSet<String> = reserved.iter().cloned().collect();
        inserted.insert("<UNK>".into());

        let existing: HashMap<&str, f64> = model
            .pieces()
            .iter()
            .map(|(s, f)| (s.as_str(), *f))
            .collect();
        let mut required: Vec<String> = required_chars.into_iter().collect();
        required.sort();
        let learned = |c: &str| existing.get(c).copied().filter(|s| s.is_finite());
        // A required char the model did not learn gets the lowest learned
        // score (plus a growing penalty, as HF tokenizers does). With no
        // learned score at all (an empty corpus), HF uses the model's
        // `min_score` of `+inf` and saves `null`, which cannot be loaded
        // back; give those chars a uniform log-probability instead.
        let base = if model.min_score.is_finite() {
            model.min_score
        } else {
            let unscored = required
                .iter()
                .filter(|c| !inserted.contains(*c) && learned(c).is_none())
                .count()
                .max(1);
            0.0 - (unscored as f64).ln()
        };
        for c in required {
            if inserted.contains(&c) {
                continue;
            }
            let score = match learned(&c) {
                Some(s) => s,
                None => {
                    let s = base + penalty;
                    penalty += PENALTY_DELTA;
                    s
                }
            };
            inserted.insert(c.clone());
            pieces.push((c, score));
        }

        // `vocab_size` is a hard cap that includes the reserved tokens;
        // `do_train` has already checked that the required chars fit.
        let budget = (self.vocab_size as usize).saturating_sub(reserved.len());
        for (token, score) in model.pieces() {
            if pieces.len() >= budget {
                break;
            }
            if inserted.contains(token) {
                continue;
            }
            inserted.insert(token.clone());
            pieces.push((token.clone(), if score.is_nan() { 0.0 } else { *score }));
        }
        pieces.sort_by(|a, b| desc(a.1, b.1));

        let unk_id = self
            .unk_token
            .as_ref()
            .and_then(|unk| reserved.iter().position(|t| t == unk));
        let mut vocab: Vec<SentencePiece> = reserved.into_iter().map(|t| (t, 0.0)).collect();
        vocab.extend(pieces);
        // Every score above is finite; `Unigram::new` checks it anyway so
        // a trained model can always be saved and loaded back.
        Unigram::new(vocab, unk_id, model.byte_fallback())
            .map_err(|e| Error::Training(format!("invalid trained model: {e}")))
    }

    /// Train on `(word, count)` pairs.
    fn do_train(
        &self,
        mut sentences: Vec<Sentence>,
        model: &mut Unigram,
    ) -> Result<Vec<AddedToken>> {
        // Hash-map order is random; sort so training is deterministic.
        sentences.sort_unstable();

        // `vocab_size` is a hard cap: the unknown token, the special
        // tokens and every required char must all fit in it.
        let required = self.required_chars(&sentences);
        let reserved = self.reserved_tokens();
        let required_extra = required.iter().filter(|c| !reserved.contains(c)).count();
        if reserved.len() + required_extra > self.vocab_size as usize {
            return Err(Error::Training(format!(
                "the vocabulary size ({}) is smaller than the number of required chars ({}) \
                 plus special tokens (including the unknown token: {})",
                self.vocab_size,
                required_extra,
                reserved.len()
            )));
        }

        // A training-time unknown piece always sits at id 0.
        let mut pieces: Vec<SentencePiece> = vec![("<UNK>".into(), f64::NAN)];
        let progress = Progress::new(self.show_progress, "Suffix array seeds", None);
        pieces.extend(self.make_seed_sentence_pieces(&sentences));
        progress.set_message(format!("Suffix array seeds: {}", pieces.len() - 1));
        progress.finish();

        let desired = (self.vocab_size as usize * 11) / 10;
        let mut current = Unigram::with_any_scores(pieces.clone(), Some(0), false)?;
        let progress = Progress::new(self.show_progress, "EM training", None);
        loop {
            for _ in 0..self.n_sub_iterations {
                let expected = self.run_e_step(&current, &sentences);
                pieces = self.run_m_step(&pieces, &expected);
                current = Unigram::with_any_scores(pieces.clone(), Some(0), false)?;
                progress.inc(1);
            }
            if pieces.len() <= desired {
                break;
            }
            progress.set_message(format!("EM training: {} → {desired} pieces", pieces.len()));
            let pruned = self.prune_sentence_pieces(&current, &pieces, &sentences);
            if pruned.len() >= pieces.len() {
                // No progress possible; avoid looping forever.
                break;
            }
            pieces = pruned;
            current = Unigram::with_any_scores(pieces.clone(), Some(0), false)?;
        }

        progress.finish();

        *model = self.finalize(&current, required)?;
        Ok(self.special_tokens.clone())
    }
}

impl Trainer for UnigramTrainer {
    type Model = Unigram;

    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn train(&self, model: &mut Unigram) -> Result<Vec<AddedToken>> {
        let sentences: Vec<Sentence> = self.words.iter().map(|(s, n)| (s.clone(), *n)).collect();
        self.do_train(sentences, model)
    }

    /// Count the words of the corpus. Replaces anything fed before.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let progress = Progress::new(self.show_progress, "Pre-processing sequences", None);
        let count = |seq: S| -> Result<HashMap<String, u32>> {
            progress.inc(1);
            let mut map = HashMap::new();
            for w in process(seq.as_ref())? {
                *map.entry(w).or_insert(0u32) += 1;
            }
            Ok(map)
        };
        let merge = |mut acc: HashMap<String, u32>,
                     m: HashMap<String, u32>|
         -> Result<HashMap<String, u32>> {
            for (k, v) in m {
                *acc.entry(k).or_insert(0) += v;
            }
            Ok(acc)
        };
        #[cfg(feature = "parallel")]
        let words = iterator
            .par_bridge()
            .map(count)
            .try_reduce(HashMap::new, merge)?;
        #[cfg(not(feature = "parallel"))]
        let words = iterator
            .map(count)
            .try_fold(HashMap::new(), |acc, m| merge(acc, m?))?;
        progress.finish();
        self.words = words;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Model;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 0.01, "{a} != {b}");
    }

    fn trainer() -> UnigramTrainerBuilder {
        UnigramTrainer::builder().show_progress(false)
    }

    #[test]
    fn suffix_tree_nodes_match_brute_force() {
        let text: Vec<u32> = "abracadabra\0banana\0abab\0"
            .chars()
            .map(|c| c as u32)
            .collect();
        let mut fast: Vec<(Vec<u32>, u32)> = suffix_tree_internal_nodes(&text)
            .into_iter()
            .map(|(s, l, f)| (text[s..s + l].to_vec(), f))
            .collect();
        fast.sort();
        // Brute force: substrings occurring ≥ 2 times whose occurrences
        // are followed by at least two distinct continuations (or end).
        let n = text.len();
        let mut occ: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
        for i in 0..n {
            for j in i + 1..=n {
                occ.entry(text[i..j].to_vec()).or_default().push(j);
            }
        }
        let mut slow: Vec<(Vec<u32>, u32)> = occ
            .into_iter()
            .filter(|(_, ends)| ends.len() >= 2)
            .filter(|(_, ends)| {
                let nexts: HashSet<Option<u32>> =
                    ends.iter().map(|&e| text.get(e).copied()).collect();
                nexts.len() >= 2 || nexts.contains(&None)
            })
            .map(|(k, ends)| (k, ends.len() as u32))
            .collect();
        slow.sort();
        assert_eq!(fast, slow);
    }

    #[test]
    fn seeds_match_hf() {
        let t = trainer().build().unwrap();
        let sentences = vec![
            ("This is a".to_string(), 1),
            ("こんにちは友達".to_string(), 1),
        ];
        assert_eq!(t.required_chars(&sentences).len(), 13);
        let table = t.make_seed_sentence_pieces(&sentences);
        let strings: Vec<&str> = table.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(
            strings,
            [
                "s", "i", " ", "達", "友", "ん", "は", "に", "ち", "こ", "h", "a", "T", "is ", "s "
            ]
        );
        let expected = [
            -2.5649493574615367,
            -2.5649493574615367,
            -2.5649493574615367,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -3.258096538021482,
            -1.4663370687934272,
            -1.8718021769015916,
        ];
        for ((_, s), e) in table.iter().zip(expected) {
            approx(*s, e);
        }
    }

    #[test]
    fn initial_alphabet_is_required() {
        let t = trainer()
            .initial_alphabet("abcdef".chars().collect())
            .build()
            .unwrap();
        let req = t.required_chars(&[("こんにちは友達".to_string(), 1)]);
        let expected: HashSet<String> = "こんにちは友達abcdef".chars().map(String::from).collect();
        assert_eq!(req, expected);
    }

    #[test]
    fn unk_token_placement() {
        let words = || vec![("The".to_string(), 12), ("are".to_string(), 11)];
        let first3 =
            |m: &Unigram| -> Vec<(String, f64)> { m.pieces().iter().take(3).cloned().collect() };

        let t = trainer()
            .special_tokens(vec![
                AddedToken::new("[SEP]", true),
                AddedToken::new("[CLS]", true),
            ])
            .unk_token("[UNK]")
            .build()
            .unwrap();
        let mut m = Unigram::default();
        t.do_train(words(), &mut m).unwrap();
        assert_eq!(
            first3(&m),
            vec![
                ("[UNK]".into(), 0.0),
                ("[SEP]".into(), 0.0),
                ("[CLS]".into(), 0.0)
            ]
        );
        assert_eq!(m.unk_id(), Some(0));

        let t = trainer()
            .special_tokens(vec![
                AddedToken::new("[SEP]", true),
                AddedToken::new("[CLS]", true),
                AddedToken::new("[UNK]", true),
            ])
            .unk_token("[UNK]")
            .build()
            .unwrap();
        let mut m = Unigram::default();
        t.do_train(words(), &mut m).unwrap();
        assert_eq!(
            first3(&m),
            vec![
                ("[SEP]".into(), 0.0),
                ("[CLS]".into(), 0.0),
                ("[UNK]".into(), 0.0)
            ]
        );
        assert_eq!(m.unk_id(), Some(2));

        let t = trainer().build().unwrap();
        let mut m = Unigram::default();
        t.do_train(words(), &mut m).unwrap();
        assert_eq!(m.pieces()[0].0, "e");
        assert_eq!(m.unk_id(), None);
    }

    #[test]
    fn special_tokens_first() {
        let t = trainer()
            .special_tokens(vec![
                AddedToken::new("[SEP]", true),
                AddedToken::new("[CLS]", true),
            ])
            .build()
            .unwrap();
        let mut m = Unigram::default();
        t.do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut m)
            .unwrap();
        let first: Vec<_> = m.pieces().iter().take(2).cloned().collect();
        assert_eq!(first, vec![("[SEP]".into(), 0.0), ("[CLS]".into(), 0.0)]);
    }

    #[test]
    fn to_log_prob_normalizes() {
        let mut a = vec![(String::new(), 1.0), (String::new(), 2.0)];
        to_log_prob(&mut a);
        approx(a[0].1, -1.098);
        approx(a[1].1, -0.405);
    }

    #[test]
    fn vocab_too_small_errors() {
        let t = trainer().vocab_size(2).build().unwrap();
        let mut m = Unigram::default();
        assert!(t.do_train(vec![("abc".into(), 1)], &mut m).is_err());
    }

    #[test]
    fn vocab_size_is_a_hard_cap_including_special_tokens() {
        // Issue #32: 2 specials + 3 required chars do not fit in 4.
        let specials = || vec![AddedToken::new("<s>", true), AddedToken::new("</s>", true)];
        let corpus = || {
            vec![
                ("ab".to_string(), 3),
                ("abc".to_string(), 2),
                ("bc".to_string(), 1),
            ]
        };
        let t = trainer()
            .vocab_size(4)
            .special_tokens(specials())
            .build()
            .unwrap();
        let err = t.do_train(corpus(), &mut Unigram::default()).unwrap_err();
        assert!(matches!(err, Error::Training(_)), "{err:?}");
        assert!(err.to_string().contains("special tokens"), "{err}");

        // The unknown token counts too: 1 unk + 2 specials + 3 chars > 5.
        let t = trainer()
            .vocab_size(5)
            .special_tokens(specials())
            .unk_token("<unk>")
            .build()
            .unwrap();
        assert!(t.do_train(corpus(), &mut Unigram::default()).is_err());

        // Exactly enough room for the reserved tokens and the chars.
        let t = trainer()
            .vocab_size(5)
            .special_tokens(specials())
            .build()
            .unwrap();
        let mut m = Unigram::default();
        t.do_train(corpus(), &mut m).unwrap();
        assert_eq!(m.vocab_size(), 5);
        let got: Vec<&str> = m.pieces().iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(&got[..2], ["<s>", "</s>"]);
        let mut chars = got[2..].to_vec();
        chars.sort_unstable();
        assert_eq!(chars, ["a", "b", "c"]);

        // Any larger budget is still never exceeded.
        for size in 6..12 {
            let t = trainer()
                .vocab_size(size)
                .special_tokens(specials())
                .unk_token("<unk>")
                .build()
                .unwrap();
            let mut m = Unigram::default();
            t.do_train(corpus(), &mut m).unwrap();
            assert!(m.vocab_size() <= size, "{size}: {}", m.vocab_size());
        }
    }

    #[test]
    fn special_and_unk_tokens_are_not_duplicated_by_corpus_pieces() {
        // Issue #34: "a" is a special token and also a corpus piece.
        let corpus = || {
            vec![
                ("a".to_string(), 1),
                ("b".to_string(), 1),
                ("ab".to_string(), 1),
            ]
        };
        let t = trainer()
            .special_tokens(vec![AddedToken::new("a", true)])
            .build()
            .unwrap();
        let mut m = Unigram::default();
        t.do_train(corpus(), &mut m).unwrap();
        let tokens: Vec<&str> = m.pieces().iter().map(|(s, _)| s.as_str()).collect();
        let unique: HashSet<&str> = tokens.iter().copied().collect();
        assert_eq!(unique.len(), tokens.len(), "duplicates in {tokens:?}");
        assert_eq!(m.token_to_id("a"), Some(0));
        assert_eq!(m.pieces()[0], ("a".to_string(), 0.0));

        // Same with an unknown token that is not a special token.
        let t = trainer().unk_token("b").build().unwrap();
        let mut m = Unigram::default();
        t.do_train(corpus(), &mut m).unwrap();
        let tokens: Vec<&str> = m.pieces().iter().map(|(s, _)| s.as_str()).collect();
        let unique: HashSet<&str> = tokens.iter().copied().collect();
        assert_eq!(unique.len(), tokens.len(), "duplicates in {tokens:?}");
        assert_eq!(m.token_to_id("b"), Some(0));
        assert_eq!(m.unk_id(), Some(0));

        // A special token listed twice gets one id, and the unk id still
        // points at the right piece.
        let t = trainer()
            .special_tokens(vec![
                AddedToken::new("<s>", true),
                AddedToken::new("<s>", true),
                AddedToken::new("<unk>", true),
            ])
            .unk_token("<unk>")
            .build()
            .unwrap();
        let mut m = Unigram::default();
        t.do_train(corpus(), &mut m).unwrap();
        assert_eq!(m.token_to_id("<s>"), Some(0));
        assert_eq!(m.token_to_id("<unk>"), Some(1));
        assert_eq!(m.unk_id(), Some(1));
    }

    #[test]
    fn builder_validates() {
        assert!(trainer().vocab_size(0).build().is_err());
        assert!(trainer().shrinking_factor(1.5).build().is_err());
        assert!(trainer().n_sub_iterations(0).build().is_err());
    }

    #[test]
    fn empty_corpus_does_not_panic() {
        let t = trainer().unk_token("<unk>").build().unwrap();
        let mut m = Unigram::default();
        t.do_train(vec![], &mut m).unwrap();
        assert_eq!(m.vocab_size(), 1);
    }

    fn corpus_words() -> Vec<String> {
        // Fixed corpus (the original examples/corpus.txt) so the expected
        // numbers below stay valid when the example file changes.
        let text = "the quick brown fox jumps over the lazy dog\n\
                    the quick brown fox jumps over the lazy dog\n\
                    hello world hello world hello world\n\
                    pack my box with five dozen liquor jugs\n\
                    pack my box with five dozen liquor jugs\n\
                    how vexingly quick daft zebras jump\n\
                    how vexingly quick daft zebras jump\n\
                    sphinx of black quartz judge my vow\n\
                    sphinx of black quartz judge my vow\n";
        let mut words = Vec::new();
        for i in 0..20 {
            for line in text.lines() {
                for w in line.split_whitespace() {
                    words.push(format!("▁{w}"));
                    if i % 3 == 0 {
                        words.push(format!("▁{}", w.to_uppercase()));
                    }
                }
            }
        }
        words
    }

    fn train_corpus(vocab_size: usize) -> Unigram {
        let mut t = trainer()
            .vocab_size(vocab_size)
            .unk_token("<unk>")
            .special_tokens(vec![AddedToken::new("<unk>", true)])
            .build()
            .unwrap();
        let words = corpus_words();
        t.feed(std::iter::once(words.join(" ")), |s| {
            Ok(s.split(' ').map(String::from).collect())
        })
        .unwrap();
        let mut m = Unigram::default();
        t.train(&mut m).unwrap();
        m
    }

    #[test]
    fn trained_model_is_deterministic_sized_and_covers_corpus() {
        let a = train_corpus(80);
        let b = train_corpus(80);
        assert_eq!(a, b, "training must be deterministic");
        assert_eq!(a.vocab_size(), 80);
        // This corpus only supports 102 pieces: HF tokenizers 0.23.2
        // stops at exactly the same 102-piece vocabulary.
        assert_eq!(train_corpus(120).vocab_size(), 102);
        let unk = a.unk_id().unwrap() as u32;
        for w in corpus_words() {
            for t in a.tokenize(&w).unwrap() {
                assert_ne!(t.id, unk, "unk while encoding {w:?}");
            }
        }
        // Every char of the corpus is a piece.
        for w in corpus_words() {
            for c in w.chars() {
                assert!(
                    a.token_to_id(&c.to_string()).is_some(),
                    "missing char {c:?}"
                );
            }
        }
    }
}
