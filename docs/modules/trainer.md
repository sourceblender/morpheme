# `trainer`

> Phase 2 work. Filled out as the BPE trainer landed. WordPiece and
> Unigram trainers land in Phase 2.1.

## Purpose

Train a `Model` from raw text corpora.

## Public API

```rust
pub trait Trainer {
    fn train<C: Corpus>(&self, corpus: C) -> Result<TrainedModel>;
}

pub struct TrainedModel {
    pub vocab: Vocab,
    pub merges: Vec<(String, String)>,
    pub end_of_word_suffix: String,
}

pub trait Corpus {
    fn for_each_line<F: FnMut(&str)>(&self, f: F);
}

impl Corpus for &str { ... }
impl Corpus for String { ... }
impl Corpus for PathBuf { ... }
impl Corpus for &[&str] { ... }

pub struct BpeTrainer { ... }
pub struct BpeTrainerBuilder { ... }
```

`BpeTrainerBuilder` exposes:

- `BpeTrainerBuilder::new(vocab_size)`
- `.end_of_word_suffix(s)`
- `.alphabet(symbols)`
- `.min_pair_frequency(n)`
- `.pre_tokenize(bool)`
- `.build() -> BpeTrainer`

## BPE algorithm (v0.1)

1. Tokenize the corpus into words (whitespace split by default;
   `pre_tokenize(false)` treats each line as a single word).
2. Compute word frequencies (`FxHashMap<String, u64>`).
3. Build initial symbols for every unique word: split into characters,
   append `end_of_word_suffix` to the last character.
4. Initial vocab = `alphabet` symbols + every unique symbol seen in
   the corpus (sorted lexicographically for determinism).
5. Pair counts: weighted by word frequency. Ties broken by the
   lexicographically larger merged token (HF / GPT-2 convention).
6. Iteratively merge the best pair:
   - Append `(a, b)` to `merges`.
   - Append `ab` to `vocab`.
   - Update every word occurrence that contains the pair:
     - Subtract the word's frequency from the pairs being destroyed.
     - Add the word's frequency to the new pairs created.
   - Stop when `vocab.len() >= vocab_size` or no pairs remain.

## Incremental updates

For each touched word, the pair-count delta is computed against the
old word_pairs list and the new symbolization — the global pair-count
map and `pair_occurrences` index are updated in O(occurrences)
rather than O(corpus). This is what makes the trainer fast on large
corpora.

## Correctness

Cross-validated against a slow naive reference trainer (rebuilds pair
counts from scratch every iteration). The two produce *identical*
merge tables across every test corpus, including:

- `aa bb cc cc aa bb`
- `hello world hello hello world`
- `the quick brown fox jumps over the lazy dog` (×2)
- `ab ba ab ba ab ba` (alphabetical-tiebreak stress)
- `a b c a b c a b c a b c`

## Test strategy

- Tiny golden tests (`train_small_corpus_produces_merges`,
  `train_then_tokenize_corpus_yields_expected_tokens`) — assert
  specific merges appear in the output.
- Round-trip — train → build Tokenizer → encode the training corpus
  → assert merged tokens appear.
- Cross-validation (`matches_naive_reference`) — assert the fast
  trainer and the naive trainer agree on `merges` and `vocab` for
  five corpora.
- Edge cases: empty corpus errors, alphabet pre-population respected,
  `min_pair_frequency` skips merges but keeps rare symbols, target
  vocab size respected.

## Known limitations

- Single-threaded. `rayon` parallelism lands in Phase 4.
- `O(n)` scan to find the best pair per iteration. A
  `BinaryHeap<(Reverse<u64>, pair)>` with lazy deletion would give
  amortized `O(log n)` per iteration; left for Phase 4 perf work.
- No special tokens / unk handling in v0.1. v0.2 will let the trainer
  pre-populate `<unk>`, `<pad>`, etc. via the `alphabet` field.
- Initial vocab is built from every symbol seen, even rare ones. The
  `min_pair_frequency` controls *merges* only, not initial vocab
  inclusion.

## Roadmap

Phase 2.1:
- `WordPieceTrainer` — frequency-weighted greedy vocab reduction.
- `UnigramTrainer` — EM over an initial vocabulary.
- Special-token support (auto-prepend `<unk>`, accept user-supplied
  `<pad>`, `<bos>`, `<eos>`).

Phase 4:
- `rayon`-parallel pair counting and merge step.
- Heap-based best-pair lookup.
- Streaming corpus input for files that don't fit in memory.

## Unigram trainer (Phase 2.2)

The `UnigramTrainer` produces a probability distribution over
subwords via Expectation-Maximization.

### Algorithm

1. **Seed vocab** — every unique character plus every substring
   (length ≤ `max_subword_len`) that appears ≥ `seed_min_frequency`
   times.
2. **Initialize** — `p ∝ freq` with a `smoothing` constant added
   to every subword (Laplace-style prior).
3. **EM loop**, `num_epochs` iterations:
   - **E-step**: forward + backward lattice passes compute the
     expected count of each subword across the corpus.
   - **M-step**: re-estimate probabilities from expected counts.
   - **Prune**: drop subwords whose expected count falls below
     `min_frequency` until `vocab_size` is reached. Pinned
     (alphabet) entries are exempt from pruning.
4. **Output** — vocab + per-subword log-probs + `min_score` floor.

### Caveats

- Forward/backward passes are O(|vocab| × n) per word, single-
  threaded. Phase 4 perf work can swap in an Aho-Corasick-style
  trie for subword matching.
- For a tiny corpus, the trainer can't generate enough seed
  candidates to reach the target vocab size — it produces as many
  as possible and stops.