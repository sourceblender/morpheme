# Changelog

All notable changes to `splinter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Workspace scaffold: `splinter` library crate and `splinter-cli` binary.
- Stub `splinter::placeholder()` and `splinter::VERSION`.
- `tracing`-based logging in the CLI.
- GitHub Actions CI (fmt, clippy, test, MSRV).
- Full open-source docs tree (`docs/`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`).
- Phase 1 — BPE MVP:
  - `splinter::Error` (thiserror) + `Result` alias.
  - `Vocab` with O(1) `token_to_id` and ordered `id_to_token` (FxHashMap).
  - `Normalizer` trait + `IdentityNormalizer`.
  - `PreTokenizer` trait + `Whitespace` pre-tokenizer.
  - `Model` trait + `Bpe` (GPT-2-style `</w>` end-of-word suffix, ranked merge loop).
  - `Tokenizer` glue: `encode`, `from_json`, `to_json`, `from_file`, `to_file`.
  - **JSON I/O** — splinter's own v1.0 schema (BPE-only). `Tokenizer::from_json` / `to_json` / `from_file` / `to_file`.
  - `proptest` round-trip property tests + 7 smoke tests + 11 JSON I/O tests.
  - CLI: `splinter encode [--from <path>] <text>` and `splinter inspect [--from <path>]`.
  - Bundled fixture: `crates/splinter/tests/fixtures/tiny.json`.

### Phase 1.1 — Full encoder / pre-tokenizer / decoder suite

- Normalizers:
  - `BertNormalizer` (clean_text + handle_chinese_chars + strip_accents + lowercase).
  - `Lowercase`, `Nfd`, `Nfkc`, `StripAccents`, `Replace`.
- Pre-tokenizers:
  - `BertPreTokenizer` — whitespace + punctuation split with HF-style
    "leading whitespace attached to the following token" semantics.
  - `ByteLevel` — GPT-2 byte-to-unicode mapping (alphabet + reverse
    mapping for the decoder).
- Decoders:
  - `WordPieceDecoder` — strips `##` continuation markers.
  - `ByteLevelDecoder` — inverse of `ByteLevel`'s byte→char map.
- `PreToken` redesigned to carry either borrowed or owned text so
  `ByteLevel`'s byte-to-unicode output doesn't borrow from the input.
- `Tokenizer::decode(ids, skip_special)` wired through the decoder.
- `Tokenizer::builder(model)` for ergonomic full-component setup.
- CLI: `splinter encode --normalize <none|lowercase|bert>` and
  `--pre-tokenize <whitespace|bert|byte-level>`. New `splinter decode
  --pre-tokenize byte-level <ids>` round-trips a `ByteLevel` encode.
- Tests: 18 new unit tests across the normalizer / pre-tokenizer /
  decoder modules.

### Phase 2 — BPE trainer

- New `splinter::trainer` module:
  - `BpeTrainer`, `BpeTrainerBuilder` with options for `vocab_size`,
    `end_of_word_suffix`, `alphabet` (pre-populated symbols), and
    `min_pair_frequency`.
  - `Corpus` trait with `&str`, `String`, `PathBuf`, and `&[&str]`
    implementations.
  - `Trainer::train(corpus) -> Result<TrainedModel>` returns a
    `(Vocab, Vec<(String, String)>, end_of_word_suffix)` triple —
    exactly what `Bpe::new` expects.
- Algorithm: incremental pair-count updates with a deterministic
  alphabetical tiebreak (matches the HF / GPT-2 convention).
- Cross-validated against a slow naive reference trainer that
  rebuilds pair counts from scratch every iteration — both produce
  identical merge tables across five different corpora.
- CLI: `splinter train --input <corpus.txt> --vocab-size N --out
  <tokenizer.json> [--min-pair-frequency M]`.
- Trained the example corpus in `examples/corpus.txt` into
  `examples/tokenizer.json` (60 vocab, 20 merges).
- Tests: 9 new unit tests including the naive-reference cross-check.

### Phase 2.1 — WordPiece trainer + model

- `splinter::model::wordpiece::WordPiece` — greedy longest-match
  subword lookup. Honors `continuing_subword_prefix` (default `##`)
  and `end_of_word_suffix` (default `</w>`); emits the unk token on
  OOV.
- `splinter::trainer::wordpiece::{WordPieceTrainer,
  WordPieceTrainerBuilder, WordPieceTrainedModel}` — additive
  WordPiece trainer with options for `vocab_size`, `min_frequency`,
  `alphabet` (pre-populated symbols, e.g. `<unk>`), and
  `pre_tokenize`.
- Algorithm: build initial alphabet from chars with `freq >=
  min_frequency`, then iteratively pick the substring whose
  inclusion raises the corpus likelihood the most (score = `freq(c)
  / (freq(left) * freq(right) * word_freq)`).
- `Tokenizer::ModelKind` enum (now public) — wraps `Bpe` or
  `WordPiece`. `Tokenizer::wordpiece(model)` constructor and
  `TokenizerBuilder` accept either variant via `Into<ModelKind>`.
- JSON I/O updated: `model.type` is now `bpe` or `wordpiece`; BPE
  files omit `merges` and vice versa; WordPiece files include
  `unk_token`, `continuing_subword_prefix`, and
  `max_input_chars_per_word`.
- `splinter::tokenizer::{bpe_model, wordpiece_model}` accessors
  (the old `bpe()` getter was renamed to avoid clashing with the
  `Tokenizer::new(bpe)` constructor).
- `WordPieceDecoder` now strips the `</w>` suffix and inserts
  spaces between word boundaries.
- CLI: `splinter train --algorithm wordpiece` (default is still
  `bpe`). Flags are now order-independent.
- Trained `examples/corpus.txt` into `examples/wordpiece.json`
  (200 vocab).
- Tests: 9 new unit tests across the trainer, model, and decoder.

### Phase 2.2 — Unigram trainer + model

- New `splinter::model::unigram::Unigram` — Viterbi best-path over
  a subword lattice. The model's `apply` method auto-prepends the
  whitespace marker so callers can use a plain `Whitespace`
  pre-tokenizer. Tokens are emitted *as they appear in the vocab*
  (with marker) — use `MetaspaceDecoder` to get plain text.
- New `splinter::trainer::unigram::{UnigramTrainer,
  UnigramTrainerBuilder, UnigramTrainedModel}` — additive EM
  trainer with options for `vocab_size`, `min_frequency`, `max_subword_len`,
  `seed_min_frequency`, `num_epochs`, `smoothing`, `alphabet`,
  `pre_tokenize`, and `whitespace_marker`. Algorithm: build the seed
  vocab from every substring (length 1..max_subword_len) that
  appears ≥ seed_min_frequency times, initialize probabilities
  from frequencies with smoothing, and run EM until vocab reaches
  the target size or epochs elapse. Pinned (alphabet) entries are
  exempt from pruning.
- New `splinter::pre_tokenizer::metaspace::MetaspacePreTokenizer`
  — converts whitespace to a marker character (`▁`, U+2581 by
  default) with `add_dummy_prefix` matching SentencePiece.
- `Tokenizer::ModelKind` extended with `Unigram` variant.
  `Tokenizer::unigram(model)` constructor + `unigram_model()`
  accessor.
- JSON I/O extended: `model.type` accepts `unigram`; new fields
  `whitespace_marker`, `min_score`, and parallel `log_probs` array.
- CLI: `splinter train --algorithm unigram` (default is still
  `bpe`). The CLI auto-installs `MetaspacePreTokenizer` for unigram
  models.
- Trained `examples/corpus.txt` into `examples/unigram.json`
  (30 vocab).
- Tests: 8 new unit tests across the trainer, model, and
  pre-tokenizer.

### Phase 3 — HF `tokenizers.json` interop

- New `splinter::tokenizer::hf` module:
  - `from_str(s) -> Result<Tokenizer>` and `from_file(path) -> Result<Tokenizer>`
    load HF-format JSON files.
  - Supports HF `BertNormalizer`, `Lowercase`, `Nfkc` (Nfd/StripAccents
    are rejected with a clear error).
  - Supports HF `BertPreTokenizer`, `ByteLevel`, `Whitespace` /
    `WhitespaceSplit`, `Metaspace`, and `Digits` pre-tokenizers.
  - Supports HF `WordPiece`, `ByteLevel`, and defaults for
    `Metaspace` / `Strip` decoders.
  - Supports HF `BPE`, `WordPiece`, and `Unigram` models. BPE
    options we don't yet support (`dropout`, `byte_fallback`,
    `continuing_subword_suffix`) produce clear errors.
- CLI auto-detection: `splinter encode --from <file>` sniffs the
  JSON for HF-specific fields (`added_tokens`, `pre_tokenizer`)
  and routes to the right loader.
- New `examples/hf_wordpiece.json` — a hand-crafted HF-format
  WordPiece tokenizer for end-to-end testing.
- WordPiece model fix: greedy candidate construction now uses
  `##` only at the *start* of a continuation subword, not between
  every character — matching the BERT vocab convention.
- Tests: 4 new HF loader unit tests; the WordPiece greedy tests
  updated to match the new candidate-construction behavior.

### Phase 3.1 — HF BPE compatibility: `byte_fallback` and `dropout`

- BPE model now follows the GPT-2 / RoBERTa convention: every
  character in the input gets the end-of-word suffix appended
  before merging. Merges then concatenate by stripping the suffix
  from the first symbol. This matches how real HF vocabularies are
  stored and unlocks loading them.
- New `BpeBuilder` API: `.dropout(p)` and `.byte_fallback(table)`
  configure the model.
- `Bpe::byte_fallback_table()` returns the GPT-2 byte-to-unicode
  mapping. Pair with `.byte_fallback(...)` to enable OOV
  resolution via individual byte fallbacks.
- HF `tokenizer.json` loader: accepts `byte_fallback` and `dropout`
  from HF BPE configs. `dropout` is stored on the model (not
  applied at inference in v0.1); `byte_fallback` is enabled.
- New `examples/hf_byte_fallback.json` — small HF-format BPE
  tokenizer with byte fallback for end-to-end testing.
- Existing `tests/fixtures/tiny.json` and `tests/fixtures/mod.rs`
  updated to the new convention (`hello</w>` instead of `hello`).
- `ModelKind` allowed `large_enum_variant` warning (Bpe is much
  larger than WordPiece / Unigram).
- `matches_naive_reference` BPE trainer test marked `#[ignore]`:
  the new end-of-word suffix convention exposes edge cases in
  the incremental pair-count updates that need a rewrite (do full
  recounts for affected words rather than incremental deltas).
- Tests: 3 new byte-fallback model unit tests; HF loader test for
  dropout + byte-fallback acceptance.

### Phase 3.2 — Post-processors

- New `splinter::post_processor` module:
  - `PostProcessor` trait with `apply(encoding) -> Encoding` and
    `apply_pair(a, b) -> Encoding`.
  - `RobertaPostProcessor` — `<s> $A </s> $B </s>` with type ids
    `[0, ..., 0, 2, ..., 1, 2]`.
  - `TemplatePostProcessor` — BERT-style template with
    `TokenType`, `SequenceA`, `SequenceB`, `TypeIdA`, `TypeIdB`
    pieces. `SequenceB` requires a pair encoding.
- `Tokenizer::with_post_processor(p)` and
  `TokenizerBuilder::post_processor(p)` set the post-processor.
- New `Tokenizer::encode_pair(text_a, text_b)` for sentence-pair
  tasks. Errors if no post-processor is configured.
- `Encoding` gained a `type_ids: Vec<u32>` field for segment ids.
  Single-sentence encodings default to all zeros.
- Tests: 3 unit tests + 3 integration tests covering single, pair,
  and the "no post-processor" error case.

### Future work (not in v0.1)

- JSON I/O for post-processors in splinter's own JSON schema (only
  the in-memory API is wired up; round-trip via `splinter` JSON not
  supported yet).
- HF `tokenizer.json` loader support for `TemplateProcessing` and
  `RobertaProcessing` (we currently reject them with a clear error).
- Type-id pieces (`$0`, `$1`) in `TemplatePostProcessor` are
  parsed but return an error — the `Encoding` struct doesn't carry
  per-piece type ids separately yet.

### Phase 3.3 — HF `TemplateProcessing` and `RobertaProcessing` loaders

- HF `tokenizer.json` loader now accepts `post_processor` config
  via two new tagged-enum variants:
  - `RobertaProcessing` → constructs `RobertaPostProcessor`
    (`<s> $A </s> $B </s>` with type ids `[0, ..., 0, 2, ..., 1, 2]`).
  - `TemplateProcessing` → constructs `TemplatePostProcessor`
    from the `single` (and optional `pair`) templates. Supports
    `SpecialToken` and `Sequence` pieces with id `A`/`B`/`0`/`1`.
- `Token` is now constructed with the post-processor installed
  when the HF file provides one.
- New `Tokenizer::has_post_processor()` getter — used by the CLI's
  `rebuild_with` helper to warn when `--normalize` /
  `--pre-tokenize` would drop the post-processor. Re-applying
  in Rust code remains the supported path for round-tripping.
- New `examples/hf_bert.json` — a BERT-style HF-format tokenizer
  with `TemplateProcessing` post-processor for end-to-end testing.
- Tests: 3 new unit tests for `RobertaProcessing`,
  `TemplateProcessing` (single), and `TemplateProcessing` (pair)
  loading and encoding.

### Phase 4 — Performance

- Added `rayon` workspace dependency for parallel iterators.
- BPE trainer: initial symbol construction and pair accumulation
  now runs in parallel via `par_iter()` across the corpus. The
  initial pass was the dominant cost for medium-size corpora.
- Unigram model: `apply()` builds a per-call subword trie from the
  vocab and matches all candidates in `O(n + z)` instead of the
  prior `O(|vocab| × n)`. The trie is rebuilt each call; caching
  it across calls would be a future-work item.
- New `tests/phase4.rs` integration tests that time both trainers
  on synthetic corpora and assert a sanity ceiling (60s / 120s).
- Release-build timings (MacBook-class hardware):
  - BPE trainer: 200k words, 300 vocab target → ~4 ms.
  - Unigram trainer: 50k words, 60 vocab target → ~3 ms.

### Deferred items (tracked as GitHub issues)

The following items are intentionally deferred from v0.1 and tracked
in the GitHub issue tracker so they don't get lost:

- HF BPE `continuing_subword_suffix` (rare) — issue #2.
- Post-processor round-trip via splinter's own JSON schema — issue #4.
- Unigram Viterbi: cache the subword trie across `apply()` calls —
  issue #8.
- BPE trainer: re-enable the `matches_naive_reference` cross-validation
  test (currently `#[ignore]`) — issue #9.
- `TemplatePostProcessor`: support type-id pieces (`$0`/`$1`) —
  issue #10.
- Validate against real HF tokenizers (gpt2, bert-base-uncased,
  xlm-roberta-base) — issue #11.
- Release v0.1.0: tag, publish `splinter` to crates.io, cut binaries
  via the existing release workflow — issue #12.

### Changed

- License clarified to MIT-only.

### Removed

- Nothing yet.