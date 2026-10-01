# Changelog

All notable changes to `morpheme` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CLI token counting with explicit special-token and padding/truncation
  behavior, bounded JSONL batch encoding/decoding, and JSON inspection.

### Fixed

- Verify Hub downloads and cached files against Git blob SHA-1 or LFS
  SHA-256 ETags; repair corrupt entries online and reject them offline.
- Save tokenizer files using atomic replacement, preserving existing
  permissions and keeping readers from seeing a partial write.

## [0.1.1] - 2026-10-01

### Fixed

- Rebind existing added/special tokens after training and allocate new ids
  above sparse vocabulary ids; report id exhaustion instead of overflowing.
- Make added-token batches atomic on allocation or normalization errors,
  and retain existing token options when trainers promote them to specials.
- Rebind post-processor and padding ids by token text after training;
  reject missing configured tokens without changing the tokenizer.
- Keep BPE unknown-token runs before subsequent byte fallback, preserving
  input order and offsets.
- Publish Hub cache blobs, snapshots and refs atomically using unique
  temporary files so concurrent downloads cannot truncate each other's data.
- Preserve pair-overflow sequence ownership without a post-processor and
  correct lookups for nonzero or absent sequence ids.
- Reject impossible special-token truncation budgets and out-of-bounds
  normalized/original offset ranges.

### Changed

- Stream training files line by line instead of retaining the complete
  corpus. File and UTF-8 errors still fail training before replacing the model.
- Document deliberate corrections to edge-case bugs shared with Hugging
  Face `tokenizers` 0.23.2 in `docs/interop.md` and ADR 0002.

## [0.1.0] - 2026-10-01

The first implementation (BPE/WordPiece/Unigram with a custom JSON
schema) was found in review to be incompatible with real Hugging Face
tokenizers and to have correctness bugs in every model and trainer. The
library was rebuilt around the Hugging Face `tokenizers` design and file
format, and is now verified against the reference implementation.

### Added

- **Hugging Face compatibility.** `tokenizer.json` is the on-disk format
  (read and write, byte-compatible with `tokenizers` 0.23, legacy forms
  accepted). Files saved by morpheme load in Python `tokenizers` and
  encode identically.
- Offset tracking through every normalization step (`NormalizedString`
  alignments), so offsets always point into the original input; byte
  offsets (`encode`) and char offsets (`encode_char_offsets`).
- Models: `Bpe` (byte fallback with `<0xNN>` tokens, `unk_token`,
  `fuse_unk`, prefix/suffix, dropout, word cache), `WordPiece` (whole-word
  unk, `from_bpe`), `WordLevel`, `Unigram` (Viterbi, unk penalty, byte
  fallback, trie built once).
- Normalizers: `BertNormalizer`, `NFC`/`NFD`/`NFKC`/`NFKD`, `Lowercase`,
  `Strip`, `StripAccents`, `Replace`, `Prepend`, `Precompiled`
  (SentencePiece charsmap), `Nmt`, `ByteLevel`, `Sequence`.
- Pre-tokenizers: `ByteLevel` (GPT-2 regex), `BertPreTokenizer`,
  `Whitespace`, `WhitespaceSplit`, `Metaspace` (`prepend_scheme`,
  `split`), `Split` (regex with look-around), `Punctuation`, `Digits`,
  `CharDelimiterSplit`, `UnicodeScripts`, `FixedLength`, `Sequence`.
- Post-processors: `TemplateProcessing` (template strings, `$A`/`$B`/
  `$0`/`$1`, type ids), `BertProcessing`, `RobertaProcessing`,
  `ByteLevel`, `Sequence` — special-token ids come from the config.
- Decoders: `ByteLevel`, `WordPiece`, `Metaspace`, `BPEDecoder`,
  `ByteFallback`, `Fuse`, `Strip`, `Replace`, `CTC`, `Sequence`.
- Added/special tokens (`lstrip`, `rstrip`, `single_word`, `normalized`),
  sequence pairs, pre-tokenized input, truncation (strategies, stride,
  overflowing encodings), padding (batch-longest, fixed, multiple-of,
  left/right), parallel `encode_batch` / `decode_batch`.
- Streaming decode (`Tokenizer::decode_stream` → `DecodeStream`, with
  `step`, `step_many` and `prefill`) for token-by-token generation;
  matches Python `tokenizers`' `DecodeStream` step for step, checked in
  the golden tests.
- Trainers: `BpeTrainer` (vocabulary and merges identical to Hugging
  Face), `WordPieceTrainer`, `WordLevelTrainer`, `UnigramTrainer`
  (pure-Rust suffix array seeding, EM, likelihood-based pruning;
  deterministic). `Tokenizer::train` / `train_from_files` feed the
  tokenizer's own normalizer and pre-tokenizer; I/O errors are reported.
- CLI (`clap`): `train` with presets (byte-level, bert, sentencepiece,
  whitespace), `encode` (pairs, JSON output, char offsets, stdin),
  `decode`, `inspect`. Commands use the full pipeline from the file.
- Testing: golden tests against 11 pinned real tokenizers
  (`scripts/hf-fixtures.txt`, `scripts/gen_golden.py`), reverse interop
  check with Python (`scripts/check_python_interop.py`), property tests,
  per-component ground-truth tests, regression tests for every defect
  found in review.
- Benchmarks: Criterion suite (`cargo bench`: encode/decode on four
  pretrained tokenizers, BPE/WordPiece/Unigram training, normalization),
  compiled in CI; `bench_encode` example reporting encode and decode
  throughput; measured encode, decode, training and peak-memory
  comparisons with Python `tokenizers` in `docs/benchmarks.md`.
- `Tokenizer::from_pretrained` (feature `hub`): download `tokenizer.json`
  from the Hugging Face Hub into the standard cache shared with Python,
  with revisions, tokens (`HF_TOKEN` / saved login), `HF_ENDPOINT`,
  offline mode and cache fallback. The CLI accepts Hub model ids for
  `-t` (plus `--revision`).
- Trainer progress bars (`show_progress`, as in Hugging Face) behind the
  default `progressbar` feature (`indicatif`); hidden when stderr is not
  a terminal. CLI `train --quiet`.
- Coverage reporting in CI (`cargo-llvm-cov`, lcov artifact).
- Release automation with [`dist`](https://opensource.axo.dev/cargo-dist/):
  a `vX.Y.Z` tag builds the CLI for macOS (arm64, x86-64), Linux (gnu
  arm64/x86-64, musl x86-64) and Windows, attaches archives, SHA-256
  checksums and shell/PowerShell installers to a GitHub Release, and
  publishes both crates to crates.io.
- The published library crate excludes tests, benchmarks and test data
  (~3 MB → 128 KB).
- Fuzzing: cargo-fuzz targets for `tokenizer.json` loading, encoding
  through real tokenizers, `NormalizedString` operations and component
  JSON (`fuzz/`), with a weekly CI workflow.

### Changed

- **Renamed from `splinter` to `morpheme`.** The `splinter` crate name
  belongs to an unrelated project on crates.io, and `splintr` is a
  different Rust tokenizer. Crates: `morpheme` and `morpheme-cli`;
  binary: `morpheme`; repository: `sourceblender/morpheme`.
- **Breaking — public API cleanup for 0.1** (Rust API guidelines):
  - Constructors: `AddedToken::from` → `AddedToken::new`,
    `Unigram::from` → `Unigram::new`, `Precompiled::from` →
    `Precompiled::from_bytes` (the old names shadowed the `From` trait).
  - Getters lose the `get_` prefix: `Model::get_vocab` / `get_vocab_size`
    → `vocab` / `vocab_size` (also on `Tokenizer`, which keeps its
    `with_added_tokens` argument), `Bpe::get_unk_token` /
    `get_continuing_subword_prefix` / `get_end_of_word_suffix` →
    `unk_token` / `continuing_subword_prefix` / `end_of_word_suffix`,
    `AddedVocabulary::get_vocab` / `get_added_tokens_decoder` /
    `get_encode_special_tokens` → `vocab` / `added_tokens_decoder` /
    `encode_special_tokens`, `NormalizedString::get_original` →
    `original`, `BpeTrainer::get_word_count` → `word_count`.
    `Unigram::vocab()` → `pieces()` (and `iter()` is gone), so it no
    longer clashes with `Model::vocab`. Lookups that take an argument
    keep `get_` (`get_range`, `get_splits`), as in std.
  - Every trainer builder's `build()` now returns `Result` and validates
    its options (`vocab_size > 0`, …), like the model builders already
    did. `max_token_length(usize)`; `UnigramTrainerBuilder::vocab_size` and
    `n_sub_iterations` take `usize` and `unk_token` takes the token
    directly.
  - Models and trainers are configured through builders and read through
    getters; their fields are private (`Bpe`, `WordPiece`, `WordLevel`,
    all trainers). `Bpe::merges()` returns a slice. New getters:
    `Bpe::{dropout, fuse_unk, byte_fallback, ignore_merges}`,
    `WordPiece::{unk_token, continuing_subword_prefix,
    max_input_chars_per_word}`, `WordLevel::unk_token`,
    `Split::pattern`, `decoders::Replace::{pattern, content}` (their
    patterns are compiled at construction, so the fields can no longer
    be changed behind the compiled regex's back).
  - Constructors accept `impl Into<String>` / `impl Into<SplitPattern>`
    consistently (`Prepend::new`, `decoders::Replace::new`,
    `BertProcessing::new(("[SEP]", 102), …)`, `RobertaProcessing::new`).
  - `#[non_exhaustive]` on `Error`, the six wrapper enums,
    `TruncationStrategy`, `PaddingStrategy`, `InputSequence`,
    `EncodeInput`, and on configuration components with public fields
    (construct them with `new`).
  - Implementation modules are private; every type is reachable from its
    module (`morpheme::normalizers::BertNormalizer`, …).
    `pre_tokenizers::byte_level` (byte tables, `process_offsets`) and
    `processors::template` stay public. The Unigram lattice, pattern
    `Invert`, the BPE trainer's `do_train` and the WordPiece trainer's
    `bpe_trainer{,_mut}` are internal.
- rustdoc: crate-level guide, feature table and a runnable example on
  every component, model, trainer and the main `Tokenizer` methods;
  docs.rs builds with all features and marks feature-gated items.
- **Breaking:** the public API now mirrors Hugging Face `tokenizers`
  (`Tokenizer`, `Encoding` accessors, `models`, `normalizers`,
  `pre_tokenizers`, `processors`, `decoders`, `trainers` modules with
  wrapper enums). The old `ModelKind`, `Vocab`, `TrainedModel` and
  custom JSON schema are gone.
- **Breaking:** BPE no longer appends `</w>` to every character; an
  end-of-word suffix is optional and applies to the last character only,
  as in Hugging Face.
- MSRV raised from 1.74 to 1.85 (required by current dependencies;
  verified in CI). Crates moved to the Rust 2024 edition and the
  MSRV-aware dependency resolver (v3).
- Dependencies: `thiserror` 2; unused `tracing`/`tracing-subscriber`/
  `anyhow` dependencies removed; crates.io metadata (description,
  keywords, categories, readme) added.
- CI: GitHub Actions updated to their current majors, least-privilege
  `permissions`, cancellation of superseded runs, blocking `cargo-deny`
  (licenses, bans, sources, RustSec advisories) via `deny.toml`, rustdoc
  and Python-interop jobs. Release publishing passes the registry token
  via the environment.
- `examples/` tokenizers regenerated with the new CLI
  (`scripts/regenerate-examples.sh`); `examples/corpus.txt` expanded.

### Fixed

- Overflowing encodings from truncation now carry the same type ids as
  the main encoding for `RobertaProcessing` (all `0`),
  `TemplateProcessing` type overrides (e.g. `$B:3`) and the default pair
  processing; Hugging Face gets the first two wrong.
- `WordLevelTrainer` no longer leaves gaps in the id range when a
  special token is repeated or also appears in the corpus.
- BPE trainer produced models that could not be loaded (merged symbols
  missing from the vocabulary) and double-counted pair frequencies.
- HF loader silently replaced normalizers, pre-tokenizers and decoders
  (wrong `type` tag casing), renumbered vocabulary ids, failed to parse
  RoBERTa post-processors and array-style merges, and panicked on bad
  merges.
- `BertNormalizer` deleted non-ASCII characters whose low byte looked
  like whitespace (e.g. `Р`, `Ġ`, `†`).
- `BertPreTokenizer`, `ByteLevel`, `Metaspace`, `ByteLevelDecoder` and
  `WordPieceDecoder` diverged from Hugging Face (glued whitespace, missing
  `Ġ`, `?` for non-ASCII bytes, missing spaces between words).
- Post-processors used special-token id 0 and wrong RoBERTa type ids.
- WordPiece and Unigram trainers produced unusable or under-sized
  vocabularies; Unigram failed on unknown characters instead of emitting
  unk.
- The CLI replaced every loaded tokenizer's pipeline with defaults.
- Training from a missing or non-UTF-8 file silently trained on nothing.
- Found by fuzzing:
  - Saving a model whose vocabulary has several tokens with the same id
    (valid in Hugging Face files) dropped all but one of them, and BPE
    rewrote its merges from ids; the reloaded tokenizer encoded
    differently or failed to load. Vocabularies and merges are now
    written exactly as loaded.
  - A whole-string normalization (e.g. NFD) after text was inserted
    before the first character (`prepend("")`, an empty-pattern
    `replace`) indexed out of bounds and panicked.

[Unreleased]: https://github.com/sourceblender/morpheme/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sourceblender/morpheme/releases/tag/v0.1.1
[0.1.0]: https://github.com/sourceblender/morpheme/releases/tag/v0.1.0
