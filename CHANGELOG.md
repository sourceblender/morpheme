# Changelog

All notable changes to `splinter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first implementation (BPE/WordPiece/Unigram with a custom JSON
schema) was found in review to be incompatible with real Hugging Face
tokenizers and to have correctness bugs in every model and trainer. The
library was rebuilt around the Hugging Face `tokenizers` design and file
format, and is now verified against the reference implementation.

### Added

- **Hugging Face compatibility.** `tokenizer.json` is the on-disk format
  (read and write, byte-compatible with `tokenizers` 0.23, legacy forms
  accepted). Files saved by splinter load in Python `tokenizers` and
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

### Changed

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
