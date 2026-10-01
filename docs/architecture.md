# splinter — Architecture (stub)

This document is intentionally short. It will grow alongside the code.

## Goals

- Pure-Rust tokenizer library + CLI.
- Compete on ergonomics and speed with `huggingface/tokenizers`.
- Support BPE, WordPiece, and Unigram tokenizers.
- Compatible with the `tokenizers` JSON model format (load + save).

## Crate layout

```
crates/splinter          # core library
apps/splinter-cli        # CLI front-end
```

The workspace keeps the library reusable in isolation. The CLI is a thin wrapper.

## Module plan (library)

- `normalizer` — text normalization (NFC, NFKC, strip, replace, BERT-style).
- `pre_tokenizer` — split text into words / whitespace / punctuation / bytes.
- `model` — BPE, WordPiece, Unigram core algorithms.
- `post_processor` — add special tokens, build NLI / seq2seq input templates.
- `decoder` — turn ids / tokens back into text.
- `trainer` — train a model from a corpus.
- `tokenizer` — top-level glue: load, tokenize, encode, decode, save.

Each module will follow the trait-per-component pattern from `tokenizers` so users can mix and match.

## Open questions

- Do we want to support loading existing `tokenizers.json` files in v0.1?
- Async or sync I/O for corpus loading during training?
- SIMD-acceleration story — `memchr`? `byteorder`? custom?

## Status

Skeleton only. `cargo test` should pass. `cargo run -p splinter-cli` should print a version banner and exit.