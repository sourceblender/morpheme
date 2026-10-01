<div align="center">

# splinter

**A Rust tokenizer library inspired by [Hugging Face `tokenizers`](https://huggingface.co/docs/tokenizers/en/index).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.74%2B-orange.svg)](https://www.rust-lang.org/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)

</div>

`splinter` aims to be a fast, ergonomic, fully-tested Rust implementation of modern subword tokenization — BPE, WordPiece, Unigram — built from scratch in idiomatic Rust, with a focus on:

- **Pure Rust** — no FFI, no system C deps.
- **Speed** — zero-copy on hot paths, careful allocation, `memchr`-friendly scanning.
- **Composability** — traits for normalizer, pre-tokenizer, model, post-processor, decoder. Mix and match.
- **Trainability** — train BPE / WordPiece / Unigram models from raw text corpora.
- **Interop** — load and save the [`tokenizers`](https://huggingface.co/docs/tokenizers/en/index) JSON format.

## Status

🚧 **Pre-alpha.** Skeleton only. Nothing useful yet. See [`docs/roadmap.md`](./docs/roadmap.md).

## Quick start

```sh
git clone https://github.com/sourceblender/splinter
cd splinter
cargo test
cargo run -p splinter-cli -- --help
```

## Project layout

```
splinter/
├── crates/splinter/        # core library
├── apps/splinter-cli/      # CLI binary
├── docs/                   # design + contributor docs
├── examples/               # sample corpus + trained tokenizer
└── .github/                # CI + issue / PR templates
```

### Train a tokenizer

```sh
splinter train \
  --input examples/corpus.txt \
  --vocab-size 1000 \
  --out examples/tokenizer.json
```

Then encode with the trained tokenizer:

```sh
splinter encode --from examples/tokenizer.json "the quick brown fox"
```

## What's in the library (v0.1)

| Module | Types |
| --- | --- |
| `normalizer` | `BertNormalizer`, `Lowercase`, `Nfd`, `Nfkc`, `StripAccents`, `Replace`, `IdentityNormalizer` |
| `pre_tokenizer` | `Whitespace`, `BertPreTokenizer`, `ByteLevel` (+ `ByteLevelAddChar`) |
| `model` | `Bpe` (GPT-2 style `</w>` end-of-word suffix) |
| `decoder` | `WordPieceDecoder`, `ByteLevelDecoder` |
| `tokenizer` | `Tokenizer` with `encode`, `decode`, `from_json` / `to_json` / `from_file` / `to_file` |

## Documentation

- [`docs/architecture.md`](./docs/architecture.md) — high-level design.
- [`docs/roadmap.md`](./docs/roadmap.md) — what we're building and in what order.
- [`docs/modules/`](./docs/modules) — per-module deep dives.
- [`docs/contributing.md`](./docs/contributing.md) — development workflow.
- [`docs/benchmarks.md`](./docs/benchmarks.md) — performance methodology.

User-facing project docs:

- [Contributing](./CONTRIBUTING.md)
- [Code of conduct](./CODE_OF_CONDUCT.md)
- [Security policy](./SECURITY.md)
- [Support](./SUPPORT.md)
- [Changelog](./CHANGELOG.md)

## License

[MIT](./LICENSE) © 2026 The splinter authors.