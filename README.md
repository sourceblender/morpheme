# splinter

A Rust tokenizer library inspired by [Hugging Face `tokenizers`](https://huggingface.co/docs/tokenizers/en/index).

`splinter` aims to be a fast, ergonomic, fully-tested alternative implementation of modern subword tokenization — BPE, WordPiece, Unigram — built from scratch in idiomatic Rust.

## Status

🚧 **Pre-alpha.** Skeleton only. Nothing works yet.

## Goals

- Pure Rust, no FFI.
- Fast: SIMD-friendly where it pays, zero-copy on hot paths.
- Modular: bring your own normalizer, pre-tokenizer, post-processor, decoder.
- Trainable from raw text corpora.
- Compatible with `tokenizers` JSON model format where it makes sense.

## Layout

```
splinter/
├── crates/
│   └── splinter/        # core library crate
├── apps/
│   └── splinter-cli/    # CLI binary
├── docs/
│   └── architecture.md  # design notes
└── Cargo.toml           # workspace root
```

## License

TBD.