# Architecture

> Status: **stub / living document.** Update as decisions land.

## Goals

- Pure-Rust tokenizer library + CLI.
- Compete on ergonomics and speed with [`huggingface/tokenizers`](https://huggingface.co/docs/tokenizers/en/index).
- Support **BPE**, **WordPiece**, and **Unigram** tokenization algorithms.
- Compatible with the `tokenizers` JSON model format (load + save).
- Trainable from raw text corpora.
- Reusable in constrained environments (`no_std` is **not** a v0.1 goal but the layout shouldn't preclude it).

## Non-goals (v0.x)

- SentencePiece's full Python API surface.
- Multimodal tokenizers (image / audio).
- A hosted model hub.

## Crate layout

```
splinter/
├── crates/splinter          # core library — the public API
└── apps/splinter-cli        # CLI front-end (thin wrapper over the library)
```

The workspace keeps the library reusable in isolation. The CLI is a thin
wrapper that exposes the library to humans and shell scripts. Future
language bindings (Python / Node / WASM) would consume `crates/splinter`
directly.

## Module plan (library)

| Module             | Responsibility                                                       |
| ------------------ | -------------------------------------------------------------------- |
| `normalizer`       | Text normalization (NFC, NFKC, NFD, strip, replace, BERT-style).    |
| `pre_tokenizer`    | Split text into words / whitespace / punctuation / bytes / splits. |
| `model`            | BPE, WordPiece, Unigram core algorithms.                            |
| `post_processor`   | Add special tokens, build NLI / seq2seq input templates.            |
| `decoder`          | Turn ids / tokens back into text (ByteLevel, WordPiece, Metaspace).  |
| `trainer`          | Train a model from a corpus.                                         |
| `tokenizer`        | Top-level glue: load, tokenize, encode, decode, save.                |

v0.1 ships every module *type* listed here, with concrete
implementations for the BPE subset (`BertNormalizer`, `BertPreTokenizer`,
`ByteLevel`, `WordPieceDecoder`, `ByteLevelDecoder`). Other variants
land across Phase 2.

Each module is composed through traits so users can mix and match. The
default `Tokenizer::default()` wires a reasonable starting configuration.

## Pipeline

```
                ┌──────────────┐
   raw text ──▶ │  Normalizer  │ ──▶ normalized text
                └──────────────┘
                       │
                       ▼
                ┌──────────────┐
   normalized ─▶ │ PreTokenizer │ ──▶ pre-tokens (words / spans)
                └──────────────┘
                       │
                       ▼
                ┌──────────────┐
   pre-tokens ─▶ │    Model     │ ──▶ token ids (per pre-token)
                └──────────────┘
                       │
                       ▼
                ┌──────────────┐
       ids ──▶  │PostProcessor │ ──▶ final ids + type ids
                └──────────────┘

   ids ──▶ Decoder ──▶ text
```

## Error model

A single `splinter::Error` enum (thiserror). Conversion to `anyhow::Error`
is provided for callers who prefer a dynamic approach. There is **no**
`unwrap()` in the public library surface — every fallible operation
returns `Result<_, Error>`.

## Performance stance

- **Hot path is encode.** Optimize for it.
- Avoid allocations during pre-tokenization when possible.
- Pre-tokenizers return ranges into the original buffer where practical.
- BPE pair merges operate over integer indices, not strings.
- Public benchmarks live in `benches/` (Criterion) and the methodology in
  [`docs/benchmarks.md`](./benchmarks.md).

## Interop with `huggingface/tokenizers`

- **Splinter's own JSON format** — v1.0 schema in
  `crates/splinter/src/tokenizer/json.rs`. BPE only. Loaded and saved
  via `Tokenizer::from_json` / `to_json` / `from_file` / `to_file`.
- **Load:** read a `tokenizer.json` and reconstruct internal state. Best
  effort in v0.1 — components without an exact analogue are skipped with
  a warning. Lands in Phase 3.
- **Save:** emit a JSON shape that `tokenizers` can read. Drift is
  tracked in [`docs/interop.md`](./interop.md).

## Open questions

- Async or sync I/O for corpus loading during training? **Tendency:** sync
  with background-thread producers for parallelism; async is left for a
  later major.
- SIMD story — `memchr` + hand-rolled ASCII fast path? **Tendency:** yes,
  with a portable fallback.
- Stable Rust MSRV. **Tendency:** `1.74` for now. Bump as needed.

## Decision log

Each substantial architectural choice gets a short ADR in
[`docs/decisions/`](./decisions). Use `docs/decisions/0000-template.md`
as the starting point.