# Architecture

## Goals

- Pure-Rust tokenizer library + CLI, no FFI.
- **Behavioral compatibility with Hugging Face `tokenizers`**: same
  `tokenizer.json` format (read and write) and the same ids, tokens,
  offsets and decoded text. Compatibility is verified, not assumed (see
  [Testing](#testing)).
- BPE, WordPiece, WordLevel and Unigram models, with trainers.
- Fast enough to be the obvious choice in Rust services and tools.

## Non-goals (v0.x)

- Python/JS bindings (see the roadmap).
- Custom, user-defined pipeline components that serialize to
  `tokenizer.json` (built-in components only; see
  [Components](#components)).

## Crate layout

```
morpheme/
├── crates/morpheme      # the library
├── apps/morpheme-cli    # `morpheme` binary: train / encode / decode / inspect
└── bindings/wasm        # `morpheme-wasm`: wasm-bindgen glue + browser example (unpublished)
```

## Pipeline

```
input text
  │
  ├─ AddedVocabulary ─ split out added/special tokens (raw matches first)
  ├─ Normalizer      ─ rewrite text, keeping byte alignments to the input
  ├─ AddedVocabulary ─ split out added tokens that match normalized text
  ├─ PreTokenizer    ─ split into words (each a NormalizedString slice)
  ├─ Model           ─ word → tokens (ids, values, offsets in the word)
  ├─ truncation      ─ optional, reserves room for special tokens
  ├─ PostProcessor   ─ add special tokens, merge pairs, set type ids
  └─ padding         ─ optional
  ▼
Encoding { ids, tokens, offsets, type_ids, word_ids,
           special_tokens_mask, attention_mask, overflowing }

ids ─▶ (skip special tokens) ─▶ Decoder ─▶ text
```

`Tokenizer::train` runs the normalizer and pre-tokenizer over a corpus,
feeds the resulting words to a trainer, replaces the model, and
registers the trainer's special tokens as added tokens.

## Core types

| Type | Role |
| --- | --- |
| `NormalizedString` | Original text, normalized text, and for every normalized byte the original byte range it came from. All normalizer/pre-tokenizer edits go through alignment-preserving operations (`transform`, `replace`, `split`, `nfd`, …), which is how offsets into the original input survive any rewrite. |
| `PreTokenizedString` | Ordered splits of the input; each split is a `NormalizedString` slice plus, once tokenized, its tokens. Splits produced by added tokens are already tokenized and skipped by later stages. |
| `Encoding` | The output; supports merging, truncation with stride/overflow, and padding. |
| `AddedVocabulary` | Added and special tokens, matched with Aho-Corasick (leftmost-longest) and honoring `lstrip`/`rstrip`/`single_word`/`normalized`. |

The alignment model is the same as Hugging Face's, so offsets match
exactly, including for normalizers that expand or delete characters.

## Components

Each stage is a trait in `morpheme::traits` — `Normalizer`,
`PreTokenizer`, `Model`, `PostProcessor`, `Decoder`, `Trainer` — and
each module has a wrapper enum (`NormalizerWrapper`,
`PreTokenizerWrapper`, `ModelWrapper`, `PostProcessorWrapper`,
`DecoderWrapper`, `TrainerWrapper`) that implements the trait by
dispatch and (de)serializes with the Hugging Face `"type"` tag. A
`Tokenizer` holds wrappers, which keeps it non-generic, `Clone`, and
fully serializable.

`ByteLevel` plays three roles (pre-tokenizer, post-processor, decoder)
and `Metaspace` two (pre-tokenizer, decoder), as in Hugging Face.

## Serialization

`tokenizer.json` is the only on-disk format. Serialization is
byte-compatible with `tokenizers` 0.23 (field names, order and
defaults); deserialization also accepts the legacy forms Hugging Face
accepts (untyped models, `"a b"` merges, Metaspace `add_prefix_space`,
…). Unknown component types are errors — never silent fallbacks.
Details: [`interop.md`](./interop.md).

## Error model

One `morpheme::Error` enum (`thiserror`). Malformed input — bad JSON,
unknown types, merges that reference missing tokens, invalid regexes,
out-of-range ids — is always an `Err`, never a panic. Encoding never
fails for lack of vocabulary when the model has an `unk_token` or byte
fallback.

## Concurrency and performance

- `Tokenizer` is `Send + Sync`; `encode_batch`/`decode_batch` and trainer
  word counting use `rayon`.
- Regexes are compiled once per component (`fancy-regex`, which supports
  the look-around used by GPT-2-style patterns).
- BPE memoizes merged words in a sharded cache; Unigram builds its trie
  once at load.

## Feature flags

- `progressbar` (default): trainer progress bars via `indicatif`.
- `parallel` (default): `rayon` for `encode_batch`, `decode_batch`, batch
  padding and the trainers. Without it the same code paths run
  sequentially on the calling thread (`#[cfg(feature = "parallel")]` picks
  `par_iter`/`par_chunks`/`par_bridge` or their `std` counterparts at each
  site), which is what single-threaded targets want.
- `hub`: see below.

The library compiles for `wasm32-unknown-unknown`. `tempfile` is a
non-wasm dependency there, so `Tokenizer::save` (atomic replace through a
temporary file) is compiled out on wasm32; `from_bytes`/`to_json` cover
the browser. CI checks `--no-default-features --target
wasm32-unknown-unknown`.

`bindings/wasm` (`morpheme-wasm`) wraps `Tokenizer` for JavaScript with
`wasm-bindgen`: `fromJson`, `encode`, `tokens`, `count`, `decode`, with
Rust errors thrown as JS `Error`s. It depends on the library with
`default-features = false` and is built with `wasm-pack build --target
web`; see `bindings/wasm/README.md` and the `www/index.html` example.

## Hub downloads

The optional `hub` feature adds `Tokenizer::from_pretrained`
(`tokenizer/hub.rs`): a blocking HTTPS client (`ureq` with rustls, no
OpenSSL) that writes the standard `huggingface_hub` cache layout. It is
off by default in the library, so the core has no network dependencies;
the CLI enables it.

## Testing

| Layer | Where |
| --- | --- |
| Golden interop vs 11 real tokenizers | `crates/morpheme/tests/hf_golden.rs`, fixtures pinned in `scripts/hf-fixtures.txt`, outputs from `scripts/gen_golden.py` |
| Reverse interop (Python loads morpheme files) | `scripts/check_python_interop.py` (CI job) |
| Per-component ground truth | unit tests next to each component |
| Properties (lossless round-trip, valid offsets) | `crates/morpheme/tests/roundtrip.rs` |
| Regressions for past defects | `crates/morpheme/tests/regressions.rs` |
| CLI | `apps/morpheme-cli/tests/cli.rs` |
