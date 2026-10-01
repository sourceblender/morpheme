<div align="center">

# morpheme

**Fast, pure-Rust subword tokenization, compatible with Hugging Face [`tokenizers`](https://huggingface.co/docs/tokenizers/en/index).**

[![crates.io](https://img.shields.io/crates/v/morpheme.svg)](https://crates.io/crates/morpheme)
[![PyPI](https://img.shields.io/pypi/v/morpheme.svg)](https://pypi.org/project/morpheme/)
[![docs.rs](https://img.shields.io/docsrs/morpheme)](https://docs.rs/morpheme)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)

</div>

`morpheme` loads, runs, trains and saves tokenizers in the Hugging Face
`tokenizer.json` format — BPE, WordPiece, WordLevel and Unigram — and
matches the reference implementation on the supported configurations and
pinned real-model tests. Deliberate edge-case corrections and unsupported
forms are documented in [the compatibility guide](./docs/interop.md).
It ships as a Rust library, a command-line tool, a Python package and a
WebAssembly (WASM) module for the browser.

- **Compatible** — reads and writes `tokenizer.json`; files trained with
  morpheme load in Python `tokenizers` and encode identically, and vice
  versa.
- **Verified** — every change is checked against 11 real, pinned
  tokenizers (BERT, GPT-2, RoBERTa, GPT-NeoX, Qwen2.5, Llama, T5,
  ALBERT, XLM-R, …) across tricky Unicode inputs.
- **Pure Rust** — no FFI; the default build has no C/C++ dependencies
  (the optional `hub` feature uses rustls, whose `ring` backend includes
  C and assembly).
- **Fast** — faster than Python `tokenizers` in our encode and training
  benchmarks (see [benchmarks](./docs/benchmarks.md)).

**Status:** v0.4.0. The library API may still change before 1.0; see the
[changelog](./CHANGELOG.md).

## Install

| Channel | Install |
| --- | --- |
| Rust library | `cargo add morpheme` (add `--features hub` to download from the Hugging Face Hub) |
| CLI | `cargo install morpheme-cli`, or a prebuilt binary (below) |
| Python (≥ 3.9) | `pip install morpheme` |
| WASM | build from source with `wasm-pack` ([`bindings/wasm`](./bindings/wasm/README.md)); not published to npm |

Prebuilt CLI binaries, via the installer scripts attached to each
[GitHub Release](https://github.com/sourceblender/morpheme/releases/latest):

```sh
# macOS (Apple silicon) and Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/sourceblender/morpheme/releases/latest/download/morpheme-cli-installer.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -c "irm https://github.com/sourceblender/morpheme/releases/latest/download/morpheme-cli-installer.ps1 | iex"
```

### Supported platforms

| Channel | Prebuilt artifacts | Elsewhere |
| --- | --- | --- |
| Rust library | — (source crate on crates.io) | any target with Rust 1.85+, including `wasm32-unknown-unknown` |
| CLI | Linux x86-64 (glibc and musl) and arm64 (glibc); macOS arm64; Windows x86-64 | `cargo install morpheme-cli` (Intel macOS included) |
| Python | abi3 wheels for CPython ≥ 3.9: Linux x86-64 and aarch64 (manylinux), macOS arm64, Windows x64 | the sdist builds with a Rust toolchain (Intel macOS included) |
| WASM | — | `wasm-pack build` in `bindings/wasm` |

## Quick start

### Rust

Load a `tokenizer.json` and encode:

```rust
use morpheme::Tokenizer;

let tokenizer = Tokenizer::from_file("bert-base-uncased/tokenizer.json")?;

let encoding = tokenizer.encode("Hello, world!", true)?;
assert_eq!(encoding.tokens(), ["[CLS]", "hello", ",", "world", "!", "[SEP]"]);
assert_eq!(encoding.ids(), [101, 7592, 1010, 2088, 999, 102]);
assert_eq!(encoding.offsets()[1], (0, 5)); // byte offsets into the input

let pair = tokenizer.encode(("How are you?", "Fine."), true)?;
println!("{:?}", pair.type_ids()); // [0, 0, 0, 0, 0, 0, 1, 1, 1]

let text = tokenizer.decode(encoding.ids(), true)?;
assert_eq!(text, "hello, world!");
```

`encode` returns byte offsets; `encode_char_offsets` returns char
offsets like the Python API. `encode_batch` applies the file's
padding/truncation settings and runs in parallel with the default
`parallel` feature.

With the `hub` feature, load straight from the Hugging Face Hub. Files
go to the standard Hugging Face cache (shared with Python) and
`HF_TOKEN`, `HF_HOME`, `HF_ENDPOINT` and `HF_HUB_OFFLINE` work as usual:

```rust
use morpheme::{FromPretrainedParameters, Tokenizer};

let tokenizer = Tokenizer::from_pretrained("google-bert/bert-base-uncased", None)?;
let pinned = Tokenizer::from_pretrained(
    "openai-community/gpt2",
    Some(FromPretrainedParameters::default().revision("607a30d783dfa663caf39e06633721c8d4cfcd7e")),
)?;
```

Train a GPT-2-style byte-level BPE tokenizer:

```rust
use morpheme::models::Bpe;
use morpheme::pre_tokenizers::ByteLevel;
use morpheme::trainers::BpeTrainer;
use morpheme::{AddedToken, Tokenizer};

let mut tokenizer = Tokenizer::new(Bpe::default())
    .with_pre_tokenizer(ByteLevel::new(false, true, true))
    .with_decoder(ByteLevel::default());

let trainer = BpeTrainer::builder()
    .vocab_size(5_000)
    .initial_alphabet(ByteLevel::alphabet())
    .special_tokens(vec![AddedToken::new("<|endoftext|>", true)])
    .build()?;
tokenizer.train_from_files(trainer, &["corpus.txt"])?;
tokenizer.save("tokenizer.json", true)?;

let ids = tokenizer.encode("Any text — even 😀 — round-trips.", false)?.ids().to_vec();
assert_eq!(tokenizer.decode(&ids, false)?, "Any text — even 😀 — round-trips.");
```

Runtime-only model settings — Unigram subword-regularization sampling
and BPE dropout — are changed in place through `Tokenizer::model_mut`;
see [Unigram](./docs/modules/unigram.md#subword-regularization-sampling)
and [BPE](./docs/modules/bpe.md#runtime-settings).

### CLI

```sh
# Train (presets: byte-level for bpe, bert for wordpiece,
# sentencepiece for unigram, whitespace for wordlevel)
morpheme train --model bpe --vocab-size 5000 --out tokenizer.json examples/corpus.txt

morpheme encode -t tokenizer.json "the quick brown fox"
morpheme encode -t tokenizer.json --json "first" --pair "second"
morpheme decode -t tokenizer.json 84,259,420,1209,513   # → "the quick brown fox"
morpheme inspect -t tokenizer.json

# Budget raw text without configured padding or truncation
morpheme count -t tokenizer.json --no-special-tokens "the quick brown fox"
# Process bounded JSONL batches with one tokenizer load
morpheme encode-batch -t tokenizer.json --input documents.jsonl --ignore-tokenizer-settings
morpheme inspect -t tokenizer.json --json

# Any Hugging Face Hub model id works in place of a path (cached locally)
morpheme encode -t google-bert/bert-base-uncased "Hello, world!"
morpheme inspect -t meta-llama/Llama-3.2-1B --revision main   # gated: set HF_TOKEN
```

Every command uses the full pipeline stored in the file; `--help` on
any command lists its options. See [CLI automation](./docs/cli.md) for
JSONL schemas, counting semantics, size limits and streaming error
behavior.

### Python

```python
import morpheme

tok = morpheme.Tokenizer.from_pretrained("google-bert/bert-base-uncased")
# or: morpheme.Tokenizer.from_file("tokenizer.json")

enc = tok.encode("Hello, world!")
print(enc.tokens)   # ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
print(enc.offsets)  # character spans, like Python `tokenizers`
print(tok.decode(enc.ids))  # hello, world!
```

The Python API is deliberately small (load, encode, decode, count,
vocabulary lookups, save); see [`bindings/python`](./bindings/python/README.md).

### WebAssembly

[`bindings/wasm`](./bindings/wasm/README.md) wraps the library for the
browser with `wasm-bindgen`. After `wasm-pack build --target web`:

```js
import init, { Tokenizer } from "./pkg/morpheme_wasm.js";

await init();
const json = await (await fetch("tokenizer.json")).text();
const tok = Tokenizer.fromJson(json);
tok.count("Hello, world!", true); // 6 with bert-base-uncased
```

## Cargo features

| Feature | Default | Enables |
| --- | --- | --- |
| `progressbar` | yes | Trainer progress bars on stderr (`indicatif`) |
| `parallel` | yes | `rayon` for `encode_batch`, `decode_batch`, batch padding and the trainers; without it they run sequentially |
| `hub` | no | `Tokenizer::from_pretrained` (native targets only; a no-op on wasm32) |

The CLI enables `hub` by default. The library compiles for
`wasm32-unknown-unknown` with any feature set; `Tokenizer::save` is
compiled out there (use `to_json`).

## What's supported

| Component | Types (Hugging Face `type` names) |
| --- | --- |
| Models | `BPE` (incl. `byte_fallback`, `dropout`, prefixes/suffixes), `WordPiece`, `WordLevel`, `Unigram` (incl. subword-regularization sampling) |
| Normalizers | `BertNormalizer`, `NFC`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `Strip`, `StripAccents`, `Replace`, `Prepend`, `Precompiled` (SentencePiece), `Nmt`, `ByteLevel`, `Sequence` |
| Pre-tokenizers | `ByteLevel`, `BertPreTokenizer`, `Whitespace`, `WhitespaceSplit`, `Metaspace`, `Split` (regex, incl. look-around), `Punctuation`, `Digits`, `CharDelimiterSplit`, `UnicodeScripts`, `FixedLength`, `Sequence` |
| Post-processors | `TemplateProcessing`, `BertProcessing`, `RobertaProcessing`, `ByteLevel`, `Sequence` |
| Decoders | `ByteLevel`, `WordPiece`, `Metaspace`, `BPEDecoder`, `ByteFallback`, `Fuse`, `Strip`, `Replace`, `CTC`, `Sequence` |
| Trainers | `BpeTrainer` (output identical to Hugging Face), `WordPieceTrainer`, `WordLevelTrainer`, `UnigramTrainer` |
| Tokenizer | added/special tokens, pairs, pre-tokenized input, truncation (with stride/overflow), padding, batch encode/decode, streaming decode |

Legacy `tokenizer.json` forms written by older `tokenizers` versions
load too. Known differences from Hugging Face are listed in
[`docs/interop.md`](./docs/interop.md).

## How it's tested

- `crates/morpheme/tests/hf_golden.rs` — 11 real tokenizers pinned by
  revision ([`scripts/hf-fixtures.txt`](./scripts/hf-fixtures.txt));
  ids, tokens, offsets, masks, word ids, decoding, streaming decode,
  pairs, batches and the serialized JSON must match outputs recorded from
  `tokenizers` 0.23.2 ([`scripts/gen_golden.py`](./scripts/gen_golden.py)).
- [`scripts/check_python_interop.py`](./scripts/check_python_interop.py) —
  the reverse direction: morpheme-trained files loaded by Python.
- Property tests (lossless byte-level round-trips, valid offsets),
  per-component ground-truth tests, regression tests for past bugs,
  `cargo-fuzz` targets, and tests for the Python and WASM bindings.

```sh
./scripts/fetch-hf-fixtures.sh   # once: downloads the pinned tokenizer.json files
cargo test --workspace --all-features
```

The full local gate, including the Python and WASM bindings, is in
[`docs/contributing.md`](./docs/contributing.md#local-gate).

## Performance

Encoding is 1.2–1.7× faster than Python `tokenizers` 0.23.2 on the same
files and inputs, and training 1.5–1.8× faster; see
[`docs/benchmarks.md`](./docs/benchmarks.md) for the numbers, caveats and
methodology. Regressions are tracked daily on a dedicated benchmark host.

## Project layout

```
morpheme/
├── crates/morpheme/     # the library
├── apps/morpheme-cli/   # the `morpheme` binary
├── bindings/python/     # the `morpheme` Python package (PyO3 + maturin)
├── bindings/wasm/       # `morpheme-wasm` browser bindings (wasm-bindgen)
├── fuzz/                # cargo-fuzz targets (separate nightly workspace)
├── docs/                # design, module and operations documentation
├── examples/            # sample corpus and trained tokenizers
└── scripts/             # fixtures, golden generation, interop, benchmarks, release smoke
```

## Documentation

- [API reference on docs.rs](https://docs.rs/morpheme).
- [`docs/architecture.md`](./docs/architecture.md) — design, crate layout and pipeline.
- [`docs/modules/`](./docs/modules/README.md) — per-component reference.
- [`docs/interop.md`](./docs/interop.md) — format compatibility and known differences.
- [`docs/cli.md`](./docs/cli.md) — CLI automation: counting, JSONL batches, inspection.
- [`docs/document-workflow.md`](./docs/document-workflow.md) — executable
  dataset preparation with exact budgets and atomic publication.
- [`docs/benchmarks.md`](./docs/benchmarks.md) — performance and methodology.
- [`docs/decisions/`](./docs/decisions/README.md) — architecture decision records.
- [`docs/roadmap.md`](./docs/roadmap.md) — what's done and what's next.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`CHANGELOG.md`](./CHANGELOG.md),
  [`SECURITY.md`](./SECURITY.md), [`SUPPORT.md`](./SUPPORT.md).

## Acknowledgements

morpheme's file format and component semantics follow Hugging Face
[`tokenizers`](https://github.com/huggingface/tokenizers) (Apache-2.0),
which served as the behavioral reference.

## License

[MIT](./LICENSE) © 2026 The morpheme authors.
