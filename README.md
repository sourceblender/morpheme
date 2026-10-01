<div align="center">

# morpheme

**Fast, pure-Rust subword tokenization, compatible with Hugging Face [`tokenizers`](https://huggingface.co/docs/tokenizers/en/index).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)

</div>

`morpheme` loads, runs, trains and saves tokenizers in the Hugging Face
`tokenizer.json` format — BPE, WordPiece, WordLevel and Unigram — and
matches the reference implementation on the supported configurations and
pinned real-model tests. Deliberate edge-case corrections and unsupported
forms are documented in [the compatibility guide](./docs/interop.md).

- **Compatible** — reads and writes `tokenizer.json`; files trained with
  morpheme load in Python `tokenizers` and encode identically, and vice
  versa.
- **Verified** — every release is checked against 11 real, pinned
  tokenizers (BERT, GPT-2, RoBERTa, GPT-NeoX, Qwen2.5, Llama, T5,
  ALBERT, XLM-R, …) across tricky Unicode inputs.
- **Pure Rust** — no FFI, no C/C++ dependencies.
- **Fast** — faster than Python `tokenizers` in our encode and training
  benchmarks (see [benchmarks](./docs/benchmarks.md)).

## Status

**v0.1.1** — correctness and reliability fixes. The library API may still change before 1.0.

```sh
cargo add morpheme                    # the library
cargo add morpheme --features hub     # + download tokenizers from the Hugging Face Hub
```

The `morpheme` CLI:

```sh
cargo install morpheme-cli
# or a prebuilt binary (macOS, Linux, Windows):
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/sourceblender/morpheme/releases/latest/download/morpheme-cli-installer.sh | sh
```

On Windows: `powershell -ExecutionPolicy Bypass -c "irm https://github.com/sourceblender/morpheme/releases/latest/download/morpheme-cli-installer.ps1 | iex"`.

## Library

Load a supported `tokenizer.json` and encode:

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

`encode` returns byte offsets; `encode_char_offsets` returns char
offsets like the Python API. `encode_batch` encodes in parallel and
applies the file's padding/truncation settings.

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

## CLI

```sh
cargo install morpheme-cli   # or, from a checkout: cargo install --path apps/morpheme-cli

# Train (presets: byte-level for bpe, bert for wordpiece,
# sentencepiece for unigram, whitespace for wordlevel)
morpheme train --model bpe --vocab-size 5000 --out tokenizer.json examples/corpus.txt

morpheme encode -t tokenizer.json "the quick brown fox"
morpheme encode -t tokenizer.json --json --pair "second" "first"
morpheme decode -t tokenizer.json 84,259,420,1209,513   # → "the quick brown fox"
morpheme inspect -t tokenizer.json

# Any Hugging Face Hub model id works in place of a path (cached locally)
morpheme encode -t google-bert/bert-base-uncased "Hello, world!"
morpheme inspect -t meta-llama/Llama-3.2-1B --revision main   # gated: set HF_TOKEN
```

Every command uses the full pipeline stored in the file; `--help` on
any command lists its options.

## What's supported

| Component | Types (Hugging Face `type` names) |
| --- | --- |
| Models | `BPE` (incl. `byte_fallback`, `dropout`, prefixes/suffixes), `WordPiece`, `WordLevel`, `Unigram` |
| Normalizers | `BertNormalizer`, `NFC`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `Strip`, `StripAccents`, `Replace`, `Prepend`, `Precompiled` (SentencePiece), `Nmt`, `ByteLevel`, `Sequence` |
| Pre-tokenizers | `ByteLevel`, `BertPreTokenizer`, `Whitespace`, `WhitespaceSplit`, `Metaspace`, `Split` (regex, incl. look-around), `Punctuation`, `Digits`, `CharDelimiterSplit`, `UnicodeScripts`, `FixedLength`, `Sequence` |
| Post-processors | `TemplateProcessing`, `BertProcessing`, `RobertaProcessing`, `ByteLevel`, `Sequence` |
| Decoders | `ByteLevel`, `WordPiece`, `Metaspace`, `BPEDecoder`, `ByteFallback`, `Fuse`, `Strip`, `Replace`, `CTC`, `Sequence` |
| Trainers | `BpeTrainer` (output identical to HF), `WordPieceTrainer`, `WordLevelTrainer`, `UnigramTrainer` |
| Tokenizer | added/special tokens, pairs, pre-tokenized input, truncation (with stride/overflow), padding, batch encode/decode, streaming decode |

Known differences from Hugging Face are listed in
[`docs/interop.md`](./docs/interop.md).

## How it's tested

- `tests/hf_golden.rs` — 11 real tokenizers pinned by revision
  ([`scripts/hf-fixtures.txt`](./scripts/hf-fixtures.txt)); ids, tokens,
  offsets, masks, word ids, decoding, pairs, batches and the serialized
  JSON must match outputs recorded from `tokenizers` 0.23.2
  ([`scripts/gen_golden.py`](./scripts/gen_golden.py)).
- [`scripts/check_python_interop.py`](./scripts/check_python_interop.py) —
  the reverse direction: morpheme-trained files loaded by Python.
- Property tests (lossless byte-level round-trips, valid offsets),
  per-component ground-truth tests, and regression tests for past bugs.

```sh
./scripts/fetch-hf-fixtures.sh   # once: downloads the pinned tokenizer.json files
cargo test --workspace
```

## Project layout

```
morpheme/
├── crates/morpheme/     # the library
├── apps/morpheme-cli/   # the `morpheme` binary
├── docs/                # design and module documentation
├── examples/            # sample corpus and trained tokenizers
└── scripts/             # fixture download, golden generation, interop check
```

## Documentation

- [`docs/architecture.md`](./docs/architecture.md) — design and pipeline.
- [`docs/modules/`](./docs/modules) — per-component reference.
- [`docs/interop.md`](./docs/interop.md) — format compatibility and known differences.
- [`docs/benchmarks.md`](./docs/benchmarks.md) — performance and methodology.
- [`docs/roadmap.md`](./docs/roadmap.md) — what's next.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`CHANGELOG.md`](./CHANGELOG.md).

## Acknowledgements

morpheme's file format and component semantics follow Hugging Face
[`tokenizers`](https://github.com/huggingface/tokenizers) (Apache-2.0),
which served as the behavioral reference.

## License

[MIT](./LICENSE) © 2026 The morpheme authors.
