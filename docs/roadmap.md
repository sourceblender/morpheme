# Roadmap

> Living document: what has shipped, and what is planned next. No dates.

## Released

### v0.1 (2026-10-01)

- [x] Hugging Face-compatible pipeline: added tokens, normalizers,
      pre-tokenizers, models, post-processors, decoders, with offset
      tracking into the original input.
- [x] BPE, WordPiece, WordLevel, Unigram models; matching trainers
      (BPE output identical to Hugging Face), with progress bars
      (`progressbar` feature).
- [x] `tokenizer.json` read/write, byte-compatible with `tokenizers`
      0.23, legacy forms accepted.
- [x] Truncation (with overflow), padding, pairs, pre-tokenized input,
      parallel batch encode/decode, streaming decode (`DecodeStream`).
- [x] Hugging Face Hub download (`from_pretrained`, `hub` feature) using
      the cache shared with Python.
- [x] CLI: `train` (presets), `encode`, `decode`, `inspect`; accepts Hub
      model ids.
- [x] Golden tests against 11 real tokenizers; reverse interop check
      with Python; property, regression and fuzz testing; coverage in CI;
      MSRV 1.85.
- [x] Criterion benchmark suite (compiled in CI); encode, decode,
      training and memory comparisons with Python.
- [x] Public API review (`pub(crate)`, `#[non_exhaustive]`, Rust API
      Guidelines naming) and rustdoc examples.
- [x] Release pipeline (`dist`): CLI binaries, checksums, installers;
      crates.io publishing of [`morpheme`](https://crates.io/crates/morpheme)
      and [`morpheme-cli`](https://crates.io/crates/morpheme-cli).

### v0.2 (2026-10-01)

- [x] CLI token budgeting (`count`), bounded JSONL batches
      (`encode-batch`, `decode-batch`) and JSON inspection.
- [x] Hash-check Hub downloads and cached files against their ETag;
      repair corrupt entries online and reject them offline.
- [x] Atomic `tokenizer.json` saves.
- [x] Executable document-budget consumer with atomic dataset publication,
      pinned/offline Hub replay, and a 10,000-record stress check.
- [x] Repeatable local performance baselines with input fingerprints,
      diverse text, isolated load memory, and opt-in timing/RSS regression
      thresholds.

### v0.3 (2026-10-01)

- [x] Accept the legacy untagged normalizer/decoder JSON forms.
- [x] Review fixes across the pipeline, trainers, Hub client and CLI
      (see the [changelog](../CHANGELOG.md#030---2026-10-01)).
- [x] Fuzz targets for the `Precompiled` parser and for decoding; all
      fuzz targets run in a weekly CI workflow.

### v0.4 (2026-10-01)

- [x] Python bindings via PyO3 and maturin, published to
      [PyPI](https://pypi.org/project/morpheme/) as `morpheme` (abi3
      wheels for Linux, macOS arm64 and Windows, plus an sdist).
- [x] WebAssembly: the library compiles for `wasm32-unknown-unknown`
      (`parallel` feature gates rayon), and `bindings/wasm` exposes
      `fromJson` / `encode` / `tokens` / `count` / `decode` with a
      browser token-counter example.
- [x] Unigram subword-regularization sampling (`alpha`, `nbest_size`)
      with seeded, order-independent draws; `Tokenizer::model_mut` and
      `Bpe::set_dropout` for runtime model settings.
- [x] Benchmark regression tracking on dedicated hardware: the
      `bench-9800x3d` self-hosted runner measures daily and on tags and
      stores every result on the `benchmarks` branch.

## Next

- [ ] Publish the WASM package to npm (today it is built from source).
- [ ] Switch PyPI uploads from the API token to trusted publishing
      (OIDC), once the project registers this repository's workflow.
- [ ] Broaden the Python API beyond load/encode/decode/count (for
      example pairs, padding and truncation settings, and training),
      driven by user requests.

## Out of scope

- Tokenizer-free models, model training, a model registry.
- Prebuilt Intel macOS CLI binaries and Python wheels: Intel Macs use
  `cargo install morpheme-cli` or the Python sdist.
