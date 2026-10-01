# Roadmap

> Living document. Phases, not dates.

## Done (v0.1.0)

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
- [x] Release pipeline (`dist`): CLI binaries for six targets, checksums,
      installers; crates.io publishing.
- [x] Released v0.1.0 (2026-10-01): [`morpheme`](https://crates.io/crates/morpheme)
      and [`morpheme-cli`](https://crates.io/crates/morpheme-cli) on crates.io,
      binaries on the [GitHub Release](https://github.com/sourceblender/morpheme/releases/tag/v0.1.0).

## Next

- [x] CLI token budgeting, bounded JSONL batches, and JSON inspection.
- [x] Exercise all four fuzz targets for five minutes in CI (2026-10-01).
- [x] Executable document-budget consumer with atomic dataset publication,
      pinned/offline Hub replay, and a 10,000-record stress check.

- [x] Unigram subword-regularization sampling (`alpha`, `nbest_size`) with
      seeded, order-independent draws (2026-10-01).
- [x] Repeatable local performance baselines with input fingerprints, diverse
      text, isolated load memory, and opt-in timing/RSS regression thresholds.
- [ ] Benchmark regression tracking on dedicated hardware.
- [x] Accept the legacy untagged normalizer/decoder JSON forms (no file
      in the fixture set needs them today).
- [x] Hash-check Hub downloads and cached files against their ETag;
      repair corrupt entries online and reject them offline.

## Ecosystem

- [x] Python bindings via PyO3 (partial): `bindings/python` builds a `morpheme`
      module with maturin (load from file / string / Hub, encode, decode, batch
      variants, count, vocab lookups, save). Platform wheels in CI and a
      published package are still to do.
- [ ] WASM target via `wasm-bindgen`. Partial: the library compiles for
      `wasm32-unknown-unknown` (`parallel` feature gates rayon with a
      sequential fallback; `save` is compiled out there) and CI checks it.
      The `wasm-bindgen` binding crate and browser example are still to do.

## Out of scope

- Tokenizer-free models, model training, a model registry.
