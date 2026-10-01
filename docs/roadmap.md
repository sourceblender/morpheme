# Roadmap

> Living document. Phases, not dates.

## Done (v0.1, unreleased)

- [x] Hugging Face-compatible pipeline: added tokens, normalizers,
      pre-tokenizers, models, post-processors, decoders, with offset
      tracking into the original input.
- [x] BPE, WordPiece, WordLevel, Unigram models; matching trainers
      (BPE output identical to Hugging Face).
- [x] `tokenizer.json` read/write, byte-compatible with `tokenizers`
      0.23, legacy forms accepted.
- [x] Truncation (with overflow), padding, pairs, pre-tokenized input,
      parallel batch encode/decode.
- [x] Streaming decode (`DecodeStream`) for token-by-token generation.
- [x] CLI: `train` (presets), `encode`, `decode`, `inspect`.
- [x] Hub download (`from_pretrained`, feature `hub`) with the shared
      Hugging Face cache; the CLI accepts model ids.
- [x] Golden tests against 11 real tokenizers; reverse interop check
      with Python; property and regression tests; MSRV 1.85 in CI.

## Next: v0.1.0 release

- [ ] Review the public API surface for 1.0-readiness (naming, what is
      `pub` vs `pub(crate)`), and add `#[non_exhaustive]` where enums
      may grow.
- [ ] Publish `splinter` and `splinter-cli` to crates.io; pre-built CLI
      binaries (release workflow exists).
- [ ] rustdoc examples on every public type; docs.rs build.

## Then

- [ ] Unigram subword-regularization sampling (`alpha`, `nbest_size`).
- [ ] Progress reporting for trainers (`show_progress`).
- [ ] Criterion benchmark suite with regression tracking in CI; decode
      and memory measurements.
- [ ] Accept the legacy untagged normalizer/decoder JSON forms (no file
      in the fixture set needs them today).

## Ecosystem

- [ ] Python bindings via PyO3.
- [ ] WASM target via `wasm-bindgen`.

## Out of scope

- Tokenizer-free models, model training, a model registry.
