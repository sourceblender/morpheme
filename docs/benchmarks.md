# Benchmarks

> Stub. Filled out as benchmarks land in `crates/splinter/benches/`.

## Methodology

- **Hardware.** Captured in `benches/hardware.json` at run time. Include
  CPU model, microarch, cores, and L2/L3 sizes.
- **Inputs.** Frozen corpora in `crates/splinter/benches/data/`:
  - `tiny.txt` — 1 KiB, smoke.
  - `wikitext-103-sample.txt` — 1 MiB, representative prose.
  - `code-corpus.txt` — 5 MiB mixed Rust + Python + Markdown.
- **Comparators.** `huggingface/tokenizers` (Rust binding) on the same
  JSON model file.
- **Statistical method.** Criterion defaults — 100 samples, 10 s warmup,
  outlier detection on.

## What we report

- `encode(text)` throughput (MB/s) and ns/word.
- `decode(ids)` throughput.
- `encode_batch(texts)` parallel speedup vs. single-thread.
- Memory: `heaptrack` or `dhat-rs` peak working set on a fixed corpus.

## Reporting regressions

A regression > **10%** on the encode hot path fails CI. The threshold
lives in `tools/bench-compare/config.toml`.

## How to run locally

```sh
cargo bench -p splinter
```

Criterion's HTML report lands in
`target/criterion/report/index.html`.