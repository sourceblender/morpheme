# Benchmarks

Rough comparisons with Python `tokenizers` 0.23.2 (whose core is
Rust), run on the same `tokenizer.json` files and inputs. Treat these
as indicative, not definitive: one machine, no Criterion statistics.

**Machine:** Apple M5, 10 cores, 16 GB · rustc 1.98.1, `--release`
(thin LTO) · Python 3.12. Recorded 2026-09-30.

## Encoding

Input: 100,000 lines (13.8 MB) of English sentences, 5–40 words each,
sampled from the words of `examples/corpus.txt`. Sequential = one
`encode` call per line; batch = one `encode_batch` call (parallel). Both
libraries produced the **same token count** for every model (checked).

| Tokenizer | splinter sequential | Python sequential | splinter batch | Python batch |
| --- | --- | --- | --- | --- |
| bert-base-uncased (WordPiece) | 10.1 MB/s | 5.8 MB/s | 40.4 MB/s | 24.4 MB/s |
| gpt2 (byte-level BPE) | 11.5 MB/s | 7.6 MB/s | 40.0 MB/s | 32.7 MB/s |
| qwen2.5 (regex split + byte-level BPE) | 10.5 MB/s | 6.9 MB/s | 36.8 MB/s | 29.9 MB/s |
| llama (SentencePiece BPE, byte fallback) | 9.6 MB/s | 6.9 MB/s | 45.9 MB/s | 31.8 MB/s |
| t5-small (Unigram, Precompiled) | 10.3 MB/s | 7.2 MB/s | 28.1 MB/s | 23.9 MB/s |

Caveats:

- The Python sequential numbers include per-call Python overhead; the
  batch column is the fairer library-to-library comparison.
- The corpus has a small vocabulary of distinct words, which favors the
  BPE word caches of both libraries. Text with more distinct words will
  be slower for both.

## Training

Input: 200,000 lines (26 MB) of synthetic text over 60,000 distinct
words; target vocabulary 8,000; presets as in the CLI (byte-level BPE,
BERT WordPiece, SentencePiece-style Unigram). Wall-clock time including
file reading.

| Trainer | splinter | Python `tokenizers` |
| --- | --- | --- |
| BPE | 0.54 s | 0.81 s |
| WordPiece | 0.50 s | 0.89 s |
| Unigram | 1.05 s | 1.72 s |

The BPE trainer produced **exactly the same vocabulary and all 7,743
merges** as Python on this corpus.

## Reproducing

```sh
./scripts/fetch-hf-fixtures.sh
cargo run --release --example bench_encode -- \
    crates/splinter/tests/data/hf/gpt2.json my-corpus.txt
```

For the Python side, load the same file with
`tokenizers.Tokenizer.from_file`, call `no_padding()` /
`no_truncation()`, and time `encode` per line and `encode_batch` over
all lines.

## Not yet measured

- Decode throughput.
- Memory usage.
- A Criterion suite with regression tracking in CI.
