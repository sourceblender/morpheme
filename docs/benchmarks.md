# Benchmarks

Rough comparisons with Python `tokenizers` 0.23.2 (whose core is
Rust), run on the same `tokenizer.json` files and inputs. Treat these
as indicative, not definitive: one machine, single runs. For
repeatable measurements of splinter itself, use the
[Criterion suite](#criterion-suite).

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

## Decoding

Same corpus: the ids produced by `encode_batch` above, decoded with
`skip_special_tokens` on. Sequential = one `decode` call per sequence;
batch = one `decode_batch` call.

| Tokenizer | splinter sequential | Python sequential | splinter batch | Python batch |
| --- | --- | --- | --- | --- |
| bert-base-uncased | 3.1 M tok/s | 2.9 M tok/s | 17.3 M tok/s | 4.9 M tok/s |
| gpt2 | 11.7 M tok/s | 11.1 M tok/s | 62.9 M tok/s | 3.1 M tok/s |
| llama | 8.3 M tok/s | 5.0 M tok/s | 28.7 M tok/s | 3.1 M tok/s |
| t5-small | 11.6 M tok/s | 11.9 M tok/s | 73.8 M tok/s | 3.0 M tok/s |

Python's `decode_batch` is slower than its own sequential loop here,
probably because of the cost of passing 100,000 id lists across the
Python boundary; the sequential columns are the closer comparison.
splinter's `decode_batch` decodes in parallel.

## Memory

Peak resident set size (`/usr/bin/time -l`, macOS) of a process that
loads the tokenizer, encodes the 13.8 MB corpus sequentially and as a
batch, keeps all 100,000 encodings, and decodes them — i.e. the
`bench_encode` example vs. the equivalent Python script.

| Tokenizer | splinter | Python `tokenizers` |
| --- | --- | --- |
| bert-base-uncased | 611 MiB | 917 MiB |
| gpt2 | 389 MiB | 666 MiB |
| llama | 439 MiB | 1,191 MiB |
| t5-small | 654 MiB | 925 MiB |

Most of this is the retained encodings (ids, tokens, offsets, masks for
~3 M tokens), not the tokenizer itself; the Python figure also includes
the interpreter and the Python-side `Encoding` objects.

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

## Criterion suite

Statistically sound, repeatable measurements live in
`crates/splinter/benches/` ([Criterion](https://github.com/bheisler/criterion.rs)):

| Bench | What it measures |
| --- | --- |
| `encode` | `encode` (per line) and `encode_batch` throughput, plus `decode` / `decode_batch`, on bert-base-uncased, gpt2, llama and t5-small (2,000 generated sentences) |
| `train` | BPE (byte-level), WordPiece (BERT) and Unigram (SentencePiece-style) training on a deterministic ~1.5 MB synthetic corpus, 4,000-token vocabulary |
| `normalize` | The alignment-tracking `NormalizedString` path: `BertNormalizer` and NFKC on mixed-script text |

```sh
./scripts/fetch-hf-fixtures.sh          # the encode bench needs the fixtures
cargo bench -p splinter                 # everything (~3 minutes)
cargo bench -p splinter --bench train   # one bench
cargo bench -p splinter -- --save-baseline main   # then compare a branch:
cargo bench -p splinter -- --baseline main
```

Inputs are generated deterministically, so runs are comparable across
machines and commits. HTML reports land in `target/criterion/`. CI
compiles the suite on every PR (`benches compile` job) but does not run
it: shared CI runners are too noisy for regression thresholds.

## Reproducing the comparisons

```sh
./scripts/fetch-hf-fixtures.sh
cargo run --release --example bench_encode -- \
    crates/splinter/tests/data/hf/gpt2.json my-corpus.txt
```

`bench_encode` reports sequential and batch encode and decode
throughput on any tokenizer and text file. For the Python side, load the
same file with `tokenizers.Tokenizer.from_file`, call `no_padding()` /
`no_truncation()`, and time `encode` per line, `encode_batch`, `decode`
per sequence and `decode_batch` over the same lines. Wrap either in
`/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) for peak
memory.

## Not yet measured

- Automated regression tracking (needs dedicated benchmark hardware).
- Memory of the tokenizer alone, separate from retained encodings.
