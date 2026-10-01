# Benchmarks

Rough comparisons with Python `tokenizers` 0.23.2 (whose core is
Rust), run on the same `tokenizer.json` files and inputs. Treat these
as indicative, not definitive: one machine, single runs. For
repeatable measurements of morpheme itself, use the
[Criterion suite](#criterion-suite).

**Machine:** Apple M5, 10 cores, 16 GB · rustc 1.98.1, `--release`
(thin LTO) · Python 3.12. Recorded 2026-09-30.

## Encoding

Input: 100,000 lines (13.8 MB) of English sentences, 5–40 words each,
sampled from the words of `examples/corpus.txt`. Sequential = one
`encode` call per line; batch = one `encode_batch` call (parallel). Both
libraries produced the **same token count** for every model (checked).

| Tokenizer | morpheme sequential | Python sequential | morpheme batch | Python batch |
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

| Tokenizer | morpheme sequential | Python sequential | morpheme batch | Python batch |
| --- | --- | --- | --- | --- |
| bert-base-uncased | 3.1 M tok/s | 2.9 M tok/s | 17.3 M tok/s | 4.9 M tok/s |
| gpt2 | 11.7 M tok/s | 11.1 M tok/s | 62.9 M tok/s | 3.1 M tok/s |
| llama | 8.3 M tok/s | 5.0 M tok/s | 28.7 M tok/s | 3.1 M tok/s |
| t5-small | 11.6 M tok/s | 11.9 M tok/s | 73.8 M tok/s | 3.0 M tok/s |

Python's `decode_batch` is slower than its own sequential loop here,
probably because of the cost of passing 100,000 id lists across the
Python boundary; the sequential columns are the closer comparison.
morpheme's `decode_batch` decodes in parallel.

## Memory

Peak resident set size (`/usr/bin/time -l`, macOS) of a process that
loads the tokenizer, encodes the 13.8 MB corpus sequentially and as a
batch, keeps all 100,000 encodings, and decodes them — i.e. the
`bench_encode` example vs. the equivalent Python script.

| Tokenizer | morpheme | Python `tokenizers` |
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

| Trainer | morpheme | Python `tokenizers` |
| --- | --- | --- |
| BPE | 0.54 s | 0.81 s |
| WordPiece | 0.50 s | 0.89 s |
| Unigram | 1.05 s | 1.72 s |

The BPE trainer produced **exactly the same vocabulary and all 7,743
merges** as Python on this corpus.

## Criterion suite

Statistically sound, repeatable measurements live in
`crates/morpheme/benches/` ([Criterion](https://github.com/bheisler/criterion.rs)):

| Bench | What it measures |
| --- | --- |
| `encode` | `encode` (per line) and `encode_batch` throughput, plus `decode` / `decode_batch`, on bert-base-uncased, gpt2, llama and t5-small (2,000 generated sentences) |
| `train` | BPE (byte-level), WordPiece (BERT) and Unigram (SentencePiece-style) training on a deterministic ~1.5 MB synthetic corpus, 4,000-token vocabulary |
| `normalize` | The alignment-tracking `NormalizedString` path: `BertNormalizer` and NFKC on mixed-script text |

```sh
./scripts/fetch-hf-fixtures.sh          # the encode bench needs the fixtures
cargo bench -p morpheme                 # everything (~3 minutes)
cargo bench -p morpheme --bench train   # one bench
cargo bench -p morpheme -- --save-baseline main   # then compare a branch:
cargo bench -p morpheme -- --baseline main
```

Inputs are generated deterministically, so runs are comparable across
machines and commits. HTML reports land in `target/criterion/`. CI
compiles the suite on every PR (`benches compile` job) but does not run
it: shared CI runners are too noisy for regression thresholds.

## Reproducing the comparisons

```sh
./scripts/fetch-hf-fixtures.sh
cargo run --release --example bench_encode -- \
    crates/morpheme/tests/data/hf/gpt2.json my-corpus.txt
```

`bench_encode` reports sequential and batch encode and decode
throughput on any tokenizer and text file. For the Python side, load the
same file with `tokenizers.Tokenizer.from_file`, call `no_padding()` /
`no_truncation()`, and time `encode` per line, `encode_batch`, `decode`
per sequence and `decode_batch` over the same lines. Wrap either in
`/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) for peak
memory.

## Local performance baselines

The baseline runner records **39 isolated workloads**, three samples each by
default: load-only processes for four tokenizer families; sequential and bounded
batch encode/decode over both repeated and diverse text; and BPE, WordPiece,
and Unigram training. Each sample starts a fresh process, so sequential encoding
does not pre-warm the cache for batch encoding. Within a sample, caches warm
naturally as input is processed.

```sh
./scripts/fetch-hf-fixtures.sh
python3 scripts/benchmark_baseline.py --output baseline-main.json
# Run again after a change on the same machine:
python3 scripts/benchmark_baseline.py --output baseline-branch.json \
    --compare baseline-main.json --threshold-percent 20
```

Comparison uses median duration and peak RSS and exits with status 1 if either
increases by more than the selected threshold. It rejects comparisons with
different machine/OS/compiler, lockfile or build configuration fingerprints,
fixture/corpus hashes, workload keys, or settings. Revision and dirty state are
recorded but allowed to differ. Investigate a flagged change with more repetitions
(`--repeats 7`) and Criterion before calling it a regression; laptop load and
thermal state still affect measurements. This is an opt-in local check. The manual
`Performance baseline` workflow uploads a JSON artifact from its shared runner
and does not enforce a threshold.

The default deterministic inputs have 2,000 lines of 24 words plus accented,
CJK, and emoji text per line. The repeated corpus samples a small English
vocabulary; the diverse corpus generates 48,000 pseudo-random 12-letter words
with seed 42, exceeding the BPE cache capacity. Four Rayon workers and 256-record
batches are explicit defaults. `--lines`, `--batch-size`, and `--threads` tune these
settings and become part of the comparison fingerprint.

All raw samples, counts, input hashes, compiler details, and machine information
are retained in JSON. Encode time excludes load and input reading; decode time
also excludes preparing the IDs. Output construction, disposal, and per-operation
batch allocation are timed. Training time includes trainer construction and
training, excluding corpus reading. Load time includes reading/parsing the model.
Peak RSS is for the whole isolated process (`time -l` on macOS, `time -v` on Linux):
load-only processes do not read a corpus; encode/decode processes include the
input string, line references, model, caches, and at most one encoding batch.
Decode RSS also includes its untimed encoding preparation. Peak RSS measures
process memory, not just model heap allocations. Other systems record timings
with RSS unavailable.

[Apple M5 baseline](baselines/apple-m5.json), recorded 2026-10-01 with rustc
1.98.1, three samples and four workers (medians; different workloads from the
older Python comparison above):

| Tokenizer | Load-only peak MiB | Repeated seq MB/s | Repeated batch MB/s | Diverse seq MB/s | Diverse batch MB/s |
| --- | --- | --- | --- | --- | --- |
| bert-base-uncased | 13.8 | 9.4 | 18.9 | 8.4 | 19.1 |
| gpt2 | 25.7 | 10.3 | 19.2 | 8.8 | 20.1 |
| llama | 27.8 | 9.4 | 21.8 | 12.1 | 28.8 |
| t5-small | 23.3 | 9.6 | 18.1 | 9.3 | 18.4 |

Dedicated-hardware trend tracking remains future work. The baseline runner and
Criterion provide the repeatable measurements needed to start that tracking.
