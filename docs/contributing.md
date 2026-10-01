# Contributing (detail)

> The high-level version is [CONTRIBUTING.md](../CONTRIBUTING.md). This
> document covers the day-to-day developer workflow inside the repo.

## Workspace layout

```
splinter/
├── crates/splinter/        # library — public API lives here
├── apps/splinter-cli/      # CLI binary
├── docs/                   # markdown documentation (this folder)
├── examples/               # sample corpus + trained tokenizers
├── scripts/                # HF fixtures, golden generation, interop check
├── fuzz/                   # cargo-fuzz targets (separate nightly workspace)
└── .github/                # CI + issue / PR templates
```

## Local setup

Requirements:

- Rust **stable** (the CI also runs `1.85`, the MSRV).
- `rustfmt` and `clippy` (installed by default with rustup).
- `cargo`, `git`.

Recommended:

- [`just`](https://github.com/casey/just) — `justfile` at the repo root
  wraps the common commands. A `Makefile` is provided as a fallback.

First run:

```sh
cargo build --workspace
cargo test --workspace
cargo run -p splinter-cli -- --help
```

## Golden tests and Hugging Face fixtures

`crates/splinter/tests/hf_golden.rs` compares splinter with Python
`tokenizers` on real tokenizer files. The files are pinned in
`scripts/hf-fixtures.txt` and downloaded (not committed) by:

```sh
./scripts/fetch-hf-fixtures.sh      # or: just fixtures
```

The expected outputs in `crates/splinter/tests/golden/` are generated
from Python and committed. Regenerate them after changing the input
sentences or the fixture list (needs [`uv`](https://docs.astral.sh/uv/)):

```sh
just golden    # uv run --with tokenizers==0.23.2 scripts/gen_golden.py
```

Never hand-edit golden files to make a test pass: a golden mismatch
means splinter disagrees with the reference implementation.

`just interop` checks the reverse direction (Python loading
splinter-trained files).

## Local gate

Before opening a PR, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

These are the same checks the CI runs. There is a `just gate` (and
`make gate`) target that runs all three.

## Module-by-module workflow

When you start work on a module:

1. Create or update the relevant deep-dive under `docs/modules/`.
2. If the work involves a design decision, write an ADR under
   `docs/decisions/`.
3. Implement. Write tests first when practical.
4. Update the relevant `docs/roadmap.md` checkbox.

## Testing

- **Unit tests** live next to the code (`#[cfg(test)] mod tests`).
- **Integration tests** live in `crates/splinter/tests/`.
- **Property tests** use `proptest` for round-trip and offset
  invariants (`crates/splinter/tests/roundtrip.rs`).
- **Regression tests** for fixed bugs go in
  `crates/splinter/tests/regressions.rs`, one test per bug, named after it.

### Coverage

CI's `coverage` job runs the whole suite under
[`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov), prints a
summary on the run page and uploads `lcov.info` as an artifact (line
coverage was 91.5% when it was added). Locally:

```sh
cargo install cargo-llvm-cov   # once; needs `rustup component add llvm-tools-preview`
cargo llvm-cov --workspace --all-features --summary-only
cargo llvm-cov --workspace --all-features --html   # target/llvm-cov/html
```

New code should come with tests; look at the uncovered lines of the
files you touched rather than chasing the total.

## Fuzzing

`fuzz/` holds [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
targets. It is a separate workspace (excluded from the main one) because
it needs nightly Rust:

| Target | What it checks |
| --- | --- |
| `load_json` | Arbitrary bytes as `tokenizer.json`: loading never panics; anything that loads can encode/decode and survives save → load unchanged. |
| `encode` | Arbitrary text through real tokenizers (BERT, GPT-2, Llama, T5, Qwen2.5): encoding never fails, offsets are valid slices, GPT-2 round-trips losslessly. |
| `normalized_string` | Random sequences of normalization ops: alignments always map back into the original text. |
| `components_json` | Arbitrary JSON for each component type: loading never panics; loaded components run and re-serialize. |

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
./scripts/fetch-hf-fixtures.sh     # real tokenizers for `encode` and the seeds
./fuzz/run-all.sh 300              # all targets in parallel, 5 minutes each
cargo +nightly fuzz run encode     # or one target, until Ctrl-C
```

Seed corpora come from `fuzz/make_corpus.py` (real component configs
and the golden inputs). Crashes land in `fuzz/artifacts/<target>/`;
`cargo +nightly fuzz fmt <target> <file>` prints the input. Fix the root
cause and add a regression test that reproduces it. A weekly CI workflow
(`.github/workflows/fuzz.yml`, also runnable by hand) fuzzes every target
for 5 minutes and uploads any crash as an artifact.

## Benchmarking

- `cargo bench -p splinter` runs the Criterion suite (`encode`, `train`,
  `normalize`); use `--save-baseline` / `--baseline` to compare a change
  against `main`. CI only compiles it.
- `cargo run --release --example bench_encode -- <tokenizer.json> <text>`
  measures encode and decode throughput on any file.
- Methodology and results: [`docs/benchmarks.md`](./benchmarks.md).

## Style

- Follow `rustfmt` defaults — the CI enforces it.
- No warnings in CI (`-D warnings`).
- Public items get `///` doc comments.
- Crate-level docs explain the *what*; module-level docs explain the
  *why*.

## Pull requests

- Branch from `main`.
- One logical change per PR.
- Reference any related issue (`Closes #123`).
- Fill out the PR template.
- Expect review from at least one maintainer.

## Releases

- Triggered by a maintainer tagging `vX.Y.Z`.
- `CHANGELOG.md` is updated at release time.
- `Cargo.toml` versions are bumped by the maintainer.
- crates.io publish is gated on a passing release workflow.