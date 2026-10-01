# Contributing (detail)

> The high-level version is [CONTRIBUTING.md](../CONTRIBUTING.md). This
> document covers the day-to-day developer workflow inside the repo.

## Workspace layout

```
morpheme/
├── crates/morpheme/        # library — public API lives here
├── apps/morpheme-cli/      # CLI binary
├── bindings/python/        # PyO3 bindings, the `morpheme` package on PyPI
├── bindings/wasm/          # wasm-bindgen bindings + browser example
├── docs/                   # markdown documentation (this folder)
├── examples/               # sample corpus + trained tokenizers
├── scripts/                # fixtures, golden generation, interop, benchmarks, release smoke
├── fuzz/                   # cargo-fuzz targets (separate nightly workspace)
└── .github/                # CI + issue / PR templates
```

## Local setup

Requirements:

- Rust **stable** (the CI also runs `1.85`, the MSRV).
- `rustfmt` and `clippy` (installed by default with rustup).
- `cargo`, `git`.

For the bindings and interop checks:

- Python 3.9+ (CI uses 3.12) for the Python bindings, the interop check
  and the script tests.
- The `wasm32-unknown-unknown` target (`rustup target add
  wasm32-unknown-unknown`), Node.js, and
  [`wasm-pack`](https://drager.github.io/wasm-pack/) or
  `wasm-bindgen-cli` 0.2.129 for the WASM bindings.

Recommended:

- [`just`](https://github.com/casey/just) — `justfile` at the repo root
  wraps the common commands. A `Makefile` is provided as a fallback.

First run:

```sh
cargo build --workspace
cargo test --workspace
cargo run -p morpheme-cli -- --help
```

## Golden tests and Hugging Face fixtures

`crates/morpheme/tests/hf_golden.rs` compares morpheme with Python
`tokenizers` on real tokenizer files. The files are pinned in
`scripts/hf-fixtures.txt` and downloaded (not committed) by:

```sh
./scripts/fetch-hf-fixtures.sh      # or: just fixtures
```

The expected outputs in `crates/morpheme/tests/golden/` are generated
from Python and committed. Regenerate them after changing the input
sentences or the fixture list (needs [`uv`](https://docs.astral.sh/uv/)):

```sh
just golden    # uv run --with tokenizers==0.23.2 scripts/gen_golden.py
```

Never hand-edit golden files to make a test pass: a golden mismatch
means morpheme disagrees with the reference implementation.

`just interop` checks the reverse direction (Python loading
morpheme-trained files).

## CI and docs-only changes

Every CI run starts with a small `detect changes` job that lists the
changed files with `git diff` (complete, unlike `paths-ignore`, which
only looks at the first 300 files). If every changed file is
documentation (`*.md`, `docs/`, `LICENSE`, issue templates,
`CODEOWNERS`, `dependabot.yml`), all other jobs are skipped. Anything
else, including a change that mixes docs and code, runs the full suite,
as do manual runs (`workflow_dispatch`) and any change the job cannot
classify.

The `ci-success` job always runs and passes only if every job passed or
was skipped as docs-only. To make CI required on `main`, require that
single check in the branch ruleset: it reports on docs-only changes
too, so they are never left pending.

## Local gate

`just gate` (or `make gate`) fetches the fixtures and runs the quick
core of CI:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI (`.github/workflows/ci.yml`) runs more than that, with
`RUSTFLAGS=-D warnings`. To reproduce it locally before a larger change,
run the jobs that apply:

```sh
./scripts/fetch-hf-fixtures.sh

# clippy, test, rustdoc (all features and the library without defaults)
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo clippy -p morpheme --all-targets --no-default-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo test -p morpheme --no-default-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
cargo bench --no-run --workspace --locked
cargo +1.85 test --workspace --locked          # MSRV

# WebAssembly: the library and the bindings must build for wasm32
cargo check -p morpheme --no-default-features --locked --target wasm32-unknown-unknown
cargo check -p morpheme --all-features --locked --target wasm32-unknown-unknown
cargo check -p morpheme-wasm --locked --target wasm32-unknown-unknown
(cd bindings/wasm && wasm-pack build --target web --release)
node bindings/wasm/tests/smoke.mjs

# Python bindings (builds the extension into .venv, then runs pytest)
python3 -m venv .venv
.venv/bin/pip install maturin pytest
(cd bindings/python && ../../.venv/bin/maturin develop --locked)
.venv/bin/pytest bindings/python/tests

# Interop with Python `tokenizers` and the script tests
.venv/bin/pip install tokenizers==0.23.2
.venv/bin/python scripts/check_python_interop.py
.venv/bin/python scripts/test_benchmark_baseline.py
.venv/bin/python scripts/check_document_workflow.py

# Hub downloads (network) and dependency policy
cargo test -p morpheme --features hub --test hub --locked -- --ignored
cargo deny check                               # needs cargo-deny
```

CI builds the WASM smoke-test package with `wasm-bindgen-cli` instead of
`wasm-pack`; both produce the same `web` target in `bindings/wasm/pkg`.

## Module-by-module workflow

When you start work on a module:

1. Create or update the relevant deep-dive under `docs/modules/`.
2. If the work involves a design decision, write an ADR under
   `docs/decisions/`.
3. Implement. Write tests first when practical.
4. Update the relevant `docs/roadmap.md` checkbox.

## Testing

- **Unit tests** live next to the code (`#[cfg(test)] mod tests`).
- **Integration tests** live in `crates/morpheme/tests/`.
- **Property tests** use `proptest` for round-trip and offset
  invariants (`crates/morpheme/tests/roundtrip.rs`).
- **Regression tests** for fixed bugs go in
  `crates/morpheme/tests/regressions.rs`, one test per bug, named after it.

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
| `precompiled` | Arbitrary bytes as a SentencePiece `precompiled_charsmap`: `from_bytes` never panics; anything that parses normalizes fixed inputs with valid alignments and round-trips through JSON. Seeded with the real T5 / ALBERT / XLM-R charsmaps. |
| `decode` | Arbitrary id sequences (in-range, out-of-range, special) through `decode` and `DecodeStream` on the real tokenizers: nothing panics, out-of-range ids are dropped, and the streamed text is a prefix of (normally equal to) the full decode. |

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
./scripts/fetch-hf-fixtures.sh     # real tokenizers for `encode` / `decode` and the seeds
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

- `cargo bench -p morpheme` runs the Criterion suite (`encode`, `train`,
  `normalize`); use `--save-baseline` / `--baseline` to compare a change
  against `main`. CI only compiles it.
- `cargo run --release --example bench_encode -- <tokenizer.json> <text>`
  measures encode and decode throughput on any file.
- `python3 scripts/benchmark_baseline.py` records and compares isolated
  local baselines; the `Benchmark tracking` workflow runs it daily on a
  dedicated host ([details](./benchmarks.md#dedicated-host-tracking)).
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
- The maintainer bumps the version in all four crates' `Cargo.toml`
  (library, CLI, both bindings) and in `bindings/python/pyproject.toml`.
- The tag runs three workflows: `Release` (`dist`) builds the CLI
  archives and installers for the GitHub Release and then publishes the
  crates to crates.io; `Python wheels` builds the wheels and sdist and
  uploads them to PyPI; `Benchmark tracking` records a result for the
  tag on the dedicated host.
- After publication, dispatch `release-smoke.yml` with the published
  version. It verifies archive checksums and exercises fresh binaries on
  macOS, Linux and Windows, plus fresh crates.io library and CLI installs.
  Locally: `python3 scripts/release_smoke.py --version 0.4.0 --target
  aarch64-apple-darwin` (requires `gh`), or use `--cli /path/to/morpheme`.
