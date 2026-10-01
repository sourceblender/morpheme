# `just` runner — https://github.com/casey/just
# Run `just` to list available recipes.

set shell := ["zsh", "-cu"]

default:
    @just --list

# Format, lint, and test. The CI gate.
gate: fixtures
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

# Build everything.
build:
    cargo build --workspace

# Run the CLI.
run *ARGS:
    cargo run -p splinter-cli -- {{ ARGS }}

# Apply formatting.
fmt:
    cargo fmt --all

# Auto-fix where possible, then gate.
fix:
    cargo fmt --all
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged

# Download the pinned Hugging Face tokenizer.json files for golden tests.
fixtures:
    ./scripts/fetch-hf-fixtures.sh

# Regenerate golden outputs from Python `tokenizers` (needs uv).
golden: fixtures
    uv run --with tokenizers==0.23.2 scripts/gen_golden.py

# Check that Python `tokenizers` loads splinter-trained files identically.
interop:
    uv run --with tokenizers==0.23.2 scripts/check_python_interop.py
