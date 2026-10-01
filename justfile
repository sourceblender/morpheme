# `just` runner — https://github.com/casey/just
# Run `just` to list available recipes.

set shell := ["zsh", "-cu"]

default:
    @just --list

# Format, lint, and test. The CI gate.
gate:
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