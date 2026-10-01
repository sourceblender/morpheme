# `make` fallback for users without `just`. Same recipes.

.PHONY: default gate build run fmt fix fixtures golden interop

default:
	@echo "Targets: gate build run fmt fix fixtures golden interop"

gate: fixtures
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace

build:
	cargo build --workspace

run:
	cargo run -p splinter-cli

fmt:
	cargo fmt --all

fix:
	cargo fmt --all
	cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged

fixtures:
	./scripts/fetch-hf-fixtures.sh

golden: fixtures
	uv run --with tokenizers==0.23.2 scripts/gen_golden.py

interop:
	uv run --with tokenizers==0.23.2 scripts/check_python_interop.py
