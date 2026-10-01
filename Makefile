# `make` fallback for users without `just`. Same recipes.

.PHONY: default gate build run fmt fix

default:
	@echo "Targets: gate build run fmt fix"

gate:
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