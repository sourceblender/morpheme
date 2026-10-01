#!/usr/bin/env bash
# Retrain the example tokenizers in examples/ from examples/corpus.txt.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo build --quiet --release -p morpheme-cli
bin="target/release/morpheme"
"$bin" train --model bpe       --vocab-size 500 --out examples/bpe.json       examples/corpus.txt
"$bin" train --model wordpiece --vocab-size 500 --out examples/wordpiece.json examples/corpus.txt
"$bin" train --model unigram   --vocab-size 500 --out examples/unigram.json   examples/corpus.txt
"$bin" train --model wordlevel --vocab-size 500 --out examples/wordlevel.json examples/corpus.txt
