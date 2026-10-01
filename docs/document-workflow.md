# Prepare documents within a token budget

The executable `budget_documents` example is a small Rust consumer for
dataset preparation. It reads `{ "id": ..., "text": ... }` JSONL records,
keeps original text/order, and writes ids, byte offsets, exact token
counts, and tokenizer provenance. Each record must fit the supplied
budget. Padding/truncation are disabled, so an oversized document cannot
silently become a shorter training/inference input. Post-processor
special tokens are included in the budget.

```sh
cargo run --release -p morpheme --example budget_documents -- \
  examples/bpe.json 512 documents.jsonl prepared.jsonl
```

A local tokenizer is loaded from the same bytes used for its SHA-256
fingerprint. Hub sources require a full pinned commit:

```sh
cargo run --release -p morpheme --features hub --example budget_documents -- \
  google-bert/bert-base-uncased 512 documents.jsonl prepared.jsonl \
  86b5e0934494bd15c9632b12f734a8a67f723594
# Repeat with the verified cached revision, without network access:
HF_HUB_OFFLINE=1 cargo run --release -p morpheme --features hub \
  --example budget_documents -- google-bert/bert-base-uncased 512 \
  documents.jsonl prepared.jsonl 86b5e0934494bd15c9632b12f734a8a67f723594
```

One record is retained at a time, limited to 8 MiB including its ending.
Malformed records, encoding errors and budget failures abort the run and
preserve any previous output. The complete new dataset is published by
atomic replacement only after every record succeeds. Empty datasets are
valid and produce an empty file. The output cannot overwrite the input
dataset or local tokenizer. Choose a new output path for each preparation
run when retaining historical datasets matters.

This prepares tokenizer text, not chat requests. Render the model's chat
template before preparation when the budget includes chat formatting.
Oversized documents require an explicit application policy (split,
summarize, or reject); this consumer rejects them.

## Use the released library independently

Copy `crates/morpheme/examples/budget_documents.rs` into a fresh Rust
application's `src/main.rs` with this manifest:

```toml
[package]
name = "document-consumer"
version = "0.1.0"
edition = "2024"

[features]
default = ["hub"]
hub = ["morpheme/hub"]

[dependencies]
morpheme = "=0.2.0"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
tempfile = "3"
```

The consumer uses the morpheme 0.2.0 API, including verification of Hub
downloads and cached files. Pin the library version and Hub revision for
repeatable preparation.

## Verify the workflow

```sh
python3 scripts/check_document_workflow.py
python3 scripts/check_document_workflow.py --hub
```

The core run trains a byte-level BPE tokenizer, prepares Unicode/empty
documents, confirms that configured truncation cannot hide oversized
inputs, checks fingerprints and atomic failure handling, and processes
10,000 distinct records. `--hub` additionally downloads a pinned public
BERT tokenizer into a fresh cache and confirms identical offline output.
The core run is part of CI; live Hub verification remains an explicit
network check.
