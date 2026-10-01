# 0004 — Bounded JSONL automation and explicit token-budget counts

- **Status:** accepted
- **Date:** 2026-10-01

## Context

The library supports parallel batches, while spawning a CLI process per
record reloads the tokenizer repeatedly. Token budgeting must also avoid
silently counting only the prefix retained by configured truncation.

## Decision

Add JSONL encode/decode commands with explicit object schemas, optional
opaque record ids, bounded record sizes and configurable parallel batch
sizes. Preserve input order, reject invalid records, and flush completed
batches. Encoding retains file settings unless explicitly disabled.

Token counts disable truncation/padding by default and explicitly control
special tokens. Applying file settings is an opt-in operation. Keep chat
template rendering outside this tokenizer CLI.

## Consequences

- Datasets can be processed with one tokenizer load and bounded batches.
- Output from completed batches can precede an error; consumers needing
  atomic dataset publication must write a temporary output and rename it
  only after success.
- Batch-longest padding is local to each batch, so callers choose batch
  size as part of their inference policy.
- Machine-readable inspection has an explicit schema version.
