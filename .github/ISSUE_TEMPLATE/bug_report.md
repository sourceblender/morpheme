---
name: Bug report
about: Something in morpheme is broken
title: "[bug] "
labels: bug
assignees: ""
---

## Summary

One-line description of the bug.

## Reproduction

<!--
A minimal reproduction is the single most helpful thing.
Paste a `cargo` snippet, a tokenizer.json (or the Hugging Face model id
it came from), or a failing test. For interop bugs, include what Python
`tokenizers` returns for the same input.
-->

```rust
// minimal code
```

## Expected

What you expected to happen.

## Actual

What actually happened. Include the full error message.

## Environment

- `morpheme` version / commit: <!-- e.g. v0.1.0, or `git rev-parse HEAD` -->
- Rust version (`rustc --version`):
- OS:
- Backtrace (if applicable): paste `RUST_BACKTRACE=full cargo run ...`

## Possible cause

Optional — your best guess at the root cause.