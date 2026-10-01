# 0001 — Adopt the Hugging Face `tokenizers` design and file format

- **Status:** accepted
- **Date:** 2026-09-30

## Context

The first implementation used its own conventions: a custom JSON
schema, an end-of-word suffix appended to every character, and a
normalizer → pre-tokenizer → model pipeline with no offset tracking. A
review found it could not encode any real-world tokenizer correctly,
and that every model and trainer had correctness bugs. The tokenizers
people actually need to run (BERT, GPT-2, Llama, T5, …) are distributed
as Hugging Face `tokenizer.json` files, and "correct" means "matches
the reference implementation".

## Decision

Rebuild splinter on the Hugging Face `tokenizers` design: the same
pipeline stages and component semantics, offsets tracked through
`NormalizedString` alignments, and `tokenizer.json` (version `1.0`) as
the only on-disk format, byte-compatible on write. Compatibility is
enforced by golden tests against real, pinned tokenizers and by a
reverse check with the Python library.

Within that design:

- **Pure Rust.** Same Unicode crates as Hugging Face (for identical
  data tables), `fancy-regex` instead of Oniguruma, and an in-house
  suffix array instead of the C++ `esaxx` for Unigram training.
- **Errors, not panics,** on malformed input.
- **Deterministic trainers,** even where Hugging Face's ordering is
  not (outputs are otherwise identical for BPE).
- **Built-in components only** in a `Tokenizer` (wrapper enums), so
  every tokenizer is `Clone`, non-generic and fully serializable.

## Consequences

- Real tokenizers load and produce reference-identical output; files
  trained with splinter work in the Python ecosystem.
- "Correct" is testable: any behavioral difference shows up as a golden
  mismatch, and intentional differences are listed in
  [`docs/interop.md`](../interop.md).
- The public API follows Hugging Face naming, which breaks the 0.0.0
  API (no published users).
- Hugging Face semantics are inherited, quirks included (e.g. BPE
  without `unk_token` drops unknown characters); deviations must be
  deliberate and documented.
- User-defined components cannot be stored in a `Tokenizer` or
  serialized.
- MSRV rises to 1.85, required by current dependencies.

## Alternatives considered

- **Fix the original design in place.** Rejected: the per-character
  `</w>` convention, the custom schema and the missing offset model were
  the root causes, so the fixes would have amounted to a rewrite anyway,
  without the reference to test against.
- **Generic `TokenizerImpl<M, N, PT, PP, D>` like Hugging Face's Rust
  crate.** Rejected for now: the wrappers cover every built-in, keep the
  type simple, and match what the Python library exposes.
- **Bind to the `tokenizers` crate.** Rejected: splinter exists to be an
  independent, pure-Rust implementation.
