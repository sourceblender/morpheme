# `post_processor`

## Purpose

Add special tokens and arrange ids / type-ids for downstream model
inputs (BERT, RoBERTa, T5 / sentence-pair NLI, etc.).

## Public API

```rust
pub trait PostProcessor: Send + Sync {
    fn process<'a>(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
    ) -> Result<Encoding>;
}
```

Concrete types in v0.1:

- `TemplateProcessing` — SentencePiece-style templates with `$A`, `$B`,
  `[CLS]`, `[SEP]` substitutions.
- `RobertaProcessing` — RoBERTa's `(seqA..sep, seqB..sep)` shape with
  type ids all zero.
- `BertProcessing` — BERT's `(cls, seqA..sep, seqB..sep)` shape with
  segment type ids.

## Algorithm

Pure structural transformation on the `Encoding` struct. No string work.

## Performance notes

This is allocation-light. We touch the encoding once and produce a new
one. We avoid double-allocations by computing the final capacity up
front.

## Test strategy

- Equivalence with HF `tokenizers` output for the canonical templates.
- Single-sequence vs. pair sequences.

## Known limitations

- No streaming post-processing.