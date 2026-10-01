# `bpe`

## Purpose

Byte-Pair Encoding — the subword algorithm used by GPT-2 / RoBERTa.

## Public API

```rust
pub struct Bpe {
    pub vocab: Vocab,
    merges: Vec<(String, String)>,
    merge_ranks: HashMap<(u32, u32), usize>,
    end_of_word_suffix: String,
}

impl Model for Bpe { ... }
```

Public helpers on `Bpe`:
- `Bpe::new(vocab, merges, end_of_word_suffix) -> Self`
- `Bpe::num_merges(&self) -> usize`
- `Bpe::end_of_word_suffix(&self) -> &str`
- `Bpe::merges_iter(&self) -> impl Iterator<Item = &(String, String)>`

## Serialization

`Bpe` is persisted via `Tokenizer::to_json` / `from_json` (the splinter
v1.0 schema). The merge list is serialized in insertion order, which
matches the rank order used internally. See `docs/interop.md` for the
schema spec.

## Algorithm

1. Split a pre-token into individual characters.
2. Repeatedly merge the highest-priority pair (lowest index in `merges`)
   until no priority pair remains.
3. Look up each resulting symbol in `vocab`. Unknowns become the unk id
   (or, with `byte_fallback`, fall back to UTF-8 bytes).

## Performance notes

- The merge loop is dominated by priority lookups. We keep the merge
  table as a dense `Vec<u32>` indexed by `(a, b)` pair id.
- Encode cache (LRU per pre-token string) catches the common case where
  the same word appears many times.
- With `dropout`, use a `SmallRng` per call.

## Test strategy

- Reference GPT-2 / RoBERTa vocab merges — golden output checks.
- Idempotence: `bpe(bpe_inv(ids)) == ids` on valid inputs.
- Dropout distribution: with dropout, frequency of rarer merges follows
  expected probability within tolerance.
- Cache correctness across threads (no shared mutable state).

## Known limitations

- Streaming BPE not yet supported — operates on a full pre-token at a
  time.