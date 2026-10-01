# `wordpiece`

## Purpose

WordPiece — the greedy longest-match subword algorithm used by BERT.

## Public API

```rust
pub struct WordPiece {
    vocab: Vocab,
    unk_token: String,
    max_input_chars_per_word: usize,
}

impl Model for WordPiece { ... }
```

## Algorithm

1. For each pre-token, find the longest suffix-prefix match starting
   from the left.
2. Greedily extend to the right using the vocabulary; if no match, mark
   the whole token as unk.
3. Continuation subwords get a `##` prefix (BERT convention).

## Performance notes

- Trie lookup. The vocab is stored in a flat `[u8]` arena with byte
  indices for cache friendliness.
- For very large vocabs we use a `HashMap<String, u32>` indexed by the
  longest-match candidate's hash, with a final equality check.

## Test strategy

- Reference BERT vocab — golden output checks.
- Boundary cases: tokens longer than `max_input_chars_per_word`, OOV
  characters, mixed scripts.

## Known limitations

- No support for the SentencePiece-style `▁` prefix (handled by the
  Metaspace pre-tokenizer, which is post-2).