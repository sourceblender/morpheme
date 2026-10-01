# `unigram`

## Purpose

Unigram (SentencePiece) — a probabilistic subword model that picks the
segmentation with the highest total log-probability.

## Public API

```rust
pub struct Unigram {
    pieces: Vec<Piece>,         // id -> (piece string, log_prob)
    vocab: HashMap<String, u32>,
    unk_id: u32,
    min_score: f64,
}

impl Model for Unigram { ... }
```

## Algorithm

1. For each pre-token, build a DAG over byte positions where each edge
   corresponds to a vocabulary piece that matches at that position.
2. Run a Viterbi best-path forward pass over the DAG, accumulating the
   log-prob for the best segmentation.
3. Backtrace to recover the segments.

## Performance notes

- The DAG construction is the hot loop. It uses prefix-match against
   the vocab. Sorted vocab + binary search keeps it predictable.
- The Viterbi pass is O(n) in pre-token length.

## Test strategy

- Reference SentencePiece vocab — golden output checks.
- Viterbi probability sums (numerical stability tests).
- DAG construction edge cases: empty pre-token, all-OOV pre-token.

## Known limitations

- No `nbest` segmentation output yet — single best only.