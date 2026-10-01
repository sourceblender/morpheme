# `models::Unigram` — SentencePiece unigram LM

Used by T5, ALBERT, XLM-RoBERTa, mBART and other SentencePiece models.
Each piece has a score (a log probability); a pre-token is split into the
sequence of pieces with the highest total score.

## Algorithm

- **Lattice.** A prefix trie over the vocabulary, built once at
  construction, finds every piece that starts at each char position.
- **Viterbi.** The best path through the lattice is found with
  SentencePiece's optimized single-pass Viterbi (an explicit-lattice
  version backs the trainer and is kept equivalent).
- **Unknown chars.** A char that no piece covers gets an unknown node
  scored `min_score - 10.0` (the lowest score in the vocabulary minus
  SentencePiece's unknown penalty). Consecutive unknown chars are fused
  into one piece. Like HF, the resulting token keeps the original text as
  its value and gets the id `unk_id`.
- **Byte fallback.** With `byte_fallback`, unknown text is emitted as
  `<0xNN>` byte pieces when those exist in the vocabulary.
- **Cache.** Encoded strings shorter than 256 bytes are cached (up to
  10 000 entries); cloning gives an empty cache.

The model does not add the `▁` word marker itself — that is the job of
the `Metaspace` pre-tokenizer (or a `Prepend`/`Replace` normalizer), as in
HF.

```rust
use morpheme::models::Unigram;
use morpheme::Model;

fn main() -> morpheme::Result<()> {
    let model = Unigram::new(
        vec![
            ("<unk>".into(), 0.0),
            ("▁".into(), -2.0),
            ("▁hello".into(), -3.0),
            ("▁he".into(), -4.0),
            ("llo".into(), -4.0),
            ("w".into(), -5.0),
            ("o".into(), -5.0),
        ],
        Some(0),
        false,
    )?;
    assert_eq!(model.encode("▁hello")?, ["▁hello"]);
    // Unknown chars: like HF, the piece keeps its text and gets the unk id;
    // consecutive unknown chars are fused.
    let toks = model.tokenize("▁hello▁xyz")?;
    let last = toks.last().unwrap();
    assert_eq!((last.id, last.value.as_str()), (0, "xyz"));
    assert!(Unigram::new(vec![("a".into(), 0.0)], Some(3), false).is_err());
    Ok(())
}
```

## API

- `Unigram::new(vocab: Vec<(String, f64)>, unk_id: Option<usize>,
  byte_fallback: bool) -> Result<Unigram>` — the id of a piece is its
  index. Fails on an empty vocabulary, an out-of-range `unk_id`, or a
  non-finite (`NaN` or infinite) score, naming the offending piece.
- `Unigram::default()` — `[("<unk>", 0.0)]` with `unk_id` 0 (HF default).
- `encode(&str) -> Result<Vec<String>>`, `unk_id()`, `byte_fallback()`,
  `pieces()` (the `(piece, score)` pairs), `clear_cache()`, plus the
  `Model` trait.
- The Viterbi lattice is internal; use `tokenize` / `encode`.

## Subword-regularization sampling

`encode` / `tokenize` can draw a random segmentation instead of the
Viterbi best one (SentencePiece `SampleEncode`, HF `Unigram(alpha,
nbest_size)`):

```rust
use morpheme::models::{ModelWrapper, Unigram};
use morpheme::Tokenizer;

fn main() -> morpheme::Result<()> {
    let pieces = vec![
        ("<unk>".to_string(), 0.0),
        ("▁hello".to_string(), -3.0),
        ("▁he".to_string(), -4.0),
        ("llo".to_string(), -4.0),
    ];
    // Builder style on a model...
    let model = Unigram::new(pieces, Some(0), false)?
        .with_sampling(0.1, -1)? // alpha, nbest_size
        .with_seed(42); // optional: reproducible draws
    assert_eq!(model.sampling(), Some((0.1, -1)));

    // ...or in place on a loaded tokenizer, through `model_mut`.
    let mut tokenizer = Tokenizer::new(model);
    if let ModelWrapper::Unigram(unigram) = tokenizer.model_mut() {
        unigram.set_sampling(0.0, -1)?; // alpha 0: back to plain Viterbi
        unigram.set_seed(None);
        assert_eq!(unigram.sampling(), None);
    }
    Ok(())
}
```

- `set_sampling(alpha, nbest_size) -> Result<()>` / `with_sampling`:
  - `nbest_size > 1`: the `nbest_size` best lattice paths are found and
    one is picked with probability proportional to `exp(alpha * score)`;
  - `nbest_size < 0`: one path is sampled from the whole lattice with
    the piece scores scaled by `alpha` (forward filtering, backward
    sampling);
  - `nbest_size` of `0` or `1`, or `alpha == 0`: plain Viterbi,
    byte-for-byte identical to not sampling (`sampling()` reports
    `None`).
  - `alpha` must be finite and `>= 0`; anything else is a config error.
    Any finite `alpha` is numerically safe: n-best weights are computed
    as `exp(alpha * (score - best_score))` with the exponent clamped, and
    if whole-lattice sampling underflows every path to `-inf` the draw
    falls back to the Viterbi path (the distribution it has collapsed
    onto).
  - Sampling accepts exactly the inputs Viterbi accepts and raises the
    same missing-`unk_id` error otherwise; it never fails where plain
    encoding succeeds.
- `set_seed(Option<u64>)` / `with_seed(u64)`: with a seed the draw for
  a given sentence is a pure function of `(seed, sentence)`, so results
  are reproducible and independent of thread or batch order
  (`encode_batch` under rayon gives the same output as sequential
  `encode`). Without a seed each call draws from a thread-local
  entropy-seeded generator. The PRNG is an in-crate splitmix64-seeded
  xorshift64*; no dependency is added.
- Sampled encodings bypass the sentence cache (the cache only ever holds
  Viterbi results) and build tokens through the same path as Viterbi,
  so offsets stay contiguous and `byte_fallback` / unknown fusing behave
  the same.
- These settings are runtime-only: they are not read from or written
  to `tokenizer.json` (HF's Unigram JSON has no such fields), and
  `Clone` carries them over. Set them on a loaded tokenizer with
  `Tokenizer::model_mut`, as above.

## Serialization

```json
{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁",-2.0],...],"byte_fallback":false}
```

Missing `"type"` or `byte_fallback` (older files) is accepted; a wrong
`"type"` is an error. Output is byte-identical to HF's for T5, ALBERT and
XLM-RoBERTa.

Scores are always finite. JSON has no `NaN` or infinity (`serde_json`
would write `null`, which neither morpheme nor HF can load back), so
serializing a model with a non-finite score is an error, and
`Tokenizer::save` fails before it touches an existing file.

See [trainer.md](./trainer.md) for `UnigramTrainer`.
