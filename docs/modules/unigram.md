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
use splinter::models::Unigram;
use splinter::Model;

fn main() -> splinter::Result<()> {
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
  index. Fails on an empty vocabulary or an out-of-range `unk_id`.
- `Unigram::default()` — `[("<unk>", 0.0)]` with `unk_id` 0 (HF default).
- `encode(&str) -> Result<Vec<String>>`, `unk_id()`, `byte_fallback()`,
  `pieces()` (the `(piece, score)` pairs), `clear_cache()`, plus the
  `Model` trait.
- The Viterbi lattice is internal; use `tokenize` / `encode`.

Not implemented: subword-regularization sampling (`alpha` /
`nbest_size` sampling). HF does not store these settings in
`tokenizer.json`, so loading files is unaffected.

## Serialization

```json
{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁",-2.0],...],"byte_fallback":false}
```

Missing `"type"` or `byte_fallback` (older files) is accepted; a wrong
`"type"` is an error. Output is byte-identical to HF's for T5, ALBERT and
XLM-RoBERTa.

See [trainer.md](./trainer.md) for `UnigramTrainer`.
