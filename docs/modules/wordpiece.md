# `models::WordPiece` — BERT subwords (and `WordLevel`)

## WordPiece

Used by BERT, DistilBERT, ELECTRA, MiniLM and other BERT descendants.

### Algorithm

For each pre-token (word):

1. If the word is longer than `max_input_chars_per_word` chars (default
   100), it becomes a single `unk_token`.
2. Otherwise, greedy longest-match-first: starting at the beginning, find
   the longest vocabulary entry that matches; for every piece after the
   first, the candidate is looked up with the `continuing_subword_prefix`
   (default `##`) prepended. Advance and repeat.
3. If any position has no match, the **whole word** becomes a single
   `unk_token` (default `[UNK]`) covering the entire word — partial
   matches are discarded, as in BERT and HF.

Real BERT vocabularies have no end-of-word marker; none is added.

```rust
use splinter::models::{WordLevel, WordPiece};
use splinter::Model;
use std::collections::HashMap;

fn main() -> splinter::Result<()> {
    let vocab: HashMap<String, u32> = [("[UNK]", 0), ("un", 1), ("##aff", 2), ("##able", 3), ("hello", 4)]
        .iter().map(|(t, i)| (t.to_string(), *i)).collect();
    let wp = WordPiece::builder().vocab(vocab.clone()).build()?;
    let v = |s: &str| -> Vec<String> { wp.tokenize(s).unwrap().into_iter().map(|t| t.value).collect() };
    assert_eq!(v("unaffable"), ["un", "##aff", "##able"]);
    // Any unmatched piece turns the whole word into a single [UNK].
    assert_eq!(v("unafx"), ["[UNK]"]);
    let unk = wp.tokenize("unafx")?;
    assert_eq!(unk[0].offsets, (0, 5));

    let wl = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    assert_eq!(wl.tokenize("hello")?[0].id, 4);
    assert_eq!(wl.tokenize("bye")?[0].value, "[UNK]");
    Ok(())
}
```

Builder: `vocab`, `unk_token` (`[UNK]`), `continuing_subword_prefix`
(`##`), `max_input_chars_per_word` (100), `build() -> Result`.
`WordPiece::from_bpe(&bpe)` turns a trained BPE model's vocabulary into
a WordPiece model (taking over its unk token and prefix) — this is how
`WordPieceTrainer` works.

If the unk token is needed but missing from the vocabulary, `tokenize`
returns an error instead of producing a bad id.

Serialization (HF field order):

```json
{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##",
 "max_input_chars_per_word":100,"vocab":{"[PAD]":0,...}}
```

Files without `"type"` (older `tokenizers`) load too; ids are preserved
as written.

Usually paired with `BertNormalizer`, `BertPreTokenizer`, the `WordPiece`
decoder and a `[CLS] $A [SEP]` post-processor — see the CLI's `bert`
preset or `tests/smoke.rs`.

## WordLevel

The simplest model: each pre-token maps to exactly one vocabulary entry,
or to `unk_token` (default `<unk>`) if it is missing. If the unk token is
needed and not in the vocabulary, encoding returns an error.

Builder: `vocab`, `unk_token`, `build() -> Result`. Serialization:

```json
{"type":"WordLevel","vocab":{"[UNK]":0,"the":1,...},"unk_token":"[UNK]"}
```

Trained by `WordLevelTrainer` ([trainer.md](./trainer.md)).
