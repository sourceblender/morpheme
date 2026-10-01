# `models::Bpe` — Byte-Pair Encoding

Used by GPT-2, RoBERTa, GPT-NeoX, Llama, Qwen and most modern LLMs.
Semantics follow Hugging Face `tokenizers` (checked against real tokenizer
files in `tests/hf_golden.rs`), with deliberate edge-case corrections
listed in [`docs/interop.md`](../interop.md).

## Algorithm

For each pre-token (word):

1. **Split into chars.** With `continuing_subword_prefix` (e.g. `##`),
   every non-initial char carries the prefix; with `end_of_word_suffix`
   (e.g. `</w>`), the last char carries the suffix. Most modern models
   use neither — byte-level models get their "start of word" marker
   (`Ġ`) from the ByteLevel pre-tokenizer instead.
2. **Unknown chars.** A char missing from the vocabulary becomes:
   - its UTF-8 bytes as `<0xNN>` tokens, if `byte_fallback` is on and
     those tokens exist (Llama / SentencePiece-converted models);
   - otherwise `unk_token`, with runs of unknown chars fused into one
     token when `fuse_unk` is on;
   - otherwise it is dropped — exactly what HF does when a model has no
     `unk_token` and no byte fallback.
3. **Merge.** Merges are ranked by their position in the merge list.
   Adjacent pairs are merged lowest-rank first using a priority queue;
   merging `a` + `b` produces `a` followed by `b` without its continuing
   prefix. Merging stops when no adjacent pair has a rank.
4. Each resulting symbol becomes a `Token { id, value, offsets }`, with
   byte offsets inside the word.

Options:

- `ignore_merges`: if the whole word is already in the vocabulary, emit
  it directly (Llama-3 style). Like HF, this shortcut only applies when
  `dropout` is `None` or `0.0`; with dropout active the word is always
  merged.
- `dropout` (BPE-dropout, `0.0..=1.0`): each merge is skipped with
  probability `p` during tokenization. Used for training-time
  augmentation; it uses an internal PRNG and bypasses the cache. `None` or
  `0.0` is deterministic (`0.0` is normalized to `None`). It can be
  changed on a loaded tokenizer; see [Runtime settings](#runtime-settings).
- **Cache:** deterministic models memoize merged words (default capacity
  10 000 words; words ≥ 256 bytes are not cached). The cache is sharded
  so parallel batch encoding does not contend on a single lock.
  `cache_capacity(0)` disables it; cloning a model gives it an empty
  cache.

## Construction

```rust
use morpheme::models::Bpe;
use morpheme::Model;
use std::collections::HashMap;

fn vocab(items: &[(&str, u32)]) -> HashMap<String, u32> {
    items.iter().map(|(t, i)| (t.to_string(), *i)).collect()
}

fn main() -> morpheme::Result<()> {
    let bpe = Bpe::builder()
        .vocab_and_merges(
            vocab(&[("<unk>", 0), ("l", 1), ("o", 2), ("w", 3), ("lo", 4), ("low", 5)]),
            vec![("l".into(), "o".into()), ("lo".into(), "w".into())],
        )
        .unk_token("<unk>")
        .build()?;
    let toks: Vec<(u32, String, (usize, usize))> =
        bpe.tokenize("lowl")?.into_iter().map(|t| (t.id, t.value, t.offsets)).collect();
    assert_eq!(toks, [(5, "low".into(), (0, 3)), (1, "l".into(), (3, 4))]);
    // Unknown chars become the unk token (fuse_unk merges runs of them).
    assert_eq!(bpe.tokenize("lz")?[1].value, "<unk>");

    // Llama-style byte fallback: unknown chars become <0xNN> byte tokens.
    let ll = Bpe::builder()
        .vocab_and_merges(vocab(&[("<unk>", 0), ("<0xC3>", 1), ("<0xA9>", 2), ("a", 3)]), vec![])
        .unk_token("<unk>")
        .byte_fallback(true)
        .build()?;
    let values: Vec<String> = ll.tokenize("aé")?.into_iter().map(|t| t.value).collect();
    assert_eq!(values, ["a", "<0xC3>", "<0xA9>"]);

    // A merge whose token is missing from the vocab is an error, not a panic.
    let bad = Bpe::builder()
        .vocab_and_merges(vocab(&[("a", 0)]), vec![("a".into(), "z".into())])
        .build();
    assert!(bad.is_err());
    Ok(())
}
```

Builder methods: `vocab_and_merges`, `unk_token`,
`continuing_subword_prefix`, `end_of_word_suffix`, `fuse_unk`,
`byte_fallback`, `ignore_merges`, `dropout`, `cache_capacity`, `build()
-> Result<Bpe>`. `Bpe::new(vocab, merges)` uses the defaults. Getters
(no `get_` prefix, per the Rust API guidelines): `unk_token`,
`continuing_subword_prefix`, `end_of_word_suffix`, `dropout`,
`fuse_unk`, `byte_fallback`, `ignore_merges`, `merges()`, plus the
`Model` trait (`token_to_id`, `id_to_token`, `vocab`, `vocab_size`).

`build()` fails (it never panics) if a merge references a token that is
not in the vocabulary, if a merged token is missing, or if `dropout` is
outside `0..=1`. Vocabulary ids are kept exactly as given — never
renumbered — so files with gaps in their ids load correctly.

## Runtime settings

`Bpe::set_dropout(Option<f32>) -> Result<()>` changes the dropout of an
existing model with the builder's validation: `None` or a value in
`[0, 1]` (`Some(0.0)` becomes `None`); anything else, including `NaN`, is
an `Error::Config` and leaves the model unchanged. The word cache is
cleared, since cached words assume the previous merge behaviour. To reach
the model inside a `Tokenizer`, use `Tokenizer::model_mut`:

```rust
use morpheme::models::{Bpe, ModelWrapper};
use morpheme::Tokenizer;

fn main() -> morpheme::Result<()> {
    let mut tokenizer = Tokenizer::new(Bpe::default());
    if let ModelWrapper::Bpe(bpe) = tokenizer.model_mut() {
        bpe.set_dropout(Some(0.1))?; // augment while preparing training data
        assert!(bpe.set_dropout(Some(1.5)).is_err());
        assert_eq!(bpe.dropout(), Some(0.1));
        bpe.set_dropout(None)?; // deterministic again
    }
    Ok(())
}
```

Unlike Unigram sampling, `dropout` is part of the Hugging Face format:
it is written to and read from `tokenizer.json`. `model_mut` is for
runtime settings only; replace the vocabulary with `set_model`, so that
added tokens and post-processor ids are rebound.

## Serialization

```json
{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
 "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
 "vocab":{"!":0,"\"":1,...},"merges":[["Ġ","t"],["Ġ","a"],...]}
```

- The vocabulary is written in id order; merges in the current HF format
  (`[a, b]` pairs).
- Loading accepts files without `"type"` (older `tokenizers`) and legacy
  `"a b"` string merges.
- A pair listed more than once in `merges` takes the rank of its last
  occurrence and is written once on re-save, as HF does.

See [trainer.md](./trainer.md) for `BpeTrainer`.
