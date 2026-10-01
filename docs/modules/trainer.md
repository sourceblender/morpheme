# `trainers` — learning a vocabulary

Trainers learn a model from text. Use them through the tokenizer so the
corpus goes through the *same* normalizer and pre-tokenizer that
encoding will use:

```rust,ignore
tokenizer.train(trainer, lines_iterator)?;           // any Iterator<Item: AsRef<str>>
tokenizer.train_from_files(trainer, &["a.txt"])?;    // lines, endings kept
```

`train` normalizes and pre-tokenizes each input, counts the resulting
words in parallel (rayon), trains, replaces the model, and registers the
trainer's special tokens as added tokens. If the tokenizer currently
holds a different kind of model (e.g. a `Bpe` and a `UnigramTrainer`),
it is replaced by a default model of the trainer's kind. Options the
trainer doesn't set (BPE `unk_token`, WordPiece/WordLevel `unk_token`)
come from the model you start with, so build it first:
`Tokenizer::new(WordLevel::builder().unk_token("[UNK]").build()?)`.

`train_from_files` reports missing files and invalid UTF-8 as errors;
nothing is silently skipped. Feeding again replaces previously fed words.

## Trainers and options

| Trainer | Options (defaults) |
| --- | --- |
| `BpeTrainer` | `vocab_size` (30 000), `min_frequency` (0), `special_tokens` ([]), `limit_alphabet` (none), `initial_alphabet` ({}), `continuing_subword_prefix` (none), `end_of_word_suffix` (none), `max_token_length` (none), `show_progress` (true) |
| `WordPieceTrainer` | same as BPE, with `continuing_subword_prefix` = `##`; trains BPE then builds the model with `WordPiece::from_bpe` (as HF does) |
| `WordLevelTrainer` | `vocab_size` (30 000), `min_frequency` (0), `special_tokens` ([]), `show_progress` |
| `UnigramTrainer` | `vocab_size` (8000), `n_sub_iterations` (2), `shrinking_factor` (0.75), `special_tokens` ([]), `initial_alphabet` ({}), `unk_token` (none), `max_piece_length` (16), `seed_size` (1 000 000), `show_progress` |

All are built with `X::builder()...build()`, which returns a `Result`:
every trainer rejects `vocab_size` 0; BPE and WordPiece also reject
`limit_alphabet` 0 and `max_token_length` 0; Unigram also rejects
`shrinking_factor` outside `(0, 1)`, `n_sub_iterations` 0 and
`max_piece_length` 0. Options are set only through the builders.
Special tokens always get the first ids, in the order given.

## Progress

With `show_progress` (the default, as in HF) and the `progressbar` cargo
feature (on by default), trainers draw progress bars on stderr:
"Pre-processing sequences" while counting words, then "Tokenize words",
"Count pairs" and "Compute merges" for BPE and WordPiece, or "Suffix
array seeds" and "EM training" (with the current/target piece count)
for Unigram. Bars are hidden automatically when stderr is not a
terminal, so logs and CI output stay clean. Pass `show_progress(false)`
to silence them, or build with `default-features = false` to drop the
`indicatif` dependency entirely. The CLI shows progress only when
stderr is a terminal; `splinter train --quiet` turns it off.

## How they work

- **BPE**: builds the alphabet (special tokens, `initial_alphabet`, the
  most frequent chars up to `limit_alphabet`), then repeatedly merges the
  most frequent adjacent pair, updating pair counts incrementally. Ties
  are broken exactly like HF (highest count, then smallest pair of ids).
  For GPT-2-style models pass `ByteLevel::alphabet()` as the initial
  alphabet so every byte is representable.
- **WordPiece**: BPE with the `##` continuation prefix, converted to a
  WordPiece vocabulary.
- **WordLevel**: keeps the most frequent words (≥ `min_frequency`).
- **Unigram** (SentencePiece): seeds the vocabulary with every char plus
  the most frequent substrings (internal nodes of the corpus suffix tree,
  found with a pure-Rust suffix array), runs EM with a digamma prior,
  prunes the pieces whose removal costs the least likelihood (shrinking
  by `shrinking_factor` per round), keeps every required char, then adds
  special tokens and cuts to `vocab_size`. It fails if `vocab_size` is
  smaller than the number of required chars.

## Example

```rust
use splinter::models::{Bpe, Unigram, WordPiece};
use splinter::normalizers::{BertNormalizer, Nfkc};
use splinter::pre_tokenizers::{BertPreTokenizer, ByteLevel, Metaspace};
use splinter::trainers::{BpeTrainer, UnigramTrainer, WordPieceTrainer};
use splinter::{AddedToken, Tokenizer};

const CORPUS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "hello world, hello again",
    "pack my box with five dozen liquor jugs",
];

fn main() -> splinter::Result<()> {
    // GPT-2 style byte-level BPE.
    let mut gpt = Tokenizer::new(Bpe::default())
        .with_pre_tokenizer(ByteLevel::new(false, true, true))
        .with_decoder(ByteLevel::default());
    let trainer = BpeTrainer::builder()
        .vocab_size(300)
        .min_frequency(1)
        .initial_alphabet(ByteLevel::alphabet())
        .special_tokens(vec![AddedToken::new("<|endoftext|>", true)])
        .show_progress(false)
        .build()?;
    gpt.train(trainer, CORPUS.iter())?;
    assert_eq!(gpt.token_to_id("<|endoftext|>"), Some(0));
    let enc = gpt.encode("hello wörld", false)?;
    assert_eq!(gpt.decode(enc.ids(), false)?, "hello wörld");

    // BERT style WordPiece.
    let mut bert = Tokenizer::new(WordPiece::default())
        .with_normalizer(BertNormalizer::default())
        .with_pre_tokenizer(BertPreTokenizer);
    let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].map(|s| AddedToken::new(s, true));
    bert.train(WordPieceTrainer::builder().vocab_size(200).special_tokens(specials.to_vec()).build()?, CORPUS.iter())?;
    assert_eq!(bert.token_to_id("[UNK]"), Some(1));

    // SentencePiece style Unigram.
    let ms = Metaspace::default();
    let mut sp = Tokenizer::new(Unigram::default())
        .with_normalizer(Nfkc)
        .with_pre_tokenizer(ms.clone())
        .with_decoder(ms);
    let trainer = UnigramTrainer::builder()
        .vocab_size(100)
        .special_tokens(vec![AddedToken::new("<unk>", true)])
        .unk_token("<unk>")
        .show_progress(false)
        .build()?;
    sp.train(trainer, CORPUS.iter())?;
    let enc = sp.encode("hello world", false)?;
    assert_eq!(sp.decode(enc.ids(), false)?, "hello world");
    Ok(())
}
```

## Parity with Hugging Face

- **BPE** produces exactly the same vocabulary ids and merges as HF's
  `BpeTrainer`: verified on unit-test corpora (special tokens, repeated
  chars, Unicode/emoji, `min_frequency`, `limit_alphabet`,
  `max_token_length`) and on a 26 MB, 200 000-line corpus (7 743 merges,
  identical vocab). Training is deterministic.
- Where HF itself is nondeterministic — symbol ids assigned in hash-map
  order when a prefix/suffix is set, `limit_alphabet` ties, Unigram's
  word iteration and required-char penalties — splinter uses a fixed
  order (sorted words, code-point tie-breaks, fixed-size parallel
  chunks), so its results are reproducible across runs and machines.
- **Unigram** is deterministic and close to HF: on test corpora the
  pieces overlap HF's completely (e.g. 8000/8000 on a 200k-word corpus),
  with score differences only where HF's own output varies between runs.
- Trained tokenizers save as standard `tokenizer.json`; Python
  `tokenizers` loads them and produces identical encodings
  (`scripts/check_python_interop.py`).
