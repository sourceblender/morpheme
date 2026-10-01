# Module deep dives

How each part of the `morpheme` pipeline works, what it accepts in
`tokenizer.json`, and how it maps to Hugging Face `tokenizers`.

| Doc | Covers |
| --- | --- |
| [tokenizer.md](./tokenizer.md) | `Tokenizer`: loading/saving, encode/decode, pairs, batches, added tokens, truncation, padding |
| [normalizer.md](./normalizer.md) | `normalizers::*`: BERT, Unicode forms, Precompiled, Replace, Prepend, … |
| [pre-tokenizer.md](./pre-tokenizer.md) | `pre_tokenizers::*`: ByteLevel, Metaspace, BERT, Whitespace, Split, … |
| [bpe.md](./bpe.md) | `models::Bpe` |
| [wordpiece.md](./wordpiece.md) | `models::WordPiece` and `models::WordLevel` |
| [unigram.md](./unigram.md) | `models::Unigram` |
| [post-processor.md](./post-processor.md) | `processors::*`: Template, BERT, RoBERTa, ByteLevel, Sequence |
| [decoder.md](./decoder.md) | `decoders::*`: ByteLevel, WordPiece, Metaspace, ByteFallback, … |
| [trainer.md](./trainer.md) | `trainers::*`: BPE, WordPiece, WordLevel, Unigram |
| [../interop.md](../interop.md) | The `tokenizer.json` format, what is verified, known deviations |

## The pipeline

```text
                       encode(input, add_special_tokens)
input ─┬─▶ added tokens   split out [CLS], <|endoftext|>, … (never split further)
       ├─▶ Normalizer     rewrite text (lowercase, NFKC, ▁ for spaces, …)
       ├─▶ PreTokenizer   split into words / pieces
       ├─▶ Model          each piece → tokens (BPE, WordPiece, WordLevel, Unigram)
       ├─▶ truncation     max_length minus the special tokens about to be added
       ├─▶ PostProcessor  add special tokens, merge pairs, set type ids
       └─▶ padding        pad to a fixed length or the longest in the batch
                       ─▶ Encoding { ids, tokens, offsets, type_ids, masks, word_ids, overflowing }

ids ─▶ id → token ─▶ (skip special tokens) ─▶ Decoder ─▶ text
```

Every component is a trait in `morpheme::traits` (`Normalizer`,
`PreTokenizer`, `Model`, `PostProcessor`, `Decoder`, `Trainer`), and every
built-in implementation is also a variant of a serde wrapper enum
(`NormalizerWrapper`, `PreTokenizerWrapper`, `ModelWrapper`,
`PostProcessorWrapper`, `DecoderWrapper`) whose JSON form is exactly the
Hugging Face one. A `Tokenizer` holds the wrappers, so any pipeline built
in code can be saved and any `tokenizer.json` can be loaded.

## Where offsets come from

Normalizers change text — `"Héllo"` may become `"hello"`, a space may
become `▁`. To still report where each token came from in the
*original* input, all text flows through
[`NormalizedString`](../../crates/morpheme/src/normalized_string.rs),
which stores, for every byte of the normalized text, the byte range of
the original text that produced it. Normalizers edit text only through
its alignment-preserving operations (`transform`, `replace`, `split`,
`prepend`, `nfkc`, …), pre-tokenizers split it into slices that keep
those alignments, and the final token offsets are mapped back through
them. The alignment model is the same as in Hugging Face `tokenizers`,
so offsets match theirs exactly (this is checked by the golden tests).

`Tokenizer::encode` reports **byte** offsets; `encode_char_offsets`
reports **char** offsets, which is what the Python `tokenizers` API
returns.
