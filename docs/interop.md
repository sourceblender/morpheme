# Interop with `huggingface/tokenizers`

> Phase 3 work. This doc covers two formats:
> 1. **Splinter's own JSON** (v1.0, implemented now).
> 2. **HF `tokenizers.json`** (Phase 3, future).

## 1. Splinter's own JSON — v1.0 (implemented)

Defined in `crates/splinter/src/tokenizer/json.rs`. Schema version
`"1.0"`. BPE only.

```json
{
  "version": "1.0",
  "model": {
    "type": "bpe",
    "dropout": null,
    "unk_token": null,
    "end_of_word_suffix": "</w>",
    "continuing_subword_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false
  },
  "vocab": ["<unk>", "a", "b", ...],
  "merges": [["a", "a</w>"], ...]
}
```

### Invariants

- `vocab` is a list of unique strings. The list order defines ids — id
  `i` is the token at index `i`. Id 0 is conventionally `<unk>` but
  the loader does not enforce that.
- `merges` is a list of `[a, b]` pairs in priority order — index 0 is
  the highest-priority merge.
- `dropout`, `byte_fallback`, and `continuing_subword_suffix` are
  reserved fields. They must be `null` / `false` in v0.1; non-null /
  true values cause a `Model(...)` error at load time.
- `version` must equal `"1.0"`. Other values cause a `Model(...)` error.

### Loader errors

Unsupported configurations raise `splinter::Error::Model(_)` with a
human-readable message. The CLI surfaces these via `anyhow`.

## 2. HF `tokenizers.json` (Phase 3)

Goal: round-trip a `tokenizer.json` produced by `huggingface/tokenizers`.

### Components we plan to support

| `tokenizers` component         | `splinter` counterpart                          |
| ------------------------------ | ----------------------------------------------- |
| `normalizer.BertNormalizer`    | `splinter::normalizer::BertNormalizer`         |
| `pre_tokenizer.BertPreTokenizer` | `splinter::pre_tokenizer::BertPreTokenizer`  |
| `pre_tokenizer.ByteLevel`      | `splinter::pre_tokenizer::ByteLevel`            |
| `model.BPE`                    | `splinter::model::bpe::Bpe`                     |
| `model.WordPiece` | `splinter::model::wordpiece::WordPiece`         |
| `model.Unigram` | `splinter::model::unigram::Unigram`             |
| `decoder.WordPiece`            | `splinter::decoder::WordPieceDecoder`           |
| `decoder.ByteLevel`            | `splinter::decoder::ByteLevelDecoder`           |
| `post_processor.TemplateProcessing` | `splinter::post_processor::TemplateProcessing` |

### Drift log

When our JSON shape diverges from `tokenizers`'s, record it here with
the reason and the PR that introduced it.

| Divergence | Why | PR |
| ---------- | --- | -- |
| _none yet_ |    |    |