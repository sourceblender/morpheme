# Interop with Hugging Face `tokenizers`

splinter's on-disk format **is** the Hugging Face `tokenizer.json`
format (`"version": "1.0"`). There is no separate splinter schema:
`Tokenizer::from_file` reads files produced by `tokenizers` /
`transformers`, and `Tokenizer::save` writes files they can load.

## What "compatible" means here

- **Same output.** For the same file and input, splinter produces the
  same ids, tokens, offsets, type ids, attention mask, special-tokens
  mask, word ids and decoded text as `tokenizers` 0.23.2.
- **Same JSON.** Every component serializes with HF's field names, field
  order and defaults, so re-saving a file gives what HF's own re-save
  gives. Top-level keys are written in HF order: `version`, `truncation`,
  `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`,
  `post_processor`, `decoder`, `model`. Vocabularies are written in id
  order, BPE merges as `[a, b]` pairs (the current HF format).
- **Ids preserved.** Vocabulary and added-token ids are kept exactly as
  written in the file — never renumbered.

### Supported components

| Kind | Types |
| --- | --- |
| model | `BPE`, `WordPiece`, `WordLevel`, `Unigram` |
| normalizer | `BertNormalizer`, `Strip`, `StripAccents`, `NFC`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `Nmt`, `Precompiled`, `Replace`, `Prepend`, `ByteLevel`, `Sequence` |
| pre_tokenizer | `BertPreTokenizer`, `ByteLevel`, `CharDelimiterSplit`, `Metaspace`, `Whitespace`, `WhitespaceSplit`, `Sequence`, `Split`, `Punctuation`, `Digits`, `UnicodeScripts`, `FixedLength` |
| post_processor | `BertProcessing`, `RobertaProcessing`, `ByteLevel`, `TemplateProcessing`, `Sequence` |
| decoder | `BPEDecoder`, `ByteLevel`, `WordPiece`, `Metaspace`, `CTC`, `Sequence`, `Replace`, `Fuse`, `Strip`, `ByteFallback` |
| other | `added_tokens` (all flags), `truncation`, `padding` |

An unknown `"type"` anywhere is a load error that names the type —
nothing is silently replaced with a default.

### Legacy forms accepted

- Model objects without `"type"` (files from older `tokenizers`): the
  type is inferred from the fields (`merges` → BPE, list-shaped `vocab`
  or `unk_id` → Unigram, `max_input_chars_per_word` /
  `continuing_subword_prefix` → WordPiece, otherwise WordLevel).
- BPE merges as `"a b"` strings.
- `Metaspace` with `add_prefix_space` instead of `prepend_scheme`, and
  without `split`; `ByteLevel` without `use_regex`; Unigram without
  `byte_fallback`.

## Offsets: bytes vs chars

Python `tokenizers` returns **char** offsets. Rust code usually wants
**byte** offsets (to slice a `&str`). `Tokenizer::encode` returns byte
offsets; `Tokenizer::encode_char_offsets` returns char offsets that match
Python exactly. The CLI has `--char-offsets`.

## How it is verified

- **`crates/splinter/tests/hf_golden.rs`** loads 11 real tokenizers,
  pinned by revision in `scripts/hf-fixtures.txt` (BERT uncased/cased,
  MiniLM with truncation+padding, GPT-2, RoBERTa, GPT-NeoX, Qwen2.5,
  Llama, T5, ALBERT, XLM-RoBERTa). For 32 tricky inputs each (empty and
  whitespace-only strings, accents, combining marks, CJK, Hangul,
  Cyrillic, Arabic, emoji with skin tones and flags, control and
  zero-width chars, 120-char words, special tokens in text, …), plus
  pairs and a padded batch, it compares every encoding field and both
  decode modes against goldens recorded from Python (`scripts/gen_golden.py`,
  output in `tests/golden/`). It also checks that every serialized
  component — and its field order — equals HF's re-serialization, and
  repeats everything after a save → load round trip. Fetch the files
  with `scripts/fetch-hf-fixtures.sh` (CI does this; set
  `SPLINTER_SKIP_HF_GOLDEN=1` to skip locally).
- **`scripts/check_python_interop.py`** checks the other direction:
  tokenizers trained by the splinter CLI (BPE, WordPiece, Unigram,
  WordLevel) are loaded by Python `tokenizers` and must encode and decode
  identically. Runs in CI.
- Component-level unit tests compare against hardcoded Python outputs,
  and the BPE trainer is checked for identical vocab and merges.

## Known deviations

Deliberate differences, all in the direction of *accepting more* or
*failing safely*:

- **Errors instead of panics.** Malformed input (merges referencing
  missing tokens, bad regexes, invalid `unk_id`, malformed Precompiled
  charsmaps, templates naming undefined special tokens, `Strip` decoder
  bounds, …) returns `splinter::Error`; HF panics in several of these
  cases.
- **Lenient defaults.** Some fields HF requires fall back to their
  defaults when missing (BertNormalizer flags, ByteLevel
  `add_prefix_space`/`trim_offsets`, Digits `individual_digits`,
  RobertaProcessing flags, TemplateProcessing `special_tokens`). Valid
  files behave identically.
- **Not accepted:** the very old untagged normalizer / decoder JSON
  without a `"type"` key.
- **Deterministic trainers.** Where HF's trainers depend on hash-map
  iteration order (BPE symbol ids with a prefix/suffix, `limit_alphabet`
  ties, Unigram), splinter uses a fixed order, so results are
  reproducible but can differ from a particular HF run in those cases.
- **Unigram subword-regularization sampling** (`alpha`, `nbest_size`) is
  not implemented; it is not part of `tokenizer.json`.
- **Hub downloads** (`from_pretrained`, feature `hub`) fetch only
  `tokenizer.json`, not the other files of a repository, and use the
  same cache layout as `huggingface_hub`.

Kept on purpose because HF does it: a BPE model with neither
`unk_token` nor `byte_fallback` silently drops characters it cannot
represent.
