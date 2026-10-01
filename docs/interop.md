# Interop with Hugging Face `tokenizers`

morpheme's on-disk format **is** the Hugging Face `tokenizer.json`
format (`"version": "1.0"`). There is no separate morpheme schema:
`Tokenizer::from_file` reads files produced by `tokenizers` /
`transformers`, and `Tokenizer::save` writes files they can load.

## What "compatible" means here

- **Same output.** For the same file and input, morpheme produces the
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
- Untagged normalizer / decoder objects (no `"type"` key, from files
  written before the tag existed): the variant is inferred from the
  fields, as in HF's untagged fallback — normalizers `BertNormalizer`
  (`clean_text`, `handle_chinese_chars`, `lowercase`), `Strip`
  (`strip_left`, `strip_right`), `Sequence` (`normalizers`),
  `Precompiled` (`precompiled_charsmap`), `Replace` (`pattern`,
  `content`), `Prepend` (`prepend`); decoders `BPEDecoder` (`suffix`),
  `WordPiece` (`prefix`, `cleanup`), `CTC` (`pad_token`,
  `word_delimiter_token`, `cleanup`), `Replace` (`pattern`, `content`),
  `Strip` (`content`, `start`, `stop`). Parameterless components (`{}`)
  and the components whose HF deserializer requires the tag (`ByteLevel`,
  `Metaspace`, decoder `Sequence`, `Fuse`, `ByteFallback`) need the
  `"type"` key, as in HF. Saving always writes the tagged form.

Unknown keys inside any component object are ignored (the same policy
for normalizers, pre-tokenizers, decoders, models and the top-level
file); only an unknown `"type"` value is an error.

## Offsets: bytes vs chars

Python `tokenizers` returns **char** offsets. Rust code usually wants
**byte** offsets (to slice a `&str`). `Tokenizer::encode` returns byte
offsets; `Tokenizer::encode_char_offsets` returns char offsets that match
Python exactly. The CLI has `--char-offsets`, and the `morpheme` Python
package always returns char offsets.

## How it is verified

- **`crates/morpheme/tests/hf_golden.rs`** loads 11 real tokenizers,
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
  `MORPHEME_SKIP_HF_GOLDEN=1` to skip locally).
- **`scripts/check_python_interop.py`** checks the other direction:
  tokenizers trained by the morpheme CLI (BPE, WordPiece, Unigram,
  WordLevel) are loaded by Python `tokenizers` and must encode and decode
  identically. Runs in CI.
- Component-level unit tests compare against hardcoded Python outputs,
  and the BPE trainer is checked for identical vocab and merges.

## Rust API naming

The file format and behavior follow Hugging Face exactly; the Rust API
follows the [Rust API guidelines](https://rust-lang.github.io/api-guidelines/)
where they differ from the `tokenizers` crate's names:

| Hugging Face (`tokenizers` crate) | morpheme |
| --- | --- |
| `get_vocab()` / `get_vocab_size()` (models and `Tokenizer`) | `vocab()` / `vocab_size()` |
| `Encoding::get_ids()`, `get_tokens()`, `get_offsets()`, … | `ids()`, `tokens()`, `offsets()`, … |
| `AddedToken::from(content, special)` | `AddedToken::new(content, special)` |
| `Unigram::from(vocab, unk_id, byte_fallback)` | `Unigram::new(vocab, unk_id, byte_fallback)` |
| `Precompiled::from(&[u8])` | `Precompiled::from_bytes(&[u8])` |
| `BPE`, `BPEDecoder`, `CTC` | `Bpe`, `BpeDecoder`, `Ctc` |
| public fields on models and trainers | builders + getters |

Methods that look something up by an argument keep `get_`
(`NormalizedString::get_range`, `PreTokenizedString::get_splits`), as in
the standard library. The JSON `"type"` names are unchanged (`"BPE"`,
`"BPEDecoder"`, `"CTC"`).

## Known deviations

Deliberate differences that accept more input, fail safely, or prevent
data corruption (see [ADR 0002](decisions/0002-correct-upstream-edge-case-bugs.md)).
Inputs that avoid these edge cases produce identical results.

### Errors instead of panics

- **Errors instead of panics.** Malformed input (merges referencing
  missing tokens, bad regexes, invalid `unk_id`, malformed Precompiled
  charsmaps, templates naming undefined special tokens, …) returns
  `morpheme::Error`; HF panics in several of these cases. A Precompiled
  charsmap whose trie size is not a multiple of 4 is rejected (HF rounds
  it down and reads the replacement pool from the wrong offset).
- **`Strip` decoder clamps.** `start` / `stop` larger than the token are
  clamped to its length and the result is returned; HF panics on an
  index underflow. Within bounds the output is identical.
- **Zero-width `Split` patterns are safe.** A `Split` regex that matches
  the empty string (`$`, `^`, `\b`, lookarounds) after a normalizer that
  changes the byte length (`NFD`, `Lowercase`, `Prepend`, …) produces the
  same tokens as without the pattern; HF panics (`NormalizedString bad
  split`). A custom `Pattern` reporting offsets inside a character is an
  error, not a panic.
- **Duplicate added-token ids are rejected.** A `tokenizer.json` whose
  `added_tokens` list the same id for two contents fails to load with an
  error naming both; HF keeps the last one but leaves the first content
  mapped to the id. Reusing an id through the API (for example after
  `set_model`) unmaps the previous content.
- **Template strings tolerate repeated whitespace, but not emptiness.**
  `TemplateProcessing` templates are split on any whitespace run, so
  `"[CLS]  $A\t[SEP]"` parses here; HF's `try_from` splits on single
  spaces and fails on the empty pieces. An empty template (`""` or an
  empty piece list) is an error in both the parser and the
  `TemplateProcessing` builder, where HF's builder would accept an empty
  `single` and produce empty encodings. Well-formed templates behave
  identically, and files are unaffected: `tokenizer.json` stores
  templates as piece lists.
- **Truncation rejects impossible special-token budgets.** Encoding with
  special tokens errors when the maximum cannot hold the required specials,
  or leaves no room for nonempty input. Empty input may use a budget exactly
  equal to the specials. Encoding without specials still uses the original
  maximum.

### Loading

- **Lenient defaults.** Some fields HF requires fall back to their
  defaults when missing (BertNormalizer flags, ByteLevel
  `add_prefix_space`/`trim_offsets`, Digits `individual_digits`,
  RobertaProcessing flags, TemplateProcessing `special_tokens`). Valid
  files behave identically.

### Encoding, truncation and offsets

- **Overflow windows are complete.** With truncation enabled, morpheme
  tokenizes the whole input and computes `overflowing` from the full
  token list, which equals HF's documented `Encoding.truncate(max_len,
  stride, direction)` + `post_process` on the full encoding. Since the
  `tokenize_with_limit` early exit in `tokenizers` 0.23, HF stops
  tokenizing a single sequence once `max_length` tokens exist (before the
  special-token budget is subtracted), so its overflow is computed on a
  truncated list and the tail of the input is silently dropped; with no
  post-processor it reports no overflow at all. Pairs whose sequences
  each fit in `max_length` match. Example (`WordLevel`, `[CLS]`/`[SEP]`
  post-processor, `max_length=5`, `only_first`, input `a b c d e f`):
  main ids are identical, overflow is `[[1,6,7,8,2]]` here and
  `[[1,6,7,2]]` in HF. Pinned by `only_first_truncation_through_tokenizer`
  and `left_truncation_through_tokenizer_with_specials_pairs_and_stride`
  in `crates/morpheme/tests/core.rs`.
- **Consistent type ids on overflow.** When truncation produces
  overflowing encodings, they get the same type ids as the main
  encoding: all `0` for `RobertaProcessing` (HF leaves `1` on the second
  sequence's overflow when `add_special_tokens` is false, which RoBERTa
  models cannot accept), and the template's type id for
  `TemplateProcessing` pieces such as `$B:3` (HF keeps the original id
  on overflow). Without overflow, outputs are identical to HF.
- **Sequence ownership is preserved.** Pair overflow retains sequence
  ranges even without a post-processor. Sequence lookups use the actual
  ids (including nonzero ids), and a missing sequence yields no match.
- **BPE fallback keeps input order.** An unknown-token run is emitted
  before a following successful byte fallback. HF can emit the fallback
  first, reordering tokens and assigning their offsets to the wrong chars.
- **Precompiled deletion at position 0.** An alignment bug shared with
  HF, corrected here on purpose (text output is identical): when a
  SentencePiece charsmap (T5, ALBERT, XLM-R, …) deletes the very first
  char of the input (for example U+0007 in T5), HF drops the removal
  from its alignment bookkeeping, so every surviving char maps back to
  the char *before* it — token offsets for such inputs are off by one
  char. morpheme counts the removal, so `"\u{7}ab"` normalizes to `"ab"`
  with `a` and `b` aligned to themselves. Deletions anywhere else were
  already correct in both.

### Added tokens

- **Added-token matches never overlap.** When an `rstrip` added token
  swallows whitespace that the next added token also starts with (for
  example `<x>` with `rstrip` and ` <y>` on `"<x> <y>"`), the second match
  starts where the first ends, so every byte belongs to exactly one
  token and offsets are `[(0,4),(4,7)]`. HF reports overlapping offsets
  (`[(0,4),(3,7)]`), and panics when the second token also has `lstrip`.
- **Safe added-token ids.** New added tokens receive ids above the
  highest occupied model or added-token id, including sparse vocabularies.
  Exhausting the `u32` id range returns an error. HF starts allocation
  at the vocabulary count, which can collide with an existing sparse id.
- **Re-adding an added token updates its flags.** `add_tokens` /
  `add_special_tokens` with the content of an existing added token but
  different flags (`lstrip`, `normalized`, `special`, …) replaces the
  stored token and counts it as added. HF compares added tokens by
  content only, so it returns 0 and keeps the old flags.
- **Retraining rebinds added tokens.** Existing added tokens retain their
  flags and are assigned ids against the newly trained model. Trainer
  special tokens use the new model's ids instead of retaining stale ids
  that can shadow ordinary tokens. Post-processor and padding ids are also
  rebound by token text. If a required configured token is absent from the
  new vocabulary, training fails without changing the tokenizer; include
  it in the trainer's special tokens or register it as an added token first.

### Trainers

- **Deterministic trainers.** Where HF's trainers depend on hash-map
  iteration order (BPE symbol ids with a prefix/suffix, `limit_alphabet`
  ties, Unigram), morpheme uses a fixed order, so results are
  reproducible but can differ from a particular HF run in those cases.
- **No id gaps from `WordLevelTrainer`.** A special token that is
  listed twice or also occurs in the corpus gets a single id; HF assigns
  it again, leaving unused ids.
- **No duplicate pieces from `UnigramTrainer`.** A special token or
  `unk_token` that also occurs as a corpus piece (or a special token
  listed twice) gets a single id, the special one. HF emits the piece a
  second time, and the duplicate shadows the special id on lookup.
- **`UnigramTrainer` enforces `vocab_size`.** The special tokens and the
  `unk_token` count toward `vocab_size`, the trained model never exceeds
  it, and a `vocab_size` too small for them plus the required chars is an
  `Error::Training`. HF only checks the required chars against
  `vocab_size` and can return more pieces than requested.
- **`WordPieceTrainer::default()` uses the `##` prefix**, the same as
  `WordPieceTrainer::builder().build()`. HF's derived `Default` has no
  continuation prefix, so it trains a vocabulary without `##` tokens.

### Decoding

- **`DecodeStream::prefill` never re-emits the prompt.** When the
  prefilled ids end inside a character (byte fallback), the step that
  completes the character emits just that character; the text before it
  is treated as already shown. HF emits the entire prompt again with that
  first chunk. Later chunks are identical. `tests/hf_golden.rs` strips
  the re-emitted prompt from HF's recorded first chunk for this case.
  Both libraries recognise an incomplete character by the trailing
  U+FFFD it decodes to. morpheme tells a real U+FFFD apart when it is
  made of several byte-fallback tokens (`<0xEF><0xBF><0xBD>`); a single
  id whose text is just U+FFFD remains ambiguous and is emitted again
  with the next chunk (HF re-emits the whole prompt in both cases).

### Runtime settings and Hub downloads

- **Unigram subword-regularization sampling** (`alpha`, `nbest_size`) is
  a runtime setting (`Unigram::set_sampling` / `set_seed`, reached on a
  loaded tokenizer through `Tokenizer::model_mut`), as in HF and
  SentencePiece; it is never read from or written to `tokenizer.json`,
  so a sampling model serializes exactly like a Viterbi one and loads
  back deterministic. See
  [`docs/modules/unigram.md`](modules/unigram.md#subword-regularization-sampling).
- **Hub downloads** (`from_pretrained`, feature `hub`) fetch only
  `tokenizer.json`, not the other files of a repository, and use the
  same cache layout as `huggingface_hub`. Downloads and cached content
  are verified against content-hash ETags; mirrors using opaque ETags
  are rejected. Copied snapshots must retain the matching blob, as in
  the standard HF cache. Corrupt entries are repaired online and
  rejected offline.

## Parity quirks kept on purpose

Behaviours that look odd but are kept because HF does them, so outputs
stay identical:

- A BPE model with neither `unk_token` nor `byte_fallback` silently
  drops characters it cannot represent.
- **Normalized special tokens are not skipped on decode.** A special
  token with `normalized: true` whose normalized form differs from its
  content (for example `<MASK>` under a `Lowercase` normalizer) decodes to
  its normalized text, and `skip_special_tokens` does not remove it,
  because the special-token set is keyed by the raw content. Special
  tokens are `normalized: false` by default, so only files that opt in
  are affected.
- **Byte-fallback offsets include prefix / suffix bytes.** With
  `byte_fallback` and a `continuing_subword_prefix` or
  `end_of_word_suffix`, an unknown char is looked up *with* the prefix /
  suffix attached, and when it falls back to bytes the prefix / suffix
  bytes are emitted as `<0xNN>` tokens too, each with a one-byte offset
  inside the word.
- **`ignore_merges` only without dropout.** The whole-word shortcut is
  consulted only when `dropout` is `null` / `0.0`; with dropout active a
  word is always merged (with vocab `{a, b, ab}`, `ignore_merges: true`
  and `dropout: 1.0`, `"ab"` gives `[a, b]`).
- **Duplicate merge pairs.** A pair listed more than once in `merges`
  takes the rank of its last occurrence and is written once on re-save
  (HF's merges are a map keyed by pair).
- `BpeTrainer` and `WordPieceTrainer` do not error when the special
  tokens and the alphabet alone exceed `vocab_size`; they learn no merges
  and return the larger vocabulary (see
  [`docs/modules/trainer.md`](modules/trainer.md#vocab_size)).
