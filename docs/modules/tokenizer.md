# `Tokenizer` — the full pipeline

`morpheme::Tokenizer` holds an optional normalizer, pre-tokenizer,
post-processor and decoder, a model, the added tokens, and optional
truncation and padding settings. It is `Clone`, `Send` and `Sync`, so one
instance can serve many threads.

## Loading and saving

| Method | |
| --- | --- |
| `Tokenizer::from_file(path)` / `from_json(&str)` / `from_bytes(..)` / `"…".parse()` | Load a Hugging Face `tokenizer.json` |
| `to_json(pretty)` / `save(path, pretty)` | Write `tokenizer.json` (byte-compatible with HF) |
| `Tokenizer::new(model)` + `with_normalizer` / `with_pre_tokenizer` / `with_post_processor` / `with_decoder` | Build in code (builder style) |
| `set_normalizer` / `set_pre_tokenizer` / `set_model` / `set_post_processor` / `set_decoder` | Replace a component in place |
| `normalizer()`, `pre_tokenizer()`, `model()`, `post_processor()`, `decoder()`, `truncation()`, `padding()`, `added_vocabulary()` | Inspect |

### From the Hugging Face Hub (feature `hub`)

`Tokenizer::from_pretrained(id, params)` downloads `tokenizer.json` for
a model id (`name` or `org/name`) and loads it. `FromPretrainedParameters`
sets the `revision` (branch, tag or commit; default `main`), `token`,
`cache_dir` and extra `User-Agent` entries.

- **Cache.** Files are stored in the standard `huggingface_hub` layout
  (`models--org--name/{blobs,refs,snapshots}`) under `HF_HUB_CACHE`,
  else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, so Python and
  morpheme share downloads. A pinned commit that is already cached is
  served without any network request.
- **Auth.** `params.token`, else `HF_TOKEN`, else the token saved by
  `huggingface-cli login` (`$HF_HOME/token`). The token is only sent to
  the Hub host, never to the CDN it redirects to.
- **Offline.** `HF_HUB_OFFLINE=1` serves from the cache only; if the Hub
  is unreachable, a cached copy is used when available.
- **Errors.** Missing, private or gated repositories, unknown revisions
  and repositories without a `tokenizer.json` return `Error::Hub` with a
  message saying which.
- `HF_ENDPOINT` points at a mirror or private Hub.

```rust,ignore
use morpheme::{FromPretrainedParameters, Tokenizer};

let params = FromPretrainedParameters::default().revision("v1.0").token("hf_...");
let tokenizer = Tokenizer::from_pretrained("my-org/my-model", Some(params))?;
```

The CLI accepts a model id anywhere it accepts a path (`-t org/name`,
with `--revision`).

## Encoding

- `encode(input, add_special_tokens)` → `Encoding` with **byte**
  offsets into the original input.
- `encode_char_offsets(input, add_special_tokens)` → the same with
  **char** offsets — what Python `tokenizers` returns.
- `encode_batch(inputs, add)` / `encode_batch_char_offsets` — parallel
  (rayon); with `BatchLongest` padding every result is padded to the
  longest.
- Inputs: `&str`, `String`, a pair `(a, b)`, or pre-tokenized words
  (`Vec<&str>`, `&[&str]`, `Vec<String>`) — each word is tokenized on its
  own, gets its index as word id, and its offsets are relative to that
  word.

`Encoding` exposes `ids()`, `tokens()`, `offsets()`, `type_ids()`,
`attention_mask()`, `special_tokens_mask()`, `word_ids()`,
`sequence_ids()`, `overflowing()`, and helpers such as
`word_to_tokens`, `token_to_word`, `char_to_token`.

The steps run in this order: added tokens are split out (tokens with
`normalized: false` — the default for special tokens — are matched in the
raw input, the others in the normalized text) → normalizer → pre-tokenizer → model → truncation →
post-processor → padding. `normalize(text)` runs only the normalizer.

## Decoding

`decode(ids, skip_special_tokens)` maps ids to tokens (added tokens
first), drops special tokens if asked, and runs the decoder (or joins
with spaces when there is none). Unknown ids are skipped.
`decode_batch(&[&[u32]], skip)` runs in parallel.

### Streaming decode

For ids that arrive one at a time (text generation), decoding each id
on its own is wrong: a character split across byte-fallback or
byte-level tokens decodes to `�`, and decoders such as `Metaspace` or
`Strip` treat the start of every call as the start of the text.
`decode_stream(skip_special_tokens)` returns a `DecodeStream` that keeps
the context it needs and returns exactly the new text per step:

```rust,ignore
let mut stream = tokenizer.decode_stream(true);
for id in generated_ids {
    if let Some(chunk) = stream.step(id)? {
        print!("{chunk}"); // None means "not a complete character yet"
    }
}
```

- `step(id)` / `step_many(&ids)` return `Some(new_text)` or `None` when
  the ids don't complete any text yet. Concatenating every chunk equals
  `decode` of all ids.
- `.prefill(&ids)` starts from ids whose text was already shown (e.g.
  the prompt): later chunks are decoded in their context but the prefill
  itself is not re-emitted. A prefill that ends mid-character is not
  considered shown, matching Hugging Face.
- Semantics match Python `tokenizers.decoders.DecodeStream` step for step
  (checked against all 11 golden fixtures, including the `None` steps).

## Added tokens

`add_tokens(&[AddedToken])` / `add_special_tokens(&[AddedToken])`.
An `AddedToken` (`AddedToken::new(content, special)`) is matched
verbatim and never split. Flags: `single_word` (only between non-word
chars), `lstrip` / `rstrip` (swallow adjacent whitespace), `normalized`
(match the normalized text instead of the raw input; defaults to
`!special`), `special` (skippable when decoding). A token already in the
model keeps the model's id; new ones get ids after the model's vocabulary.
When loading `tokenizer.json`, ids are taken from the file as written.
`set_encode_special_tokens(true)` makes special tokens tokenize like
plain text.

## Truncation and padding

```rust,ignore
TruncationParams { max_length: 512, stride: 0, strategy: LongestFirst, direction: Right }
PaddingParams { strategy: BatchLongest | Fixed(n), direction: Right, pad_to_multiple_of: None,
                pad_id: 0, pad_type_id: 0, pad_token: "[PAD]" }
```

- Truncation reserves room for the special tokens the post-processor
  will add. Removed tokens become `overflowing()` encodings of at most
  `max_length` tokens overlapping by `stride`; they are post-processed
  and padded too.
- Pair strategies: `LongestFirst` (shrink the longer sequence first),
  `OnlyFirst`, `OnlySecond`. `set_truncation` rejects a stride that is not
  smaller than the effective max length.
- Padding sets `attention_mask` 0 and `special_tokens_mask` 1 for pad
  tokens. Both settings are stored in and loaded from `tokenizer.json`.

## Example

```rust
use morpheme::models::WordLevel;
use morpheme::pre_tokenizers::Whitespace;
use morpheme::processors::BertProcessing;
use morpheme::{AddedToken, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use std::collections::HashMap;

fn main() -> morpheme::Result<()> {
    let vocab: HashMap<String, u32> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "héllo", "world", "a", "b", "c"]
        .iter().enumerate().map(|(i, t)| (t.to_string(), i as u32)).collect();
    let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    let mut tok = Tokenizer::new(model)
        .with_pre_tokenizer(Whitespace)
        .with_post_processor(BertProcessing::new(("[SEP]", 3), ("[CLS]", 2)));
    tok.add_special_tokens(&["[PAD]", "[UNK]", "[CLS]", "[SEP]"].map(|t| AddedToken::new(t, true)))?;

    // Byte offsets (Rust) vs char offsets (what Python returns).
    let enc = tok.encode("héllo world", true)?;
    assert_eq!(enc.tokens(), &["[CLS]", "héllo", "world", "[SEP]"]);
    assert_eq!(enc.offsets()[2], (7, 12));
    assert_eq!(tok.encode_char_offsets("héllo world", true)?.offsets()[2], (6, 11));

    // Pre-tokenized input: one word id per given word.
    let enc = tok.encode(vec!["a", "b"], false)?;
    assert_eq!(enc.word_ids(), &[Some(0), Some(1)]);

    // Decoding (no decoder set: tokens joined with spaces).
    assert_eq!(tok.decode(&[2, 4, 5, 3], true)?, "héllo world");

    // Truncation keeps room for the special tokens; padding pads batches.
    tok.set_truncation(Some(TruncationParams { max_length: 4, ..Default::default() }))?;
    tok.set_padding(Some(PaddingParams { strategy: PaddingStrategy::BatchLongest, ..Default::default() }));
    let batch = tok.encode_batch(vec!["a b c", "a"], true)?;
    assert_eq!(batch[0].tokens(), &["[CLS]", "a", "b", "[SEP]"]);
    // Overflowing parts are post-processed and padded too.
    assert_eq!(batch[0].overflowing()[0].tokens(), &["[CLS]", "c", "[SEP]", "[PAD]"]);
    assert_eq!(batch[1].ids(), &[2, 6, 3, 0]);
    assert_eq!(batch[1].attention_mask(), &[1, 1, 1, 0]);

    // Save and load (tokenizer.json, Hugging Face format).
    let json = tok.to_json(true)?;
    let back = Tokenizer::from_json(&json)?;
    assert_eq!(back.encode("a b", true)?, tok.encode("a b", true)?);
    Ok(())
}
```

## Training

`train(trainer, iterator)` and `train_from_files(trainer, &paths)` — see
[trainer.md](./trainer.md).
