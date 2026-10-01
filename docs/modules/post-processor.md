# `processors` — special tokens and pairs

After the model (and truncation), the post-processor adds special tokens,
merges a sentence pair into one encoding, and sets type ids. Special
tokens get offsets `(0, 0)`, word id `None` and `special_tokens_mask` 1.
With `encode(..., add_special_tokens = false)` nothing is inserted.

```rust,ignore
pub trait PostProcessor {
    fn added_tokens(&self, is_pair: bool) -> usize; // reserved by truncation
    fn process_encodings(&self, encodings: Vec<Encoding>, add_special_tokens: bool) -> Result<Vec<Encoding>>;
    fn process(&self, encoding: Encoding, pair: Option<Encoding>, add_special_tokens: bool) -> Result<Encoding>; // provided
}
```

Without a post-processor, a pair is simply concatenated (sequence B keeps
type id 1).

## Built-in post-processors

| Rust type | `"type"` in JSON | Parameters | Layout |
| --- | --- | --- | --- |
| `TemplateProcessing` | `TemplateProcessing` | `single`, `pair` templates; `special_tokens` | Anything, e.g. `[CLS] $A [SEP]` / `[CLS] $A [SEP] $B:1 [SEP]:1` |
| `BertProcessing` | `BertProcessing` | `sep` (`["[SEP]",102]`), `cls` (`["[CLS]",101]`) | `[CLS] A [SEP]`, `[CLS] A [SEP] B [SEP]` (B and its `[SEP]` get type id 1) |
| `RobertaProcessing` | `RobertaProcessing` | `sep` (`["</s>",2]`), `cls` (`["<s>",0]`), `trim_offsets` (true), `add_prefix_space` (true) | `<s> A </s>`, `<s> A </s></s> B </s>`, all type ids 0 |
| `pre_tokenizers::ByteLevel` | `ByteLevel` | `trim_offsets`, `add_prefix_space` | No tokens added; trims the whitespace carried by `Ġ` tokens out of their offsets |
| `Sequence` | `Sequence` | `processors: [...]` | Several in order (e.g. ByteLevel trimming then a template) |

Special-token ids always come from the configuration — they are never
looked up or defaulted. Overflowing parts produced by truncation are
processed too.

### Template syntax

Pieces are separated by whitespace:

- `$A` (or `$a`, `$`) — the first sequence; `$B` / `$b` — the second;
  `$0`, `$1`, … — the first sequence with that type id;
- anything else — a special token, which must be listed in
  `special_tokens` (a key may expand to several tokens/ids);
- a `:N` suffix sets the type id (`$B:1`, `[SEP]:1`).

`TemplateProcessing::builder()` validates on `build()`: the pair template
must use both `$A` and `$B`, the single template must not use `$B`, and
every special token used must be provided — otherwise you get an error.

## Example

```rust
use splinter::models::WordLevel;
use splinter::pre_tokenizers::Whitespace;
use splinter::processors::{BertProcessing, RobertaProcessing, TemplateProcessing};
use splinter::Tokenizer;
use std::collections::HashMap;

fn main() -> splinter::Result<()> {
    let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[CLS]", 1), ("[SEP]", 2), ("hello", 3), ("world", 4)]
        .iter().map(|(t, i)| (t.to_string(), *i)).collect();
    let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    let base = Tokenizer::new(model).with_pre_tokenizer(Whitespace);

    let template = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")?
        .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")?
        .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
        .build()?;
    let tok = base.clone().with_post_processor(template);
    let enc = tok.encode(("hello", "world"), true)?;
    assert_eq!(enc.tokens(), &["[CLS]", "hello", "[SEP]", "world", "[SEP]"]);
    assert_eq!(enc.type_ids(), &[0, 0, 0, 1, 1]);
    assert_eq!(enc.special_tokens_mask(), &[1, 0, 1, 0, 1]);
    assert_eq!(enc.offsets()[0], (0, 0));

    // Same layout as BertProcessing:
    let bert = base.clone().with_post_processor(BertProcessing::new(("[SEP]", 2), ("[CLS]", 1)));
    assert_eq!(bert.encode(("hello", "world"), true)?.ids(), enc.ids());

    // RoBERTa: <s> A </s></s> B </s>, all type ids 0.
    let rob = base.with_post_processor(RobertaProcessing::new(("[SEP]", 2), ("[CLS]", 1)));
    let e = rob.encode(("hello", "world"), true)?;
    assert_eq!(e.tokens(), &["[CLS]", "hello", "[SEP]", "[SEP]", "world", "[SEP]"]);
    assert_eq!(e.type_ids(), &[0, 0, 0, 0, 0, 0]);
    Ok(())
}
```

## JSON notes and deviations

- Serialized like HF: templates as lists of `{"SpecialToken":{...}}` /
  `{"Sequence":{...}}`, `special_tokens` sorted by key, tuples as
  `["[SEP]", 102]`.
- A template that names a special token missing from the config (only
  possible via hand-written JSON) returns an error when encoding; HF
  panics.
- Template strings split on any run of whitespace (HF splits on single
  spaces, producing empty pieces for double spaces).
- Missing `trim_offsets`/`add_prefix_space` (RoBERTa) default to true and
  missing `special_tokens` (Template) default to empty, instead of
  failing to load.
- `TemplateProcessing::default()` uses the pair template `$A:0 $B:1`
  (HF's default drops sequence B).
