# `pre_tokenizer`

## Purpose

Split normalized text into *pre-tokens* — the units the model will
further split into subwords.

## Public API

```rust
pub trait PreTokenizer: Send + Sync {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<PreToken<'a>>>;
}

pub struct PreToken<'a> {
    pub text: &'a str,
    pub span: (usize, usize), // byte offsets into the original buffer
}
```

Concrete pre-tokenizers in v0.1:

- `Whitespace` — split on `\\s+`.
- `BertPreTokenizer` — whitespace + punctuation split.
- `ByteLevel` — GPT-2 style byte-to-unicode mapping.
- `Digits` — isolated digits.

## Algorithm

Pre-tokenizers run after normalization. They return `PreToken` values
that borrow from the input buffer. The model layer then operates on the
`&str` slices.

## Performance notes

- `Whitespace` and `BertPreTokenizer` are byte-scan operations. We use
  `memchr` for the ASCII whitespace fast path.
- `ByteLevel` allocates a `String` per pre-token (the byte-to-unicode
  transform is non-trivial).

## Test strategy

- Span correctness — `(end - start) == text.len()` for every token.
- Equivalence with HF `tokenizers` for shared cases.
- Adversarial: zero-length matches, mixed scripts, RTL.

## Known limitations

- No language-aware segmentation (e.g. jieba). Use a sibling crate.