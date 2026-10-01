# `decoder`

## Purpose

Turn ids / tokens back into text. The inverse of the encode pipeline.

## Public API

```rust
pub trait Decoder: Send + Sync {
    fn decode(&self, tokens: &[String]) -> Result<String>;
}
```

Concrete decoders in v0.1:

- `WordPieceDecoder` — joins `##` subwords.
- `ByteLevelDecoder` — inverse of the GPT-2 byte-to-unicode map.
- `MetaspaceDecoder` — SentencePiece's `▁` → space conversion.
- `Strip` — strip leading / trailing / both for special tokens.

## Algorithm

Each decoder owns its inverse transform. They compose via
`DecoderSequence`.

## Performance notes

The decoder is the reverse hot path; large decodes (entire documents)
get a streaming variant in v0.2.

## Test strategy

- Round-trip: `decode(encode(text)) == text` for every supported
  tokenizer config.
- Empty input, single token, all-special-tokens.

## Known limitations

- No streaming variant yet.