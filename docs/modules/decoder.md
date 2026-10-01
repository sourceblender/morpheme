# `decoders` — tokens back to text

`Tokenizer::decode(ids, skip_special_tokens)` maps ids to token strings
(added tokens first, then the model), drops special tokens if asked, and
hands the list to the decoder. Without a decoder, tokens are joined with
spaces.

```rust,ignore
pub trait Decoder {
    /// Transform the token list; the text is the concatenation of the result.
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;
    fn decode(&self, tokens: Vec<String>) -> Result<String>; // provided
}
```

Decoders compose: `Sequence` runs each decoder's `decode_chain` on the
output of the previous one.

## Built-in decoders

| Rust type | `"type"` in JSON | Parameters (defaults) | Behavior |
| --- | --- | --- | --- |
| `pre_tokenizers::ByteLevel` | `ByteLevel` | (pre-tokenizer fields) | Map byte-level chars back to bytes and decode UTF-8 (`Ġ` → space) |
| `WordPiece` | `WordPiece` | `prefix` (`##`), `cleanup` (true) | Glue `##` continuations to the previous token, separate words with spaces; `cleanup` removes spaces before punctuation and in English contractions |
| `pre_tokenizers::Metaspace` | `Metaspace` | (pre-tokenizer fields) | `▁` → space, dropping the prefix space according to `prepend_scheme` |
| `BpeDecoder` | `BPEDecoder` | `suffix` (`</w>`) | Replace the end-of-word suffix with a space (none after the last token) |
| `ByteFallback` | `ByteFallback` | — | Turn runs of `<0xNN>` tokens into the UTF-8 text they encode (invalid bytes → one `�` each) |
| `Fuse` | `Fuse` | — | Concatenate all tokens into one |
| `Strip` | `Strip` | `content`, `start`, `stop` | Remove up to `start` leading / `stop` trailing `content` chars from each token |
| `Replace` | `Replace` | `pattern`, `content` | Replace a pattern in every token (e.g. `▁` → `" "`) |
| `Ctc` | `CTC` | `pad_token` (`<pad>`), `word_delimiter_token` (`\|`), `cleanup` (true) | CTC (speech): collapse repeats, drop padding, delimiter → space |
| `Sequence` | `Sequence` | `decoders: [...]` | Apply several in order |

## Example

```rust
use morpheme::decoders::{ByteFallback, Fuse, Replace, Sequence, Strip, WordPiece};
use morpheme::pre_tokenizers::{ByteLevel, Metaspace};
use morpheme::Decoder;

fn s(v: &[&str]) -> Vec<String> { v.iter().map(|x| x.to_string()).collect() }

fn main() -> morpheme::Result<()> {
    assert_eq!(WordPiece::default().decode(s(&["hello", "world", "##s", "!"]))?, "hello worlds!");
    assert_eq!(ByteLevel::default().decode(s(&["Hello", "Ġw", "Ã¶", "rld"]))?, "Hello wörld");
    assert_eq!(Metaspace::default().decode(s(&["▁Hello", "▁world"]))?, "Hello world");

    // The Llama decoder chain: ▁ -> space, <0xNN> bytes -> text, join, drop one leading space.
    let llama = Sequence::new(vec![
        Replace::new("▁", " ")?.into(),
        ByteFallback::new().into(),
        Fuse::new().into(),
        Strip::new(' ', 1, 0).into(),
    ]);
    assert_eq!(llama.decode(s(&["▁Hi", "▁", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]))?, "Hi 😀");
    Ok(())
}
```

## Notes

- Decoding is lossless where the pipeline is: byte-level BPE always
  round-trips (`decode(encode(s)) == s`, property-tested); WordPiece and
  SentencePiece pipelines lose what their normalizers removed (case,
  accents, repeated spaces).
- `decoders::Replace` is its own struct with the same JSON shape as the
  `Replace` normalizer.
- `BpeDecoder` on an empty list and `Strip` with `stop` beyond the token
  return results instead of panicking (HF panics).
- Legacy untagged decoder JSON (no `"type"`) is accepted for the decoders
  HF can tell apart by their fields (`BPEDecoder`, `WordPiece`, `CTC`,
  `Replace`, `Strip`); `ByteLevel`, `Metaspace`, `Sequence`, `Fuse` and
  `ByteFallback` need the `"type"` key, as in HF. Saving always writes
  the tagged form. An unknown `"type"` is a load error; unknown keys
  inside a decoder object are ignored.
