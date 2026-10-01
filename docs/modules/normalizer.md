# `normalizers` — rewriting text

A normalizer rewrites the input before it is split (lowercasing, Unicode
normalization, accent stripping, replacing spaces, …). It implements

```rust,ignore
pub trait Normalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}
```

and works only through `NormalizedString`'s alignment-preserving
operations, so offsets into the original input stay exact (see
[README](./README.md#where-offsets-come-from)).

## Built-in normalizers

| Rust type | `"type"` in JSON | Parameters (defaults) | What it does |
| --- | --- | --- | --- |
| `BertNormalizer` | `BertNormalizer` | `clean_text` (true), `handle_chinese_chars` (true), `strip_accents` (`null` = follow `lowercase`), `lowercase` (true) | BERT: drop control chars and map whitespace to `' '`, put spaces around CJK ideographs (not Hangul/kana), strip accents, lowercase |
| `Lowercase` | `Lowercase` | — | Unicode lowercase (one char may become several) |
| `Nfc`, `Nfd`, `Nfkc`, `Nfkd` | `NFC`, `NFD`, `NFKC`, `NFKD` | — | Unicode normalization forms |
| `StripAccents` | `StripAccents` | — | Remove combining marks (use after `NFD`/`NFKD`) |
| `Strip` | `Strip` | `strip_left`, `strip_right` | Trim leading/trailing whitespace |
| `Replace` | `Replace` | `pattern` (`{"String": …}` or `{"Regex": …}`), `content` | Replace every match |
| `Prepend` | `Prepend` | `prepend` | Prepend a string to non-empty input (Llama's leading `▁`) |
| `Nmt` | `Nmt` | — | SentencePiece `nmt_nfkc` cleanup: drop most C0 controls, map odd spaces/separators to `' '` |
| `Precompiled` | `Precompiled` | `precompiled_charsmap` (base64) | SentencePiece's compiled normalization rules (T5, ALBERT, XLM-R) |
| `ByteLevel` | `ByteLevel` | — | Map every UTF-8 byte to its GPT-2 printable char |
| `Sequence` | `Sequence` | `normalizers: [...]` | Apply several in order |

Unicode tables come from the same crates Hugging Face uses
(`unicode-normalization-alignments`, `unicode_categories`), so results
agree character for character. `Precompiled` is a pure-Rust reader of the
darts-clone trie inside SentencePiece models; it applies rules per
grapheme cluster and, like HF, the *shortest* matching rule wins.
A malformed charsmap is an error, never a panic.

## Example

```rust
use splinter::normalizers::{BertNormalizer, Lowercase, Nfd, Prepend, Replace, Sequence, StripAccents};
use splinter::{NormalizedString, Normalizer, NormalizerWrapper, OffsetRange};

fn main() -> splinter::Result<()> {
    // BERT: clean control chars, space out CJK, strip accents, lowercase.
    let mut n = NormalizedString::from("Héllo\u{0}  世界");
    BertNormalizer::default().normalize(&mut n)?;
    assert_eq!(n.get(), "hello   世  界 ");

    // Offsets survive: "e" (normalized 1..2) came from "é" (original 1..3).
    assert_eq!(n.get_range_original(OffsetRange::Normalized(1..2)), Some("é"));

    // Compose with Sequence (this is what HF writes as {"type":"Sequence",...}).
    let seq = Sequence::new(vec![Nfd.into(), StripAccents.into(), Lowercase.into()]);
    let mut n = NormalizedString::from("Crème Brûlée");
    seq.normalize(&mut n)?;
    assert_eq!(n.get(), "creme brulee");

    // Llama-style: prepend ▁ and replace spaces.
    let llama = Sequence::new(vec![
        Prepend::new("▁").into(),
        Replace::new(" ", "▁")?.into(),
    ]);
    let mut n = NormalizedString::from("Hey friend");
    llama.normalize(&mut n)?;
    assert_eq!(n.get(), "▁Hey▁friend");

    // From tokenizer.json.
    let w: NormalizerWrapper = serde_json::from_str(r#"{"type":"NFKC"}"#)?;
    let mut n = NormalizedString::from("ﬁ①");
    w.normalize(&mut n)?;
    assert_eq!(n.get(), "fi1");
    Ok(())
}
```

## JSON notes

- Output is byte-compatible with HF: same field names and order, e.g.
  `{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}`.
- `BertNormalizer` fields that are missing fall back to the defaults
  above (HF requires them; this only accepts more files).
- An unknown or missing `"type"` is a load error. The very old untagged
  normalizer format (no `"type"` key) is not accepted.
- `Replace` with an invalid regex is a load error.
