# `pre_tokenizers` — splitting text into words

A pre-tokenizer splits the normalized text into pieces that the model
tokenizes independently. Each piece becomes one "word" (its index is the
token's word id). It implements

```rust,ignore
pub trait PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()>;
}
```

Pieces are slices of `NormalizedString`, so a pre-tokenizer may also
rewrite text (ByteLevel maps bytes, Metaspace replaces spaces) without
losing offsets.

## Built-in pre-tokenizers

| Rust type | `"type"` in JSON | Parameters (defaults) | Behavior |
| --- | --- | --- | --- |
| `BertPreTokenizer` | `BertPreTokenizer` | — | Split on whitespace (removed); isolate every punctuation char (ASCII punctuation + Unicode `P*`) |
| `Whitespace` | `Whitespace` | — | Regex `\w+|[^\w\s]+`: `"Hey man!"` → `Hey`, `man`, `!` |
| `WhitespaceSplit` | `WhitespaceSplit` | — | Split on whitespace only: `"Hey man!"` → `Hey`, `man!` |
| `ByteLevel` | `ByteLevel` | `add_prefix_space` (true), `trim_offsets` (true), `use_regex` (true) | GPT-2: split with the GPT-2 regex, then map every byte to a printable char (space → `Ġ`, newline → `Ċ`) |
| `Metaspace` | `Metaspace` | `replacement` (`▁`), `prepend_scheme` (`always` / `first` / `never`), `split` (true) | SentencePiece: spaces → `▁`, optionally prepend `▁`, split so every piece starts with it |
| `Split` | `Split` | `pattern` (`{"String"}`/`{"Regex"}`), `behavior`, `invert` (false) | Split on a pattern; `behavior` is `Removed`, `Isolated`, `MergedWithPrevious`, `MergedWithNext` or `Contiguous` |
| `Punctuation` | `Punctuation` | `behavior` (`Isolated`) | Split on punctuation |
| `Digits` | `Digits` | `individual_digits` (false) | Isolate digits, one per piece or in runs |
| `CharDelimiterSplit` | `CharDelimiterSplit` | `delimiter` | Split on one char (removed) |
| `UnicodeScripts` | `UnicodeScripts` | — | Split where the Unicode script changes (kana/`ー` count as Han, as in SentencePiece) |
| `FixedLength` | `FixedLength` | `length` (5) | Chunks of `length` chars |
| `Sequence` | `Sequence` | `pretokenizers: [...]` | Apply several in order |

`ByteLevel` also implements `Decoder` and `PostProcessor` (offset
trimming), and `Metaspace` also implements `Decoder` — the same struct is
used in all those positions, as in HF. `ByteLevel::alphabet()` returns the
256 byte-level chars (use it as a BPE trainer's `initial_alphabet`), and
`pre_tokenizers::byte_level::{bytes_char, char_bytes}` expose the byte
tables. Regexes are compiled once, when the component is built or loaded.

## Example

```rust
use splinter::pre_tokenizers::{BertPreTokenizer, ByteLevel, Digits, Metaspace, PrependScheme, Sequence, Split, Whitespace, WhitespaceSplit};
use splinter::{OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};

fn pieces(pt: &impl PreTokenizer, s: &str) -> Vec<(String, (usize, usize))> {
    let mut pts = PreTokenizedString::from(s);
    pt.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (s.to_owned(), o))
        .collect()
}

fn main() -> splinter::Result<()> {
    let p = pieces(&Whitespace, "Hey man!");
    assert_eq!(p, [("Hey".into(), (0, 3)), ("man".into(), (4, 7)), ("!".into(), (7, 8))]);
    let p = pieces(&WhitespaceSplit, "Hey man!");
    assert_eq!(p[1].0, "man!");
    let words: Vec<String> = pieces(&BertPreTokenizer, "Hey friend!  How?!").into_iter().map(|p| p.0).collect();
    assert_eq!(words, ["Hey", "friend", "!", "How", "?", "!"]);

    // GPT-2: regex split, bytes mapped to printable chars (space -> Ġ).
    let gpt2 = ByteLevel::new(false, true, true);
    let words: Vec<String> = pieces(&gpt2, "Hello world, it's 😀").into_iter().map(|p| p.0).collect();
    assert_eq!(words, ["Hello", "Ġworld", ",", "Ġit", "'s", "ĠðŁĺĢ"]);

    // SentencePiece: spaces become ▁ and every word keeps its marker.
    let m = Metaspace::new('▁', PrependScheme::Always, true);
    let words: Vec<String> = pieces(&m, "Hello big world").into_iter().map(|p| p.0).collect();
    assert_eq!(words, ["▁Hello", "▁big", "▁world"]);

    // Llama-3 / Qwen style: a regex Split followed by other pre-tokenizers.
    let seq = Sequence::new(vec![
        Split::new(splinter::pattern::SplitPattern::Regex(r"\p{N}{1,3}".into()), SplitDelimiterBehavior::Isolated, false)?.into(),
        Digits::new(true).into(),
    ]);
    let words: Vec<String> = pieces(&seq, "abc12345").into_iter().map(|p| p.0).collect();
    assert_eq!(words, ["abc", "1", "2", "3", "4", "5"]);
    Ok(())
}
```

## JSON notes

- Legacy forms HF accepts are accepted too: `Metaspace` with
  `add_prefix_space` instead of `prepend_scheme` (`false` → `never`) and
  without `split`; `ByteLevel` without `use_regex`. Missing
  `add_prefix_space`/`trim_offsets` (ByteLevel) and `individual_digits`
  (Digits) fall back to defaults instead of failing.
- Re-saving upgrades legacy forms exactly as HF's own re-save does.
- An unknown `"type"` or an invalid `Split` regex is a load error.
