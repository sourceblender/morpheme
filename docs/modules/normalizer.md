# `normalizer`

## Purpose

Mutate raw input text into a canonical form before pre-tokenization.

## Public API

```rust
pub trait Normalizer: Send + Sync {
    fn normalize(&self, text: &str) -> Result<Cow<'_, str>>;
}
```

Concrete normalizers in v0.1:

- `BertNormalizer` — clean_text, handle_chinese_chars, strip_accents, lowercase.
- `Lowercase` — ASCII lowercase fold.
- `Nfd`, `Nfc`, `Nfkc`, `Nfkd` — Unicode normalization.
- `StripAccents` — strip combining marks.
- `Replace(pattern, content)` — pattern-based string replacement.

## Algorithm

Each normalizer is a pure function over a string slice. They are
designed to compose; the public builder `NormalizerSequence` chains them.

## Performance notes

- Avoid intermediate `String` allocations when possible. `BertNormalizer`
  in particular can stay zero-copy for inputs without Chinese chars.
- The hot path is `encode(text).normalize()`. We benchmark every
  normalizer in isolation in `benches/normalizers.rs`.

## Test strategy

- Golden tests against HF `tokenizers` outputs.
- Property tests: `normalize(x).normalize(x) == normalize(x)`.
- Unicode edge cases (zero-width joiner, combining marks, RTL).

## Known limitations

- No streaming normalizer yet.
- No locale-aware casing.