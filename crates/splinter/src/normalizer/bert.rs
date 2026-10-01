//! `BertNormalizer` — the BERT-style cleaner used by the WordPiece
//! pipeline. Combines the four cleaning steps `tokenizers` exposes:
//!
//! - `clean_text` — strip control characters, optionally replace
//!   whitespace runs with a single space, and (optionally) remove
//!   accent-free diacritics.
//! - `handle_chinese_chars` — pad every CJK character with spaces.
//! - `strip_accents` — strip combining marks (independent of
//!   `lowercase`).
//! - `lowercase` — ASCII lowercase.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

use super::Normalizer;
use crate::error::Result;

/// Configuration for [`BertNormalizer`]. Mirrors the HF `BertNormalizer`
/// argument set.
#[derive(Debug, Clone)]
pub struct BertNormalizerOpts {
    /// Strip control characters (`\t`, `\n`, `\r`, `\0`, `\u000b`,
    /// `\u000c`) and replace every whitespace run with a single space.
    pub clean_text: bool,
    /// Pad every CJK character with spaces. BERT-style.
    pub handle_chinese_chars: bool,
    /// Strip combining marks. Independent of `lowercase`.
    pub strip_accents: bool,
    /// ASCII lowercase.
    pub lowercase: bool,
}

impl Default for BertNormalizerOpts {
    fn default() -> Self {
        // Defaults match the BERT WordPiece preprocessing pipeline.
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: true,
            lowercase: true,
        }
    }
}

/// BERT-style normalizer. See module docs of [`crate::normalizer`].
#[derive(Debug, Clone, Default)]
pub struct BertNormalizer(BertNormalizerOpts);

impl BertNormalizer {
    /// Build a `BertNormalizer` from explicit options.
    pub fn new(opts: BertNormalizerOpts) -> Self {
        Self(opts)
    }

    /// Builder-style: turn lowercase folding on or off.
    pub fn with_lowercase(mut self, yes: bool) -> Self {
        self.0.lowercase = yes;
        self
    }

    /// Builder-style: turn `clean_text` on or off.
    pub fn with_clean_text(mut self, yes: bool) -> Self {
        self.0.clean_text = yes;
        self
    }

    /// Builder-style: turn `strip_accents` on or off.
    pub fn with_strip_accents(mut self, yes: bool) -> Self {
        self.0.strip_accents = yes;
        self
    }

    /// Builder-style: turn `handle_chinese_chars` on or off.
    pub fn with_handle_chinese_chars(mut self, yes: bool) -> Self {
        self.0.handle_chinese_chars = yes;
        self
    }
}

fn is_cjk(c: char) -> bool {
    let u = c as u32;
    // CJK Unified Ideographs, Extension A–F, Compatibility Ideographs,
    // and the Hangul Syllables block — covers everything BERT's
    // `handle_chinese_chars` is documented to handle.
    matches!(u,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2EBEF
        | 0xF900..=0xFAFF
        | 0xAC00..=0xD7A3
    )
}

fn is_control_char(c: char) -> bool {
    let u = c as u32;
    // The control chars BERT's cleaner drops: \t, \n, \r, \0, \v, \f,
    // and 0x7F (DEL).
    matches!(u, 0x00 | 0x09 | 0x0A | 0x0B | 0x0C | 0x0D | 0x7F)
}

fn is_ascii_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn unicode_general_category_is_mn(c: char) -> bool {
    // We avoid pulling in the `unicode-general-category` crate by
    // enumerating the Mn blocks ourselves. Covers the Latin, Greek,
    // Cyrillic, Hebrew, Arabic, and the common punctuation range.
    let u = c as u32;
    matches!(u,
        0x0300..=0x036F
        | 0x0483..=0x0489
        | 0x0591..=0x05BD
        | 0x05BF
        | 0x05C1..=0x05C2
        | 0x05C4..=0x05C5
        | 0x05C7
        | 0x0610..=0x061A
        | 0x064B..=0x065F
        | 0x0670
        | 0x06D6..=0x06DC
        | 0x06DF..=0x06E4
        | 0x06E7..=0x06E8
        | 0x06EA..=0x06ED
        | 0x200B..=0x200F
        | 0x202A..=0x202E
        | 0x2060..=0x2064
        | 0x206A..=0x206F
    )
}

impl Normalizer for BertNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Result<Cow<'a, str>> {
        let opts = &self.0;

        // Fast path: scan for any work. If we don't need to mutate
        // anything, return a borrow.
        let mut needs_work = false;
        for ch in text.chars() {
            if opts.clean_text && (is_control_char(ch) || is_ascii_whitespace(ch as u8)) {
                needs_work = true;
                break;
            }
            if opts.handle_chinese_chars && is_cjk(ch) {
                needs_work = true;
                break;
            }
            if opts.lowercase && ch.is_uppercase() {
                needs_work = true;
                break;
            }
            if opts.strip_accents {
                // We can't cheaply tell without iterating; assume
                // any accented char would trigger work.
                if ch as u32 > 0x7F {
                    needs_work = true;
                    break;
                }
            }
        }
        if !needs_work && !opts.strip_accents {
            return Ok(Cow::Borrowed(text));
        }
        if !needs_work && opts.strip_accents {
            // strip_accents only triggers work if some char has marks.
            let any_marks = text.chars().any(|c| {
                // NFD once and check if any combining marks drop.
                let mut count = 0;
                for d in c.nfd() {
                    if unicode_general_category_is_mn(d) {
                        count += 1;
                    }
                }
                count > 0
            });
            if !any_marks {
                return Ok(Cow::Borrowed(text));
            }
        }

        let mut chars = text.chars().peekable();
        let mut out = String::with_capacity(text.len());
        let mut prev_space = false;
        while let Some(ch) = chars.next() {
            let next_is_cjk_or_text = chars
                .peek()
                .map(|&c| !is_control_char(c) && !is_ascii_whitespace(c as u8))
                .unwrap_or(false);
            if opts.clean_text && is_control_char(ch) {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
                continue;
            }
            if opts.handle_chinese_chars && is_cjk(ch) {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
                out.push(ch);
                if next_is_cjk_or_text {
                    out.push(' ');
                    prev_space = true;
                }
                continue;
            }
            if opts.clean_text && is_ascii_whitespace(ch as u8) {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
                continue;
            }
            prev_space = false;

            let mut ch = ch;
            if opts.strip_accents {
                let decomposed: String = ch.nfd().collect();
                let stripped: String = decomposed
                    .chars()
                    .filter(|c| !unicode_general_category_is_mn(*c))
                    .collect();
                if stripped != decomposed {
                    // Keep the base char; we lose any further
                    // stripping beyond the first base, which matches
                    // what `tokenizers` does for BERT.
                    ch = stripped.chars().next().unwrap_or(ch);
                }
            }
            if opts.lowercase {
                for low in ch.to_lowercase() {
                    out.push(low);
                }
            } else {
                out.push(ch);
            }
        }

        Ok(Cow::Owned(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lowercases() {
        // The default BertNormalizer enables `lowercase`. Whitespace
        // runs are collapsed and `clean_text` drops control chars;
        // punctuation stays attached to neighboring word chars in
        // the *output* (the surrounding spaces are pre-tokenizer
        // concerns, not normalizer concerns).
        let n = BertNormalizer::default();
        let out = n.normalize("Hello, World!").unwrap();
        assert_eq!(out, "hello, world!");
    }

    #[test]
    fn no_lowercase_keeps_case() {
        let n = BertNormalizer::new(BertNormalizerOpts {
            clean_text: false,
            handle_chinese_chars: false,
            strip_accents: false,
            lowercase: false,
        });
        let out = n.normalize("Hello, World!").unwrap();
        assert!(out.contains("Hello"));
        assert!(out.contains("World"));
    }

    #[test]
    fn clean_text_collapses_whitespace() {
        let n = BertNormalizer::default();
        let out = n.normalize("a\t\tb").unwrap();
        assert_eq!(out, "a b");
    }

    #[test]
    fn cjk_padding() {
        let n = BertNormalizer::default();
        let out = n.normalize("hi你好").unwrap();
        assert_eq!(out, "hi 你 好");
    }

    #[test]
    fn lowercase_only() {
        let n = BertNormalizer::new(BertNormalizerOpts {
            clean_text: false,
            handle_chinese_chars: false,
            strip_accents: false,
            lowercase: true,
        });
        assert_eq!(n.normalize("Hello").unwrap(), "hello");
    }
}
