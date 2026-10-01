//! BERT's normalizer.

use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::traits::Normalizer;

/// BERT whitespace: `\t`, `\n`, `\r` and anything Unicode calls
/// whitespace.
pub(crate) fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

/// BERT control chars: Unicode "Other" categories, except the
/// whitespace controls `\t`, `\n`, `\r`.
pub(crate) fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => c.is_other(),
    }
}

/// CJK ideographs (the ranges BERT's original implementation uses —
/// notably *not* Hangul, Hiragana or Katakana).
pub(crate) fn is_chinese_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// BERT's normalizer: optional control-char cleanup, spacing around CJK
/// ideographs, accent stripping and lowercasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BertNormalizer {
    /// Remove control chars and map all whitespace to a plain space.
    pub clean_text: bool,
    /// Put spaces around CJK ideographs.
    pub handle_chinese_chars: bool,
    /// Strip accents. `None` means "same as `lowercase`".
    pub strip_accents: Option<bool>,
    /// Lowercase the text.
    pub lowercase: bool,
}

impl Default for BertNormalizer {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    }
}

impl BertNormalizer {
    /// Build a BERT normalizer.
    pub fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> Self {
        Self {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        }
    }

    fn do_clean_text(&self, normalized: &mut NormalizedString) {
        normalized
            .filter(|c| !(c == '\0' || c == '\u{fffd}' || is_control(c)))
            .map(|c| if is_whitespace(c) { ' ' } else { c });
    }

    fn do_handle_chinese_chars(&self, normalized: &mut NormalizedString) {
        let mut dest: Vec<(char, isize)> = Vec::with_capacity(normalized.len());
        for c in normalized.get().chars() {
            if is_chinese_char(c) {
                dest.extend([(' ', 0), (c, 1), (' ', 1)]);
            } else {
                dest.push((c, 0));
            }
        }
        normalized.transform(dest, 0);
    }

    fn do_strip_accents(&self, normalized: &mut NormalizedString) {
        normalized.nfd().filter(|c| !c.is_mark_nonspacing());
    }
}

impl Normalizer for BertNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.clean_text {
            self.do_clean_text(normalized);
        }
        if self.handle_chinese_chars {
            self.do_handle_chinese_chars(normalized);
        }
        if self.strip_accents.unwrap_or(self.lowercase) {
            self.do_strip_accents(normalized);
        }
        if self.lowercase {
            normalized.lowercase();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;

    fn norm(n: &BertNormalizer, s: &str) -> String {
        let mut ns = NormalizedString::from(s);
        n.normalize(&mut ns).unwrap();
        ns.get().to_owned()
    }

    // Expected outputs obtained from Python `tokenizers` 0.23.2:
    // normalizers.BertNormalizer(...).normalize_str(...)
    #[test]
    fn uncased_matches_hf() {
        let n = BertNormalizer::default();
        let cases = [
            (
                "Café naïve résumé façade Ångström",
                "cafe naive resume facade angstrom",
            ),
            ("你好，世界！", " 你  好 ， 世  界 ！"),
            (
                "한국어 텍스트",
                // NFD decomposes Hangul syllables into jamo.
                "\u{1112}\u{1161}\u{11ab}\u{1100}\u{116e}\u{11a8}\u{110b}\u{1165} \u{1110}\u{1166}\u{11a8}\u{1109}\u{1173}\u{1110}\u{1173}",
            ),
            ("Россия и Українa", "россия и украінa"),
            ("ÀÉÎÕÜ àéîõü", "aeiou aeiou"),
            ("İa", "ia"),
            ("™\u{1e}g", "™g"),
            ("ﬁ ½ ①", "ﬁ ½ ①"),
            ("control\u{0}char\u{7}bell", "controlcharbell"),
            ("zero\u{200b}width and\u{a0}nbsp", "zerowidth and nbsp"),
            ("tabs\tand\nnewlines\r\nmixed", "tabs and newlines  mixed"),
            ("Emoji: 😀👍🏽", "emoji: 😀👍🏽"),
            ("e\u{301} combining", "e combining"),
            ("†Р Ġbyte", "†р gbyte"),
        ];
        for (input, expected) in cases {
            assert_eq!(norm(&n, input), expected, "input {input:?}");
        }
    }

    #[test]
    fn cased_matches_hf() {
        let n = BertNormalizer::new(true, true, None, false);
        let cases = [
            ("Café naïve", "Café naïve"),
            ("你好", " 你  好 "),
            ("ÀÉÎÕÜ àéîõü", "ÀÉÎÕÜ àéîõü"),
        ];
        for (input, expected) in cases {
            assert_eq!(norm(&n, input), expected, "input {input:?}");
        }
    }

    #[test]
    fn explicit_strip_accents_overrides_lowercase() {
        let n = BertNormalizer::new(true, true, Some(true), false);
        assert_eq!(norm(&n, "Café"), "Cafe");
        let n = BertNormalizer::new(true, true, Some(false), true);
        assert_eq!(norm(&n, "Café"), "café");
    }

    #[test]
    fn offsets_map_back_to_original() {
        let n = BertNormalizer::default();
        let mut ns = NormalizedString::from("Ab你c");
        n.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "ab 你 c");
        // The ideograph (normalized 3..6) maps to original 2..5.
        assert_eq!(
            ns.convert_offsets(OffsetRange::Normalized(3..6)),
            Some(2..5)
        );
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(0..2)),
            Some("Ab")
        );
    }

    #[test]
    fn missing_fields_take_defaults() {
        let n: BertNormalizer = serde_json::from_str(r#"{"lowercase": false}"#).unwrap();
        assert_eq!(n, BertNormalizer::new(true, true, None, false));
    }
}
