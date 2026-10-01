//! Split where the Unicode script changes (SentencePiece behavior).

mod scripts;

use crate::error::{Error, Result};
use crate::normalized_string::OffsetRange;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Splits a piece wherever the script of consecutive chars changes.
/// Spaces belong to any script; Hiragana, Katakana and `ー` count as Han
/// (as in SentencePiece).
///
/// # Example
///
/// ```
/// use morpheme::{OffsetType, PreTokenizedString, PreTokenizer};
///
/// fn split(pt: &impl PreTokenizer, text: &str) -> Vec<String> {
///     let mut s = PreTokenizedString::from(text);
///     pt.pre_tokenize(&mut s).unwrap();
///     s.get_splits(OffsetType::Byte).into_iter().map(|(w, _, _)| w.to_owned()).collect()
/// }
///
/// use morpheme::pre_tokenizers::UnicodeScripts;
///
/// // Splits where the script changes (Han, Hiragana and Katakana count as one).
/// assert_eq!(split(&UnicodeScripts::new(), "東京abcТест"), ["東京", "abc", "Тест"]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnicodeScripts;

super::impl_unit_serde!(UnicodeScripts);

impl UnicodeScripts {
    /// Build a `UnicodeScripts` pre-tokenizer.
    pub fn new() -> Self {
        Self
    }
}

fn fixed_script(c: char) -> u8 {
    if c == '\u{30FC}' {
        return scripts::HAN;
    }
    if c == ' ' {
        return scripts::ANY;
    }
    match scripts::get_script(c) {
        scripts::HIRAGANA | scripts::KATAKANA => scripts::HAN,
        s => s,
    }
}

impl PreTokenizer for UnicodeScripts {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            let mut last: Option<u8> = None;
            let mut cuts: Vec<usize> = Vec::new();
            for (offset, c) in normalized.get().char_indices() {
                let script = fixed_script(c);
                if script != scripts::ANY && last != Some(scripts::ANY) && last != Some(script) {
                    cuts.push(offset);
                }
                if script != scripts::ANY {
                    last = Some(script);
                }
            }
            cuts.push(normalized.len());
            cuts.windows(2)
                .map(|w| {
                    normalized
                        .slice(OffsetRange::Normalized(w[0]..w[1]))
                        .ok_or_else(|| Error::PreTokenizer("UnicodeScripts: bad slice".into()))
                })
                .collect::<Result<Vec<_>>>()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn scripts_table() {
        assert_eq!(scripts::get_script('京'), scripts::HAN);
        assert_eq!(scripts::get_script('い'), scripts::HIRAGANA);
        assert_eq!(scripts::get_script('グ'), scripts::KATAKANA);
        assert_eq!(scripts::get_script('a'), scripts::get_script('A'));
        assert_eq!(scripts::get_script('0'), scripts::get_script('$'));
        assert_ne!(scripts::get_script('a'), scripts::get_script('0'));
    }

    #[test]
    fn basic() {
        assert_eq!(
            splits(&UnicodeScripts, "どこで生れ。Yes"),
            vec![("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21))]
        );
    }

    #[test]
    fn spaces_are_neutral() {
        assert_eq!(
            splits(&UnicodeScripts, "Apples are りんご 林檎"),
            vec![("Apples are ", (0, 11)), ("りんご 林檎", (11, 27))]
        );
    }
}
