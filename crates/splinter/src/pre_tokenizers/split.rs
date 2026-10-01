//! Split on a string or regex pattern.

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pattern::{Invert, SplitPattern, SysRegex};
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// Splits on `pattern` with the given delimiter `behavior`. With
/// `invert`, the pattern describes the pieces to keep rather than the
/// delimiters.
///
/// # Example
///
/// ```
/// use splinter::{OffsetType, PreTokenizedString, PreTokenizer};
///
/// fn split(pt: &impl PreTokenizer, text: &str) -> Vec<String> {
///     let mut s = PreTokenizedString::from(text);
///     pt.pre_tokenize(&mut s).unwrap();
///     s.get_splits(OffsetType::Byte).into_iter().map(|(w, _, _)| w.to_owned()).collect()
/// }
///
/// use splinter::pattern::SplitPattern;
/// use splinter::pre_tokenizers::Split;
/// use splinter::SplitDelimiterBehavior;
///
/// let numbers = Split::new(SplitPattern::Regex(r"\d+".into()), SplitDelimiterBehavior::Isolated, false)?;
/// assert_eq!(split(&numbers, "ab12cd"), ["ab", "12", "cd"]);
///
/// let words = Split::new(" ", SplitDelimiterBehavior::Removed, false)?;
/// assert_eq!(split(&words, "a b"), ["a", "b"]);
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct Split {
    /// The pattern (read with [`Split::pattern`]; it is compiled once at
    /// construction, so it cannot be changed afterwards).
    pub(crate) pattern: SplitPattern,
    /// What to do with matches.
    pub behavior: SplitDelimiterBehavior,
    /// Invert matches and non-matches.
    pub invert: bool,
    #[serde(skip)]
    regex: SysRegex,
}

impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.behavior == other.behavior
            && self.invert == other.invert
    }
}

impl<'de> Deserialize<'de> for Split {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            pattern: SplitPattern,
            behavior: SplitDelimiterBehavior,
            #[serde(default)]
            invert: bool,
        }
        let h = Helper::deserialize(deserializer)?;
        Split::new(h.pattern, h.behavior, h.invert).map_err(serde::de::Error::custom)
    }
}

impl Split {
    /// Build a `Split`; fails if the regex does not compile.
    pub fn new(
        pattern: impl Into<SplitPattern>,
        behavior: SplitDelimiterBehavior,
        invert: bool,
    ) -> Result<Self> {
        let pattern = pattern.into();
        let regex = pattern.to_regex()?;
        Ok(Self {
            pattern,
            behavior,
            invert,
            regex,
        })
    }

    /// The pattern being split on.
    pub fn pattern(&self) -> &SplitPattern {
        &self.pattern
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        if self.invert {
            pretokenized.split(|_, s| s.split(Invert(&self.regex), self.behavior))
        } else {
            pretokenized.split(|_, s| s.split(&self.regex, self.behavior))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;
    use SplitDelimiterBehavior::*;

    #[test]
    fn behaviors() {
        let s = |b| Split::new(SplitPattern::Regex(r"\s+".into()), b, false).unwrap();
        assert_eq!(
            splits(&s(Removed), "How are  you?"),
            vec![("How", (0, 3)), ("are", (4, 7)), ("you?", (9, 13))]
        );
        assert_eq!(
            splits(&s(MergedWithPrevious), "How are  you?"),
            vec![("How ", (0, 4)), ("are  ", (4, 9)), ("you?", (9, 13))]
        );
        assert_eq!(
            splits(&s(MergedWithNext), "How are  you?"),
            vec![("How", (0, 3)), (" are", (3, 7)), ("  you?", (7, 13))]
        );
    }

    #[test]
    fn invert_keeps_matches() {
        let s = Split::new(SplitPattern::Regex(r"\w+".into()), Removed, true).unwrap();
        assert_eq!(
            splits(&s, "Hey, man"),
            vec![("Hey", (0, 3)), ("man", (5, 8))]
        );
    }

    #[test]
    fn string_pattern_is_literal() {
        let s = Split::new(SplitPattern::String("a.b".into()), Isolated, false).unwrap();
        assert_eq!(
            splits(&s, "xa.bxaxb"),
            vec![("x", (0, 1)), ("a.b", (1, 4)), ("xaxb", (4, 8))]
        );
    }

    #[test]
    fn bad_regex_is_an_error() {
        let json = r#"{"pattern":{"Regex":"("},"behavior":"Isolated","invert":false}"#;
        assert!(serde_json::from_str::<Split>(json).is_err());
    }
}
