//! Replace every match of a pattern with a string.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::pattern::{SplitPattern, SysRegex};
use crate::traits::Normalizer;

#[derive(Deserialize)]
struct ReplaceConfig {
    pattern: SplitPattern,
    content: String,
}

impl TryFrom<ReplaceConfig> for Replace {
    type Error = crate::Error;

    fn try_from(c: ReplaceConfig) -> Result<Self> {
        Replace::new(c.pattern, c.content)
    }
}

/// Replace every match of `pattern` (a literal string or a regex) with
/// `content`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "ReplaceConfig")]
pub struct Replace {
    pattern: SplitPattern,
    content: String,
    #[serde(skip)]
    regex: SysRegex,
}

impl PartialEq for Replace {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.content == other.content
    }
}

impl Replace {
    /// Build a `Replace` normalizer. Fails if the regex is invalid.
    pub fn new(pattern: impl Into<SplitPattern>, content: impl Into<String>) -> Result<Self> {
        let pattern = pattern.into();
        let regex = pattern.to_regex()?;
        Ok(Self {
            pattern,
            content: content.into(),
            regex,
        })
    }

    /// The pattern being replaced.
    pub fn pattern(&self) -> &SplitPattern {
        &self.pattern
    }

    /// The replacement string.
    pub fn content(&self) -> &str {
        &self.content
    }
}

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.replace(&self.regex, &self.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;

    fn norm(r: &Replace, s: &str) -> String {
        let mut ns = NormalizedString::from(s);
        r.normalize(&mut ns).unwrap();
        ns.get().to_owned()
    }

    // Expected outputs from Python `tokenizers` 0.23.2.
    #[test]
    fn string_and_regex_match_hf() {
        let r = Replace::new(SplitPattern::String(" ".into()), "▁").unwrap();
        assert_eq!(norm(&r, "Hey friend  x"), "Hey▁friend▁▁x");
        let r = Replace::new(SplitPattern::Regex(r"\s+".into()), " ").unwrap();
        assert_eq!(norm(&r, "a  b\t\tc"), "a b c");
        // Regex metacharacters in a String pattern are literal.
        let r = Replace::new(SplitPattern::String("a.b".into()), "X").unwrap();
        assert_eq!(norm(&r, "a.b axb"), "X axb");
        let r = Replace::new(SplitPattern::String("``".into()), "\"").unwrap();
        assert_eq!(norm(&r, "``hi''"), "\"hi''");
    }

    #[test]
    fn replacement_aligns_to_match() {
        let r = Replace::new(SplitPattern::String(" ".into()), "▁").unwrap();
        let mut ns = NormalizedString::from("a b");
        r.normalize(&mut ns).unwrap();
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(1..4)),
            Some(" ")
        );
    }

    #[test]
    fn invalid_regex_errors() {
        assert!(Replace::new(SplitPattern::Regex("(".into()), "").is_err());
    }
}
