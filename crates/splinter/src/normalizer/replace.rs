//! `Replace` — simple substring replacement.
//!
//! More powerful than `BertNormalizer::clean_text`'s whitespace
//! collapse: matches an exact string and substitutes a replacement.
//! Use [`RegexReplace`](crate::normalizer::regex_replace) for pattern
//! replacement (lands in Phase 1.2 once `regex` is added to deps).

use std::borrow::Cow;

use super::Normalizer;
use crate::error::Result;

/// Substring-replacement normalizer.
#[derive(Debug, Clone)]
pub struct Replace {
    pattern: String,
    content: String,
}

impl Replace {
    /// Build a `Replace` from `pattern` and `content`.
    pub fn new(pattern: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            content: content.into(),
        }
    }
}

impl Normalizer for Replace {
    fn normalize<'a>(&self, text: &'a str) -> Result<Cow<'a, str>> {
        if self.pattern.is_empty() || !text.contains(&self.pattern) {
            return Ok(Cow::Borrowed(text));
        }
        Ok(Cow::Owned(text.replace(&self.pattern, &self.content)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces() {
        let r = Replace::new("\n", " ");
        assert_eq!(r.normalize("a\nb").unwrap(), "a b");
    }

    #[test]
    fn empty_pattern_passthrough() {
        let r = Replace::new("", "x");
        let out = r.normalize("hello").unwrap();
        assert!(matches!(out, Cow::Borrowed(_)));
    }
}
