//! Replace a pattern in every token.

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;
use crate::pattern::{Pattern, SplitPattern, SysRegex};
use crate::traits::Decoder;

/// Replaces every match of `pattern` in each token with `content`
/// (e.g. `▁` → `" "` for SentencePiece models).
///
/// # Example
///
/// ```
/// use splinter::decoders::Replace;
/// use splinter::Decoder;
///
/// let d = Replace::new("▁", " ")?;
/// assert_eq!(d.decode(vec!["▁Hello".to_string(), "▁world".to_string()])?, " Hello world");
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct Replace {
    /// What to replace.
    pub(crate) pattern: SplitPattern,
    /// The replacement.
    pub(crate) content: String,
    #[serde(skip)]
    regex: SysRegex,
}

impl PartialEq for Replace {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.content == other.content
    }
}

impl<'de> Deserialize<'de> for Replace {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            pattern: SplitPattern,
            content: String,
        }
        let h = Helper::deserialize(deserializer)?;
        Replace::new(h.pattern, h.content).map_err(serde::de::Error::custom)
    }
}

impl Replace {
    /// Build a `Replace` decoder; fails if the regex does not compile.
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

    /// The replacement text.
    pub fn content(&self) -> &str {
        &self.content
    }
}

impl Decoder for Replace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .into_iter()
            .map(|token| {
                let mut out = String::with_capacity(token.len());
                for ((s, e), is_match) in (&self.regex).find_matches(&token)? {
                    if is_match {
                        out.push_str(&self.content);
                    } else {
                        out.push_str(&token[s..e]);
                    }
                }
                Ok(out)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces() {
        let d = Replace::new(SplitPattern::String("▁".into()), " ").unwrap();
        let out = d
            .decode_chain(vec!["▁Hey".into(), "▁friend".into(), "!".into()])
            .unwrap();
        assert_eq!(out, vec![" Hey", " friend", "!"]);
    }
}
