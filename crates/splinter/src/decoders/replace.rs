//! Replace a pattern in every token.

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;
use crate::pattern::{Pattern, SplitPattern, SysRegex};
use crate::traits::Decoder;

/// Replaces every match of `pattern` in each token with `content`
/// (e.g. `▁` → `" "` for SentencePiece models).
#[derive(Debug, Clone, Serialize)]
pub struct Replace {
    /// What to replace.
    pub pattern: SplitPattern,
    /// The replacement.
    pub content: String,
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
    pub fn new(pattern: SplitPattern, content: impl Into<String>) -> Result<Self> {
        let regex = pattern.to_regex()?;
        Ok(Self {
            pattern,
            content: content.into(),
            regex,
        })
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
