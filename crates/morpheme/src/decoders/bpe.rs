//! Decoder for BPE models with an end-of-word suffix.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::traits::Decoder;

fn default_suffix() -> String {
    "</w>".into()
}

/// Replaces the end-of-word `suffix` with a space (nothing on the last
/// token).
///
/// # Example
///
/// ```
/// use morpheme::decoders::BpeDecoder;
/// use morpheme::Decoder;
///
/// let d = BpeDecoder::new("</w>");
/// assert_eq!(d.decode(vec!["hel".to_string(), "lo</w>".to_string(), "world</w>".to_string()])?, "hello world");
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BpeDecoder {
    /// The end-of-word suffix (default `</w>`).
    #[serde(default = "default_suffix")]
    pub suffix: String,
}

impl Default for BpeDecoder {
    fn default() -> Self {
        Self::new(default_suffix())
    }
}

impl BpeDecoder {
    /// Build a `BpeDecoder`.
    pub fn new(suffix: impl Into<String>) -> Self {
        Self {
            suffix: suffix.into(),
        }
    }
}

impl Decoder for BpeDecoder {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let last = tokens.len().saturating_sub(1);
        Ok(tokens
            .into_iter()
            .enumerate()
            .map(|(i, t)| t.replace(&self.suffix, if i == last { "" } else { " " }))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes() {
        let d = BpeDecoder::default();
        let out = d
            .decode(vec!["My</w>".into(), "na".into(), "me</w>".into()])
            .unwrap();
        assert_eq!(out, "My name");
        assert_eq!(d.decode(vec![]).unwrap(), "");
    }
}
