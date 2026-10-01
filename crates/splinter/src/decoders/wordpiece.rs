//! WordPiece (BERT) decoder.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::traits::Decoder;

fn default_prefix() -> String {
    "##".into()
}

fn default_true() -> bool {
    true
}

/// Joins WordPiece tokens: continuation tokens (starting with `prefix`)
/// are glued to the previous token, others are separated by a space.
/// With `cleanup`, removes spaces before punctuation and in common
/// English contractions.
///
/// # Example
///
/// ```
/// use splinter::decoders::WordPiece;
/// use splinter::Decoder;
///
/// let d = WordPiece::new("##", true);
/// assert_eq!(d.decode(vec!["hello".to_string(), "world".to_string(), "##s".to_string(), "!".to_string()])?, "hello worlds!");
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WordPiece {
    /// The continuing-subword prefix (default `##`).
    #[serde(default = "default_prefix")]
    pub prefix: String,
    /// Apply tokenization-space cleanup.
    #[serde(default = "default_true")]
    pub cleanup: bool,
}

impl Default for WordPiece {
    fn default() -> Self {
        Self::new(default_prefix(), true)
    }
}

impl WordPiece {
    /// Build a WordPiece decoder.
    pub fn new(prefix: impl Into<String>, cleanup: bool) -> Self {
        Self {
            prefix: prefix.into(),
            cleanup,
        }
    }
}

/// Undo the spaces a whitespace-joining decoder puts before punctuation
/// and inside English contractions.
pub(crate) fn cleanup(s: &str) -> String {
    s.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

impl Decoder for WordPiece {
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        for (i, token) in tokens.iter_mut().enumerate() {
            if i != 0 {
                *token = match token.strip_prefix(&self.prefix) {
                    Some(rest) => rest.to_owned(),
                    None => format!(" {token}"),
                };
            }
            if self.cleanup {
                *token = cleanup(token);
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes() {
        let d = WordPiece::default();
        let tokens = ["##uelo", "Ara", "##új", "##o", "No", "##guera"];
        let out = d
            .decode(tokens.iter().map(|s| s.to_string()).collect())
            .unwrap();
        assert_eq!(out, "##uelo Araújo Noguera");
    }

    #[test]
    fn cleanup_contractions_and_punctuation() {
        let d = WordPiece::default();
        let tokens = ["i", "do", "n't", "know", ",", "do", "you", "?"];
        let out = d
            .decode(tokens.iter().map(|s| s.to_string()).collect())
            .unwrap();
        assert_eq!(out, "i don't know, do you?");
    }
}
