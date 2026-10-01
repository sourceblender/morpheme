//! CTC decoder (speech models such as Wav2Vec2).

use serde::{Deserialize, Serialize};

use super::wordpiece::cleanup;
use crate::error::Result;
use crate::traits::Decoder;

/// Collapses repeated tokens, removes `pad_token`, and (with `cleanup`)
/// turns `word_delimiter_token` into spaces.
///
/// # Example
///
/// ```
/// use morpheme::decoders::Ctc;
/// use morpheme::Decoder;
///
/// // Speech-model output: repeats collapse, pads drop, `|` separates words.
/// let d = Ctc::new("<pad>", "|", true);
/// assert_eq!(d.decode(vec!["h".to_string(), "h".to_string(), "<pad>".to_string(), "i".to_string(), "|".to_string(), "y".to_string(), "o".to_string(), "<pad>".to_string(), "u".to_string()])?, "hi you");
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Ctc {
    /// The CTC blank / padding token.
    pub pad_token: String,
    /// The word delimiter token.
    pub word_delimiter_token: String,
    /// Apply cleanup (spacing fixes, delimiter → space).
    pub cleanup: bool,
}

impl Default for Ctc {
    fn default() -> Self {
        Self::new("<pad>", "|", true)
    }
}

impl Ctc {
    /// Build a `Ctc` decoder.
    pub fn new(
        pad_token: impl Into<String>,
        word_delimiter_token: impl Into<String>,
        cleanup: bool,
    ) -> Self {
        Self {
            pad_token: pad_token.into(),
            word_delimiter_token: word_delimiter_token.into(),
            cleanup,
        }
    }
}

impl Decoder for Ctc {
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        tokens.dedup();
        Ok(tokens
            .into_iter()
            .filter_map(|t| {
                let mut r = t.replace(&self.pad_token, "");
                if self.cleanup {
                    r = cleanup(&r).replace(&self.word_delimiter_token, " ");
                }
                (!r.is_empty()).then_some(r)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes() {
        let tokens = "<pad> <pad> h e e l l <pad> l o o o <pad> | w o r l d"
            .split(' ')
            .map(String::from)
            .collect();
        assert_eq!(Ctc::default().decode(tokens).unwrap(), "hello world");
    }
}
