//! Strip a char from the edges of every token.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::traits::Decoder;

/// Removes up to `start` leading and `stop` trailing occurrences of
/// `content` from each token.
///
/// # Example
///
/// ```
/// use morpheme::decoders::Strip;
/// use morpheme::Decoder;
///
/// // Remove one `' '` from the start of each token.
/// assert_eq!(Strip::new(' ', 1, 0).decode_chain(vec![" hi".to_string(), "  there".to_string()])?, ["hi", " there"]);
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Strip {
    /// The char to strip.
    pub content: char,
    /// Max occurrences to strip at the start.
    pub start: usize,
    /// Max occurrences to strip at the end.
    pub stop: usize,
}

impl Strip {
    /// Build a `Strip` decoder.
    pub fn new(content: char, start: usize, stop: usize) -> Self {
        Self {
            content,
            start,
            stop,
        }
    }
}

impl Decoder for Strip {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .into_iter()
            .map(|token| {
                let chars: Vec<char> = token.chars().collect();
                let start = chars
                    .iter()
                    .take(self.start)
                    .take_while(|&&c| c == self.content)
                    .count();
                let stop = chars[start..]
                    .iter()
                    .rev()
                    .take(self.stop)
                    .take_while(|&&c| c == self.content)
                    .count();
                chars[start..chars.len() - stop].iter().collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips() {
        let d = Strip::new('H', 1, 0);
        assert_eq!(
            d.decode_chain(vec!["Hey".into(), " friend!".into(), "HHH".into()])
                .unwrap(),
            vec!["ey", " friend!", "HH"]
        );
        let d = Strip::new('y', 0, 1);
        assert_eq!(
            d.decode_chain(vec!["Hey".into(), "yyy".into(), "".into()])
                .unwrap(),
            vec!["He", "yy", ""]
        );
    }
}
