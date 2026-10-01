//! Join all tokens into one.

use crate::error::Result;
use crate::traits::Decoder;

/// Concatenates every token into a single one.
///
/// # Example
///
/// ```
/// use splinter::decoders::Fuse;
/// use splinter::Decoder;
///
/// assert_eq!(Fuse::new().decode_chain(vec!["hel".to_string(), "lo".to_string()])?, ["hello"]);
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fuse;

crate::pre_tokenizers::impl_unit_serde!(Fuse);

impl Fuse {
    /// Build a `Fuse` decoder.
    pub fn new() -> Self {
        Self
    }
}

impl Decoder for Fuse {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(vec![tokens.concat()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuses() {
        let out = Fuse
            .decode_chain(vec!["Hey".into(), " friend".into()])
            .unwrap();
        assert_eq!(out, vec!["Hey friend"]);
    }
}
