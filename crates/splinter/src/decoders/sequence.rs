//! Run several decoders in order.

use serde::{Deserialize, Serialize};

use super::DecoderWrapper;
use crate::error::Result;
use crate::traits::Decoder;

/// Applies each decoder's `decode_chain` in turn.
///
/// # Example
///
/// ```
/// use splinter::decoders::{ByteFallback, Fuse, Replace, Sequence, Strip};
/// use splinter::Decoder;
///
/// // Llama's decoder: `▁` → space, byte fallback, fuse, strip the leading space.
/// let d = Sequence::new(vec![
///     Replace::new("▁", " ")?.into(),
///     ByteFallback::new().into(),
///     Fuse::new().into(),
///     Strip::new(' ', 1, 0).into(),
/// ]);
/// assert_eq!(d.decode(vec!["▁Hello".to_string(), "▁w".to_string(), "<0xC3>".to_string(), "<0xA9>".to_string()])?, "Hello wé");
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    decoders: Vec<DecoderWrapper>,
}

impl Sequence {
    /// Build a sequence.
    pub fn new(decoders: Vec<DecoderWrapper>) -> Self {
        Self { decoders }
    }

    /// The decoders, in order.
    pub fn decoders(&self) -> &[DecoderWrapper] {
        &self.decoders
    }

    /// Mutable access to the decoders.
    pub fn decoders_mut(&mut self) -> &mut Vec<DecoderWrapper> {
        &mut self.decoders
    }
}

impl Decoder for Sequence {
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        for d in &self.decoders {
            tokens = d.decode_chain(tokens)?;
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::{Ctc, Fuse};

    #[test]
    fn chains() {
        let seq = Sequence::new(vec![Ctc::default().into(), Fuse.into()]);
        let tokens = "▁ ▁ H H i i ▁ y o u".split(' ').map(String::from).collect();
        assert_eq!(seq.decode_chain(tokens).unwrap(), vec!["▁Hi▁you"]);
    }
}
