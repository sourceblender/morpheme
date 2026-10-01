//! Prepend a string to non-empty input.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::traits::Normalizer;

/// Prepend `prepend` to the text (empty text is left alone). Used by
/// Llama-style tokenizers to add the leading `▁`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prepend {
    /// The string to prepend.
    pub prepend: String,
}

impl Prepend {
    /// Build a `Prepend` normalizer.
    pub fn new(prepend: String) -> Self {
        Self { prepend }
    }
}

impl Normalizer for Prepend {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            normalized.prepend(&self.prepend);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;

    #[test]
    fn prepend_matches_hf() {
        let p = Prepend::new("▁".into());
        let mut ns = NormalizedString::from("hey");
        p.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "▁hey");
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(0..4)),
            Some("h")
        );

        let mut empty = NormalizedString::from("");
        p.normalize(&mut empty).unwrap();
        assert_eq!(empty.get(), "");
    }
}
