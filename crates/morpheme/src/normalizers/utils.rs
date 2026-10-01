//! `Lowercase` and `Sequence`.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::normalizers::NormalizerWrapper;
use crate::pre_tokenizers::impl_unit_serde;
use crate::traits::Normalizer;

/// Apply several normalizers in order.
///
/// # Example
///
/// ```
/// use morpheme::{NormalizedString, Normalizer};
///
/// fn normalize(n: &impl Normalizer, text: &str) -> String {
///     let mut s = NormalizedString::from(text);
///     n.normalize(&mut s).unwrap();
///     s.get().to_owned()
/// }
///
/// use morpheme::normalizers::{Lowercase, Nfd, Sequence, StripAccents};
///
/// let n = Sequence::new(vec![Nfd.into(), StripAccents.into(), Lowercase.into()]);
/// assert_eq!(normalize(&n, "Ångström"), "angstrom");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    normalizers: Vec<NormalizerWrapper>,
}

impl Sequence {
    /// Build a sequence.
    pub fn new(normalizers: Vec<NormalizerWrapper>) -> Self {
        Self { normalizers }
    }
}

impl AsRef<[NormalizerWrapper]> for Sequence {
    fn as_ref(&self) -> &[NormalizerWrapper] {
        &self.normalizers
    }
}

impl AsMut<[NormalizerWrapper]> for Sequence {
    fn as_mut(&mut self) -> &mut [NormalizerWrapper] {
        &mut self.normalizers
    }
}

impl IntoIterator for Sequence {
    type Item = NormalizerWrapper;
    type IntoIter = std::vec::IntoIter<NormalizerWrapper>;

    fn into_iter(self) -> Self::IntoIter {
        self.normalizers.into_iter()
    }
}

impl Normalizer for Sequence {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        for n in &self.normalizers {
            n.normalize(normalized)?;
        }
        Ok(())
    }
}

/// Unicode-aware lowercase (one char may expand into several).
///
/// # Example
///
/// ```
/// use morpheme::{NormalizedString, Normalizer};
///
/// fn normalize(n: &impl Normalizer, text: &str) -> String {
///     let mut s = NormalizedString::from(text);
///     n.normalize(&mut s).unwrap();
///     s.get().to_owned()
/// }
///
/// use morpheme::normalizers::Lowercase;
///
/// assert_eq!(normalize(&Lowercase, "HeLLo ÀÉ"), "hello àé");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lowercase;

impl_unit_serde!(Lowercase);

impl Normalizer for Lowercase {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.lowercase();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;
    use crate::normalizers::{Prepend, Replace};
    use crate::pattern::SplitPattern;

    fn norm<N: Normalizer>(n: &N, s: &str) -> String {
        let mut ns = NormalizedString::from(s);
        n.normalize(&mut ns).unwrap();
        ns.get().to_owned()
    }

    // Expected outputs from Python `tokenizers` 0.23.2.
    #[test]
    fn lowercase_matches_hf() {
        assert_eq!(norm(&Lowercase, "İa"), "i\u{307}a");
        assert_eq!(norm(&Lowercase, "ΑΣ HELLO"), "ασ hello");
    }

    #[test]
    fn llama_sequence_matches_hf() {
        let seq = Sequence::new(vec![
            Prepend::new("▁").into(),
            Replace::new(SplitPattern::String(" ".into()), "▁")
                .unwrap()
                .into(),
        ]);
        assert_eq!(norm(&seq, "Hey friend!"), "▁Hey▁friend!");
        assert_eq!(norm(&seq, " x"), "▁▁x");
        assert_eq!(norm(&seq, ""), "");

        let mut ns = NormalizedString::from("Hey friend!");
        seq.normalize(&mut ns).unwrap();
        // The leading "▁" is aligned to the first char of the original.
        assert_eq!(
            ns.convert_offsets(OffsetRange::Normalized(0..3)),
            Some(0..1)
        );
        // "▁friend" (normalized 6..15) is " friend" in the original.
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(6..15)),
            Some(" friend")
        );
    }
}
