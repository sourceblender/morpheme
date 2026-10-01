//! Whitespace stripping and accent removal.

use serde::{Deserialize, Serialize};
use unicode_normalization_alignments::char::is_combining_mark;

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::traits::Normalizer;

/// Remove leading and/or trailing whitespace.
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
/// use morpheme::normalizers::Strip;
///
/// assert_eq!(normalize(&Strip::new(true, true), "  hi  "), "hi");
/// assert_eq!(normalize(&Strip::new(false, true), "  hi  "), "  hi");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Strip {
    /// Strip leading whitespace.
    pub strip_left: bool,
    /// Strip trailing whitespace.
    pub strip_right: bool,
}

impl Strip {
    /// Build a `Strip` normalizer.
    pub fn new(strip_left: bool, strip_right: bool) -> Self {
        Self {
            strip_left,
            strip_right,
        }
    }
}

impl Normalizer for Strip {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        match (self.strip_left, self.strip_right) {
            (true, true) => {
                normalized.strip();
            }
            (true, false) => {
                normalized.lstrip();
            }
            (false, true) => {
                normalized.rstrip();
            }
            (false, false) => {}
        }
        Ok(())
    }
}

/// Remove combining marks. Usually preceded by [`super::Nfd`] or
/// [`super::Nfkd`] so accented letters are decomposed first.
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
/// use morpheme::normalizers::{Nfd, Sequence, StripAccents};
///
/// // Accents are separate combining marks only after NFD.
/// let strip = Sequence::new(vec![Nfd.into(), StripAccents.into()]);
/// assert_eq!(normalize(&strip, "café naïve"), "cafe naive");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StripAccents;

impl Normalizer for StripAccents {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.filter(|c| !is_combining_mark(c));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;
    use crate::normalizers::Nfd;

    fn norm<N: Normalizer>(n: &N, s: &str) -> String {
        let mut ns = NormalizedString::from(s);
        n.normalize(&mut ns).unwrap();
        ns.get().to_owned()
    }

    // Expected outputs from Python `tokenizers` 0.23.2.
    #[test]
    fn strip_matches_hf() {
        assert_eq!(norm(&Strip::new(true, true), "  ab  "), "ab");
        assert_eq!(norm(&Strip::new(true, true), "\t\n x \u{3000}"), "x");
        assert_eq!(norm(&Strip::new(true, false), "  ab  "), "ab  ");
        assert_eq!(norm(&Strip::new(false, true), "  ab  "), "  ab");
        assert_eq!(norm(&Strip::new(true, true), "   "), "");
    }

    #[test]
    fn strip_offsets() {
        let mut ns = NormalizedString::from("  ab  ");
        Strip::new(true, true).normalize(&mut ns).unwrap();
        assert_eq!(
            ns.convert_offsets(OffsetRange::Normalized(0..2)),
            Some(2..4)
        );
    }

    #[test]
    fn strip_accents_after_nfd_matches_hf() {
        let cases = [
            (
                "Café naïve résumé façade Ångström",
                "Cafe naive resume facade Angstrom",
            ),
            ("ÀÉÎÕÜ àéîõü", "AEIOU aeiou"),
            ("Россия и Українa", "Россия и Украінa"),
            ("†Р Ġbyte", "†Р Gbyte"),
            ("İa", "Ia"),
            ("Emoji: 😀👍🏽", "Emoji: 😀👍🏽"),
        ];
        for (input, expected) in cases {
            let mut ns = NormalizedString::from(input);
            Nfd.normalize(&mut ns).unwrap();
            StripAccents.normalize(&mut ns).unwrap();
            assert_eq!(ns.get(), expected, "input {input:?}");
        }
    }

    #[test]
    fn strip_accents_offsets() {
        let mut ns = NormalizedString::from("é!");
        Nfd.normalize(&mut ns).unwrap();
        StripAccents.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "e!");
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(0..1)),
            Some("é")
        );
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(1..2)),
            Some("!")
        );
    }
}
