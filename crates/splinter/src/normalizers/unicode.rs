//! Unicode normalization forms and the NMT cleanup normalizer.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::traits::Normalizer;

/// Unicode canonical decomposition (NFD).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nfd;

impl Normalizer for Nfd {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
    }
}

/// Unicode compatibility decomposition (NFKD).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nfkd;

impl Normalizer for Nfkd {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkd();
        Ok(())
    }
}

/// Unicode canonical composition (NFC).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nfc;

impl Normalizer for Nfc {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfc();
        Ok(())
    }
}

/// Unicode compatibility composition (NFKC).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nfkc;

impl Normalizer for Nfkc {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkc();
        Ok(())
    }
}

/// The cleanup SentencePiece's `nmt_nfkc` rules apply before NFKC:
/// drop most C0 control chars and map assorted spaces/separators
/// (including `▁` itself) to a plain space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nmt;

impl Normalizer for Nmt {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized
            .filter(|c| {
                !matches!(
                    c as u32,
                    0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F
                )
            })
            .map(|c| match c as u32 {
                0x0009
                | 0x000A
                | 0x000C
                | 0x000D
                | 0x1680
                | 0x200B..=0x200F
                | 0x2028
                | 0x2029
                | 0x2581
                | 0xFEFF
                | 0xFFFD => ' ',
                _ => c,
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized_string::OffsetRange;

    fn norm<N: Normalizer>(n: N, s: &str) -> String {
        let mut ns = NormalizedString::from(s);
        n.normalize(&mut ns).unwrap();
        ns.get().to_owned()
    }

    // Expected outputs from Python `tokenizers` 0.23.2.
    #[test]
    fn forms_match_hf() {
        assert_eq!(norm(Nfkd, "ﬁ ½ ①"), "fi 1\u{2044}2 1");
        assert_eq!(norm(Nfkc, "ﬁ ½ ①"), "fi 1\u{2044}2 1");
        assert_eq!(norm(Nfkc, "你好，世界！"), "你好,世界!");
        assert_eq!(norm(Nfkc, "™\u{1e}g"), "TM\u{1e}g");
        assert_eq!(norm(Nfkc, "and\u{a0}nbsp"), "and nbsp");
        assert_eq!(norm(Nfc, "e\u{301} x"), "\u{e9} x");
        assert_eq!(norm(Nfd, "\u{e9}"), "e\u{301}");
    }

    #[test]
    fn nmt_matches_hf() {
        assert_eq!(
            norm(Nmt, "a\u{1}b\tc\u{200b}d\u{2581}e\u{feff}"),
            "ab c d e "
        );
        assert_eq!(norm(Nmt, "x\u{7f}y"), "xy");
    }

    #[test]
    fn nfkc_expansion_offsets() {
        let mut ns = NormalizedString::from("a™b");
        Nfkc.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), "aTMb");
        // Both "T" and "M" come from "™" (original bytes 1..4).
        assert_eq!(
            ns.convert_offsets(OffsetRange::Normalized(1..2)),
            Some(1..4)
        );
        assert_eq!(
            ns.convert_offsets(OffsetRange::Normalized(2..3)),
            Some(1..4)
        );
        assert_eq!(
            ns.get_range_original(OffsetRange::Normalized(3..4)),
            Some("b")
        );
    }
}
