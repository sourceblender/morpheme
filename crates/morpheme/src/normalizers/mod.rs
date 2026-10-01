//! Normalizers: rewrite the input text (lowercasing, Unicode
//! normalization, accent stripping, …) while keeping alignments to the
//! original so offsets stay exact.
//!
//! Every normalizer here matches the Hugging Face `tokenizers`
//! implementation of the same name, both in behavior and in its
//! `tokenizer.json` representation.

mod bert;
mod byte_level;
mod precompiled;
mod prepend;
mod replace;
mod strip;
mod unicode;
mod utils;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::traits::Normalizer;

pub use bert::BertNormalizer;
pub use byte_level::ByteLevel;
pub use precompiled::Precompiled;
pub use prepend::Prepend;
pub use replace::Replace;
pub use strip::{Strip, StripAccents};
pub use unicode::{Nfc, Nfd, Nfkc, Nfkd, Nmt};
pub use utils::{Lowercase, Sequence};

/// Any built-in normalizer. This is what `tokenizer.json`'s
/// `"normalizer"` field (de)serializes to; the `"type"` key selects the
/// variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum NormalizerWrapper {
    /// BERT's normalizer.
    BertNormalizer(BertNormalizer),
    /// Strip whitespace on the left and/or right.
    Strip(Strip),
    /// Remove combining marks.
    StripAccents(StripAccents),
    /// Unicode NFC.
    #[serde(rename = "NFC")]
    Nfc(Nfc),
    /// Unicode NFD.
    #[serde(rename = "NFD")]
    Nfd(Nfd),
    /// Unicode NFKC.
    #[serde(rename = "NFKC")]
    Nfkc(Nfkc),
    /// Unicode NFKD.
    #[serde(rename = "NFKD")]
    Nfkd(Nfkd),
    /// Apply several normalizers in order.
    Sequence(Sequence),
    /// Unicode lowercase.
    Lowercase(Lowercase),
    /// NMT-style control character cleanup.
    Nmt(Nmt),
    /// SentencePiece precompiled character map.
    Precompiled(Precompiled),
    /// Replace a pattern with a string.
    Replace(Replace),
    /// Prepend a string.
    Prepend(Prepend),
    /// Map each byte to its GPT-2 byte-level char.
    ByteLevel(ByteLevel),
}

impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        match self {
            NormalizerWrapper::BertNormalizer(n) => n.normalize(normalized),
            NormalizerWrapper::Strip(n) => n.normalize(normalized),
            NormalizerWrapper::StripAccents(n) => n.normalize(normalized),
            NormalizerWrapper::Nfc(n) => n.normalize(normalized),
            NormalizerWrapper::Nfd(n) => n.normalize(normalized),
            NormalizerWrapper::Nfkc(n) => n.normalize(normalized),
            NormalizerWrapper::Nfkd(n) => n.normalize(normalized),
            NormalizerWrapper::Sequence(n) => n.normalize(normalized),
            NormalizerWrapper::Lowercase(n) => n.normalize(normalized),
            NormalizerWrapper::Nmt(n) => n.normalize(normalized),
            NormalizerWrapper::Precompiled(n) => n.normalize(normalized),
            NormalizerWrapper::Replace(n) => n.normalize(normalized),
            NormalizerWrapper::Prepend(n) => n.normalize(normalized),
            NormalizerWrapper::ByteLevel(n) => n.normalize(normalized),
        }
    }
}

macro_rules! impl_from {
    ($($ty:ident => $variant:ident),* $(,)?) => {
        $(
            impl From<$ty> for NormalizerWrapper {
                fn from(n: $ty) -> Self {
                    NormalizerWrapper::$variant(n)
                }
            }
        )*
    };
}

impl_from!(
    BertNormalizer => BertNormalizer,
    Strip => Strip,
    StripAccents => StripAccents,
    Nfc => Nfc,
    Nfd => Nfd,
    Nfkc => Nfkc,
    Nfkd => Nfkd,
    Sequence => Sequence,
    Lowercase => Lowercase,
    Nmt => Nmt,
    Precompiled => Precompiled,
    Replace => Replace,
    Prepend => Prepend,
    ByteLevel => ByteLevel,
);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn roundtrip(v: Value) {
        let n: NormalizerWrapper = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(serde_json::to_value(&n).unwrap(), v);
    }

    #[test]
    fn unit_normalizers_roundtrip() {
        for ty in [
            "NFC",
            "NFD",
            "NFKC",
            "NFKD",
            "Lowercase",
            "StripAccents",
            "Nmt",
            "ByteLevel",
        ] {
            roundtrip(json!({ "type": ty }));
        }
    }

    #[test]
    fn configured_normalizers_roundtrip() {
        roundtrip(json!({
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": true,
            "strip_accents": null,
            "lowercase": false
        }));
        roundtrip(json!({"type": "Strip", "strip_left": true, "strip_right": false}));
        roundtrip(json!({"type": "Prepend", "prepend": "▁"}));
        roundtrip(json!({"type": "Replace", "pattern": {"String": " "}, "content": "▁"}));
        roundtrip(json!({"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}));
        roundtrip(json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
            ]
        }));
    }

    #[test]
    fn serialization_field_order_matches_hf() {
        let n: NormalizerWrapper = BertNormalizer::default().into();
        assert_eq!(
            serde_json::to_string(&n).unwrap(),
            r#"{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}"#
        );
        let n: NormalizerWrapper = Lowercase.into();
        assert_eq!(
            serde_json::to_string(&n).unwrap(),
            r#"{"type":"Lowercase"}"#
        );
    }

    #[test]
    fn unknown_type_is_an_error() {
        let err = serde_json::from_value::<NormalizerWrapper>(json!({"type": "Bogus"}));
        assert!(err.is_err());
        let err = serde_json::from_value::<NormalizerWrapper>(json!({"clean_text": true}));
        assert!(err.is_err());
    }

    #[test]
    fn invalid_regex_is_an_error_not_a_panic() {
        let err = serde_json::from_value::<NormalizerWrapper>(
            json!({"type": "Replace", "pattern": {"Regex": "("}, "content": ""}),
        );
        assert!(err.is_err());
    }

    /// Every normalizer config in the real fixture files must load and
    /// re-serialize identically.
    #[test]
    fn real_fixture_configs_roundtrip() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hf");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!(
                "skipping: {} missing (scripts/fetch-hf-fixtures.sh)",
                dir.display()
            );
            return;
        };
        let mut checked = 0;
        for entry in entries {
            let path = entry.unwrap().path();
            let raw = std::fs::read_to_string(&path).unwrap();
            let file: Value = serde_json::from_str(&raw).unwrap();
            let n = &file["normalizer"];
            if n.is_null() {
                continue;
            }
            let parsed: NormalizerWrapper = serde_json::from_value(n.clone())
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(
                &serde_json::to_value(&parsed).unwrap(),
                n,
                "{}",
                path.display()
            );
            checked += 1;
        }
        assert!(checked > 0);
    }
}
