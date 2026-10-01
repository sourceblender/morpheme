//! Normalizers: rewrite the input text (lowercasing, Unicode
//! normalization, accent stripping, …) while keeping alignments to the
//! original so offsets stay exact.
//!
//! Every normalizer here matches the Hugging Face `tokenizers`
//! implementation of the same name, both in behavior and in its
//! `tokenizer.json` representation.

mod bert;
pub(crate) mod byte_level;
mod precompiled;
mod prepend;
mod replace;
mod strip;
mod unicode;
mod utils;

use serde::de::{self, DeserializeOwned};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

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
///
/// Loading also accepts the legacy untagged forms (no `"type"` key) that
/// Hugging Face `tokenizers` still reads: the variant is inferred from the
/// fields, exactly as HF's untagged fallback does. Saving always writes
/// the tagged form. Unknown keys are ignored, as everywhere else.
#[derive(Debug, Clone, PartialEq, Serialize)]
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

/// The `"type"` names, in the order HF lists them.
const TYPES: &[&str] = &[
    "BertNormalizer",
    "Strip",
    "StripAccents",
    "NFC",
    "NFD",
    "NFKC",
    "NFKD",
    "Sequence",
    "Lowercase",
    "Nmt",
    "Precompiled",
    "Replace",
    "Prepend",
    "ByteLevel",
];

/// Deserialize a component from the remaining keys of its JSON object.
pub(crate) fn from_fields<T: DeserializeOwned, E: de::Error>(
    fields: Map<String, Value>,
) -> std::result::Result<T, E> {
    serde_json::from_value(Value::Object(fields)).map_err(E::custom)
}

/// Deserialize a JSON object, split off its `"type"` key, and hand the
/// rest to `tagged` (type present) or `legacy` (type absent).
pub(crate) fn deserialize_component<'de, D, T>(
    deserializer: D,
    what: &'static str,
    tagged: impl FnOnce(&str, Map<String, Value>) -> std::result::Result<T, D::Error>,
    legacy: impl FnOnce(Map<String, Value>) -> std::result::Result<T, D::Error>,
) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    let Value::Object(mut fields) = value else {
        return Err(de::Error::custom(format!(
            "invalid type: {}, expected a {what} object",
            json_type_name(&value)
        )));
    };
    match fields.remove("type") {
        Some(Value::String(ty)) => tagged(&ty, fields),
        Some(other) => Err(de::Error::custom(format!(
            "invalid type: {}, expected the {what} \"type\" to be a string",
            json_type_name(&other)
        ))),
        None => legacy(fields),
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a sequence",
        Value::Object(_) => "a map",
    }
}

/// Whether every key in `required` is present.
pub(crate) fn has_fields(fields: &Map<String, Value>, required: &[&str]) -> bool {
    required.iter().all(|k| fields.contains_key(*k))
}

impl<'de> Deserialize<'de> for NormalizerWrapper {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserialize_component(
            deserializer,
            "normalizer",
            |ty, fields| {
                Ok(match ty {
                    "BertNormalizer" => Self::BertNormalizer(from_fields(fields)?),
                    "Strip" => Self::Strip(from_fields(fields)?),
                    "StripAccents" => Self::StripAccents(from_fields(fields)?),
                    "NFC" => Self::Nfc(from_fields(fields)?),
                    "NFD" => Self::Nfd(from_fields(fields)?),
                    "NFKC" => Self::Nfkc(from_fields(fields)?),
                    "NFKD" => Self::Nfkd(from_fields(fields)?),
                    "Sequence" => Self::Sequence(from_fields(fields)?),
                    "Lowercase" => Self::Lowercase(from_fields(fields)?),
                    "Nmt" => Self::Nmt(from_fields(fields)?),
                    "Precompiled" => Self::Precompiled(from_fields(fields)?),
                    "Replace" => Self::Replace(from_fields(fields)?),
                    "Prepend" => Self::Prepend(from_fields(fields)?),
                    "ByteLevel" => Self::ByteLevel(from_fields(fields)?),
                    other => return Err(de::Error::unknown_variant(other, TYPES)),
                })
            },
            |fields| {
                // The legacy untagged forms, tried in the order of HF's
                // untagged fallback enum. Only normalizers with fields can
                // be told apart; the parameterless ones (`{}`) cannot, and
                // HF rejects them too.
                Ok(
                    if has_fields(
                        &fields,
                        &["clean_text", "handle_chinese_chars", "lowercase"],
                    ) {
                        Self::BertNormalizer(from_fields(fields)?)
                    } else if has_fields(&fields, &["strip_left", "strip_right"]) {
                        Self::Strip(from_fields(fields)?)
                    } else if has_fields(&fields, &["normalizers"]) {
                        Self::Sequence(from_fields(fields)?)
                    } else if has_fields(&fields, &["precompiled_charsmap"]) {
                        Self::Precompiled(from_fields(fields)?)
                    } else if has_fields(&fields, &["pattern", "content"]) {
                        Self::Replace(from_fields(fields)?)
                    } else if has_fields(&fields, &["prepend"]) {
                        Self::Prepend(from_fields(fields)?)
                    } else {
                        return Err(de::Error::custom(
                            "missing field `type` and the data does not match any legacy \
                             untagged normalizer form",
                        ));
                    },
                )
            },
        )
    }
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
        let err =
            serde_json::from_value::<NormalizerWrapper>(json!({"type": "Bogus"})).unwrap_err();
        assert!(err.to_string().contains("Bogus"), "{err}");
        let err = serde_json::from_value::<NormalizerWrapper>(json!({"clean_text": true}));
        assert!(err.is_err());
        let err = serde_json::from_value::<NormalizerWrapper>(json!({"type": 3}));
        assert!(err.is_err());
        let err = serde_json::from_value::<NormalizerWrapper>(json!("NFC"));
        assert!(err.is_err());
    }

    /// Extra keys are ignored for every normalizer, as for pre-tokenizers
    /// and decoders.
    #[test]
    fn unknown_fields_are_ignored() {
        for ty in TYPES {
            let v = match *ty {
                "BertNormalizer" => json!({
                    "type": ty, "clean_text": true, "handle_chinese_chars": true,
                    "strip_accents": null, "lowercase": true, "x": 1
                }),
                "Strip" => json!({"type": ty, "strip_left": true, "strip_right": true, "x": 1}),
                "Sequence" => json!({"type": ty, "normalizers": [], "x": 1}),
                "Precompiled" => continue,
                "Replace" => {
                    json!({"type": ty, "pattern": {"String": " "}, "content": "_", "x": 1})
                }
                "Prepend" => json!({"type": ty, "prepend": "_", "x": 1}),
                _ => json!({"type": ty, "x": 1}),
            };
            let n: NormalizerWrapper =
                serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("{ty}: {e}"));
            let mut expected = v.as_object().unwrap().clone();
            expected.remove("x");
            assert_eq!(serde_json::to_value(&n).unwrap(), Value::Object(expected));
        }
    }

    // Ported from Hugging Face `tokenizers` 0.23.2
    // (`normalizers/mod.rs`, `post_processor_deserialization_no_type`).
    #[test]
    fn legacy_untagged_forms_load() {
        let json = r#"{"strip_left":false, "strip_right":true}"#;
        let n: NormalizerWrapper = serde_json::from_str(json).unwrap();
        assert!(matches!(n, NormalizerWrapper::Strip(_)));
        // It is written back in the current tagged form.
        assert_eq!(
            serde_json::to_string(&n).unwrap(),
            r#"{"type":"Strip","strip_left":false,"strip_right":true}"#
        );

        // Fields of a pre-tokenizer: not a normalizer.
        let json = r#"{"trim_offsets":true, "add_prefix_space":true}"#;
        let err = serde_json::from_str::<NormalizerWrapper>(json).unwrap_err();
        assert!(err.to_string().contains("legacy untagged"), "{err}");

        let json = r#"{"prepend":"a"}"#;
        let n: NormalizerWrapper = serde_json::from_str(json).unwrap();
        assert!(matches!(n, NormalizerWrapper::Prepend(_)));

        let json = r#"{"clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}"#;
        let n: NormalizerWrapper = serde_json::from_str(json).unwrap();
        assert!(matches!(n, NormalizerWrapper::BertNormalizer(_)));
        // HF requires BertNormalizer's three bool fields in the untagged form.
        let json = r#"{"clean_text":true,"lowercase":true}"#;
        assert!(serde_json::from_str::<NormalizerWrapper>(json).is_err());

        let json = r#"{"pattern":{"String":" "},"content":"▁"}"#;
        let n: NormalizerWrapper = serde_json::from_str(json).unwrap();
        assert!(matches!(n, NormalizerWrapper::Replace(_)));

        let json = r#"{"normalizers":[{"prepend":"▁"},{"type":"NFKC"}]}"#;
        let n: NormalizerWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(
            serde_json::to_string(&n).unwrap(),
            r#"{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"NFKC"}]}"#
        );

        let t5 =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hf/t5-small.json");
        if let Ok(raw) = std::fs::read_to_string(t5) {
            let file: Value = serde_json::from_str(&raw).unwrap();
            let mut legacy = file["normalizer"].as_object().unwrap().clone();
            assert_eq!(legacy.remove("type"), Some(json!("Precompiled")));
            let n: NormalizerWrapper = serde_json::from_value(Value::Object(legacy)).unwrap();
            assert!(matches!(n, NormalizerWrapper::Precompiled(_)));
            assert_eq!(serde_json::to_value(&n).unwrap(), file["normalizer"]);
        }
    }

    // Ported from Hugging Face `tokenizers` 0.23.2
    // (`normalizers/mod.rs`, `normalizer_serialization`).
    #[test]
    fn legacy_untagged_errors_match_hf() {
        let json = r#"{"type":"Sequence","normalizers":[]}"#;
        assert!(serde_json::from_str::<NormalizerWrapper>(json).is_ok());

        // `{}` matches nothing: the parameterless normalizers need a type.
        let json = r#"{"type":"Sequence","normalizers":[{}]}"#;
        let err = serde_json::from_str::<NormalizerWrapper>(json).unwrap_err();
        assert!(err.to_string().contains("legacy untagged"), "{err}");

        let json = r#"{"replacement":"▁","prepend_scheme":"always"}"#;
        let err = serde_json::from_str::<NormalizerWrapper>(json).unwrap_err();
        assert!(err.to_string().contains("legacy untagged"), "{err}");

        let json = r#"{"type":"Sequence","prepend_scheme":"always"}"#;
        let err = serde_json::from_str::<NormalizerWrapper>(json).unwrap_err();
        assert!(
            err.to_string().contains("missing field `normalizers`"),
            "{err}"
        );
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
