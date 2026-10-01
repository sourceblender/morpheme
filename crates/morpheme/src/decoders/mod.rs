//! Decoders: turn tokens back into text.
//!
//! Decoders compose: [`Sequence`] runs several in order, each one
//! transforming the token list ([`Decoder::decode_chain`]); the final
//! text is the concatenation of the last list.

mod bpe;
mod byte_fallback;
mod ctc;
mod fuse;
mod replace;
mod sequence;
mod strip;
mod wordpiece;

use serde::de;
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;
use crate::normalizers::{deserialize_component, from_fields, has_fields};
use crate::pre_tokenizers::{ByteLevel, Metaspace};
use crate::traits::Decoder;

pub use bpe::BpeDecoder;
pub use byte_fallback::ByteFallback;
pub use ctc::Ctc;
pub use fuse::Fuse;
pub use replace::Replace;
pub use sequence::Sequence;
pub use strip::Strip;
pub use wordpiece::WordPiece;

/// Any built-in decoder. Serializes to the `tokenizer.json`
/// representation, tagged with `"type"`.
///
/// Loading also accepts the legacy untagged forms (no `"type"` key) that
/// Hugging Face `tokenizers` still reads: the variant is inferred from the
/// fields, exactly as HF's untagged fallback does. Saving always writes
/// the tagged form. Unknown keys are ignored, as everywhere else.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum DecoderWrapper {
    /// See [`BpeDecoder`].
    #[serde(rename = "BPEDecoder")]
    Bpe(BpeDecoder),
    /// See [`ByteLevel`].
    ByteLevel(ByteLevel),
    /// See [`WordPiece`].
    WordPiece(WordPiece),
    /// See [`Metaspace`].
    Metaspace(Metaspace),
    /// See [`Ctc`].
    #[serde(rename = "CTC")]
    Ctc(Ctc),
    /// See [`Sequence`].
    Sequence(Sequence),
    /// See [`Replace`].
    Replace(Replace),
    /// See [`Fuse`].
    Fuse(Fuse),
    /// See [`Strip`].
    Strip(Strip),
    /// See [`ByteFallback`].
    ByteFallback(ByteFallback),
}

/// The `"type"` names, in the order HF lists them.
const TYPES: &[&str] = &[
    "BPEDecoder",
    "ByteLevel",
    "WordPiece",
    "Metaspace",
    "CTC",
    "Sequence",
    "Replace",
    "Fuse",
    "Strip",
    "ByteFallback",
];

impl<'de> Deserialize<'de> for DecoderWrapper {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserialize_component(
            deserializer,
            "decoder",
            |ty, fields| {
                Ok(match ty {
                    "BPEDecoder" => Self::Bpe(from_fields(fields)?),
                    "ByteLevel" => Self::ByteLevel(from_fields(fields)?),
                    "WordPiece" => Self::WordPiece(from_fields(fields)?),
                    "Metaspace" => Self::Metaspace(from_fields(fields)?),
                    "CTC" => Self::Ctc(from_fields(fields)?),
                    "Sequence" => Self::Sequence(from_fields(fields)?),
                    "Replace" => Self::Replace(from_fields(fields)?),
                    "Fuse" => Self::Fuse(from_fields(fields)?),
                    "Strip" => Self::Strip(from_fields(fields)?),
                    "ByteFallback" => Self::ByteFallback(from_fields(fields)?),
                    other => return Err(de::Error::unknown_variant(other, TYPES)),
                })
            },
            |fields| {
                // The legacy untagged forms, tried in the order of HF's
                // untagged fallback enum. `ByteLevel`, `Metaspace`,
                // `Sequence`, `Fuse` and `ByteFallback` require the type
                // key in HF too, so only these five can be untagged.
                Ok(if has_fields(&fields, &["suffix"]) {
                    Self::Bpe(from_fields(fields)?)
                } else if has_fields(&fields, &["prefix", "cleanup"]) {
                    Self::WordPiece(from_fields(fields)?)
                } else if has_fields(&fields, &["pad_token", "word_delimiter_token", "cleanup"]) {
                    Self::Ctc(from_fields(fields)?)
                } else if has_fields(&fields, &["pattern", "content"]) {
                    Self::Replace(from_fields(fields)?)
                } else if has_fields(&fields, &["content", "start", "stop"]) {
                    Self::Strip(from_fields(fields)?)
                } else {
                    return Err(de::Error::custom(
                        "missing field `type` and the data does not match any legacy \
                         untagged decoder form",
                    ));
                })
            },
        )
    }
}

impl Decoder for DecoderWrapper {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match self {
            DecoderWrapper::Bpe(d) => d.decode_chain(tokens),
            DecoderWrapper::ByteLevel(d) => d.decode_chain(tokens),
            DecoderWrapper::WordPiece(d) => d.decode_chain(tokens),
            DecoderWrapper::Metaspace(d) => d.decode_chain(tokens),
            DecoderWrapper::Ctc(d) => d.decode_chain(tokens),
            DecoderWrapper::Sequence(d) => d.decode_chain(tokens),
            DecoderWrapper::Replace(d) => d.decode_chain(tokens),
            DecoderWrapper::Fuse(d) => d.decode_chain(tokens),
            DecoderWrapper::Strip(d) => d.decode_chain(tokens),
            DecoderWrapper::ByteFallback(d) => d.decode_chain(tokens),
        }
    }
}

macro_rules! impl_from {
    ($($ty:ident => $variant:ident),*) => {
        $(
            impl From<$ty> for DecoderWrapper {
                fn from(d: $ty) -> Self {
                    DecoderWrapper::$variant(d)
                }
            }
        )*
    };
}

impl_from!(
    BpeDecoder => Bpe,
    ByteLevel => ByteLevel,
    WordPiece => WordPiece,
    Metaspace => Metaspace,
    Ctc => Ctc,
    Sequence => Sequence,
    Replace => Replace,
    Fuse => Fuse,
    Strip => Strip,
    ByteFallback => ByteFallback
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_decoder_sequence() {
        let json = r#"{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}"#;
        let d: DecoderWrapper = serde_json::from_str(json).unwrap();
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(serde_json::to_value(&d).unwrap(), v);
        let tokens = [
            "▁Hey",
            "▁friend",
            "<0xF0>",
            "<0x9F>",
            "<0x98>",
            "<0x80>",
            "!",
        ];
        let out = d
            .decode(tokens.iter().map(|s| s.to_string()).collect())
            .unwrap();
        assert_eq!(out, "Hey friend😀!");
    }

    #[test]
    fn unknown_type_is_an_error() {
        let err = serde_json::from_str::<DecoderWrapper>(r#"{"type":"Nope"}"#).unwrap_err();
        assert!(err.to_string().contains("Nope"), "{err}");
    }

    /// Extra keys are ignored for every decoder.
    #[test]
    fn unknown_fields_are_ignored() {
        for ty in TYPES {
            let v = match *ty {
                "Metaspace" => serde_json::json!({"type": ty, "replacement": "▁", "x": 1}),
                "CTC" => serde_json::json!({
                    "type": ty, "pad_token": "<pad>", "word_delimiter_token": "|",
                    "cleanup": true, "x": 1
                }),
                "Sequence" => serde_json::json!({"type": ty, "decoders": [], "x": 1}),
                "Replace" => serde_json::json!({
                    "type": ty, "pattern": {"String": "▁"}, "content": " ", "x": 1
                }),
                "Strip" => {
                    serde_json::json!({"type": ty, "content": " ", "start": 1, "stop": 0, "x": 1})
                }
                _ => serde_json::json!({"type": ty, "x": 1}),
            };
            let d: DecoderWrapper =
                serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("{ty}: {e}"));
            let out = serde_json::to_value(&d).unwrap();
            assert!(out.get("x").is_none(), "{ty}: {out}");
            assert_eq!(out["type"], v["type"]);
        }
    }

    // Ported from Hugging Face `tokenizers` 0.23.2 (`decoders/mod.rs`,
    // `decoder_serialization`, `decoder_serialization_other_no_arg`).
    #[test]
    fn hf_decoder_serialization() {
        let oldjson = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        let olddecoder: DecoderWrapper = serde_json::from_str(oldjson).unwrap();
        let oldserialized = serde_json::to_string(&olddecoder).unwrap();
        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        assert_eq!(oldserialized, json);

        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(serde_json::to_string(&decoder).unwrap(), json);

        let json = r#"{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(serde_json::to_string(&decoder).unwrap(), json);
    }

    // Ported from Hugging Face `tokenizers` 0.23.2 (`decoders/mod.rs`,
    // `decoder_serialization_no_decode`).
    #[test]
    fn legacy_untagged_errors_match_hf() {
        // `{}` matches nothing: the parameterless decoders need a type.
        let json = r#"{"type":"Sequence","decoders":[{},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"}]}"#;
        let err = serde_json::from_str::<DecoderWrapper>(json).unwrap_err();
        assert!(err.to_string().contains("legacy untagged"), "{err}");

        // Metaspace requires its type in HF as well.
        let json = r#"{"replacement":"▁","prepend_scheme":"always"}"#;
        let err = serde_json::from_str::<DecoderWrapper>(json).unwrap_err();
        assert!(err.to_string().contains("legacy untagged"), "{err}");

        let json = r#"{"type":"Sequence","prepend_scheme":"always"}"#;
        let err = serde_json::from_str::<DecoderWrapper>(json).unwrap_err();
        assert!(
            err.to_string().contains("missing field `decoders`"),
            "{err}"
        );
    }

    #[test]
    fn legacy_untagged_forms_load() {
        let cases: [(&str, &str); 5] = [
            (
                r#"{"suffix":"</w>"}"#,
                r#"{"type":"BPEDecoder","suffix":"</w>"}"#,
            ),
            (
                "{\"prefix\":\"##\",\"cleanup\":true}",
                "{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":true}",
            ),
            (
                r#"{"pad_token":"<pad>","word_delimiter_token":"|","cleanup":true}"#,
                r#"{"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":true}"#,
            ),
            (
                r#"{"pattern":{"String":"▁"},"content":" "}"#,
                r#"{"type":"Replace","pattern":{"String":"▁"},"content":" "}"#,
            ),
            (
                r#"{"content":" ","start":1,"stop":0}"#,
                r#"{"type":"Strip","content":" ","start":1,"stop":0}"#,
            ),
        ];
        for (legacy, tagged) in cases {
            let d: DecoderWrapper =
                serde_json::from_str(legacy).unwrap_or_else(|e| panic!("{legacy}: {e}"));
            assert_eq!(serde_json::to_string(&d).unwrap(), tagged);
        }
        // Nested inside a tagged Sequence.
        let json = r#"{"type":"Sequence","decoders":[{"pattern":{"String":"▁"},"content":" "},{"type":"Fuse"}]}"#;
        let d: DecoderWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(d.decode(vec!["▁a".into(), "▁b".into()]).unwrap(), " a b");
    }
}
