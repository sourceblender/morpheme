//! Decoders: turn tokens back into text.
//!
//! Decoders compose: [`Sequence`] runs several in order, each one
//! transforming the token list ([`Decoder::decode_chain`]); the final
//! text is the concatenation of the last list.

pub mod bpe;
pub mod byte_fallback;
pub mod ctc;
pub mod fuse;
pub mod replace;
pub mod sequence;
pub mod strip;
pub mod wordpiece;

use serde::{Deserialize, Serialize};

use crate::error::Result;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
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
}
