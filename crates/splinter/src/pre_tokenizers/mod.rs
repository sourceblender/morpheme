//! Pre-tokenizers: split normalized text into the pieces ("words") the
//! model tokenizes independently.
//!
//! Every pre-tokenizer works on a [`PreTokenizedString`] through
//! [`NormalizedString`](crate::NormalizedString) operations, so offsets
//! back to the original input are preserved exactly as in Hugging Face
//! `tokenizers`.

pub mod bert;
pub mod byte_level;
pub mod delimiter;
pub mod digits;
pub mod fixed_length;
pub mod metaspace;
pub mod punctuation;
pub mod sequence;
pub mod split;
pub mod unicode_scripts;
pub mod whitespace;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

pub use bert::BertPreTokenizer;
pub use byte_level::ByteLevel;
pub use delimiter::CharDelimiterSplit;
pub use digits::Digits;
pub use fixed_length::FixedLength;
pub use metaspace::{Metaspace, PrependScheme};
pub use punctuation::Punctuation;
pub use sequence::Sequence;
pub use split::Split;
pub use unicode_scripts::UnicodeScripts;
pub use whitespace::{Whitespace, WhitespaceSplit};

/// Any built-in pre-tokenizer. Serializes to the `tokenizer.json`
/// representation, tagged with `"type"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PreTokenizerWrapper {
    /// See [`BertPreTokenizer`].
    BertPreTokenizer(BertPreTokenizer),
    /// See [`ByteLevel`].
    ByteLevel(ByteLevel),
    /// See [`CharDelimiterSplit`].
    CharDelimiterSplit(CharDelimiterSplit),
    /// See [`Metaspace`].
    Metaspace(Metaspace),
    /// See [`Whitespace`].
    Whitespace(Whitespace),
    /// See [`Sequence`].
    Sequence(Sequence),
    /// See [`Split`].
    Split(Split),
    /// See [`Punctuation`].
    Punctuation(Punctuation),
    /// See [`WhitespaceSplit`].
    WhitespaceSplit(WhitespaceSplit),
    /// See [`Digits`].
    Digits(Digits),
    /// See [`UnicodeScripts`].
    UnicodeScripts(UnicodeScripts),
    /// See [`FixedLength`].
    FixedLength(FixedLength),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        match self {
            PreTokenizerWrapper::BertPreTokenizer(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::ByteLevel(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::CharDelimiterSplit(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::Metaspace(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::Whitespace(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::Sequence(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::Split(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::Punctuation(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::WhitespaceSplit(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::Digits(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::UnicodeScripts(p) => p.pre_tokenize(pretokenized),
            PreTokenizerWrapper::FixedLength(p) => p.pre_tokenize(pretokenized),
        }
    }
}

macro_rules! impl_from {
    ($($ty:ident),*) => {
        $(
            impl From<$ty> for PreTokenizerWrapper {
                fn from(p: $ty) -> Self {
                    PreTokenizerWrapper::$ty(p)
                }
            }
        )*
    };
}

impl_from!(
    BertPreTokenizer,
    ByteLevel,
    CharDelimiterSplit,
    Metaspace,
    Whitespace,
    Sequence,
    Split,
    Punctuation,
    WhitespaceSplit,
    Digits,
    UnicodeScripts,
    FixedLength
);

/// Implements `Serialize` / `Deserialize` for a parameterless unit struct
/// as an empty map, so it nests inside the `"type"`-tagged wrappers
/// (`{"type": "Whitespace"}`). Any extra keys are ignored.
macro_rules! impl_unit_serde {
    ($ty:ident) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                use serde::ser::SerializeStruct;
                serializer.serialize_struct(stringify!($ty), 0)?.end()
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                struct V;
                impl<'de> serde::de::Visitor<'de> for V {
                    type Value = $ty;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str(concat!("a ", stringify!($ty), " object"))
                    }
                    fn visit_map<A: serde::de::MapAccess<'de>>(
                        self,
                        mut map: A,
                    ) -> std::result::Result<$ty, A::Error> {
                        while map
                            .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                            .is_some()
                        {}
                        Ok($ty)
                    }
                    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<$ty, E> {
                        Ok($ty)
                    }
                }
                deserializer.deserialize_any(V)
            }
        }
    };
}
pub(crate) use impl_unit_serde;
