//! `tokenizer.json` (de)serialization, byte-compatible with Hugging Face
//! `tokenizers`.

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{PaddingParams, Tokenizer, TruncationParams};
use crate::added_vocabulary::{AddedTokenWithId, AddedVocabulary};
use crate::decoders::DecoderWrapper;
use crate::models::ModelWrapper;
use crate::normalizers::NormalizerWrapper;
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::processors::PostProcessorWrapper;
use crate::traits::Normalizer;

const VERSION: &str = "1.0";

impl Serialize for Tokenizer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Tokenizer", 9)?;
        s.serialize_field("version", VERSION)?;
        s.serialize_field("truncation", &self.truncation)?;
        s.serialize_field("padding", &self.padding)?;
        s.serialize_field("added_tokens", &self.added_vocabulary.tokens_with_ids())?;
        s.serialize_field("normalizer", &self.normalizer)?;
        s.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        s.serialize_field("post_processor", &self.post_processor)?;
        s.serialize_field("decoder", &self.decoder)?;
        s.serialize_field("model", &self.model)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Tokenizer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_struct(
            "Tokenizer",
            &[
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            ],
            TokenizerVisitor,
        )
    }
}

struct TokenizerVisitor;

impl<'de> Visitor<'de> for TokenizerVisitor {
    type Value = Tokenizer;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a tokenizer.json object")
    }

    fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Tokenizer, V::Error> {
        let mut truncation: Option<TruncationParams> = None;
        let mut padding: Option<PaddingParams> = None;
        let mut added: Vec<AddedTokenWithId> = Vec::new();
        let mut normalizer: Option<NormalizerWrapper> = None;
        let mut pre_tokenizer: Option<PreTokenizerWrapper> = None;
        let mut post_processor: Option<PostProcessorWrapper> = None;
        let mut decoder: Option<DecoderWrapper> = None;
        let mut model: Option<ModelWrapper> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "version" => {
                    let v: String = map.next_value()?;
                    if v != VERSION {
                        return Err(de::Error::custom(format!(
                            "unsupported tokenizer.json version {v:?} (expected {VERSION:?})"
                        )));
                    }
                }
                "truncation" => truncation = map.next_value()?,
                "padding" => padding = map.next_value()?,
                "added_tokens" => added = map.next_value()?,
                "normalizer" => normalizer = map.next_value()?,
                "pre_tokenizer" => pre_tokenizer = map.next_value()?,
                "post_processor" => post_processor = map.next_value()?,
                "decoder" => decoder = map.next_value()?,
                "model" => model = Some(map.next_value()?),
                _ => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }

        let model = model.ok_or_else(|| de::Error::missing_field("model"))?;
        let mut added_vocabulary = AddedVocabulary::new();
        added_vocabulary
            .add_tokens_with_ids(&added, normalizer.as_ref().map(|n| n as &dyn Normalizer))
            .map_err(de::Error::custom)?;

        Ok(Tokenizer {
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
            added_vocabulary,
            truncation,
            padding,
        })
    }
}
