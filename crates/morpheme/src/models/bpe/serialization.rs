//! `tokenizer.json` (de)serialization for [`Bpe`], byte-compatible with
//! Hugging Face `tokenizers`.

use rustc_hash::FxHashMap;
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::model::{Bpe, BpeBuilder, Merges, Vocab};

/// Serializes `token -> id` as a JSON object ordered by id (then token).
///
/// Every entry is written, including tokens that share an id (Hugging
/// Face accepts such files), so a save → load round trip is lossless.
pub(crate) struct OrderedVocab<'a>(pub(crate) &'a Vocab);

impl Serialize for OrderedVocab<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries: Vec<(&String, &u32)> = self.0.iter().collect();
        entries.sort_unstable_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
        serializer.collect_map(entries)
    }
}

/// `id -> token`. When several tokens share an id, the smallest token
/// wins, so the mapping doesn't depend on hash-map iteration order.
pub(crate) fn reverse_vocab(vocab: &Vocab) -> FxHashMap<u32, String> {
    let mut reversed: FxHashMap<u32, String> = FxHashMap::default();
    for (token, &id) in vocab {
        match reversed.get(&id) {
            Some(existing) if existing <= token => {}
            _ => {
                reversed.insert(id, token.clone());
            }
        }
    }
    reversed
}

impl Serialize for Bpe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("BPE", 10)?;
        s.serialize_field("type", "BPE")?;
        s.serialize_field("dropout", &self.dropout)?;
        s.serialize_field("unk_token", &self.unk_token)?;
        s.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        s.serialize_field("end_of_word_suffix", &self.end_of_word_suffix)?;
        s.serialize_field("fuse_unk", &self.fuse_unk)?;
        s.serialize_field("byte_fallback", &self.byte_fallback)?;
        s.serialize_field("ignore_merges", &self.ignore_merges)?;
        s.serialize_field("vocab", &OrderedVocab(&self.vocab))?;
        s.serialize_field("merges", &self.merge_list)?;
        s.end()
    }
}

/// Merges are `[["a", "b"], ...]` in current files and `["a b", ...]`
/// in files written before `tokenizers` 0.20.
#[derive(Deserialize)]
#[serde(untagged)]
enum MergesRepr {
    Pairs(Vec<(String, String)>),
    Legacy(Vec<String>),
}

fn parse_legacy_merges(lines: Vec<String>) -> Result<Merges, String> {
    lines
        .into_iter()
        .filter(|l| !l.starts_with("#version"))
        .enumerate()
        .map(|(i, line)| {
            let mut parts = line.split(' ');
            match (parts.next(), parts.next(), parts.next()) {
                (Some(a), Some(b), None) => Ok((a.to_owned(), b.to_owned())),
                _ => Err(format!("invalid merge at index {i}: {line:?}")),
            }
        })
        .collect()
}

impl<'de> Deserialize<'de> for Bpe {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(BpeVisitor)
    }
}

struct BpeVisitor;

impl<'de> Visitor<'de> for BpeVisitor {
    type Value = Bpe;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a BPE model")
    }

    fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Bpe, V::Error> {
        let mut builder = BpeBuilder::new();
        let mut vocab: Option<Vocab> = None;
        let mut merges: Option<MergesRepr> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    let ty: String = map.next_value()?;
                    if ty != "BPE" {
                        return Err(de::Error::invalid_value(de::Unexpected::Str(&ty), &"BPE"));
                    }
                }
                "dropout" => {
                    if let Some(p) = map.next_value::<Option<f32>>()? {
                        builder = builder.dropout(p);
                    }
                }
                "unk_token" => {
                    if let Some(t) = map.next_value::<Option<String>>()? {
                        builder = builder.unk_token(t);
                    }
                }
                "continuing_subword_prefix" => {
                    if let Some(t) = map.next_value::<Option<String>>()? {
                        builder = builder.continuing_subword_prefix(t);
                    }
                }
                "end_of_word_suffix" => {
                    if let Some(t) = map.next_value::<Option<String>>()? {
                        builder = builder.end_of_word_suffix(t);
                    }
                }
                "fuse_unk" => {
                    if let Some(v) = map.next_value::<Option<bool>>()? {
                        builder = builder.fuse_unk(v);
                    }
                }
                "byte_fallback" => {
                    if let Some(v) = map.next_value::<Option<bool>>()? {
                        builder = builder.byte_fallback(v);
                    }
                }
                "ignore_merges" => {
                    if let Some(v) = map.next_value::<Option<bool>>()? {
                        builder = builder.ignore_merges(v);
                    }
                }
                "vocab" => vocab = Some(map.next_value()?),
                "merges" => merges = Some(map.next_value()?),
                _ => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        let vocab = vocab.ok_or_else(|| de::Error::missing_field("vocab"))?;
        let merges = match merges.ok_or_else(|| de::Error::missing_field("merges"))? {
            MergesRepr::Pairs(p) => p,
            MergesRepr::Legacy(lines) => parse_legacy_merges(lines).map_err(de::Error::custom)?,
        };
        builder
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Bpe {
        let vocab: Vocab = [("<unk>", 0), ("a", 1), ("b", 2), ("ab", 3)]
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        Bpe::builder()
            .vocab_and_merges(vocab, vec![("a".into(), "b".into())])
            .unk_token("<unk>")
            .ignore_merges(true)
            .build()
            .unwrap()
    }

    #[test]
    fn serializes_like_hf() {
        let json = serde_json::to_string(&sample()).unwrap();
        assert_eq!(
            json,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]}"#
        );
        let back: Bpe = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sample());
    }

    #[test]
    fn accepts_legacy_merges_and_missing_type() {
        let json = r#"{"dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let bpe: Bpe = serde_json::from_str(json).unwrap();
        assert_eq!(bpe.merges(), vec![("a".to_string(), "b".to_string())]);
    }

    #[test]
    fn rejects_bad_merges_and_wrong_type() {
        let bad = r#"{"vocab":{"a":0},"merges":["a z"]}"#;
        assert!(serde_json::from_str::<Bpe>(bad).is_err());
        let bad = r#"{"vocab":{"a":0},"merges":["a b c"]}"#;
        assert!(serde_json::from_str::<Bpe>(bad).is_err());
        let wrong = r#"{"type":"WordPiece","vocab":{},"merges":[]}"#;
        assert!(serde_json::from_str::<Bpe>(wrong).is_err());
    }

    /// HF re-serializes a duplicated merge pair once, at the position of
    /// its last occurrence.
    #[test]
    fn duplicate_merges_are_written_once() {
        let json = r#"{"vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4},"merges":[["a","b"],["b","c"],["a","b"]]}"#;
        let bpe: Bpe = serde_json::from_str(json).unwrap();
        let out = serde_json::to_value(&bpe).unwrap();
        assert_eq!(out["merges"], serde_json::json!([["b", "c"], ["a", "b"]]));
        // Reloading the re-serialized form gives the same model.
        let back: Bpe = serde_json::from_value(out).unwrap();
        assert_eq!(back, bpe);
    }

    #[test]
    fn preserves_sparse_ids() {
        let json = r#"{"vocab":{"a":0,"b":5,"ab":9},"merges":[["a","b"]]}"#;
        let bpe: Bpe = serde_json::from_str(json).unwrap();
        let out = serde_json::to_value(&bpe).unwrap();
        assert_eq!(out["vocab"], serde_json::json!({"a":0,"b":5,"ab":9}));
    }
}
