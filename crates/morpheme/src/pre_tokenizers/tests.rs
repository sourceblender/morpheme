//! Cross-cutting tests: ground truth from Python `tokenizers` 0.23.2
//! (`testdata/ground_truth.json`, produced by
//! `testdata/gen_ground_truth.py`) and serialization
//! round-trips of the pre-tokenizer / decoder configs found in real
//! `tokenizer.json` files.

use serde_json::Value;

use super::PreTokenizerWrapper;
use crate::Offsets;
use crate::decoders::DecoderWrapper;
use crate::pre_tokenized_string::{OffsetType, PreTokenizedString};
use crate::traits::{Decoder, PreTokenizer};

/// Pre-tokenize `input` and return `(piece, byte offsets)`. Pieces are
/// leaked so tests can compare against `&str` literals.
pub(crate) fn splits(pt: &dyn PreTokenizer, input: &str) -> Vec<(&'static str, Offsets)> {
    let mut pts = PreTokenizedString::from(input);
    pt.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (&*Box::leak(s.to_owned().into_boxed_str()), o))
        .collect()
}

fn ground_truth() -> Value {
    serde_json::from_str(include_str!("testdata/ground_truth.json")).unwrap()
}

#[test]
fn pre_tokenizers_match_python() {
    let truth = ground_truth();
    let mut checked = 0;
    for entry in truth["pre_tokenizers"].as_array().unwrap() {
        let config = &entry["config"];
        let pt: PreTokenizerWrapper = serde_json::from_value(config.clone())
            .unwrap_or_else(|e| panic!("cannot load {config}: {e}"));
        for case in entry["cases"].as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let mut pts = PreTokenizedString::from(input);
            pt.pre_tokenize(&mut pts).unwrap();
            let got: Vec<Value> = pts
                .get_splits(OffsetType::Char)
                .into_iter()
                .map(|(s, (a, b), _)| serde_json::json!([s, [a, b]]))
                .collect();
            assert_eq!(
                Value::Array(got),
                case["splits"],
                "config {config}, input {input:?}"
            );
            checked += 1;
        }
    }
    assert!(checked > 400, "only {checked} cases checked");
}

#[test]
fn decoders_match_python() {
    let truth = ground_truth();
    for entry in truth["decoders"].as_array().unwrap() {
        let config = &entry["config"];
        let d: DecoderWrapper = serde_json::from_value(config.clone())
            .unwrap_or_else(|e| panic!("cannot load {config}: {e}"));
        for case in entry["cases"].as_array().unwrap() {
            let tokens: Vec<String> = serde_json::from_value(case["tokens"].clone()).unwrap();
            let got = d.decode(tokens.clone());
            match case.get("output") {
                Some(want) => assert_eq!(
                    got.unwrap(),
                    want.as_str().unwrap(),
                    "config {config}, tokens {tokens:?}"
                ),
                None => assert!(got.is_err(), "config {config}, tokens {tokens:?}"),
            }
        }
    }
}

/// Every pre-tokenizer / decoder config in the real fixtures loads and
/// re-serializes exactly like Python `tokenizers` re-serializes it
/// (legacy fields are upgraded the same way).
#[test]
fn fixture_configs_roundtrip_like_python() {
    let truth = ground_truth();
    let fixtures = truth["fixtures"].as_object().unwrap();
    assert!(fixtures.len() >= 10);
    for (name, f) in fixtures {
        for (field, is_pretok) in [("pre_tokenizer", true), ("decoder", false)] {
            let raw = &f[field][0];
            let expected = &f[field][1];
            if raw.is_null() {
                continue;
            }
            let reserialized = if is_pretok {
                let p: PreTokenizerWrapper = serde_json::from_value(raw.clone())
                    .unwrap_or_else(|e| panic!("{name} {field}: {e}"));
                serde_json::to_value(&p).unwrap()
            } else {
                let d: DecoderWrapper = serde_json::from_value(raw.clone())
                    .unwrap_or_else(|e| panic!("{name} {field}: {e}"));
                serde_json::to_value(&d).unwrap()
            };
            assert_eq!(&reserialized, expected, "{name} {field}");
        }
    }
}

/// Re-serialized JSON keeps HF's field order (byte-compatible output).
#[test]
fn serialization_field_order_matches_hf() {
    let bl: PreTokenizerWrapper = super::ByteLevel::default().into();
    assert_eq!(
        serde_json::to_string(&bl).unwrap(),
        r#"{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#
    );
    let ms: PreTokenizerWrapper = super::Metaspace::default().into();
    assert_eq!(
        serde_json::to_string(&ms).unwrap(),
        r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#
    );
    let ws: PreTokenizerWrapper = super::Whitespace.into();
    assert_eq!(
        serde_json::to_string(&ws).unwrap(),
        r#"{"type":"Whitespace"}"#
    );
    let d: DecoderWrapper = crate::decoders::BpeDecoder::default().into();
    assert_eq!(
        serde_json::to_string(&d).unwrap(),
        r#"{"type":"BPEDecoder","suffix":"</w>"}"#
    );
}

#[test]
fn unknown_pre_tokenizer_type_is_an_error() {
    let err = serde_json::from_str::<PreTokenizerWrapper>(r#"{"type":"Bogus"}"#).unwrap_err();
    assert!(err.to_string().contains("Bogus"), "{err}");
}

#[test]
fn all_variants_roundtrip() {
    let all: Vec<PreTokenizerWrapper> = vec![
        super::BertPreTokenizer.into(),
        super::ByteLevel::new(false, false, false).into(),
        super::CharDelimiterSplit::new('x').into(),
        super::Metaspace::new('_', super::PrependScheme::First, false).into(),
        super::Whitespace.into(),
        super::Sequence::new(vec![
            super::Whitespace.into(),
            super::Digits::new(true).into(),
        ])
        .into(),
        super::Split::new(
            crate::pattern::SplitPattern::Regex(r"\d+".into()),
            crate::SplitDelimiterBehavior::Contiguous,
            true,
        )
        .unwrap()
        .into(),
        super::Punctuation::new(crate::SplitDelimiterBehavior::MergedWithNext).into(),
        super::WhitespaceSplit.into(),
        super::Digits::new(false).into(),
        super::UnicodeScripts.into(),
        super::FixedLength::new(7).into(),
    ];
    for p in all {
        let json = serde_json::to_string(&p).unwrap();
        let back: PreTokenizerWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p, "{json}");
    }
    let decoders: Vec<DecoderWrapper> = vec![
        crate::decoders::BpeDecoder::new("@@").into(),
        super::ByteLevel::default().into(),
        crate::decoders::WordPiece::new("##", false).into(),
        super::Metaspace::default().into(),
        crate::decoders::Ctc::default().into(),
        crate::decoders::Sequence::new(vec![crate::decoders::Fuse.into()]).into(),
        crate::decoders::Replace::new(crate::pattern::SplitPattern::String("a".into()), "b")
            .unwrap()
            .into(),
        crate::decoders::Fuse.into(),
        crate::decoders::Strip::new(' ', 1, 2).into(),
        crate::decoders::ByteFallback.into(),
    ];
    for d in decoders {
        let json = serde_json::to_string(&d).unwrap();
        let back: DecoderWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d, "{json}");
    }
}
