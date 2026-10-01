//! Round-trip tests for `Tokenizer::from_json` / `to_json`.

mod fixtures;

use std::io::Write;

use splinter::Tokenizer;

const FIXTURE_PATH: &str = "tests/fixtures/tiny.json";

#[test]
fn loads_fixture_from_disk() {
    let t = Tokenizer::from_file(FIXTURE_PATH).expect("fixture should load");
    let enc = t.encode("aa bb hello").expect("encode should succeed");
    assert_eq!(enc.tokens, vec!["aa</w>", "bb</w>", "hello</w>"]);
}

#[test]
fn loads_fixture_from_string() {
    let raw = std::fs::read_to_string(FIXTURE_PATH).expect("read fixture");
    let t = Tokenizer::from_json(&raw).expect("parse fixture");
    let enc = t.encode("aa bb hello").expect("encode should succeed");
    assert_eq!(enc.tokens, vec!["aa</w>", "bb</w>", "hello</w>"]);
}

#[test]
fn round_trip_through_string_preserves_encodings() {
    let t = fixtures::tokenizer();
    let original = t.encode("aa bb hello").unwrap();

    let json = t.to_json().expect("serialize");
    let reloaded = Tokenizer::from_json(&json).expect("deserialize");

    let after = reloaded.encode("aa bb hello").unwrap();
    assert_eq!(original.ids, after.ids);
    assert_eq!(original.tokens, after.tokens);
    assert_eq!(original.offsets, after.offsets);
}

#[test]
fn round_trip_through_file_preserves_encodings() {
    let t = fixtures::tokenizer();
    let original = t.encode("aa bb hello aabbb").unwrap();

    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(t.to_json().expect("serialize").as_bytes())
        .expect("write");
    tmp.flush().expect("flush");

    let reloaded = Tokenizer::from_file(tmp.path()).expect("reload");
    let after = reloaded.encode("aa bb hello aabbb").unwrap();

    assert_eq!(original.ids, after.ids);
    assert_eq!(original.tokens, after.tokens);
    assert_eq!(original.offsets, after.offsets);
}

#[test]
fn pretty_and_compact_outputs_parse_equivalently() {
    let t = fixtures::tokenizer();
    let compact = t.to_json().unwrap();

    // Pretty-print by re-serializing the compact JSON through serde_json.
    let value: serde_json::Value = serde_json::from_str(&compact).unwrap();
    let pretty = serde_json::to_string_pretty(&value).unwrap();

    let a = Tokenizer::from_json(&compact).unwrap();
    let b = Tokenizer::from_json(&pretty).unwrap();
    let enc_a = a.encode("hello").unwrap();
    let enc_b = b.encode("hello").unwrap();
    assert_eq!(enc_a.ids, enc_b.ids);
    assert_eq!(enc_a.tokens, enc_b.tokens);
}

#[test]
fn rejects_unknown_schema_version() {
    let raw = r#"{
        "version": "99.0",
        "model": {
            "type": "bpe",
            "dropout": null,
            "unk_token": null,
            "end_of_word_suffix": "</w>",
            "continuing_subword_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false
        },
        "vocab": ["<unk>", "a", "a</w>"],
        "merges": [["a", "a</w>"]]
    }"#;
    let err = Tokenizer::from_json(raw).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("99.0") || msg.contains("schema version"),
        "unexpected error: {msg}"
    );
}

#[test]
fn rejects_non_bpe_model() {
    let raw = r#"{
        "version": "1.0",
        "model": {
            "type": "wordpiece",
            "dropout": null,
            "unk_token": null,
            "end_of_word_suffix": "</w>",
            "continuing_subword_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false
        },
        "vocab": ["<unk>"],
        "merges": []
    }"#;
    let err = Tokenizer::from_json(raw).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("wordpiece") || msg.contains("model type"));
}

#[test]
fn rejects_dropout() {
    let raw = r#"{
        "version": "1.0",
        "model": {
            "type": "bpe",
            "dropout": 0.1,
            "unk_token": null,
            "end_of_word_suffix": "</w>",
            "continuing_subword_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false
        },
        "vocab": ["<unk>", "a", "a</w>"],
        "merges": [["a", "a</w>"]]
    }"#;
    let err = Tokenizer::from_json(raw).unwrap_err();
    assert!(format!("{err}").contains("dropout"));
}

#[test]
fn rejects_byte_fallback() {
    let raw = r#"{
        "version": "1.0",
        "model": {
            "type": "bpe",
            "dropout": null,
            "unk_token": null,
            "end_of_word_suffix": "</w>",
            "continuing_subword_suffix": null,
            "fuse_unk": false,
            "byte_fallback": true
        },
        "vocab": ["<unk>", "a", "a</w>"],
        "merges": [["a", "a</w>"]]
    }"#;
    let err = Tokenizer::from_json(raw).unwrap_err();
    assert!(format!("{err}").contains("byte_fallback"));
}

#[test]
fn rejects_continuing_subword_suffix() {
    // Use triple-hash raw strings so the `##` and `"##"` in the JSON
    // don't terminate the raw string literal.
    let raw = r###"{
        "version": "1.0",
        "model": {
            "type": "bpe",
            "dropout": null,
            "unk_token": null,
            "end_of_word_suffix": "</w>",
            "continuing_subword_suffix": "##",
            "fuse_unk": false,
            "byte_fallback": false
        },
        "vocab": ["<unk>", "a", "a</w>", "a##"],
        "merges": [["a", "a</w>"]]
    }"###;
    let err = Tokenizer::from_json(raw).unwrap_err();
    assert!(format!("{err}").contains("continuing_subword_suffix"));
}

#[test]
fn rejects_duplicate_vocab_token() {
    let raw = r#"{
        "version": "1.0",
        "model": {
            "type": "bpe",
            "dropout": null,
            "unk_token": null,
            "end_of_word_suffix": "</w>",
            "continuing_subword_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false
        },
        "vocab": ["<unk>", "a", "a</w>", "a"],
        "merges": []
    }"#;
    let err = Tokenizer::from_json(raw).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("duplicate") || msg.contains("a"));
}
