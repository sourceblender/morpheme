//! `tokenizer.json` loading/saving: round-trips, legacy formats, and
//! errors (never panics) on malformed input.

use morpheme::models::Bpe;
use morpheme::normalizers::Lowercase;
use morpheme::pre_tokenizers::Whitespace;
use morpheme::trainers::BpeTrainer;
use morpheme::{AddedToken, Tokenizer};

fn trained() -> Tokenizer {
    let mut tok = Tokenizer::new(Bpe::builder().unk_token("[UNK]").build().unwrap())
        .with_normalizer(Lowercase)
        .with_pre_tokenizer(Whitespace);
    let trainer = BpeTrainer::builder()
        .vocab_size(120)
        .special_tokens(vec![AddedToken::new("[UNK]", true)])
        .build()
        .unwrap();
    tok.train(
        trainer,
        include_str!("../../../examples/corpus.txt").lines(),
    )
    .unwrap();
    tok
}

#[test]
fn save_and_load_preserves_every_component() {
    let tok = trained();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    tok.save(&path, true).unwrap();
    let loaded = Tokenizer::from_file(&path).unwrap();
    assert_eq!(loaded.to_json(false).unwrap(), tok.to_json(false).unwrap());
    for text in ["THE Quick brown fox", "unseen ZEBRA"] {
        let a = tok.encode(text, true).unwrap();
        let b = loaded.encode(text, true).unwrap();
        assert_eq!(a, b);
    }
    // The normalizer survived the round trip.
    assert_eq!(
        loaded.encode("FOX", false).unwrap().tokens(),
        tok.encode("fox", false).unwrap().tokens()
    );
}

#[test]
fn pretty_and_compact_are_equivalent() {
    let tok = trained();
    let a = Tokenizer::from_json(&tok.to_json(true).unwrap()).unwrap();
    let b = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(a.to_json(false).unwrap(), b.to_json(false).unwrap());
}

fn minimal(model: &str) -> String {
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{model}}}"#
    )
}

#[test]
fn legacy_untyped_bpe_with_string_merges_loads() {
    let json = minimal(r#"{"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}"#);
    let tok = Tokenizer::from_json(&json).unwrap();
    assert_eq!(tok.encode("ab", false).unwrap().ids(), &[2]);
}

/// Unknown keys are ignored the same way in every component kind (the
/// policy of the top-level file and of Hugging Face's own derives).
#[test]
fn unknown_fields_are_ignored_consistently_across_components() {
    let model = r#"{"type":"WordLevel","vocab":{"[UNK]":0,"hi":1},"unk_token":"[UNK]"}"#;
    let components = [
        ("normalizer", r#"{"type":"NFD","x":1}"#),
        ("normalizer", r#"{"type":"Lowercase","future_flag":true}"#),
        (
            "normalizer",
            r#"{"type":"Strip","strip_left":true,"strip_right":true,"x":1}"#,
        ),
        ("pre_tokenizer", r#"{"type":"Whitespace","x":1}"#),
        (
            "pre_tokenizer",
            r#"{"type":"Punctuation","behavior":"Isolated","x":1}"#,
        ),
        ("decoder", r#"{"type":"Fuse","x":1}"#),
        (
            "decoder",
            r#"{"type":"WordPiece","prefix":"@@","cleanup":true,"x":1}"#,
        ),
    ];
    for (field, component) in components {
        let json = format!(r#"{{"version":"1.0","{field}":{component},"model":{model}}}"#);
        let tok = Tokenizer::from_json(&json).unwrap_or_else(|e| panic!("{json}: {e}"));
        // The extra key is dropped on save, nothing else changes.
        let out: serde_json::Value = serde_json::from_str(&tok.to_json(false).unwrap()).unwrap();
        let mut expected: serde_json::Value = serde_json::from_str(component).unwrap();
        expected.as_object_mut().unwrap().remove("x");
        expected.as_object_mut().unwrap().remove("future_flag");
        assert_eq!(out[field], expected, "{json}");
    }
    // Only the "type" value itself is checked strictly.
    for field in ["normalizer", "pre_tokenizer", "decoder"] {
        let json = format!(r#"{{"version":"1.0","{field}":{{"type":"Bogus"}},"model":{model}}}"#);
        let err = Tokenizer::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("Bogus"), "{field}: {err}");
    }
}

/// Files written before the `"type"` tag existed: the variant is inferred
/// from the fields, as in Hugging Face's untagged fallback. Saving writes
/// the current tagged form.
#[test]
fn legacy_untagged_normalizer_and_decoder_load() {
    let json = r#"{"version":"1.0",
        "normalizer":{"normalizers":[{"strip_left":true,"strip_right":true},{"type":"Lowercase"}]},
        "pre_tokenizer":{"type":"WhitespaceSplit"},
        "decoder":{"prefix":"@@","cleanup":true},
        "model":{"type":"WordPiece","vocab":{"[UNK]":0,"hi":1,"@@s":2},"unk_token":"[UNK]",
                 "continuing_subword_prefix":"@@","max_input_chars_per_word":100}}"#;
    let tok = Tokenizer::from_json(json).unwrap();
    let enc = tok.encode("  HI  ", false).unwrap();
    assert_eq!(enc.tokens(), ["hi"]);
    assert_eq!(tok.decode(&[1, 2], false).unwrap(), "his");
    let out: serde_json::Value = serde_json::from_str(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(
        out["normalizer"],
        serde_json::json!({"type":"Sequence","normalizers":[
            {"type":"Strip","strip_left":true,"strip_right":true},{"type":"Lowercase"}]})
    );
    assert_eq!(
        out["decoder"],
        serde_json::json!({"type":"WordPiece","prefix":"@@","cleanup":true})
    );

    // Parameterless components cannot be inferred from `{}` (HF rejects
    // them as well), and a tagged object with the wrong fields is an error
    // that names the missing field.
    for (field, component, needle) in [
        ("normalizer", r#"{}"#, "legacy untagged"),
        ("decoder", r#"{}"#, "legacy untagged"),
        (
            "decoder",
            r#"{"replacement":"▁","prepend_scheme":"always"}"#,
            "legacy untagged",
        ),
        (
            "normalizer",
            r#"{"type":"Sequence","prepend_scheme":"always"}"#,
            "missing field `normalizers`",
        ),
        (
            "decoder",
            r#"{"type":"Sequence","prepend_scheme":"always"}"#,
            "missing field `decoders`",
        ),
    ] {
        let json = format!(
            r#"{{"version":"1.0","{field}":{component},"model":{{"type":"WordLevel","vocab":{{}},"unk_token":"x"}}}}"#
        );
        let err = Tokenizer::from_json(&json).unwrap_err();
        assert!(err.to_string().contains(needle), "{json}: {err}");
    }
}

#[test]
fn ids_are_preserved_not_renumbered() {
    let json = minimal(r#"{"type":"WordLevel","vocab":{"[UNK]":0,"hi":5},"unk_token":"[UNK]"}"#);
    let tok = Tokenizer::from_json(&json).unwrap();
    assert_eq!(tok.encode("hi", false).unwrap().ids(), &[5]);
}

#[test]
fn malformed_inputs_are_errors_not_panics() {
    let cases = [
        // Merge referencing a token missing from the vocab.
        minimal(r#"{"type":"BPE","vocab":{"a":0},"merges":[["a","z"]]}"#),
        // Unknown component type.
        minimal(r#"{"type":"Mystery","vocab":{}}"#),
        r#"{"version":"1.0","normalizer":{"type":"Nope"},"model":{"type":"WordLevel","vocab":{},"unk_token":"x"}}"#.to_owned(),
        // Unsupported version.
        r#"{"version":"9.9","model":{"type":"WordLevel","vocab":{},"unk_token":"x"}}"#.to_owned(),
        // Missing model.
        r#"{"version":"1.0"}"#.to_owned(),
        // Bad regex.
        r#"{"version":"1.0","pre_tokenizer":{"type":"Split","pattern":{"Regex":"("},"behavior":"Isolated","invert":false},"model":{"type":"WordLevel","vocab":{},"unk_token":"x"}}"#.to_owned(),
        // Unigram unk_id out of range.
        minimal(r#"{"type":"Unigram","unk_id":5,"vocab":[["a",0.0]]}"#),
        "not json".to_owned(),
    ];
    for json in cases {
        let result = std::panic::catch_unwind(|| Tokenizer::from_json(&json));
        match result {
            Ok(Ok(_)) => panic!("accepted malformed input: {json}"),
            Ok(Err(_)) => {}
            Err(_) => panic!("panicked on: {json}"),
        }
    }
}

#[test]
fn added_tokens_round_trip_with_flags() {
    let mut tok = trained();
    tok.add_tokens(&[AddedToken::new("<mask>", false).lstrip(true)])
        .unwrap();
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    let a = reloaded.added_vocabulary().tokens_with_ids();
    let mask = a.iter().find(|t| t.token.content == "<mask>").unwrap();
    assert!(mask.token.lstrip && !mask.token.special);
    assert_eq!(
        reloaded
            .encode("the <mask>", false)
            .unwrap()
            .tokens()
            .last()
            .unwrap(),
        " <mask>"
    );
}
