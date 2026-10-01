//! `tokenizer.json` loading/saving: round-trips, legacy formats, and
//! errors (never panics) on malformed input.

use splinter::models::Bpe;
use splinter::normalizers::Lowercase;
use splinter::pre_tokenizers::Whitespace;
use splinter::trainers::BpeTrainer;
use splinter::{AddedToken, Tokenizer};

fn trained() -> Tokenizer {
    let mut tok = Tokenizer::new(Bpe::builder().unk_token("[UNK]").build().unwrap())
        .with_normalizer(Lowercase)
        .with_pre_tokenizer(Whitespace);
    let trainer = BpeTrainer::builder()
        .vocab_size(120)
        .special_tokens(vec![AddedToken::from("[UNK]", true)])
        .build();
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
    tok.add_tokens(&[AddedToken::from("<mask>", false).lstrip(true)])
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
