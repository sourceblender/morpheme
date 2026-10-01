//! Arbitrary bytes as `tokenizer.json`: loading must never panic, and a
//! tokenizer that loads must survive use and a save → load round trip.

#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use morpheme::Tokenizer;

const TEXTS: [&str; 4] = ["Hello, world!", "  ünïcödé 你好 😀 ", "", "[CLS] a<mask>b"];

fuzz_target!(|data: &[u8]| {
    let Ok(mut tok) = Tokenizer::from_bytes(data) else {
        return;
    };
    common::within_budget(&mut tok);
    for text in TEXTS {
        common::exercise(&tok, text, "pair text");
    }
    let json = tok
        .to_json(false)
        .expect("a loaded tokenizer must serialize");
    let mut reloaded = Tokenizer::from_json(&json)
        .unwrap_or_else(|e| panic!("own JSON failed to reload: {e}\n{json}"));
    common::within_budget(&mut reloaded);
    // BPE dropout is random by design; outputs can't be compared.
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    if value["model"]["dropout"].as_f64().is_some_and(|p| p > 0.0) {
        return;
    }
    for text in TEXTS {
        assert_eq!(
            tok.encode(text, true).ok(),
            reloaded.encode(text, true).ok(),
            "save → load changed the encoding of {text:?}"
        );
    }
});
