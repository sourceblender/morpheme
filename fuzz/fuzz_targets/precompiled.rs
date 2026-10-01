//! Raw bytes as a SentencePiece `precompiled_charsmap` blob (T5, ALBERT,
//! XLM-R, …): `Precompiled::from_bytes` never panics, and anything that
//! parses normalizes a few fixed inputs with valid alignments and
//! round-trips through JSON unchanged.
//!
//! Seeds are the real charsmaps from the HF fixtures (`make_corpus.py`).

#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use morpheme::NormalizedString;
use morpheme::Normalizer;
use morpheme::normalizers::Precompiled;

/// Inputs that exercise the rule kinds real charsmaps contain: plain
/// ASCII, combining marks (NFKC-style composition), a deleted control
/// char at position 0 (issue #31), non-BMP chars, compatibility
/// ligatures and half-width kana.
const INPUTS: &[&str] = &[
    "hello world",
    "e\u{301}a\u{308} ab\u{300}c",
    "\u{7}ab",
    "\u{7}",
    "x\u{7}\u{7}ab",
    "\u{FEFF}ab",
    "😀a\u{1F600}b\u{10FFFF}",
    "\u{FB01}abc",
    "ｶﾞｷﾞ ＡＢＣ",
    " \u{A0}\u{2003}\t\n",
];

fuzz_target!(|data: &[u8]| {
    let Ok(p) = Precompiled::from_bytes(data) else {
        return;
    };
    assert_eq!(
        p.precompiled_charsmap(),
        data,
        "from_bytes altered the blob"
    );

    for input in INPUTS {
        let mut n = NormalizedString::from(*input);
        p.normalize(&mut n)
            .expect("Precompiled::normalize is infallible");
        common::check_alignment(&n);
        assert_eq!(
            p.normalize_str(input),
            n.get(),
            "normalize_str disagrees with normalize for {input:?}"
        );
    }

    // {"precompiled_charsmap": "<base64>"} round trip.
    let json = serde_json::to_string(&p).expect("serialize");
    let back: Precompiled = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("re-load of serialized charsmap failed: {e}\n{json}"));
    assert_eq!(back, p, "charsmap changed across JSON round trip");
});
