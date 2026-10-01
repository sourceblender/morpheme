//! Arbitrary text through real tokenizers (BERT, GPT-2, Llama, T5,
//! Qwen2.5): encoding never fails or panics, offsets are valid slices,
//! and byte-level BPE (GPT-2) round-trips losslessly.

#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (String, String)| {
    let (text, pair) = input;
    for (name, tok) in common::fixtures() {
        // These tokenizers can represent any input: encoding must succeed.
        let enc = tok
            .encode(text.as_str(), true)
            .unwrap_or_else(|e| panic!("{name}: encode failed: {e}"));
        common::check_offsets(&enc, &[&text]);
        let char_enc = tok
            .encode_char_offsets(text.as_str(), true)
            .unwrap_or_else(|e| panic!("{name}: encode_char_offsets failed: {e}"));
        assert_eq!(enc.ids(), char_enc.ids(), "{name}: offset type changed ids");
        let decoded = tok
            .decode(enc.ids(), false)
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
        if *name == "gpt2" {
            assert_eq!(decoded, text, "gpt2: byte-level round trip lost data");
        }
        let pair_enc = tok
            .encode((text.as_str(), pair.as_str()), true)
            .unwrap_or_else(|e| panic!("{name}: pair encode failed: {e}"));
        common::check_offsets(&pair_enc, &[&text, &pair]);
    }
});
