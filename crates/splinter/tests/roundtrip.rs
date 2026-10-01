//! Property tests for the encode pipeline.

mod fixtures;

use proptest::prelude::*;

/// Strategy: words made of `a` and `b` only (the only single-char
/// symbols in the test vocab that participate in merges).
fn arb_word() -> impl Strategy<Value = String> {
    prop::collection::vec(prop_oneof![Just('a'), Just('b')], 1..8)
        .prop_map(|v| v.into_iter().collect())
}

/// Strategy: sentences of `arb_word` joined by single spaces.
fn arb_sentence() -> impl Strategy<Value = String> {
    prop::collection::vec(arb_word(), 1..6).prop_map(|words| words.join(" "))
}

proptest! {
    #[test]
    fn encode_is_deterministic(s in arb_sentence()) {
        let t = fixtures::tokenizer();
        let a = t.encode(&s).unwrap();
        let b = t.encode(&s).unwrap();
        prop_assert_eq!(a.ids, b.ids);
        prop_assert_eq!(a.tokens, b.tokens);
        prop_assert_eq!(a.offsets, b.offsets);
    }

    #[test]
    fn every_token_resolves_to_an_id(s in arb_sentence()) {
        let t = fixtures::tokenizer();
        let enc = t.encode(&s).unwrap();
        let vocab = t.bpe_model().expect("bpe").vocab.len();
        for id in enc.ids {
            prop_assert!((id as usize) < vocab);
        }
    }

    #[test]
    fn tokens_per_pretoken_in_range(s in arb_sentence()) {
        let t = fixtures::tokenizer();
        let enc = t.encode(&s).unwrap();
        // Every pre-token in the input must produce at least one
        // output token, and the total token count must not exceed
        // the number of characters across all pre-tokens (the
        // degenerate upper bound when no merges apply).
        let pre_tokens: Vec<&str> = s.split_ascii_whitespace().collect();
        let char_total: usize = pre_tokens.iter().map(|w| w.len()).sum();
        prop_assert!(enc.ids.len() >= pre_tokens.len());
        prop_assert!(enc.ids.len() <= char_total);
    }
}
