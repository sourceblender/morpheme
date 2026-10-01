//! End-to-end smoke tests against the hand-crafted vocab.

mod fixtures;

#[test]
fn encode_empty_string_yields_empty() {
    let t = fixtures::tokenizer();
    let enc = t.encode("").unwrap();
    assert!(enc.is_empty());
    assert_eq!(enc.len(), 0);
}

#[test]
fn encode_single_letter() {
    let t = fixtures::tokenizer();
    let enc = t.encode("a").unwrap();
    // Pre-token "a" is one character; the end-of-word suffix is
    // appended to the (only) symbol, giving "a</w>".
    assert_eq!(enc.tokens, vec!["a</w>"]);
    assert_eq!(enc.ids.len(), 1);
    assert_eq!(enc.offsets, vec![(0, 1)]);
}

#[test]
fn encode_two_letter_word_uses_merge() {
    let t = fixtures::tokenizer();
    let enc = t.encode("aa").unwrap();
    // Pre-token "aa" -> symbols ["a", "a</w>"] -> merge -> "aa</w>".
    assert_eq!(enc.tokens, vec!["aa</w>"]);
    assert_eq!(enc.ids.len(), 1);
    assert_eq!(enc.offsets, vec![(0, 2)]);
}

#[test]
fn encode_two_words_splits_on_whitespace() {
    let t = fixtures::tokenizer();
    let enc = t.encode("aa bb").unwrap();
    // Pre-token "aa" -> "aa</w>" via merge; pre-token "bb" -> "bb</w>".
    assert_eq!(enc.tokens, vec!["aa</w>", "bb</w>"]);
    assert_eq!(enc.ids.len(), 2);
    assert_eq!(enc.offsets, vec![(0, 2), (3, 5)]);
}

#[test]
fn encode_hello_uses_merges() {
    let t = fixtures::tokenizer();
    let enc = t.encode("hello").unwrap();
    assert_eq!(enc.tokens.len(), 1);
    assert!(enc.tokens[0].ends_with("</w>"));
}

#[test]
fn encode_collapses_runs_of_whitespace() {
    let t = fixtures::tokenizer();
    let enc = t.encode("aa   bb").unwrap();
    assert_eq!(enc.tokens, vec!["aa</w>", "bb</w>"]);
    assert_eq!(enc.ids.len(), 2);
}

#[test]
fn unknown_characters_fail() {
    let t = fixtures::tokenizer();
    // '1' is not in the vocab; the pre-token "a1" becomes symbols
    // ["a", "1</w>"], and "1</w>" is OOV.
    let err = t.encode("a1").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("1</w>") || msg.contains("token not in vocabulary"));
}
