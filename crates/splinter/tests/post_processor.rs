//! Integration tests for post-processors.

use splinter::{RobertaPostProcessor, Tokenizer, WordPiece};

fn build_tokenizer() -> Tokenizer {
    // Small vocab: <s>=0, </s>=2, plus a few words.
    let tokens = vec![
        "<s>".to_string(),
        "<unk>".to_string(),
        "</s>".to_string(),
        "the</w>".to_string(),
        "quick</w>".to_string(),
        "brown</w>".to_string(),
        "fox</w>".to_string(),
        "hello</w>".to_string(),
    ];
    let vocab = splinter::Vocab::from_tokens(tokens).unwrap();
    let wp = WordPiece::new(vocab, "##", "<unk>", 100);
    let tok = Tokenizer::builder(splinter::ModelKind::WordPiece(wp)).build();
    tok.with_post_processor(Box::new(RobertaPostProcessor))
}

#[test]
fn roberta_single_inserts_specials() {
    let t = build_tokenizer();
    let enc = t.encode("the quick").unwrap();
    assert_eq!(enc.tokens[0], "<s>");
    assert_eq!(enc.tokens[enc.tokens.len() - 1], "</s>");
    assert_eq!(enc.ids[0], 0);
    assert_eq!(enc.ids[enc.ids.len() - 1], 2);
    // Type ids: <s>=0, sentence=0, </s>=2
    assert_eq!(enc.type_ids, vec![0, 0, 0, 2]);
}

#[test]
fn roberta_pair_uses_two_segments() {
    let t = build_tokenizer();
    let enc = t.encode_pair("the", "brown").unwrap();
    // <s> the </s> brown </s>
    assert_eq!(
        enc.tokens,
        vec!["<s>", "the</w>", "</s>", "brown</w>", "</s>"]
    );
    // type ids: 0, 0, 2, 1, 2
    assert_eq!(enc.type_ids, vec![0, 0, 2, 1, 2]);
}

#[test]
fn encode_pair_without_post_processor_errors() {
    let tokens = vec!["<unk>".to_string(), "the</w>".to_string()];
    let vocab = splinter::Vocab::from_tokens(tokens).unwrap();
    let wp = WordPiece::new(vocab, "##", "<unk>", 100);
    let tok = Tokenizer::builder(splinter::ModelKind::WordPiece(wp)).build();
    let err = tok.encode_pair("the", "the").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("post-processor"));
}
