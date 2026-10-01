//! Post-processing, truncation and padding through the full pipeline.

use std::collections::HashMap;

use morpheme::models::WordLevel;
use morpheme::pre_tokenizers::Whitespace;
use morpheme::processors::{BertProcessing, TemplateProcessing};
use morpheme::{
    AddedToken, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

fn tokenizer() -> Tokenizer {
    let words = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h",
    ];
    let vocab: HashMap<String, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]")
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
    tok.add_special_tokens(&["[PAD]", "[UNK]", "[CLS]", "[SEP]"].map(|s| AddedToken::new(s, true)))
        .unwrap();
    tok.with_post_processor(BertProcessing::new(("[SEP]", 3), ("[CLS]", 2)))
}

#[test]
fn bert_processing_single_and_pair() {
    let tok = tokenizer();
    let enc = tok.encode("a b", true).unwrap();
    assert_eq!(enc.ids(), &[2, 4, 5, 3]);
    assert_eq!(enc.special_tokens_mask(), &[1, 0, 0, 1]);
    assert_eq!(enc.word_ids(), &[None, Some(0), Some(1), None]);

    let pair = tok.encode(("a b", "c"), true).unwrap();
    assert_eq!(pair.ids(), &[2, 4, 5, 3, 6, 3]);
    assert_eq!(pair.type_ids(), &[0, 0, 0, 0, 1, 1]);
    assert_eq!(
        pair.sequence_ids(),
        vec![None, Some(0), Some(0), None, Some(1), None]
    );

    let raw = tok.encode("a b", false).unwrap();
    assert_eq!(raw.ids(), &[4, 5]);
}

#[test]
fn template_processing_with_custom_ids() {
    let tok = tokenizer().with_post_processor(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
            .build()
            .unwrap(),
    );
    assert_eq!(tok.encode("c", true).unwrap().ids(), &[2, 6, 3]);
}

#[test]
fn truncation_reserves_room_for_special_tokens_and_overflows() {
    let mut tok = tokenizer();
    tok.set_truncation(Some(TruncationParams {
        max_length: 5,
        stride: 1,
        strategy: TruncationStrategy::LongestFirst,
        direction: TruncationDirection::Right,
    }))
    .unwrap();
    let enc = tok.encode("a b c d e f", true).unwrap();
    assert_eq!(enc.tokens(), &["[CLS]", "a", "b", "c", "[SEP]"]);
    let overflow: Vec<Vec<String>> = enc
        .overflowing()
        .iter()
        .map(|o| o.tokens().to_vec())
        .collect();
    assert_eq!(
        overflow,
        vec![
            vec!["[CLS]", "c", "d", "e", "[SEP]"],
            vec!["[CLS]", "e", "f", "[SEP]"]
        ]
    );
}

#[test]
fn truncation_longest_first_on_pairs() {
    let mut tok = tokenizer();
    tok.set_truncation(Some(TruncationParams {
        max_length: 7,
        ..Default::default()
    }))
    .unwrap();
    let enc = tok.encode(("a b c d e", "f g"), true).unwrap();
    // 7 - 3 specials = 4 tokens: the longer sequence gives way.
    assert_eq!(
        enc.tokens(),
        &["[CLS]", "a", "b", "[SEP]", "f", "g", "[SEP]"]
    );
}

#[test]
fn invalid_stride_is_rejected() {
    let mut tok = tokenizer();
    let err = tok.set_truncation(Some(TruncationParams {
        max_length: 4,
        stride: 2,
        ..Default::default()
    }));
    assert!(err.is_err());
}

#[test]
fn padding_batch_longest_and_fixed() {
    let mut tok = tokenizer();
    tok.set_padding(Some(PaddingParams {
        pad_id: 0,
        pad_token: "[PAD]".into(),
        ..Default::default()
    }));
    let batch = tok.encode_batch(vec!["a", "a b c"], true).unwrap();
    assert_eq!(batch[0].ids(), &[2, 4, 3, 0, 0]);
    assert_eq!(batch[0].attention_mask(), &[1, 1, 1, 0, 0]);
    assert_eq!(batch[1].len(), 5);

    tok.set_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(6),
        direction: PaddingDirection::Left,
        pad_to_multiple_of: Some(4),
        ..Default::default()
    }));
    let enc = tok.encode("a", true).unwrap();
    assert_eq!(enc.ids(), &[0, 0, 0, 0, 0, 2, 4, 3][..]);
}
