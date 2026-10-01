//! Regression tests for the defects found in the v0.1 review. Each test
//! names the original bug.

use std::collections::HashMap;

use morpheme::decoders::{self, DecoderWrapper};
use morpheme::models::{Bpe, Unigram, WordPiece};
use morpheme::normalizers::BertNormalizer;
use morpheme::pre_tokenizers::{BertPreTokenizer, ByteLevel, Metaspace, PrependScheme};
use morpheme::processors::RobertaProcessing;
use morpheme::trainers::BpeTrainer;
use morpheme::{
    Decoder, Model, NormalizedString, Normalizer, OffsetType, PreTokenizedString, PreTokenizer,
    Tokenizer,
};

fn splits(pt: &dyn PreTokenizer, s: &str) -> Vec<String> {
    let mut pts = PreTokenizedString::from(s);
    pt.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetType::Byte)
        .into_iter()
        .map(|(s, _, _)| s.to_owned())
        .collect()
}

#[test]
fn bpe_trainer_output_is_loadable_and_usable() {
    // Was: merge_pair stored un-stripped symbols, so Bpe::new panicked
    // with "merge symbol ... missing from vocab".
    let mut tok =
        Tokenizer::new(Bpe::default()).with_pre_tokenizer(morpheme::pre_tokenizers::Whitespace);
    let trainer = BpeTrainer::builder().vocab_size(40).build().unwrap();
    tok.train(trainer, ["aaaa aaaa aaa abab the the th"].into_iter())
        .unwrap();
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    let enc = reloaded.encode("aaaa the th", false).unwrap();
    assert_eq!(enc.tokens(), &["aaaa", "the", "th"]);
}

#[test]
fn bert_normalizer_keeps_non_ascii_letters() {
    // Was: `ch as u8` truncation deleted Р (U+0420), Ġ, † …
    let mut n = NormalizedString::from("Россия aĠbĊc†d");
    BertNormalizer::default().normalize(&mut n).unwrap();
    assert_eq!(n.get(), "россия agbcc†d");
}

#[test]
fn bert_pre_tokenizer_matches_hf() {
    // Was: whitespace glued to the next token, punctuation runs merged.
    assert_eq!(
        splits(&BertPreTokenizer, "hello, world!! ok"),
        ["hello", ",", "world", "!", "!", "ok"]
    );
}

#[test]
fn byte_level_decoder_handles_multibyte() {
    // Was: per-byte from_utf8 turned every non-ASCII byte into '?'.
    let bl = ByteLevel::new(false, true, true);
    let pieces = splits(&bl, "héllo 😀");
    assert_eq!(pieces, ["hÃ©llo", "ĠðŁĺĢ"]);
    assert_eq!(bl.decode(pieces).unwrap(), "héllo 😀");
}

#[test]
fn byte_level_marks_leading_space() {
    // Was: Ġ never attached, so GPT-2 vocab entries like "Ġworld" never
    // matched.
    assert_eq!(
        splits(&ByteLevel::default(), "hello world"),
        ["Ġhello", "Ġworld"]
    );
    assert_eq!(
        splits(&ByteLevel::new(false, true, true), "hello world"),
        ["hello", "Ġworld"]
    );
}

#[test]
fn metaspace_marks_every_word() {
    // Was: only the first word got ▁.
    let m = Metaspace::new('▁', PrependScheme::Always, true);
    assert_eq!(splits(&m, "hello big world"), ["▁hello", "▁big", "▁world"]);
}

#[test]
fn wordpiece_decoder_inserts_spaces() {
    // Was: "helloworlds".
    let d = DecoderWrapper::from(decoders::WordPiece::new("##", true));
    let out = d
        .decode(vec!["hello".into(), "world".into(), "##s".into()])
        .unwrap();
    assert_eq!(out, "hello worlds");
}

#[test]
fn roberta_pair_layout_and_type_ids() {
    // Was: hard-coded ids, type id 2, single </s> separator.
    let vocab: HashMap<String, u32> = [("<s>", 0), ("</s>", 2), ("a", 5), ("b", 6)]
        .iter()
        .map(|(t, i)| (t.to_string(), *i))
        .collect();
    let model = morpheme::models::WordLevel::builder()
        .vocab(vocab)
        .unk_token("a")
        .build()
        .unwrap();
    let tok = Tokenizer::new(model)
        .with_pre_tokenizer(morpheme::pre_tokenizers::WhitespaceSplit)
        .with_post_processor(RobertaProcessing::new(("</s>", 2), ("<s>", 0)));
    let enc = tok.encode(("a", "b"), true).unwrap();
    assert_eq!(enc.tokens(), &["<s>", "a", "</s>", "</s>", "b", "</s>"]);
    assert_eq!(enc.type_ids(), &[0, 0, 0, 0, 0, 0]);
}

#[test]
fn unigram_emits_unk_instead_of_failing() {
    // Was: any unseen char failed the whole encode.
    let model = Unigram::new(
        vec![
            ("<unk>".into(), 0.0),
            ("▁a".into(), -1.0),
            ("a".into(), -2.0),
        ],
        Some(0),
        false,
    )
    .unwrap();
    let toks = model.tokenize("▁ax").unwrap();
    // Like HF: the unknown piece keeps its text and gets the unk id.
    let last = toks.last().unwrap();
    assert_eq!((last.id, last.value.as_str()), (0, "x"));
}

#[test]
fn wordpiece_unmatched_word_is_a_single_unk() {
    // Was: partial matches kept (["h", "##e", "<unk>"]); real BERT
    // vocabs (no </w>) couldn't match at all.
    let vocab: HashMap<String, u32> = [("[UNK]", 0), ("hello", 1), ("h", 2), ("##e", 3)]
        .iter()
        .map(|(t, i)| (t.to_string(), *i))
        .collect();
    let wp = WordPiece::builder().vocab(vocab).build().unwrap();
    let values = |s: &str| -> Vec<String> {
        wp.tokenize(s)
            .unwrap()
            .into_iter()
            .map(|t| t.value)
            .collect()
    };
    assert_eq!(values("hello"), ["hello"]);
    assert_eq!(values("hex"), ["[UNK]"]);
}

#[test]
fn hf_component_tags_are_recognized() {
    // Was: snake_case tags made every real component fall back silently.
    let json = r###"{"version":"1.0","added_tokens":[],
        "normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
        "pre_tokenizer":{"type":"BertPreTokenizer"},
        "decoder":{"type":"WordPiece","prefix":"##","cleanup":true},
        "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
                 "vocab":{"[UNK]":0,"hello":1,",":2,"world":3}}}"###;
    let tok = Tokenizer::from_json(json).unwrap();
    assert_eq!(
        tok.encode("HELLO, World", false).unwrap().tokens(),
        &["hello", ",", "world"]
    );
}

#[test]
fn bpe_byte_fallback_uses_hex_byte_tokens() {
    // Was: wired to the GPT-2 byte table instead of <0xNN> tokens.
    let vocab: HashMap<String, u32> = [("<unk>", 0), ("<0xC3>", 1), ("<0xA9>", 2), ("a", 3)]
        .iter()
        .map(|(t, i)| (t.to_string(), *i))
        .collect();
    let bpe = Bpe::builder()
        .vocab_and_merges(vocab, vec![])
        .unk_token("<unk>")
        .byte_fallback(true)
        .build()
        .unwrap();
    let values: Vec<String> = bpe
        .tokenize("aé")
        .unwrap()
        .into_iter()
        .map(|t| t.value)
        .collect();
    assert_eq!(values, ["a", "<0xC3>", "<0xA9>"]);
}

#[test]
fn vocab_with_shared_ids_round_trips_losslessly() {
    // Found by fuzzing: two tokens sharing an id (HF accepts this) lost
    // one of them on save, so the reloaded tokenizer encoded differently
    // and BPE merges could reference a vanished token.
    for model in [
        r#"{"type":"BPE","vocab":{"a":0,"b":1,"ab":1},"merges":[["a","b"]]}"#,
        r#"{"type":"WordLevel","vocab":{"x":0,"y":0,"[UNK]":1},"unk_token":"[UNK]"}"#,
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"x":1,"y":1},"unk_token":"[UNK]",
              "continuing_subword_prefix":"##","max_input_chars_per_word":100}"###,
    ] {
        let json = format!(
            r#"{{"version":"1.0","pre_tokenizer":{{"type":"WhitespaceSplit"}},"model":{model}}}"#
        );
        let tok = Tokenizer::from_json(&json).unwrap();
        let saved = tok.to_json(false).unwrap();
        let reloaded = Tokenizer::from_json(&saved).unwrap();
        assert_eq!(reloaded.to_json(false).unwrap(), saved, "{model}");
        assert_eq!(reloaded.vocab(false), tok.vocab(false), "{model}");
        for text in ["ab", "x y", "a b"] {
            assert_eq!(
                tok.encode(text, false).unwrap(),
                reloaded.encode(text, false).unwrap(),
                "{model}: {text}"
            );
        }
    }
}

/// Tokenizer over a tiny whole-word vocab, truncating only the second
/// sequence with a stride so that it overflows.
fn overflowing_pair_tokenizer(
    post_processor: morpheme::PostProcessorWrapper,
    max_length: usize,
) -> Tokenizer {
    let words = [
        "<s>", "</s>", "[CLS]", "[SEP]", "[UNK]", "a", "b", "c", "d", "e", "f", "g", "h",
    ];
    let vocab: HashMap<String, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = morpheme::models::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]")
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(model)
        .with_pre_tokenizer(morpheme::pre_tokenizers::WhitespaceSplit)
        .with_post_processor(post_processor);
    tok.set_truncation(Some(morpheme::TruncationParams {
        max_length,
        stride: 1,
        strategy: morpheme::TruncationStrategy::OnlySecond,
        ..Default::default()
    }))
    .unwrap();
    tok
}

fn overflow_type_ids(e: &morpheme::Encoding) -> Vec<Vec<u32>> {
    e.overflowing()
        .iter()
        .map(|o| o.type_ids().to_vec())
        .collect()
}

#[test]
fn roberta_overflow_type_ids_are_all_zero() {
    // PR #14 review: with `add_special_tokens = false`, the second
    // sequence's overflow kept type id 1 (Hugging Face does the same),
    // which RoBERTa's single-row type embedding cannot accept.
    let tok = overflowing_pair_tokenizer(RobertaProcessing::new(("</s>", 1), ("<s>", 0)).into(), 6);
    let enc = tok.encode(("a b", "c d e f g h"), false).unwrap();
    assert_eq!(enc.type_ids(), &[0; 6]);
    assert!(!enc.overflowing().is_empty());
    for ids in overflow_type_ids(&enc) {
        assert!(ids.iter().all(|t| *t == 0), "{ids:?}");
    }
}

#[test]
fn template_type_ids_apply_to_overflow() {
    // PR #14 review: `$B:3` was applied to the main encoding only; the
    // overflow kept type id 1 (Hugging Face does the same).
    let template = morpheme::processors::TemplateProcessing::builder()
        .try_single("[CLS] $A:2 [SEP]")
        .unwrap()
        .try_pair("[CLS] $A:2 [SEP] $B:3 [SEP]")
        .unwrap()
        .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
        .build()
        .unwrap();
    let tok = overflowing_pair_tokenizer(template.into(), 9);
    let enc = tok.encode(("a b", "c d e f g h"), true).unwrap();
    assert_eq!(enc.type_ids(), &[0, 2, 2, 0, 3, 3, 3, 3, 0]);
    assert_eq!(overflow_type_ids(&enc), vec![vec![0, 2, 2, 0, 3, 3, 3, 0]]);
}

/// A post-processor that adds nothing, so `process` is the trait's
/// default implementation.
struct Passthrough;

impl morpheme::PostProcessor for Passthrough {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }
    fn process_encodings(
        &self,
        encodings: Vec<morpheme::Encoding>,
        _add_special_tokens: bool,
    ) -> morpheme::Result<Vec<morpheme::Encoding>> {
        Ok(encodings)
    }
}

#[test]
fn default_pair_processing_sets_overflow_type_ids() {
    // PR #14 review: the default `process` set sequence ids on overflow
    // but not type ids, so a pair whose overflow was built with type 0
    // kept 0 instead of 1.
    use morpheme::PostProcessor;
    let mut second = morpheme::Encoding::from_tokens(
        vec![
            morpheme::Token::new(7, "c".into(), (0, 1)),
            morpheme::Token::new(8, "d".into(), (2, 3)),
        ],
        0,
    );
    second.truncate(1, 0, morpheme::TruncationDirection::Right);
    let first =
        morpheme::Encoding::from_tokens(vec![morpheme::Token::new(5, "a".into(), (0, 1))], 0);
    let merged = Passthrough.process(first, Some(second), false).unwrap();
    assert_eq!(merged.type_ids(), &[0, 1]);
    assert_eq!(overflow_type_ids(&merged), vec![vec![0, 1]]);
}

#[test]
fn word_level_trainer_never_leaves_id_holes() {
    // PR #14 review: a special token listed twice or also present in the
    // corpus got two ids, leaving gaps (Hugging Face does the same).
    let model = morpheme::models::WordLevel::builder()
        .unk_token("[UNK]")
        .build()
        .unwrap();
    let mut tok =
        Tokenizer::new(model).with_pre_tokenizer(morpheme::pre_tokenizers::WhitespaceSplit);
    let trainer = morpheme::trainers::WordLevelTrainer::builder()
        .vocab_size(100)
        .special_tokens(
            ["[UNK]", "[PAD]", "[UNK]"]
                .map(|t| morpheme::AddedToken::new(t, true))
                .to_vec(),
        )
        .build()
        .unwrap();
    tok.train(trainer, ["[UNK] a a b [UNK] [UNK] c"].into_iter())
        .unwrap();
    let mut vocab: Vec<(String, u32)> = tok.vocab(false).into_iter().collect();
    vocab.sort_by_key(|(_, id)| *id);
    let expected: Vec<(String, u32)> = [("[UNK]", 0), ("[PAD]", 1), ("a", 2), ("b", 3), ("c", 4)]
        .iter()
        .map(|(t, i)| (t.to_string(), *i))
        .collect();
    assert_eq!(vocab, expected);
}
