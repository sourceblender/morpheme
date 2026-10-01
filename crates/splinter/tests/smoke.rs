//! End-to-end: build tokenizers in code, train them, encode, decode.

use splinter::decoders::{self, DecoderWrapper};
use splinter::models::{Bpe, Unigram, WordLevel, WordPiece};
use splinter::normalizers::{BertNormalizer, Nfkc};
use splinter::pre_tokenizers::{BertPreTokenizer, ByteLevel, Metaspace, PrependScheme, Whitespace};
use splinter::processors::TemplateProcessing;
use splinter::trainers::{BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer};
use splinter::{AddedToken, Tokenizer};

const CORPUS: &str = include_str!("../../../examples/corpus.txt");

fn lines() -> impl Iterator<Item = &'static str> + Send {
    CORPUS.lines()
}

#[test]
fn byte_level_bpe_round_trips_any_text() {
    let mut tok = Tokenizer::new(Bpe::default())
        .with_pre_tokenizer(ByteLevel::new(false, true, true))
        .with_decoder(ByteLevel::default())
        .with_post_processor(ByteLevel::new(false, true, true));
    let trainer = BpeTrainer::builder()
        .vocab_size(400)
        .initial_alphabet(ByteLevel::alphabet())
        .special_tokens(vec![AddedToken::from("<|endoftext|>", true)])
        .build();
    tok.train(trainer, lines()).unwrap();

    for text in [
        "the quick brown fox",
        "Unseen wörds, émojis 😀 and 中文 are fine",
        "  spaces\tand\nnewlines  ",
        "",
    ] {
        let enc = tok.encode(text, true).unwrap();
        assert_eq!(tok.decode(enc.ids(), false).unwrap(), text, "{text:?}");
        for (token_offsets, _) in enc.offsets().iter().zip(enc.tokens()) {
            assert!(text.get(token_offsets.0..token_offsets.1).is_some());
        }
    }
    let enc = tok.encode("hello<|endoftext|>", true).unwrap();
    assert_eq!(enc.tokens().last().unwrap(), "<|endoftext|>");
    assert_eq!(tok.decode(enc.ids(), true).unwrap(), "hello");
}

#[test]
fn bert_style_wordpiece() {
    let mut tok = Tokenizer::new(WordPiece::default())
        .with_normalizer(BertNormalizer::default())
        .with_pre_tokenizer(BertPreTokenizer)
        .with_decoder(DecoderWrapper::from(decoders::WordPiece::new("##", true)));
    let specials: Vec<AddedToken> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        .iter()
        .map(|s| AddedToken::from(*s, true))
        .collect();
    let trainer = WordPieceTrainer::builder()
        .vocab_size(300)
        .special_tokens(specials)
        .build();
    tok.train(trainer, lines()).unwrap();
    let cls = tok.token_to_id("[CLS]").unwrap();
    let sep = tok.token_to_id("[SEP]").unwrap();
    assert_eq!((tok.token_to_id("[PAD]"), cls, sep), (Some(0), 2, 3));
    tok.set_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", cls), ("[SEP]", sep)])
            .build()
            .unwrap()
            .into(),
    ));

    let enc = tok.encode("The Quick, brown FOX!", true).unwrap();
    assert_eq!(enc.tokens().first().unwrap(), "[CLS]");
    assert_eq!(enc.ids()[0], cls);
    assert_eq!(enc.tokens().last().unwrap(), "[SEP]");
    // Lowercased by the normalizer, punctuation split off.
    assert!(enc.tokens().contains(&",".to_owned()));
    assert_eq!(
        tok.decode(enc.ids(), true).unwrap(),
        "the quick, brown fox!"
    );

    let pair = tok.encode(("hello there", "general"), true).unwrap();
    let first_b = pair.type_ids().iter().position(|t| *t == 1).unwrap();
    assert_eq!(pair.tokens()[first_b - 1], "[SEP]");
    assert_eq!(*pair.type_ids().last().unwrap(), 1);
}

#[test]
fn sentencepiece_style_unigram() {
    let marker = Metaspace::new('▁', PrependScheme::Always, true);
    let mut tok = Tokenizer::new(Unigram::default())
        .with_normalizer(Nfkc)
        .with_pre_tokenizer(marker.clone())
        .with_decoder(marker);
    let trainer = UnigramTrainer::builder()
        .vocab_size(150)
        .special_tokens(vec![AddedToken::from("<unk>", true)])
        .unk_token(Some("<unk>".into()))
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, lines()).unwrap();
    assert!(tok.get_vocab_size(false) <= 150);

    let unk = tok.token_to_id("<unk>").unwrap();
    for line in lines().filter(|l| !l.trim().is_empty()) {
        let enc = tok.encode(line, false).unwrap();
        assert!(
            !enc.ids().contains(&unk),
            "unk while encoding training line {line:?}"
        );
        assert_eq!(
            tok.decode(enc.ids(), false).unwrap(),
            line.split_whitespace().collect::<Vec<_>>().join(" ")
        );
    }
    // Characters never seen in training map to <unk>, they don't fail.
    let enc = tok.encode("zebra ⌘", false).unwrap();
    assert!(enc.ids().contains(&unk));
}

#[test]
fn word_level() {
    let model = WordLevel::builder().unk_token("[UNK]").build().unwrap();
    let mut tok = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
    let trainer = WordLevelTrainer::builder()
        .vocab_size(1000)
        .special_tokens(vec![AddedToken::from("[UNK]", true)])
        .build();
    tok.train(trainer, lines()).unwrap();
    let enc = tok.encode("the fox flibbertigibbet.", false).unwrap();
    assert_eq!(enc.tokens(), &["the", "fox", "[UNK]", "."]);
}

#[test]
fn training_from_missing_file_reports_the_error() {
    let mut tok = Tokenizer::new(Bpe::default());
    let err = tok
        .train_from_files(BpeTrainer::default(), &["/definitely/not/here.txt"])
        .unwrap_err();
    assert!(err.to_string().contains("not/here.txt"), "{err}");
}
