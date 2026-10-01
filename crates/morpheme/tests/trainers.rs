//! End-to-end trainer behavior through `Tokenizer::train`.

use morpheme::models::{Bpe, Unigram, WordPiece};
use morpheme::pre_tokenizers::{ByteLevel, Whitespace};
use morpheme::trainers::{BpeTrainer, UnigramTrainer, WordPieceTrainer};
use morpheme::{AddedToken, Error, Tokenizer};

const CORPUS: &[&str] = &["playing played player plays", "replay replayed"];

/// Issue #33: `WordPieceTrainer::default()` must train the same way as
/// `WordPieceTrainer::builder().build()`, with `##` continuation tokens.
#[test]
fn default_wordpiece_trainer_produces_continuation_tokens() {
    let mut tok = Tokenizer::new(WordPiece::default()).with_pre_tokenizer(Whitespace);
    tok.train(WordPieceTrainer::default(), CORPUS.iter())
        .unwrap();

    let vocab = tok.vocab(false);
    let continuation = vocab.keys().filter(|k| k.starts_with("##")).count();
    assert!(continuation > 0, "no ## tokens in {vocab:?}");

    // Every corpus word (and a word built from its pieces) encodes
    // without the unknown token.
    for word in ["playing", "replayed", "plays", "player"] {
        let enc = tok.encode(word, false).unwrap();
        assert!(
            !enc.tokens().contains(&"[UNK]".to_string()),
            "{word}: {:?}",
            enc.tokens()
        );
        let rebuilt: String = enc
            .tokens()
            .iter()
            .map(|t| t.trim_start_matches("##"))
            .collect();
        assert_eq!(rebuilt, word);
    }

    // The result is identical to the builder's defaults.
    let mut via_builder = Tokenizer::new(WordPiece::default()).with_pre_tokenizer(Whitespace);
    via_builder
        .train(WordPieceTrainer::builder().build().unwrap(), CORPUS.iter())
        .unwrap();
    assert_eq!(tok.vocab(false), via_builder.vocab(false));
}

/// `BpeTrainer::default()` trains a usable model too.
#[test]
fn default_bpe_trainer_produces_usable_model() {
    let mut tok = Tokenizer::new(Bpe::default()).with_pre_tokenizer(Whitespace);
    tok.train(BpeTrainer::default(), CORPUS.iter()).unwrap();
    assert!(tok.vocab_size(false) > 10);
    // With the default (30 000) budget every corpus word is fully merged.
    for word in ["playing", "replayed"] {
        assert_eq!(tok.encode(word, false).unwrap().tokens(), [word]);
    }
    let enc = tok.encode("plays played", false).unwrap();
    assert_eq!(tok.decode(enc.ids(), false).unwrap(), "plays played");
}

/// Issue #32: for Unigram, `vocab_size` is a hard cap that includes the
/// special tokens, and a budget too small to hold them plus the required
/// chars is an error rather than a silent overshoot.
#[test]
fn unigram_vocab_size_is_a_hard_cap_including_special_tokens() {
    let specials = || vec![AddedToken::new("<s>", true), AddedToken::new("</s>", true)];
    let corpus = || ["ab ab ab abc abc bc"].into_iter();

    let mut tok = Tokenizer::new(Unigram::default()).with_pre_tokenizer(Whitespace);
    let trainer = UnigramTrainer::builder()
        .vocab_size(4)
        .special_tokens(specials())
        .show_progress(false)
        .build()
        .unwrap();
    let err = tok.train(trainer, corpus()).unwrap_err();
    assert!(matches!(err, Error::Training(_)), "{err:?}");
    // Training failed before touching the tokenizer.
    assert_eq!(tok.vocab_size(true), 1);

    for size in 5..10 {
        let mut tok = Tokenizer::new(Unigram::default()).with_pre_tokenizer(Whitespace);
        let trainer = UnigramTrainer::builder()
            .vocab_size(size)
            .special_tokens(specials())
            .show_progress(false)
            .build()
            .unwrap();
        tok.train(trainer, corpus()).unwrap();
        assert!(
            tok.vocab_size(true) <= size,
            "{size}: {}",
            tok.vocab_size(true)
        );
        assert_eq!(tok.token_to_id("<s>"), Some(0));
        assert_eq!(tok.token_to_id("</s>"), Some(1));
    }
}

/// Issue #34: a special token that also occurs in the corpus gets one id.
#[test]
fn unigram_special_token_that_is_also_a_piece_is_not_duplicated() {
    let mut tok = Tokenizer::new(Unigram::default()).with_pre_tokenizer(Whitespace);
    let trainer = UnigramTrainer::builder()
        .special_tokens(vec![AddedToken::new("a", true)])
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, ["a b ab"].into_iter()).unwrap();

    let vocab = tok.vocab(false);
    assert_eq!(vocab.len(), tok.vocab_size(false), "duplicate entries");
    assert_eq!(tok.token_to_id("a"), Some(0));
    assert_eq!(tok.vocab_size(true), tok.vocab_size(false));
    // Round-tripping through JSON keeps the vocabulary duplicate-free.
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(reloaded.token_to_id("a"), Some(0));
    assert_eq!(reloaded.vocab(false), vocab);
}

/// Documented HF-parity behavior: BPE performs no merges, and does not
/// error, when the special tokens and alphabet alone exceed `vocab_size`.
#[test]
fn bpe_keeps_hf_behavior_when_alphabet_exceeds_vocab_size() {
    let mut tok =
        Tokenizer::new(Bpe::default()).with_pre_tokenizer(ByteLevel::new(false, true, true));
    let trainer = BpeTrainer::builder()
        .vocab_size(5)
        .initial_alphabet(ByteLevel::alphabet())
        .special_tokens(vec![AddedToken::new("<|endoftext|>", true)])
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, CORPUS.iter()).unwrap();
    assert_eq!(tok.vocab_size(true), 257);
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(0));
    // No merges: every token is a single byte-level char.
    let enc = tok.encode("play", false).unwrap();
    assert_eq!(enc.tokens().len(), 4);
}
