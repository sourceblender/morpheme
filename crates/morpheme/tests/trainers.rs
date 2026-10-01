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

// ---------------------------------------------------------------------
// Issue #47: behavioural tests for the Unigram trainer knobs and BPE
// parity with both affixes.
// ---------------------------------------------------------------------

const UNIGRAM_CORPUS: &[&str] = &[
    "internationalization internationalization internationalization",
    "international nation national nationalization",
    "organization organizations organize organized",
    "the nation and the organization are international",
];

fn unigram_trainer() -> morpheme::trainers::UnigramTrainerBuilder {
    UnigramTrainer::builder()
        .vocab_size(40)
        .show_progress(false)
        .unk_token("<unk>")
        .special_tokens(vec![AddedToken::new("<unk>", true)])
}

fn train_unigram(trainer: UnigramTrainer) -> Tokenizer {
    let mut tok = Tokenizer::new(Unigram::default()).with_pre_tokenizer(Whitespace);
    tok.train(trainer, UNIGRAM_CORPUS.iter()).unwrap();
    tok
}

/// Non-special pieces of the trained vocabulary.
fn pieces(tok: &Tokenizer) -> Vec<String> {
    let mut v: Vec<String> = tok
        .vocab(false)
        .into_keys()
        .filter(|p| p != "<unk>")
        .collect();
    v.sort();
    v
}

/// Every corpus word encodes without `<unk>` and decodes back.
fn assert_covers_corpus(tok: &Tokenizer) {
    for word in UNIGRAM_CORPUS.iter().flat_map(|s| s.split(' ')) {
        let enc = tok.encode(word, false).unwrap();
        assert!(!enc.tokens().is_empty(), "{word}: empty encoding");
        assert!(
            !enc.tokens().iter().any(|t| t == "<unk>"),
            "{word}: {:?}",
            enc.tokens()
        );
        assert_eq!(enc.tokens().concat(), word);
    }
}

/// `max_piece_length` caps the length of every learned piece in chars,
/// and the cap is observable: the default (16) learns longer pieces from
/// the same corpus.
#[test]
fn unigram_max_piece_length_caps_piece_length() {
    let capped = train_unigram(unigram_trainer().max_piece_length(3).build().unwrap());
    let longest = pieces(&capped)
        .iter()
        .map(|p| p.chars().count())
        .max()
        .unwrap();
    assert!(longest <= 3, "{:?}", pieces(&capped));
    assert!(capped.vocab_size(true) <= 40);
    assert_covers_corpus(&capped);
    // Long words are now spelled out from short pieces.
    let enc = capped.encode("internationalization", false).unwrap();
    assert!(enc.tokens().len() >= 7, "{:?}", enc.tokens());

    let default = train_unigram(unigram_trainer().build().unwrap());
    let longest = pieces(&default)
        .iter()
        .map(|p| p.chars().count())
        .max()
        .unwrap();
    assert!(longest > 3, "{:?}", pieces(&default));
    assert!(longest <= 16);

    // Multi-byte chars count as one each.
    let mut tok = Tokenizer::new(Unigram::default()).with_pre_tokenizer(Whitespace);
    tok.train(
        unigram_trainer().max_piece_length(2).build().unwrap(),
        ["ééé ééé ééé 日本語 日本語 日本語"].into_iter(),
    )
    .unwrap();
    let longest = pieces(&tok).iter().map(|p| p.chars().count()).max();
    assert_eq!(longest, Some(2), "{:?}", pieces(&tok));
    assert!(
        pieces(&tok)
            .iter()
            .any(|p| p == "éé" || p == "日本" || p == "本語")
    );

    assert!(unigram_trainer().max_piece_length(0).build().is_err());
}

/// `seed_size` bounds the initial candidate set. The seeds always start
/// with the required single chars, so a `seed_size` no larger than the
/// alphabet leaves nothing but chars to prune, while a large seed set
/// yields multi-char pieces.
#[test]
fn unigram_seed_size_limits_candidate_pieces() {
    let chars_only = train_unigram(unigram_trainer().seed_size(1).build().unwrap());
    let p = pieces(&chars_only);
    assert!(p.iter().all(|s| s.chars().count() == 1), "{p:?}");
    let alphabet: std::collections::HashSet<char> = UNIGRAM_CORPUS
        .iter()
        .flat_map(|s| s.chars())
        .filter(|c| !c.is_whitespace())
        .collect();
    assert_eq!(p.len(), alphabet.len());
    assert_covers_corpus(&chars_only);

    let seeded = train_unigram(unigram_trainer().seed_size(1_000).build().unwrap());
    let p = pieces(&seeded);
    assert!(p.iter().any(|s| s.chars().count() > 1), "{p:?}");
    assert!(seeded.vocab_size(true) <= 40);
    assert_covers_corpus(&seeded);

    // A tiny seed set is a subset of a larger one's vocabulary.
    let small = train_unigram(unigram_trainer().seed_size(30).build().unwrap());
    let large = train_unigram(unigram_trainer().seed_size(1_000).build().unwrap());
    assert!(small.vocab_size(true) <= large.vocab_size(true));
}

/// `shrinking_factor` and `n_sub_iterations` change how aggressively the
/// vocabulary is pruned per round, never the invariants: the result is
/// deterministic, within `vocab_size`, covers the corpus, and the
/// special-token ids stay fixed.
#[test]
fn unigram_shrinking_factor_and_n_sub_iterations_are_deterministic() {
    for (factor, sub) in [(0.5, 1), (0.5, 4), (0.95, 1), (0.95, 4), (0.75, 2)] {
        let trainer = || {
            unigram_trainer()
                .shrinking_factor(factor)
                .n_sub_iterations(sub)
                .build()
                .unwrap()
        };
        let first = train_unigram(trainer());
        let second = train_unigram(trainer());
        assert_eq!(
            first.vocab(true),
            second.vocab(true),
            "factor {factor}, sub-iterations {sub}"
        );
        assert_eq!(
            first.to_json(false).unwrap(),
            second.to_json(false).unwrap(),
            "factor {factor}, sub-iterations {sub}: scores differ"
        );
        assert!(
            first.vocab_size(true) <= 40,
            "factor {factor}: {}",
            first.vocab_size(true)
        );
        assert_eq!(first.token_to_id("<unk>"), Some(0));
        assert_covers_corpus(&first);
        // Whole sentences encode without `<unk>` and keep word boundaries.
        for s in UNIGRAM_CORPUS {
            let enc = first.encode(*s, false).unwrap();
            assert!(!enc.tokens().iter().any(|t| t == "<unk>"), "{s}");
            let words = enc.word_ids().iter().flatten().max().map(|w| w + 1);
            assert_eq!(words, Some(s.split(' ').count() as u32), "{s}");
        }
    }

    // A gentler shrinking factor never ends with a smaller vocabulary
    // than an aggressive one on the same corpus.
    let aggressive = train_unigram(unigram_trainer().shrinking_factor(0.5).build().unwrap());
    let gentle = train_unigram(unigram_trainer().shrinking_factor(0.95).build().unwrap());
    assert!(gentle.vocab_size(true) >= aggressive.vocab_size(true));

    // Validation: the factor must be strictly inside (0, 1) and at least
    // one sub-iteration is required.
    for bad in [0.0, 1.0, -0.5, 1.5] {
        assert!(unigram_trainer().shrinking_factor(bad).build().is_err());
    }
    assert!(unigram_trainer().n_sub_iterations(0).build().is_err());
}

/// BPE training with both a `continuing_subword_prefix` and an
/// `end_of_word_suffix` matches Python `tokenizers` 0.23.2:
///
/// ```python
/// tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.Whitespace()
/// tok.train_from_iterator(["the then there these thesis", "the the the"],
///     trainers.BpeTrainer(vocab_size=50, continuing_subword_prefix="##", end_of_word_suffix="</w>"))
/// ```
///
/// Affixed symbols get their ids in hash-map order in HF itself, so the
/// vocabulary is compared as a set (as the in-crate suffix parity test
/// does). Those ids also break ties between equal-frequency pairs, and
/// the tie order differs between HF runs (four runs of the script above
/// gave two different orders for `(##i, ##s</w>)` / `(##r, ##e</w>)`), so
/// the merge list is compared as a set too, with the unique top merge and
/// the resulting encodings exact.
#[test]
fn bpe_trainer_with_prefix_and_suffix_matches_python() {
    let mut tok = Tokenizer::new(Bpe::default()).with_pre_tokenizer(Whitespace);
    let trainer = BpeTrainer::builder()
        .vocab_size(50)
        .show_progress(false)
        .continuing_subword_prefix("##")
        .end_of_word_suffix("</w>")
        .build()
        .unwrap();
    tok.train(
        trainer,
        ["the then there these thesis", "the the the"].into_iter(),
    )
    .unwrap();

    let expected_vocab: std::collections::HashSet<&str> = [
        "e",
        "h",
        "i",
        "n",
        "r",
        "s",
        "t",
        "##h",
        "##e",
        "##s",
        "##i",
        "##s</w>",
        "##r",
        "##e</w>",
        "##n</w>",
        "th",
        "the",
        "the</w>",
        "thes",
        "##is</w>",
        "##re</w>",
        "then</w>",
        "there</w>",
        "these</w>",
        "thesis</w>",
    ]
    .into();
    let vocab = tok.vocab(false);
    assert_eq!(
        vocab
            .keys()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>(),
        expected_vocab
    );
    // The single-char alphabet is laid out first, in sorted order.
    for (i, c) in ["e", "h", "i", "n", "r", "s", "t"].iter().enumerate() {
        assert_eq!(vocab[*c], i as u32, "{c}");
    }

    let expected_merges = [
        ("t", "##h"),
        ("th", "##e"),
        ("th", "##e</w>"),
        ("the", "##s"),
        ("##i", "##s</w>"),
        ("##r", "##e</w>"),
        ("the", "##n</w>"),
        ("the", "##re</w>"),
        ("thes", "##e</w>"),
        ("thes", "##is</w>"),
    ]
    .map(|(a, b)| (a.to_string(), b.to_string()));
    let model = serde_json::to_value(tok.model()).unwrap();
    let merges: Vec<(String, String)> = serde_json::from_value(model["merges"].clone()).unwrap();
    assert_eq!(merges.len(), expected_merges.len(), "{merges:?}");
    assert_eq!(merges[0], expected_merges[0]);
    assert_eq!(
        merges.iter().collect::<std::collections::HashSet<_>>(),
        expected_merges
            .iter()
            .collect::<std::collections::HashSet<_>>()
    );
    // The two equal-frequency ties sit at the same ranks in either order.
    let rank = |a: &str, b: &str| merges.iter().position(|(x, y)| x == a && y == b).unwrap();
    assert_eq!(rank("th", "##e").min(rank("th", "##e</w>")), 1);
    assert_eq!(rank("th", "##e").max(rank("th", "##e</w>")), 2);
    assert_eq!(rank("##i", "##s</w>").min(rank("##r", "##e</w>")), 4);
    assert_eq!(rank("##i", "##s</w>").max(rank("##r", "##e</w>")), 5);
    assert_eq!(model["continuing_subword_prefix"], "##");
    assert_eq!(model["end_of_word_suffix"], "</w>");

    type Case<'a> = (&'a str, &'a [&'a str], &'a [(usize, usize)]);
    let cases: &[Case] = &[
        ("the", &["the</w>"], &[(0, 3)]),
        ("then", &["then</w>"], &[(0, 4)]),
        ("thesis", &["thesis</w>"], &[(0, 6)]),
        (
            "theses",
            &["thes", "##e", "##s</w>"],
            &[(0, 4), (4, 5), (5, 6)],
        ),
        ("x", &[], &[]),
    ];
    for (input, tokens, offsets) in cases {
        let enc = tok.encode(*input, false).unwrap();
        assert_eq!(enc.tokens(), *tokens, "{input:?}");
        assert_eq!(enc.offsets(), *offsets, "{input:?}");
    }

    // The trained affixes survive a JSON round trip.
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(
        reloaded.encode("theses", false).unwrap().tokens(),
        ["thes", "##e", "##s</w>"]
    );
}
