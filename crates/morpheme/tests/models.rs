//! Model behaviour through the public API (issue #47): BPE cache
//! invalidation and concurrency, `end_of_word_suffix` alone,
//! `byte_fallback` with a prefix, and WordPiece `max_input_chars_per_word`
//! edge cases. Expected values come from Python `tokenizers` 0.23.2.
//!
//! Already covered by the model's own unit tests (PR #58) and therefore not
//! repeated here: a duplicated merge pair takes its last rank, and
//! `ignore_merges` is ignored under dropout.

use std::collections::HashMap;

use morpheme::decoders::BpeDecoder;
use morpheme::models::{Bpe, Unigram, WordPiece};
use morpheme::pre_tokenizers::{Whitespace, WhitespaceSplit};
use morpheme::trainers::BpeTrainer;
use morpheme::{Encoding, Model, Tokenizer};

/// `(input, tokens, ids, offsets)` as printed by the Python scripts.
type FullCase<'a> = (&'a str, &'a [&'a str], &'a [u32], &'a [(usize, usize)]);
/// `(input, tokens, ids)`.
type IdCase<'a> = (&'a str, &'a [&'a str], &'a [u32]);
/// `(input, tokens, offsets)`.
type OffsetCase<'a> = (&'a str, &'a [&'a str], &'a [(usize, usize)]);

fn vocab(entries: &[(&str, u32)]) -> HashMap<String, u32> {
    entries.iter().map(|(t, i)| (t.to_string(), *i)).collect()
}

fn merges(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn tokens_ids_offsets(e: &Encoding) -> (Vec<&str>, Vec<u32>, Vec<(usize, usize)>) {
    (
        e.tokens().iter().map(String::as_str).collect(),
        e.ids().to_vec(),
        e.offsets().to_vec(),
    )
}

/// Retraining replaces the merges, so a word cached under the old merges
/// must not be served from the cache afterwards.
#[test]
fn retraining_invalidates_the_bpe_word_cache() {
    let bpe = Bpe::builder()
        .vocab_and_merges(
            vocab(&[("a", 0), ("b", 1), ("ab", 2)]),
            merges(&[("a", "b")]),
        )
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(bpe).with_pre_tokenizer(Whitespace);
    // Populate the cache.
    for _ in 0..3 {
        assert_eq!(tok.encode("ab", false).unwrap().tokens(), ["ab"]);
    }

    // The new vocabulary has room for the alphabet only: no merges.
    let trainer = BpeTrainer::builder()
        .vocab_size(2)
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, ["ab ab a b"].into_iter()).unwrap();
    assert_eq!(tok.vocab_size(false), 2);
    assert_eq!(tok.token_to_id("ab"), None);

    let enc = tok.encode("ab", false).unwrap();
    assert_eq!(enc.tokens(), ["a", "b"]);
    assert_eq!(
        enc.ids(),
        [tok.token_to_id("a").unwrap(), tok.token_to_id("b").unwrap()]
    );
    // Same at the model level, and the result is stable on repeat.
    let direct: Vec<String> = tok
        .model()
        .tokenize("ab")
        .unwrap()
        .into_iter()
        .map(|t| t.value)
        .collect();
    assert_eq!(direct, ["a", "b"]);
    assert_eq!(tok.encode("ab", false).unwrap(), enc);

    // And the other way round: a merge learned by training is used even
    // though the word was cached unmerged before.
    let mut tok = Tokenizer::new(Bpe::default()).with_pre_tokenizer(Whitespace);
    let trainer = BpeTrainer::builder()
        .vocab_size(2)
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, ["ab ab a b"].into_iter()).unwrap();
    assert_eq!(tok.encode("ab", false).unwrap().tokens(), ["a", "b"]);
    let trainer = BpeTrainer::builder()
        .vocab_size(3)
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, ["ab ab a b"].into_iter()).unwrap();
    assert_eq!(tok.encode("ab", false).unwrap().tokens(), ["ab"]);
}

/// A large batch with many repeated words, built from `distinct` base
/// words so the caches see both hits and misses.
fn repeated_batch(words: &[&str], len: usize) -> Vec<String> {
    (0..len)
        .map(|i| {
            let a = words[i % words.len()];
            let b = words[(i * 7 + 3) % words.len()];
            let c = words[(i * 13 + 5) % words.len()];
            match i % 4 {
                0 => a.to_string(),
                1 => format!("{a} {b}"),
                2 => format!("{a} {b} {c} {a}"),
                _ => format!("{c}{a} {b}{b}"),
            }
        })
        .collect()
}

/// Encode the batch from several threads at once (exercising the sharded
/// BPE cache / the Unigram cache) and check every result against the
/// sequential, single-threaded `encode`.
fn assert_concurrent_batches_match_sequential(tok: &Tokenizer, inputs: &[String]) {
    let expected: Vec<Encoding> = inputs
        .iter()
        .map(|s| tok.encode(s.as_str(), false).unwrap())
        .collect();
    assert!(
        expected.iter().any(|e| e.len() > 1),
        "the batch should produce multi-token encodings"
    );

    let run_threads = |tok: &Tokenizer| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|t| {
                    let expected = &expected;
                    scope.spawn(move || {
                        // Each thread starts at a different offset so the
                        // caches are hit in different orders.
                        let rotated: Vec<&str> = inputs
                            .iter()
                            .cycle()
                            .skip(t * 101)
                            .take(inputs.len())
                            .map(String::as_str)
                            .collect();
                        let got = tok.encode_batch(rotated, false).unwrap();
                        for (i, enc) in got.iter().enumerate() {
                            let want = &expected[(i + t * 101) % inputs.len()];
                            assert_eq!(enc, want, "thread {t}, item {i}");
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    };

    // Cold caches: `Bpe::clone` and `Unigram::clone` start with an empty
    // cache, so the threads race on insertion as well as lookup.
    let cold = tok.clone();
    run_threads(&cold);
    // Warm caches (filled by the sequential pass above): lookups only.
    run_threads(tok);

    // The results are still the same afterwards, on both.
    for t in [&cold, tok] {
        let again = t
            .encode_batch(inputs.iter().map(String::as_str).collect(), false)
            .unwrap();
        assert_eq!(again, expected);
    }
}

#[test]
fn concurrent_encode_batch_matches_sequential_for_bpe() {
    let corpus = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "sphinx of black quartz judge my vow",
        "how vexingly quick daft zebras jump",
    ];
    let mut tok = Tokenizer::new(Bpe::default()).with_pre_tokenizer(Whitespace);
    let trainer = BpeTrainer::builder()
        .vocab_size(60)
        .show_progress(false)
        .build()
        .unwrap();
    tok.train(trainer, corpus.iter()).unwrap();

    let words: Vec<&str> = corpus.iter().flat_map(|s| s.split(' ')).collect();
    let inputs = repeated_batch(&words, 4000);
    assert_concurrent_batches_match_sequential(&tok, &inputs);
}

#[test]
fn concurrent_encode_batch_matches_sequential_for_unigram() {
    let pieces: Vec<(String, f64)> = [
        ("<unk>", -10.0),
        ("the", -1.0),
        ("quick", -2.0),
        ("brown", -2.5),
        ("fox", -2.0),
        ("jump", -2.0),
        ("s", -3.0),
        ("over", -2.0),
        ("lazy", -2.5),
        ("dog", -2.0),
        ("t", -4.0),
        ("h", -4.0),
        ("e", -4.0),
        ("q", -4.0),
        ("u", -4.0),
        ("i", -4.0),
        ("c", -4.0),
        ("k", -4.0),
        ("b", -4.0),
        ("r", -4.0),
        ("o", -4.0),
        ("w", -4.0),
        ("n", -4.0),
        ("f", -4.0),
        ("x", -4.0),
        ("j", -4.0),
        ("m", -4.0),
        ("p", -4.0),
        ("v", -4.0),
        ("l", -4.0),
        ("a", -4.0),
        ("z", -4.0),
        ("y", -4.0),
        ("d", -4.0),
        ("g", -4.0),
    ]
    .map(|(p, s)| (p.to_string(), s))
    .into();
    let model = Unigram::new(pieces, Some(0), false).unwrap();
    let tok = Tokenizer::new(model).with_pre_tokenizer(WhitespaceSplit);

    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "foxes", "quickly",
        "doggo", "ü",
    ];
    let inputs = repeated_batch(&words, 4000);
    assert_concurrent_batches_match_sequential(&tok, &inputs);
}

/// `end_of_word_suffix` without a `continuing_subword_prefix`: only the
/// last char of a word carries the suffix, merges are looked up on the
/// suffixed symbol, and the `BPEDecoder` strips it again.
///
/// ```python
/// BPE({"a":0,"b":1,"b</w>":2,"ab</w>":3,"ab":4}, [("a","b</w>"),("a","b")], end_of_word_suffix="</w>")
/// ```
#[test]
fn bpe_end_of_word_suffix_alone() {
    let bpe = Bpe::builder()
        .vocab_and_merges(
            vocab(&[("a", 0), ("b", 1), ("b</w>", 2), ("ab</w>", 3), ("ab", 4)]),
            merges(&[("a", "b</w>"), ("a", "b")]),
        )
        .end_of_word_suffix("</w>")
        .build()
        .unwrap();
    assert_eq!(bpe.end_of_word_suffix(), Some("</w>"));
    assert_eq!(bpe.continuing_subword_prefix(), None);
    let tok = Tokenizer::new(bpe)
        .with_pre_tokenizer(WhitespaceSplit)
        .with_decoder(BpeDecoder::new("</w>"));

    let cases: &[FullCase] = &[
        ("ab", &["ab</w>"], &[3], &[(0, 2)]),
        ("abb", &["ab", "b</w>"], &[4, 2], &[(0, 2), (2, 3)]),
        ("aab", &["a", "ab</w>"], &[0, 3], &[(0, 1), (1, 3)]),
        (
            "ab abb",
            &["ab</w>", "ab", "b</w>"],
            &[3, 4, 2],
            &[(0, 2), (3, 5), (5, 6)],
        ),
    ];
    for (input, tokens, ids, offsets) in cases {
        let enc = tok.encode(*input, false).unwrap();
        assert_eq!(
            tokens_ids_offsets(&enc),
            (tokens.to_vec(), ids.to_vec(), offsets.to_vec()),
            "{input:?}"
        );
        assert_eq!(tok.decode(enc.ids(), false).unwrap(), *input);
    }

    // The option round-trips through tokenizer.json.
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(
        reloaded.encode("ab abb", false).unwrap(),
        tok.encode("ab abb", false).unwrap()
    );
}

/// Documented parity quirk (`docs/interop.md`): with `byte_fallback` and a
/// `continuing_subword_prefix`, an unknown non-initial char is looked up
/// *with* the prefix attached, and when it falls back to bytes the prefix
/// bytes (`#` = `<0x23>`) are emitted as byte tokens too. A char that is
/// in the vocabulary with its prefix, or whose bytes are missing, is
/// unaffected.
///
/// ```python
/// BPE(vocab, [], byte_fallback=True, continuing_subword_prefix="##", unk_token="<unk>")
/// t.encode("aé").tokens   # ['a', '<0x23>', '<0x23>', '<0xC3>', '<0xA9>']
/// ```
#[test]
fn bpe_byte_fallback_with_prefix_emits_prefix_bytes() {
    let v = vocab(&[
        ("a", 0),
        ("##a", 1),
        ("<0x23>", 2),
        ("<0xC3>", 3),
        ("<0xA9>", 4),
        ("é", 5),
        ("<unk>", 6),
        ("##b", 7),
    ]);
    let bpe = Bpe::builder()
        .vocab_and_merges(v.clone(), vec![])
        .byte_fallback(true)
        .continuing_subword_prefix("##")
        .unk_token("<unk>")
        .build()
        .unwrap();
    let tok = Tokenizer::new(bpe).with_pre_tokenizer(WhitespaceSplit);

    let cases: &[IdCase] = &[
        (
            "aé",
            &["a", "<0x23>", "<0x23>", "<0xC3>", "<0xA9>"],
            &[0, 2, 2, 3, 4],
        ),
        (
            "aéb",
            &["a", "<0x23>", "<0x23>", "<0xC3>", "<0xA9>", "##b"],
            &[0, 2, 2, 3, 4, 7],
        ),
        // Word-initial: no prefix, and `é` is in the vocabulary.
        ("éa", &["é", "##a"], &[5, 1]),
        ("ab", &["a", "##b"], &[0, 7]),
        // `##ü` is not in the vocabulary and `<0xC3><0xBC>` is incomplete:
        // the char becomes the unk token (prefix bytes are not emitted).
        ("aü", &["a", "<unk>"], &[0, 6]),
    ];
    for (input, tokens, ids) in cases {
        let enc = tok.encode(*input, false).unwrap();
        assert_eq!(enc.tokens(), *tokens, "{input:?}");
        assert_eq!(enc.ids(), *ids, "{input:?}");
        // At the model level each byte token covers exactly one byte, as
        // documented. (After the tokenizer re-aligns them to the word's
        // char boundaries they are a known quirk in both libraries and
        // are not asserted: HF gives `(1,2),(1,2),(3,4),(4,5)` for "aé".)
        let direct = tok.model().tokenize(input).unwrap();
        assert_eq!(
            direct.iter().map(|t| t.id).collect::<Vec<_>>(),
            *ids,
            "{input:?}"
        );
        for t in &direct {
            if t.value.starts_with("<0x") {
                assert_eq!(t.offsets.1 - t.offsets.0, 1, "{input:?}: {t:?}");
            }
        }
    }

    // Without the prefix the same vocabulary finds `é` directly.
    let plain = Bpe::builder()
        .vocab_and_merges(v, vec![])
        .byte_fallback(true)
        .unk_token("<unk>")
        .build()
        .unwrap();
    let tok = Tokenizer::new(plain).with_pre_tokenizer(WhitespaceSplit);
    let enc = tok.encode("aé", false).unwrap();
    assert_eq!(enc.tokens(), ["a", "é"]);
    assert_eq!(enc.offsets(), [(0, 1), (1, 3)]);
}

/// `max_input_chars_per_word` counts chars, not bytes: `éé` (2 chars,
/// 4 bytes) still fits a limit of 2, `ééé` does not. The unk token covers
/// the whole word.
///
/// ```python
/// WordPiece({"[UNK]":0,"é":1,"##é":2,"a":3,"##a":4,"##b":5}, unk_token="[UNK]", max_input_chars_per_word=2)
/// ```
#[test]
fn wordpiece_max_input_chars_per_word_counts_chars_not_bytes() {
    let wp = WordPiece::builder()
        .vocab(vocab(&[
            ("[UNK]", 0),
            ("é", 1),
            ("##é", 2),
            ("a", 3),
            ("##a", 4),
            ("##b", 5),
        ]))
        .unk_token("[UNK]")
        .max_input_chars_per_word(2)
        .build()
        .unwrap();
    assert_eq!(wp.max_input_chars_per_word(), 2);
    let tok = Tokenizer::new(wp).with_pre_tokenizer(WhitespaceSplit);

    let cases: &[FullCase] = &[
        ("éé", &["é", "##é"], &[1, 2], &[(0, 1), (1, 2)]),
        ("ééé", &["[UNK]"], &[0], &[(0, 3)]),
        ("éa", &["é", "##a"], &[1, 4], &[(0, 1), (1, 2)]),
        ("aéé", &["[UNK]"], &[0], &[(0, 3)]),
    ];
    for (input, tokens, ids, offsets) in cases {
        let enc = tok.encode_char_offsets(*input, false).unwrap();
        assert_eq!(
            tokens_ids_offsets(&enc),
            (tokens.to_vec(), ids.to_vec(), offsets.to_vec()),
            "{input:?}"
        );
    }
    // Byte offsets of the unk token span the whole (multi-byte) word.
    assert_eq!(tok.encode("ééé", false).unwrap().offsets(), [(0, 6)]);

    // The limit survives a JSON round trip.
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(reloaded.encode("ééé", false).unwrap().tokens(), ["[UNK]"]);
    assert_eq!(reloaded.encode("éé", false).unwrap().tokens(), ["é", "##é"]);
}

/// `max_input_chars_per_word = 0` turns every non-empty word into the unk
/// token (even a single in-vocabulary char), while empty input stays
/// empty.
///
/// ```python
/// WordPiece({"[UNK]":0,"a":1,"##a":2}, unk_token="[UNK]", max_input_chars_per_word=0)
/// ```
#[test]
fn wordpiece_max_input_chars_per_word_zero_makes_every_word_unk() {
    let wp = WordPiece::builder()
        .vocab(vocab(&[("[UNK]", 0), ("a", 1), ("##a", 2)]))
        .unk_token("[UNK]")
        .max_input_chars_per_word(0)
        .build()
        .unwrap();
    assert_eq!(wp.max_input_chars_per_word(), 0);
    assert!(wp.tokenize("").unwrap().is_empty());
    let tok = Tokenizer::new(wp).with_pre_tokenizer(WhitespaceSplit);

    let cases: &[OffsetCase] = &[
        ("a", &["[UNK]"], &[(0, 1)]),
        ("aa", &["[UNK]"], &[(0, 2)]),
        ("a aa", &["[UNK]", "[UNK]"], &[(0, 1), (2, 4)]),
        ("", &[], &[]),
        (" ", &[], &[]),
    ];
    for (input, tokens, offsets) in cases {
        let enc = tok.encode(*input, false).unwrap();
        assert_eq!(enc.tokens(), *tokens, "{input:?}");
        assert_eq!(enc.offsets(), *offsets, "{input:?}");
        assert!(enc.ids().iter().all(|&id| id == 0), "{input:?}");
    }

    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(reloaded.encode("a", false).unwrap().tokens(), ["[UNK]"]);
}
