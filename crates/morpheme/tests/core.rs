//! Core pipeline behaviour through the public `Tokenizer` API (issue #47).
//! Expected values come from Python `tokenizers` 0.23.2 (see each test).

use morpheme::processors::BertProcessing;
use morpheme::{AddedToken, Encoding, Error, Tokenizer, TruncationDirection, TruncationParams};
use morpheme::{TruncationStrategy, pre_tokenizers::WhitespaceSplit};

/// `[UNK]`=0, `[CLS]`=1, `[SEP]`=2, `a`..`h` = 3..=10, with BERT special
/// tokens, so every truncation budget includes 2 (single) or 3 (pair)
/// specials.
fn bert_like_tokenizer() -> Tokenizer {
    let vocab = [
        ("[UNK]", 0),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("a", 3),
        ("b", 4),
        ("c", 5),
        ("d", 6),
        ("e", 7),
        ("f", 8),
        ("g", 9),
        ("h", 10),
    ]
    .map(|(t, i)| (t.to_string(), i))
    .into();
    let model = morpheme::models::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]")
        .build()
        .unwrap();
    Tokenizer::new(model)
        .with_pre_tokenizer(WhitespaceSplit)
        .with_post_processor(BertProcessing::new(("[SEP]", 2), ("[CLS]", 1)))
}

fn truncation(
    max_length: usize,
    stride: usize,
    strategy: TruncationStrategy,
    direction: TruncationDirection,
) -> TruncationParams {
    TruncationParams {
        direction,
        max_length,
        strategy,
        stride,
    }
}

/// `(ids, type_ids, overflowing ids, overflowing type_ids)`, the shape
/// printed by the Python script that produced the expectations.
type Shape = (Vec<u32>, Vec<u32>, Vec<Vec<u32>>, Vec<Vec<u32>>);

fn shape(e: &Encoding) -> Shape {
    (
        e.ids().to_vec(),
        e.type_ids().to_vec(),
        e.overflowing().iter().map(|o| o.ids().to_vec()).collect(),
        e.overflowing()
            .iter()
            .map(|o| o.type_ids().to_vec())
            .collect(),
    )
}

/// `TruncationStrategy::OnlyFirst` through the tokenizer: the budget
/// includes the specials, only the first sequence is cut, and a pair whose
/// first sequence cannot absorb the cut is an error.
///
/// ```python
/// t.post_processor = BertProcessing(("[SEP]", 2), ("[CLS]", 1))
/// t.enable_truncation(5, strategy="only_first"); t.encode("a b c d e f")
/// t.enable_truncation(8, strategy="only_first"); t.encode("a b c d e f", "g h")
/// t.enable_truncation(5, strategy="only_first"); t.encode("a", "b c d e f")  # raises
/// ```
///
/// Overflow note: `tokenizers` 0.23.2 stops tokenizing a *single*
/// sequence once `max_length` tokens exist (`tokenize_with_limit`), so its
/// overflow for the first case above is the truncated `[CLS] d e [SEP]`.
/// morpheme tokenizes the whole input, and its overflow matches HF's own
/// `Encoding.truncate(3, 0, "right")` + `post_process` on the full
/// encoding (`[CLS] d e f [SEP]`), which is what is pinned here. Pairs are
/// not affected as long as each sequence fits in `max_length`.
#[test]
fn only_first_truncation_through_tokenizer() {
    let mut tok = bert_like_tokenizer();
    tok.set_truncation(Some(truncation(
        5,
        0,
        TruncationStrategy::OnlyFirst,
        TruncationDirection::Right,
    )))
    .unwrap();
    let single = tok.encode("a b c d e f", true).unwrap();
    assert_eq!(
        shape(&single),
        (
            vec![1, 3, 4, 5, 2],
            vec![0; 5],
            vec![vec![1, 6, 7, 8, 2]],
            vec![vec![0; 5]],
        )
    );

    tok.set_truncation(Some(truncation(
        8,
        0,
        TruncationStrategy::OnlyFirst,
        TruncationDirection::Right,
    )))
    .unwrap();
    let pair = tok.encode(("a b c d e f", "g h"), true).unwrap();
    assert_eq!(
        shape(&pair),
        (
            vec![1, 3, 4, 5, 2, 9, 10, 2],
            vec![0, 0, 0, 0, 0, 1, 1, 1],
            vec![vec![1, 6, 7, 8, 2, 9, 10, 2]],
            vec![vec![0, 0, 0, 0, 0, 1, 1, 1]],
        )
    );
    // The second sequence is never touched, even when it is the longer
    // one: with 8 tokens of content and a budget of 7, the only token
    // removed comes from the first sequence.
    // t.enable_truncation(10, strategy="only_first"); t.encode("a b", "c d e f g h")
    tok.set_truncation(Some(truncation(
        10,
        0,
        TruncationStrategy::OnlyFirst,
        TruncationDirection::Right,
    )))
    .unwrap();
    let pair = tok.encode(("a b", "c d e f g h"), true).unwrap();
    assert_eq!(
        shape(&pair),
        (
            vec![1, 3, 2, 5, 6, 7, 8, 9, 10, 2],
            vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            vec![vec![1, 4, 2, 5, 6, 7, 8, 9, 10, 2]],
            vec![vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1]],
        )
    );
    // And when the first sequence cannot absorb the whole cut (budget 5
    // for 8 tokens: 3 to remove from 2) it is an error, as in HF.
    tok.set_truncation(Some(truncation(
        8,
        0,
        TruncationStrategy::OnlyFirst,
        TruncationDirection::Right,
    )))
    .unwrap();
    let err = tok.encode(("a b", "c d e f g h"), true).unwrap_err();
    assert!(matches!(err, Error::Truncation(_)), "{err:?}");

    // Budget 5 with 3 specials leaves 2 tokens for 6: `a` alone is too
    // short to absorb 4 removals.
    tok.set_truncation(Some(truncation(
        5,
        0,
        TruncationStrategy::OnlyFirst,
        TruncationDirection::Right,
    )))
    .unwrap();
    let err = tok.encode(("a", "b c d e f"), true).unwrap_err();
    assert!(matches!(err, Error::Truncation(_)), "{err:?}");
    // The error is specific to the strategy: the same input fits under
    // `LongestFirst`.
    tok.set_truncation(Some(truncation(
        5,
        0,
        TruncationStrategy::LongestFirst,
        TruncationDirection::Right,
    )))
    .unwrap();
    assert_eq!(tok.encode(("a", "b c d e f"), true).unwrap().len(), 5);
}

/// `TruncationDirection::Left` through the tokenizer, with specials, pairs
/// and a stride: the kept window is the *end* of the sequence, overflowing
/// windows walk backwards with `stride` overlap, and every window gets the
/// specials and type ids of a full pair.
///
/// ```python
/// t.enable_truncation(8, stride=1, strategy="only_first", direction="left")
/// t.encode("a b c d e f", "g h")
/// t.enable_truncation(5, stride=1, strategy="longest_first", direction="left")
/// t.encode("a b c d e f")
/// t.enable_truncation(7, stride=1, strategy="longest_first", direction="left")
/// t.encode("a b c d e f", "g h")
/// t.enable_truncation(7, stride=0, strategy="only_second", direction="left")
/// t.encode("a b", "c d e f g h")
/// ```
#[test]
fn left_truncation_through_tokenizer_with_specials_pairs_and_stride() {
    let mut tok = bert_like_tokenizer();

    tok.set_truncation(Some(truncation(
        8,
        1,
        TruncationStrategy::OnlyFirst,
        TruncationDirection::Left,
    )))
    .unwrap();
    let pair = tok.encode(("a b c d e f", "g h"), true).unwrap();
    assert_eq!(
        shape(&pair),
        (
            vec![1, 6, 7, 8, 2, 9, 10, 2],
            vec![0, 0, 0, 0, 0, 1, 1, 1],
            vec![vec![1, 4, 5, 6, 2, 9, 10, 2], vec![1, 3, 4, 2, 9, 10, 2]],
            vec![vec![0, 0, 0, 0, 0, 1, 1, 1], vec![0, 0, 0, 0, 1, 1, 1]],
        )
    );
    // Offsets still point into the original text.
    assert_eq!(pair.offsets()[1], (6, 7));
    assert_eq!(pair.token_to_sequence(1), Some(0));
    assert_eq!(pair.token_to_sequence(5), Some(1));

    tok.set_truncation(Some(truncation(
        5,
        1,
        TruncationStrategy::LongestFirst,
        TruncationDirection::Left,
    )))
    .unwrap();
    // Single sequence: windows of 3 walking left with an overlap of 1,
    // `d e f`, `b c d`, `a b`. This is HF's `Encoding.truncate(3, 1,
    // "left")` + `post_process` result; HF's own `encode` only produces
    // the first overflow window because of its early exit (see
    // `only_first_truncation_through_tokenizer`).
    let single = tok.encode("a b c d e f", true).unwrap();
    assert_eq!(
        shape(&single),
        (
            vec![1, 6, 7, 8, 2],
            vec![0; 5],
            vec![vec![1, 4, 5, 6, 2], vec![1, 3, 4, 2]],
            vec![vec![0; 5], vec![0; 4]],
        )
    );

    tok.set_truncation(Some(truncation(
        7,
        1,
        TruncationStrategy::LongestFirst,
        TruncationDirection::Left,
    )))
    .unwrap();
    let pair = tok.encode(("a b c d e f", "g h"), true).unwrap();
    assert_eq!(
        shape(&pair),
        (
            vec![1, 7, 8, 2, 9, 10, 2],
            vec![0, 0, 0, 0, 1, 1, 1],
            vec![
                vec![1, 6, 7, 2, 9, 10, 2],
                vec![1, 5, 6, 2, 9, 10, 2],
                vec![1, 4, 5, 2, 9, 10, 2],
                vec![1, 3, 4, 2, 9, 10, 2],
            ],
            vec![vec![0, 0, 0, 0, 1, 1, 1]; 4],
        )
    );

    tok.set_truncation(Some(truncation(
        7,
        0,
        TruncationStrategy::OnlySecond,
        TruncationDirection::Left,
    )))
    .unwrap();
    let pair = tok.encode(("a b", "c d e f g h"), true).unwrap();
    assert_eq!(
        shape(&pair),
        (
            vec![1, 3, 4, 2, 9, 10, 2],
            vec![0, 0, 0, 0, 1, 1, 1],
            vec![vec![1, 3, 4, 2, 7, 8, 2], vec![1, 3, 4, 2, 5, 6, 2]],
            vec![vec![0, 0, 0, 0, 1, 1, 1]; 2],
        )
    );

    // The direction survives a JSON round trip.
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(reloaded.encode(("a b", "c d e f g h"), true).unwrap(), pair);
}

/// `single_word` boundaries are Unicode word boundaries (`\w` is
/// Unicode): a letter such as `é`, `日` or a combining mark next to the
/// token keeps it from matching, while punctuation does not.
///
/// ```python
/// t.add_tokens([AddedToken("ab", single_word=True)])
/// t.encode("éab ab").tokens   # ['[UNK]', 'ab']
/// t.encode("ab-ab").tokens    # ['ab', '[UNK]', 'ab']
/// ```
#[test]
fn single_word_respects_unicode_word_boundaries() {
    let mut tok = bert_like_tokenizer();
    tok.set_post_processor(None);
    tok.add_tokens(&[AddedToken::new("ab", false).single_word(true)])
        .unwrap();
    let ab = tok.token_to_id("ab").unwrap();

    type Case<'a> = (&'a str, &'a [&'a str], &'a [(usize, usize)]);
    let cases: &[Case] = &[
        ("ab ab", &["ab", "ab"], &[(0, 2), (3, 5)]),
        ("éab ab", &["[UNK]", "ab"], &[(0, 3), (4, 6)]),
        ("abé ab", &["[UNK]", "ab"], &[(0, 3), (4, 6)]),
        ("日ab ab", &["[UNK]", "ab"], &[(0, 3), (4, 6)]),
        ("ab_ab ab", &["[UNK]", "ab"], &[(0, 5), (6, 8)]),
        ("ab\u{301}ab ab", &["[UNK]", "ab"], &[(0, 5), (6, 8)]),
        ("ab-ab", &["ab", "[UNK]", "ab"], &[(0, 2), (2, 3), (3, 5)]),
    ];
    for (input, tokens, offsets) in cases {
        let enc = tok.encode_char_offsets(*input, false).unwrap();
        assert_eq!(enc.tokens(), *tokens, "{input:?}");
        assert_eq!(enc.offsets(), *offsets, "{input:?}");
        let matched = enc.ids().iter().filter(|&&id| id == ab).count();
        assert_eq!(
            matched,
            tokens.iter().filter(|t| **t == "ab").count(),
            "{input:?}"
        );
    }
}
