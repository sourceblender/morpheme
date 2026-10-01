//! Pipeline behaviour through the public API (issue #47): the `Contiguous`
//! and `MergedWithPrevious` split behaviours, loaded from `tokenizer.json`
//! and run through `Tokenizer::encode`. Expected values come from Python
//! `tokenizers` 0.23.2 (the same configs are also in the pre-tokenizer
//! ground-truth set, `src/pre_tokenizers/testdata/ground_truth.json`).

use morpheme::pre_tokenizers::{PreTokenizerWrapper, Split};
use morpheme::{
    OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior, Tokenizer,
    pattern::SplitPattern,
};

fn splits(pt: &dyn PreTokenizer, s: &str) -> Vec<(String, (usize, usize))> {
    let mut pts = PreTokenizedString::from(s);
    pt.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetType::Byte)
        .into_iter()
        .map(|(piece, offsets, _)| (piece.to_string(), offsets))
        .collect()
}

fn owned(v: &[(&str, (usize, usize))]) -> Vec<(String, (usize, usize))> {
    v.iter().map(|(s, o)| (s.to_string(), *o)).collect()
}

/// A WordLevel tokenizer whose vocabulary is exactly the pieces the split
/// is expected to produce, so `encode` shows the pieces as tokens.
fn tokenizer_json(pre_tokenizer: &serde_json::Value, pieces: &[&str]) -> String {
    let vocab: serde_json::Map<String, serde_json::Value> = std::iter::once("[UNK]")
        .chain(pieces.iter().copied())
        .enumerate()
        .map(|(i, p)| (p.to_string(), serde_json::json!(i)))
        .collect();
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    })
    .to_string()
}

/// ```python
/// pre_tokenizers.Split(" ", "contiguous").pre_tokenize_str("How are  you?")
/// # [('How', (0, 3)), (' ', (3, 4)), ('are', (4, 7)), ('  ', (7, 9)), ('you?', (9, 13))]
/// ```
#[test]
fn contiguous_split_keeps_delimiter_runs_together() {
    let expected = [
        ("How", (0, 3)),
        (" ", (3, 4)),
        ("are", (4, 7)),
        ("  ", (7, 9)),
        ("you?", (9, 13)),
    ];
    let config = serde_json::json!({
        "type": "Split",
        "pattern": {"String": " "},
        "behavior": "Contiguous",
        "invert": false,
    });

    let loaded: PreTokenizerWrapper = serde_json::from_value(config.clone()).unwrap();
    assert_eq!(splits(&loaded, "How are  you?"), owned(&expected));
    let built = Split::new(
        SplitPattern::String(" ".into()),
        SplitDelimiterBehavior::Contiguous,
        false,
    )
    .unwrap();
    assert_eq!(splits(&built, "How are  you?"), owned(&expected));
    assert_eq!(serde_json::to_value(&loaded).unwrap(), config);

    let pieces: Vec<&str> = ["How", " ", "are", "  ", "you?"].to_vec();
    let tok = Tokenizer::from_json(&tokenizer_json(&config, &pieces)).unwrap();
    let enc = tok.encode("How are  you?", false).unwrap();
    assert_eq!(enc.tokens(), pieces);
    assert_eq!(
        enc.offsets(),
        expected.iter().map(|(_, o)| *o).collect::<Vec<_>>()
    );
    assert_eq!(enc.ids(), [1, 2, 3, 4, 5]);
    // A run at either edge and the empty / all-delimiter inputs.
    assert_eq!(
        splits(&loaded, "  a "),
        owned(&[("  ", (0, 2)), ("a", (2, 3)), (" ", (3, 4))])
    );
    assert_eq!(splits(&loaded, "   "), owned(&[("   ", (0, 3))]));
    assert_eq!(splits(&loaded, ""), vec![]);
}

/// ```python
/// pre_tokenizers.Split(" ", "merged_with_previous").pre_tokenize_str("How are  you?")
/// # [('How ', (0, 4)), ('are ', (4, 8)), (' ', (8, 9)), ('you?', (9, 13))]
/// ```
#[test]
fn merged_with_previous_split_attaches_delimiter_to_previous_piece() {
    let expected = [
        ("How ", (0, 4)),
        ("are ", (4, 8)),
        (" ", (8, 9)),
        ("you?", (9, 13)),
    ];
    let config = serde_json::json!({
        "type": "Split",
        "pattern": {"String": " "},
        "behavior": "MergedWithPrevious",
        "invert": false,
    });

    let loaded: PreTokenizerWrapper = serde_json::from_value(config.clone()).unwrap();
    assert_eq!(splits(&loaded, "How are  you?"), owned(&expected));
    let built = Split::new(
        SplitPattern::String(" ".into()),
        SplitDelimiterBehavior::MergedWithPrevious,
        false,
    )
    .unwrap();
    assert_eq!(splits(&built, "How are  you?"), owned(&expected));
    assert_eq!(serde_json::to_value(&loaded).unwrap(), config);

    let pieces: Vec<&str> = ["How ", "are ", " ", "you?"].to_vec();
    let tok = Tokenizer::from_json(&tokenizer_json(&config, &pieces)).unwrap();
    let enc = tok.encode("How are  you?", false).unwrap();
    assert_eq!(enc.tokens(), pieces);
    assert_eq!(
        enc.offsets(),
        expected.iter().map(|(_, o)| *o).collect::<Vec<_>>()
    );
    assert_eq!(enc.ids(), [1, 2, 3, 4]);
    // A leading delimiter has nothing to attach to and stays on its own.
    assert_eq!(
        splits(&loaded, " a b"),
        owned(&[(" ", (0, 1)), ("a ", (1, 3)), ("b", (3, 4))])
    );
    assert_eq!(splits(&loaded, "a "), owned(&[("a ", (0, 2))]));
    assert_eq!(splits(&loaded, ""), vec![]);

    // With a regex, a whole run is one match and attaches as a unit.
    let regex = Split::new(
        SplitPattern::Regex(r"\s+".into()),
        SplitDelimiterBehavior::MergedWithPrevious,
        false,
    )
    .unwrap();
    assert_eq!(
        splits(&regex, "How are  you?"),
        owned(&[("How ", (0, 4)), ("are  ", (4, 9)), ("you?", (9, 13))])
    );
}
