//! Shared helpers for the fuzz targets.

#![allow(dead_code)]

use std::path::Path;
use std::sync::OnceLock;

use morpheme::{Encoding, NormalizedString, OffsetRange, PaddingStrategy, Tokenizer};

/// `check_alignment` skips strings longer than this (bytes): the check is
/// quadratic, and anything this big is a work budget, not a bug.
pub const ALIGNMENT_CHECK_MAX_LEN: usize = 4096 * 4;

/// Every normalized char must map back to a valid slice of the original
/// text, and original ranges must map into bounds of the normalized text.
pub fn check_alignment(n: &NormalizedString) {
    // `convert_offsets` is a linear scan, so checking every char is
    // quadratic: skip the check for results far over the budget.
    if n.len() > ALIGNMENT_CHECK_MAX_LEN {
        return;
    }
    let original = n.original();
    let normalized = n.get();
    let (start, end) = n.offsets_original();
    assert!(start <= end, "offsets_original reversed");
    for (b, c) in normalized.char_indices() {
        let range = b..b + c.len_utf8();
        let mapped = n
            .convert_offsets(OffsetRange::Normalized(range.clone()))
            .unwrap_or_else(|| panic!("no mapping for normalized {range:?} in {normalized:?}"));
        assert!(
            original.get(mapped.clone()).is_some(),
            "normalized {range:?} ({c:?}) maps to {mapped:?}, not a valid slice of {original:?}"
        );
    }
    // Mapping original ranges into normalized text must stay in bounds too.
    for (b, c) in original.char_indices() {
        if let Some(r) = n.convert_offsets(OffsetRange::Original(b..b + c.len_utf8())) {
            assert!(
                r.start <= r.end && r.end <= normalized.len(),
                "original→normalized {r:?} out of bounds"
            );
        }
    }
}

/// Real tokenizers from `crates/morpheme/tests/data/hf` (fetched by
/// `scripts/fetch-hf-fixtures.sh`); missing files are skipped.
pub fn fixtures() -> &'static [(&'static str, Tokenizer)] {
    static FIXTURES: OnceLock<Vec<(&'static str, Tokenizer)>> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/morpheme/tests/data/hf");
        ["bert-base-uncased", "gpt2", "llama", "t5-small", "qwen2.5"]
            .into_iter()
            .filter_map(|name| {
                let path = dir.join(format!("{name}.json"));
                path.exists().then(|| {
                    let tok = Tokenizer::from_file(&path)
                        .unwrap_or_else(|e| panic!("fixture {name} failed to load: {e}"));
                    (name, tok)
                })
            })
            .collect()
    })
}

/// Keep fuzzed configurations within a sane work budget: padding or
/// truncation to billions of tokens is a resource limit, not a bug.
pub fn within_budget(tok: &mut Tokenizer) {
    if let Some(p) = tok.padding() {
        let too_big = matches!(p.strategy, PaddingStrategy::Fixed(n) if n > 4096)
            || p.pad_to_multiple_of.is_some_and(|m| m > 4096);
        if too_big {
            tok.set_padding(None);
        }
    }
}

/// Byte offsets of real (non-special) tokens must slice `input` on char
/// boundaries.
pub fn check_offsets(enc: &Encoding, inputs: &[&str]) {
    let max = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    for (i, &(start, end)) in enc.offsets().iter().enumerate() {
        assert!(start <= end, "token {i}: reversed offsets {start}..{end}");
        assert!(
            end <= max,
            "token {i}: offsets {start}..{end} past input end {max}"
        );
        if enc.special_tokens_mask()[i] == 0
            && let Some(seq) = enc.token_to_sequence(i)
            && let Some(s) = inputs.get(seq)
        {
            assert!(
                s.get(start..end).is_some(),
                "token {i} ({:?}): offsets {start}..{end} not a valid slice of {s:?}",
                enc.tokens()[i]
            );
        }
    }
}

/// Run a tokenizer through its whole public surface on `text` (and a
/// pair), checking basic invariants. Errors are fine; panics are bugs.
pub fn exercise(tok: &Tokenizer, text: &str, pair: &str) {
    for add_special in [true, false] {
        if let Ok(enc) = tok.encode(text, add_special) {
            check_offsets(&enc, &[text]);
            let _ = tok.decode(enc.ids(), false);
            let _ = tok.decode(enc.ids(), true);
        }
        if let Ok(enc) = tok.encode((text, pair), add_special) {
            check_offsets(&enc, &[text, pair]);
        }
        let _ = tok.encode_char_offsets(text, add_special);
    }
    let words: Vec<&str> = text.split_whitespace().take(16).collect();
    let _ = tok.encode(words, true);
}
