//! Arbitrary id sequences (in-range, out-of-range and special-token ids)
//! through `decode` and `DecodeStream` on real tokenizers (BERT, GPT-2,
//! Llama, T5, Qwen2.5): nothing panics, out-of-range ids are dropped
//! rather than failing, and streaming yields a prefix of the full decode
//! that is the whole text once no trailing byte-fallback is pending.

#![no_main]

#[path = "common.rs"]
mod common;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use morpheme::Tokenizer;

#[derive(Arbitrary, Debug)]
enum Id {
    /// Any id at all.
    Raw(u32),
    /// Reduced modulo the vocabulary size.
    InRange(u32),
    /// One of the tokenizer's special tokens (`[CLS]`, `</s>`, …).
    Special(u8),
    /// Just past the vocabulary.
    Beyond(u16),
}

#[derive(Arbitrary, Debug)]
struct Input {
    tokenizer: u8,
    ids: Vec<Id>,
}

/// Work budget for one input: `DecodeStream` re-decodes a growing window
/// per step, so cap the sequence length.
const MAX_IDS: usize = 256;

fn special_ids(tok: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = tok
        .added_vocabulary()
        .added_tokens_decoder()
        .iter()
        .filter(|(_, t)| t.special)
        .map(|(&id, _)| id)
        .collect();
    ids.sort_unstable();
    ids
}

fn resolve(tok: &Tokenizer, specials: &[u32], ids: &[Id]) -> Vec<u32> {
    let vocab = tok.vocab_size(true) as u32;
    ids.iter()
        .take(MAX_IDS)
        .map(|id| match *id {
            Id::Raw(x) => x,
            Id::InRange(x) => x % vocab,
            Id::Special(i) => specials
                .get(i as usize % specials.len().max(1))
                .copied()
                .unwrap_or(vocab),
            Id::Beyond(x) => vocab.saturating_add(x as u32),
        })
        .collect()
}

/// Concatenating every chunk a `DecodeStream` yields must equal
/// `decode` of the same ids — unless the full text ends in U+FFFD, which
/// the stream withholds as a possibly incomplete byte-fallback character.
/// In that case nothing is asserted: the stream may have yielded less
/// (a pending character) or text that is no longer a prefix, because
/// `ByteFallback` replaces a whole byte run that is invalid UTF-8 with one
/// U+FFFD per byte (as Hugging Face does), so Llama ids `[z, z, z, 0xFC]`
/// decode to `"����"` after the stream already emitted `"zzz"`. `Err`
/// from `step` is a documented outcome (a decoder that rewrites
/// already-emitted text), not a bug.
fn check_stream(name: &str, tok: &Tokenizer, ids: &[u32], skip: bool, full: &str) {
    let mut stream = tok.decode_stream(skip);
    let mut out = String::new();
    for &id in ids {
        match stream.step(id) {
            Ok(Some(chunk)) => out.push_str(&chunk),
            Ok(None) => {}
            Err(_) => return,
        }
    }
    if !full.ends_with('\u{FFFD}') {
        assert_eq!(
            out, full,
            "{name} (skip={skip}): stream output differs from decode for {ids:?}"
        );
    }
}

fuzz_target!(|input: Input| {
    let fixtures = common::fixtures();
    if fixtures.is_empty() {
        return;
    }
    let (name, tok) = &fixtures[input.tokenizer as usize % fixtures.len()];
    let specials = special_ids(tok);
    let ids = resolve(tok, &specials, &input.ids);
    let vocab = tok.vocab_size(true) as u32;

    let full = tok
        .decode(&ids, false)
        .unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
    let skipped = tok
        .decode(&ids, true)
        .unwrap_or_else(|e| panic!("{name}: decode (skip special) failed: {e}"));

    // Out-of-range ids are dropped, never an error or a different text.
    let in_range: Vec<u32> = ids.iter().copied().filter(|&id| id < vocab).collect();
    if in_range.len() != ids.len() {
        let pruned = tok
            .decode(&in_range, false)
            .unwrap_or_else(|e| panic!("{name}: decode of in-range ids failed: {e}"));
        assert_eq!(pruned, full, "{name}: out-of-range ids changed the decode");
    }
    if specials.is_empty() || !ids.iter().any(|id| specials.contains(id)) {
        assert_eq!(
            skipped, full,
            "{name}: skip_special_tokens changed a text with no specials"
        );
    }

    check_stream(name, tok, &ids, false, &full);
    check_stream(name, tok, &ids, true, &skipped);
});
