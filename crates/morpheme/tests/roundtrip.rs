//! Property tests: lossless round-trips and offset validity.

use std::sync::OnceLock;

use morpheme::models::Bpe;
use morpheme::normalizers::BertNormalizer;
use morpheme::pre_tokenizers::{BertPreTokenizer, ByteLevel};
use morpheme::trainers::BpeTrainer;
use morpheme::{NormalizedString, Normalizer, OffsetRange, Tokenizer};
use proptest::prelude::*;

fn byte_level() -> &'static Tokenizer {
    static TOK: OnceLock<Tokenizer> = OnceLock::new();
    TOK.get_or_init(|| {
        let mut tok = Tokenizer::new(Bpe::default())
            .with_pre_tokenizer(ByteLevel::new(false, true, true))
            .with_decoder(ByteLevel::default());
        let trainer = BpeTrainer::builder()
            .vocab_size(500)
            .initial_alphabet(ByteLevel::alphabet())
            .build()
            .unwrap();
        tok.train(
            trainer,
            include_str!("../../../examples/corpus.txt").lines(),
        )
        .unwrap();
        tok
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Byte-level BPE is lossless for every string.
    #[test]
    fn byte_level_decode_inverts_encode(s in any::<String>()) {
        let tok = byte_level();
        let enc = tok.encode(s.as_str(), false).unwrap();
        prop_assert_eq!(tok.decode(enc.ids(), false).unwrap(), s);
    }

    /// Byte offsets always slice the original input on char boundaries,
    /// in non-decreasing order.
    #[test]
    fn offsets_are_valid_slices(s in "\\PC{0,40}") {
        let tok = byte_level();
        let enc = tok.encode(s.as_str(), false).unwrap();
        let mut last = 0;
        for &(start, end) in enc.offsets() {
            prop_assert!(start <= end && end <= s.len());
            prop_assert!(s.is_char_boundary(start) && s.is_char_boundary(end));
            prop_assert!(start >= last || start == end);
            last = start;
        }
    }

    /// Char offsets are consistent with byte offsets.
    #[test]
    fn char_offsets_match_byte_offsets(s in "\\PC{0,40}") {
        let tok = byte_level();
        let bytes = tok.encode(s.as_str(), false).unwrap();
        let chars = tok.encode_char_offsets(s.as_str(), false).unwrap();
        for (&(bs, be), &(cs, ce)) in bytes.offsets().iter().zip(chars.offsets()) {
            prop_assert_eq!(s[..bs].chars().count(), cs);
            prop_assert_eq!(s[..be].chars().count(), ce);
        }
    }

    /// Normalized → original offset mapping always lands on the original
    /// text that produced each normalized char.
    #[test]
    fn bert_normalizer_alignments_point_into_original(s in "\\PC{0,30}") {
        let mut n = NormalizedString::from(s.as_str());
        BertNormalizer::default().normalize(&mut n).unwrap();
        let normalized = n.get().to_owned();
        for (b, c) in normalized.char_indices() {
            let r = n.convert_offsets(OffsetRange::Normalized(b..b + c.len_utf8()));
            if let Some(r) = r {
                prop_assert!(s.get(r.clone()).is_some(), "{:?} -> {:?}", c, r);
            }
        }
    }

    /// BERT pre-tokenization never produces whitespace inside a token.
    #[test]
    fn bert_pre_tokenizer_drops_whitespace(s in "\\PC{0,40}") {
        let tok = Tokenizer::new(morpheme::models::WordLevel::default())
            .with_pre_tokenizer(BertPreTokenizer);
        let mut pts = morpheme::PreTokenizedString::from(s.as_str());
        morpheme::PreTokenizer::pre_tokenize(tok.pre_tokenizer().unwrap(), &mut pts).unwrap();
        for (piece, _, _) in pts.get_splits(morpheme::OffsetType::Byte) {
            prop_assert!(!piece.chars().any(char::is_whitespace), "{:?}", piece);
        }
    }
}
