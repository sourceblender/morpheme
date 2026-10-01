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

fn word_level_tokenizer(vocab: &[(&str, u32)]) -> Tokenizer {
    let model = morpheme::models::WordLevel::builder()
        .vocab(vocab.iter().map(|(s, id)| (s.to_string(), *id)).collect())
        .unk_token("[UNK]")
        .build()
        .unwrap();
    Tokenizer::new(model).with_pre_tokenizer(morpheme::pre_tokenizers::WhitespaceSplit)
}

fn word_level_trainer() -> morpheme::trainers::WordLevelTrainer {
    morpheme::trainers::WordLevelTrainer::builder()
        .show_progress(false)
        .special_tokens(vec![
            morpheme::AddedToken::new("[UNK]", true),
            morpheme::AddedToken::new("[SEP]", true),
        ])
        .build()
        .unwrap()
}

#[test]
fn saving_tokenizers_is_atomic_for_readers_and_preserves_existing_files_on_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    let tok = word_level_tokenizer(&[("[UNK]", 0), ("a", 1)]);
    tok.save(&path, false).unwrap();
    let initial = std::fs::read(&path).unwrap();
    assert!(tok.save(dir.path(), true).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), initial);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            for _ in 0..100 {
                let loaded = Tokenizer::from_file(&path).unwrap();
                assert_eq!(loaded.encode("a", false).unwrap().ids(), &[1]);
            }
        });
        for pretty in [true, false].into_iter().cycle().take(20) {
            tok.save(&path, pretty).unwrap();
        }
        reader.join().unwrap();
    });
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        tok.save(&path, true).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }
}

#[test]
fn retraining_reassigns_added_tokens_without_corrupting_model_ids() {
    use morpheme::AddedToken;
    let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("a", 1), ("[SEP]", 2)]);
    tok.add_tokens(&[AddedToken::new("[SEP]", false)
        .lstrip(true)
        .rstrip(true)
        .single_word(true)
        .normalized(false)])
        .unwrap();
    tok.add_tokens(&[AddedToken::new("extra", false).single_word(true)])
        .unwrap();
    tok.set_encode_special_tokens(true);
    tok.train(word_level_trainer(), ["a b a"].into_iter())
        .unwrap();
    assert!(tok.added_vocabulary().encode_special_tokens());
    tok.set_encode_special_tokens(false);
    assert_eq!(tok.token_to_id("[SEP]"), tok.model().token_to_id("[SEP]"));
    assert_eq!(tok.token_to_id("a"), Some(2));
    assert_eq!(tok.token_to_id("extra"), Some(4));
    assert_eq!(
        tok.encode("a [SEP] extra", false).unwrap().ids(),
        &[2, 1, 4]
    );
    assert_eq!(tok.decode(&[2, 1, 4], true).unwrap(), "a extra");
    assert!(tok.added_vocabulary().added_tokens_decoder()[&4].single_word);
    let sep = &tok.added_vocabulary().added_tokens_decoder()[&1];
    assert!(sep.special && sep.lstrip && sep.rstrip && sep.single_word);
    assert!(!sep.normalized);
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(
        reloaded.encode("a [SEP] extra", false).unwrap(),
        tok.encode("a [SEP] extra", false).unwrap()
    );
}

#[test]
fn failed_token_addition_preserves_vocabulary_and_matchers() {
    use morpheme::{AddedToken, AddedVocabulary};
    let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("last", u32::MAX - 1)]);
    let before = tok.to_json(false).unwrap();
    assert!(
        tok.add_tokens(&[
            AddedToken::new("extra", false),
            AddedToken::new("overflow", false),
        ])
        .is_err()
    );
    assert_eq!(tok.to_json(false).unwrap(), before);
    assert_eq!(tok.token_to_id("extra"), None);
    tok.add_tokens(&[AddedToken::new("extra", false)]).unwrap();
    assert_eq!(tok.encode("extra", false).unwrap().ids(), &[u32::MAX]);

    struct FailingNormalizer;
    impl Normalizer for FailingNormalizer {
        fn normalize(&self, _: &mut NormalizedString) -> morpheme::Result<()> {
            Err(morpheme::Error::Config("normalization failed".into()))
        }
    }
    let model = word_level_tokenizer(&[("[UNK]", 0)]);
    let mut added = AddedVocabulary::new();
    added
        .add_tokens(&[AddedToken::new("existing", false)], model.model(), None)
        .unwrap();
    let before = serde_json::to_value(added.tokens_with_ids()).unwrap();
    assert!(
        added
            .add_tokens(
                &[AddedToken::new("new", false)],
                model.model(),
                Some(&FailingNormalizer)
            )
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(added.tokens_with_ids()).unwrap(),
        before
    );
    assert_eq!(added.token_to_id("new", model.model()), None);
}

#[test]
fn retraining_rebinds_processors_and_padding_by_token_text() {
    use morpheme::processors::{
        BertProcessing, RobertaProcessing, Sequence, SpecialToken, TemplateProcessing,
    };
    let template = TemplateProcessing::builder()
        .try_single("prefix $A [SEP]")
        .unwrap()
        .try_pair("prefix $A [SEP] $B:1 [SEP]:1")
        .unwrap()
        .special_tokens(vec![
            SpecialToken::new(
                "prefix".into(),
                vec![11, 12],
                vec!["[CLS]".into(), "[PAD]".into()],
            )
            .unwrap(),
            SpecialToken::from(("[SEP]", 10)),
        ])
        .build()
        .unwrap();
    let processors = [
        BertProcessing::new(("[SEP]", 10), ("[CLS]", 11)).into(),
        RobertaProcessing::new(("[SEP]", 10), ("[CLS]", 11)).into(),
        Sequence::new(vec![
            Sequence::new(vec![
                morpheme::pre_tokenizers::ByteLevel::default().into(),
                template.into(),
            ])
            .into(),
        ])
        .into(),
    ];
    for processor in processors {
        let mut tok = word_level_tokenizer(&[
            ("[UNK]", 0),
            ("[SEP]", 10),
            ("[CLS]", 11),
            ("[PAD]", 12),
            ("a", 13),
        ]);
        tok.set_post_processor(Some(processor));
        tok.set_padding(Some(morpheme::PaddingParams {
            strategy: morpheme::PaddingStrategy::Fixed(8),
            pad_id: 12,
            ..Default::default()
        }));
        let trainer = morpheme::trainers::WordLevelTrainer::builder()
            .show_progress(false)
            .special_tokens(
                ["[UNK]", "[CLS]", "[PAD]", "[SEP]"]
                    .map(|s| morpheme::AddedToken::new(s, true))
                    .to_vec(),
            )
            .build()
            .unwrap();
        tok.train(trainer, ["a b a"].into_iter()).unwrap();
        assert_eq!(tok.padding().unwrap().pad_id, 2);
        let encoded = tok.encode(("a", "b"), true).unwrap();
        for (token, id) in encoded.tokens().iter().zip(encoded.ids()) {
            assert_eq!(tok.token_to_id(token), Some(*id));
        }
        assert_eq!(encoded.len(), 8);
        assert_eq!(tok.decode(encoded.ids(), true).unwrap(), "a b");
        let loaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
        assert_eq!(loaded.encode(("a", "b"), true).unwrap(), encoded);
    }
}

#[test]
fn retraining_missing_configured_tokens_leaves_tokenizer_unchanged() {
    for padding in [false, true] {
        let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("a", 1), ("[CLS]", 2), ("[PAD]", 3)]);
        if padding {
            tok.set_padding(Some(morpheme::PaddingParams {
                pad_id: 3,
                ..Default::default()
            }));
        } else {
            tok.set_post_processor(Some(
                morpheme::processors::BertProcessing::new(("[SEP]", 4), ("[CLS]", 2)).into(),
            ));
        }
        let before = tok.to_json(false).unwrap();
        let error = tok
            .train(word_level_trainer(), ["a b a"].into_iter())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(if padding { "[PAD]" } else { "[CLS]" })
        );
        assert_eq!(tok.to_json(false).unwrap(), before);
    }
}

#[test]
fn adding_tokens_to_sparse_vocab_preserves_round_trips() {
    let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("a", 2)]);
    tok.add_tokens(&[morpheme::AddedToken::new("x", false)])
        .unwrap();
    assert_eq!(tok.token_to_id("x"), Some(3));
    let encoded = tok.encode("a x", false).unwrap();
    assert_eq!(encoded.ids(), &[2, 3]);
    assert_eq!(tok.decode(encoded.ids(), false).unwrap(), "a x");
}

#[test]
fn added_token_id_exhaustion_is_an_error_not_a_panic() {
    use morpheme::AddedToken;
    let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("last", u32::MAX - 1)]);
    assert_eq!(
        tok.add_tokens(&[AddedToken::new("extra", false)]).unwrap(),
        1
    );
    assert_eq!(tok.token_to_id("extra"), Some(u32::MAX));
    assert_eq!(
        tok.add_tokens(&[AddedToken::new("extra", false)]).unwrap(),
        0
    );
    assert_eq!(
        tok.add_special_tokens(&[AddedToken::new("last", true)])
            .unwrap(),
        1
    );
    assert!(
        tok.add_tokens(&[AddedToken::new("overflow", false)])
            .is_err()
    );
    assert_eq!(tok.token_to_id("overflow"), None);
    let mut loaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert!(
        loaded
            .add_tokens(&[AddedToken::new("overflow", false)])
            .is_err()
    );
    let mut full = word_level_tokenizer(&[("[UNK]", 0), ("last", u32::MAX)]);
    assert!(
        full.add_tokens(&[AddedToken::new("overflow", false)])
            .is_err()
    );
}

#[test]
fn bpe_flushes_unknown_runs_before_byte_fallback() {
    let vocab: HashMap<String, u32> = [("[UNK]", 0), ("<0x62>", 1), ("a", 2)]
        .map(|(s, id)| (s.to_string(), id))
        .into();
    for fuse_unk in [false, true] {
        let tok = Tokenizer::new(
            Bpe::builder()
                .vocab_and_merges(vocab.clone(), vec![])
                .unk_token("[UNK]")
                .byte_fallback(true)
                .fuse_unk(fuse_unk)
                .build()
                .unwrap(),
        );
        let encoded = tok.encode("zzba", false).unwrap();
        let expected = if fuse_unk {
            vec![0, 1, 2]
        } else {
            vec![0, 0, 1, 2]
        };
        assert_eq!(encoded.ids(), expected);
        let n = encoded.len();
        assert_eq!(&encoded.offsets()[n - 2..], &[(2, 3), (3, 4)]);
        // Cached inference must retain the same order and offsets.
        assert_eq!(tok.encode("zzba", false).unwrap(), encoded);
    }
}

#[test]
fn pair_overflow_without_processor_keeps_sequence_ownership() {
    let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    tok.set_truncation(Some(morpheme::TruncationParams {
        max_length: 3,
        strategy: morpheme::TruncationStrategy::OnlySecond,
        ..Default::default()
    }))
    .unwrap();
    let encoded = tok.encode(("a", "b c d"), false).unwrap();
    let overflow = &encoded.overflowing()[0];
    assert_eq!(overflow.tokens(), &["a", "d"]);
    assert_eq!(overflow.sequence_ids(), vec![Some(0), Some(1)]);
    assert_eq!(overflow.token_to_chars(1), Some((1, (4, 5))));
    assert_eq!(overflow.word_to_tokens(2, 1), Some((1, 2)));
    assert_eq!(overflow.char_to_token(4, 1), Some(1));
    assert_eq!(overflow.char_to_token(4, 0), None);
}

#[test]
fn truncation_rejects_impossible_special_token_budgets() {
    let mut tok = word_level_tokenizer(&[("[UNK]", 0), ("a", 1)]);
    tok.set_post_processor(Some(morpheme::processors::BertProcessing::default().into()));
    for max_length in [0, 1, 2] {
        tok.set_truncation(Some(morpheme::TruncationParams {
            max_length,
            ..Default::default()
        }))
        .unwrap();
        assert!(tok.encode("a", true).is_err());
        assert!(tok.encode(("a", "a"), true).is_err());
        assert!(tok.encode("a", false).is_ok());
    }
    assert_eq!(tok.encode("", true).unwrap().len(), 2);
    tok.set_truncation(Some(morpheme::TruncationParams {
        max_length: 3,
        ..Default::default()
    }))
    .unwrap();
    let encoded = tok.encode("a a a", true).unwrap();
    assert_eq!(encoded.len(), 3);
    assert!(encoded.overflowing().iter().all(|e| e.len() <= 3));
    assert!(tok.encode(("a", "a"), true).is_err());
    assert_eq!(tok.encode(("", ""), true).unwrap().len(), 3);
    // Deserialized settings must receive the same runtime validation.
    let loaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert!(loaded.encode(("a", "a"), true).is_err());
}

#[test]
fn file_training_streams_line_endings_and_reports_late_errors() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.txt");
    let second = dir.path().join("second.txt");
    std::fs::write(&first, "a\r\nb\n").unwrap();
    std::fs::write(&second, "c\nlast").unwrap();
    // No pre-tokenizer: each complete line, including its ending, is a token.
    let initial = morpheme::models::WordLevel::builder()
        .unk_token("[UNK]")
        .build()
        .unwrap();
    let mut files = Tokenizer::new(initial.clone());
    let mut iterator = Tokenizer::new(initial);
    files
        .train_from_files(word_level_trainer(), &[&first, &second])
        .unwrap();
    iterator
        .train(
            word_level_trainer(),
            ["a\r\n", "b\n", "c\n", "last"].into_iter(),
        )
        .unwrap();
    assert_eq!(
        files.to_json(false).unwrap(),
        iterator.to_json(false).unwrap()
    );
    let before = files.to_json(false).unwrap();
    std::fs::write(&second, b"valid\n\xff\n").unwrap();
    let err = files
        .train_from_files(word_level_trainer(), &[&first, &second])
        .unwrap_err();
    assert!(err.to_string().contains("second.txt"));
    assert_eq!(files.to_json(false).unwrap(), before);
    let missing = dir.path().join("missing.txt");
    assert!(
        files
            .train_from_files(word_level_trainer(), &[&first, &missing])
            .is_err()
    );
    assert_eq!(files.to_json(false).unwrap(), before);
}

#[test]
fn sequence_lookups_handle_nonzero_and_missing_ids() {
    let mut encoding = morpheme::Encoding::new(
        vec![1],
        vec![1],
        vec!["a".into()],
        vec![Some(0)],
        vec![(0, 1)],
        vec![0],
        vec![1],
        vec![],
    );
    assert_eq!(encoding.char_to_token(0, 1), None);
    assert_eq!(encoding.word_to_tokens(0, 1), None);
    for sequence in [1, 7] {
        let mut encoding = encoding.clone();
        encoding.set_sequence_id(sequence);
        assert_eq!(encoding.sequence_ids(), vec![Some(sequence)]);
        assert_eq!(encoding.token_to_sequence(0), Some(sequence));
        assert_eq!(encoding.char_to_token(0, sequence), Some(0));
        assert_eq!(encoding.word_to_tokens(0, sequence), Some((0, 1)));
        assert_eq!(encoding.char_to_token(0, 0), None);
        assert_eq!(encoding.word_to_tokens(0, 0), None);
    }
    encoding.set_sequence_id(7);
    encoding.pad(3, 0, 0, "[PAD]", morpheme::PaddingDirection::Left);
    assert_eq!(encoding.sequence_ids(), vec![None, None, Some(7)]);
}

#[test]
fn normalized_offset_conversion_rejects_invalid_ranges() {
    use morpheme::OffsetRange;
    let normalized = NormalizedString::from("a");
    for range in [99..99, 0..99, std::ops::Range { start: 2, end: 1 }] {
        assert_eq!(
            normalized.convert_offsets(OffsetRange::Normalized(range.clone())),
            None
        );
        assert_eq!(
            normalized.convert_offsets(OffsetRange::Original(range)),
            None
        );
    }
    assert_eq!(
        normalized.convert_offsets(OffsetRange::Normalized(1..1)),
        Some(1..1)
    );
    assert_eq!(
        normalized.convert_offsets(OffsetRange::Original(..=usize::MAX)),
        None
    );
    assert_eq!(
        normalized.convert_offsets(OffsetRange::Normalized((
            std::ops::Bound::Excluded(usize::MAX),
            std::ops::Bound::Unbounded
        ))),
        None
    );
}

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
