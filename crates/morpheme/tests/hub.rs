//! Network tests for `Tokenizer::from_pretrained` (feature `hub`).
//!
//! Ignored by default; CI runs them with
//! `cargo test -p morpheme --features hub --test hub -- --ignored`.
#![cfg(feature = "hub")]

use std::path::Path;

use morpheme::{FromPretrainedParameters, Tokenizer};

/// Public repositories need no credentials: an empty token opts out of
/// `HF_TOKEN` and the `huggingface-cli login` token file, so the
/// developer's real token is never sent by these tests.
fn anonymous() -> FromPretrainedParameters {
    FromPretrainedParameters::default().token("")
}

/// Revision of google-bert/bert-base-uncased pinned in
/// scripts/hf-fixtures.txt.
const BERT_REVISION: &str = "86b5e0934494bd15c9632b12f734a8a67f723594";

#[test]
#[ignore = "downloads from the Hugging Face Hub"]
fn hub_download_matches_pinned_fixture() {
    let cache = tempfile::tempdir().unwrap();
    let params = anonymous().revision(BERT_REVISION).cache_dir(cache.path());
    let downloaded =
        Tokenizer::from_pretrained("google-bert/bert-base-uncased", Some(params.clone())).unwrap();

    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hf/bert-base-uncased.json");
    if local.exists() {
        let local = Tokenizer::from_file(local).unwrap();
        for text in ["Hello, world!", "Café naïve 你好 😀", ""] {
            assert_eq!(
                downloaded.encode(text, true).unwrap(),
                local.encode(text, true).unwrap(),
                "{text:?}"
            );
        }
    } else {
        let enc = downloaded.encode("Hello, world!", true).unwrap();
        assert_eq!(enc.ids(), [101, 7592, 1010, 2088, 999, 102]);
    }

    // The file landed in the standard cache layout and is reused offline.
    let snapshot = cache
        .path()
        .join("models--google-bert--bert-base-uncased/snapshots")
        .join(BERT_REVISION)
        .join("tokenizer.json");
    assert!(snapshot.is_file());
    let again = Tokenizer::from_pretrained("google-bert/bert-base-uncased", Some(params)).unwrap();
    assert_eq!(again.vocab_size(true), downloaded.vocab_size(true));
}

#[test]
#[ignore = "downloads from the Hugging Face Hub"]
fn hub_missing_repo_is_a_clear_error() {
    let cache = tempfile::tempdir().unwrap();
    let params = anonymous().cache_dir(cache.path());
    let err = Tokenizer::from_pretrained("morpheme-tests/definitely-not-a-repo-7f3a", Some(params))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not exist") || err.contains("token"),
        "{err}"
    );
}
