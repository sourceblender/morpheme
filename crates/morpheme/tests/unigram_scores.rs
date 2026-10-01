//! Issue #86: Unigram scores must be finite so every saved
//! `tokenizer.json` can be loaded back.

use std::collections::HashSet;

use morpheme::models::{ModelWrapper, Unigram};
use morpheme::trainers::UnigramTrainer;
use morpheme::{Error, Tokenizer};

fn unigram_scores(tok: &Tokenizer) -> Vec<(String, f64)> {
    let ModelWrapper::Unigram(model) = tok.model() else {
        panic!("expected a Unigram model");
    };
    model.pieces().to_vec()
}

/// Training on an empty corpus used to give the required chars an
/// infinite score, saved as `null`, which could not be loaded back.
#[test]
fn empty_corpus_training_round_trips_through_save_and_load() {
    let trainer = UnigramTrainer::builder()
        .show_progress(false)
        .initial_alphabet(HashSet::from(['x']))
        .unk_token("<unk>")
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(Unigram::default());
    tok.train(trainer, std::iter::empty::<&str>()).unwrap();

    let scores = unigram_scores(&tok);
    assert_eq!(
        scores.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        ["<unk>", "x"]
    );
    assert!(scores.iter().all(|(_, s)| s.is_finite()), "{scores:?}");

    let json = tok.to_json(false).unwrap();
    assert!(!json.contains("null]"), "{json}");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    tok.save(&path, false).unwrap();
    let loaded = Tokenizer::from_file(&path).unwrap();
    assert_eq!(loaded.to_json(false).unwrap(), json);
    assert_eq!(unigram_scores(&loaded), scores);
    let enc = loaded.encode("xy", false).unwrap();
    assert_eq!(enc.tokens(), ["x", "y"]);
    assert_eq!(enc.ids(), [1, 0]);
}

/// Several unlearned chars share a uniform log-probability (with HF's
/// growing tie-break penalty), all finite.
#[test]
fn empty_corpus_with_several_required_chars_has_finite_scores() {
    let trainer = UnigramTrainer::builder()
        .show_progress(false)
        .initial_alphabet(HashSet::from(['a', 'b', 'c', 'd']))
        .unk_token("<unk>")
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(Unigram::default());
    tok.train(trainer, std::iter::empty::<&str>()).unwrap();
    let scores = unigram_scores(&tok);
    assert_eq!(scores.len(), 5);
    for (piece, score) in &scores[1..] {
        assert!(score.is_finite(), "{piece}: {score}");
        assert!((score + 4f64.ln()).abs() < 1e-3, "{piece}: {score}");
    }
    Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
}

#[test]
fn unigram_new_rejects_non_finite_scores() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = Unigram::new(
            vec![("<unk>".into(), 0.0), ("y".into(), bad)],
            Some(0),
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("\"y\"") && msg.contains("id 1"), "{msg}");
    }
    // A finite vocabulary is still fine.
    Unigram::new(
        vec![("<unk>".into(), 0.0), ("y".into(), -1e300)],
        Some(0),
        false,
    )
    .unwrap();
}

/// `NaN` and infinity cannot appear in valid JSON; `null` and
/// out-of-range numbers are parse errors (as in HF tokenizers).
#[test]
fn non_numeric_scores_in_json_are_rejected() {
    for score in ["null", "1e400", "\"NaN\""] {
        let json =
            format!(r#"{{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["x",{score}]]}}"#);
        assert!(serde_json::from_str::<Unigram>(&json).is_err(), "{score}");
    }
}

/// Every real Unigram fixture still loads and re-serializes unchanged
/// (skipped when `scripts/fetch-hf-fixtures.sh` has not been run).
#[test]
fn every_fixture_round_trips() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hf");
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("skipping: {dir} not found (run scripts/fetch-hf-fixtures.sh)");
        return;
    };
    let mut seen = 0;
    for entry in entries {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let tok = Tokenizer::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let json = tok.to_json(false).unwrap();
        let back = Tokenizer::from_json(&json).unwrap();
        assert_eq!(back.to_json(false).unwrap(), json, "{}", path.display());
        if let ModelWrapper::Unigram(m) = tok.model() {
            assert!(m.pieces().iter().all(|(_, s)| s.is_finite()));
        }
        seen += 1;
    }
    eprintln!("round-tripped {seen} fixtures");
}
