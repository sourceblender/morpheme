//! Parity tests against real Hugging Face `tokenizer.json` files
//! (downloaded by `scripts/fetch-hf-fixtures.sh` into the gitignored
//! `tests/data/hf/`). Tests skip with a message when a file is missing.

use std::path::PathBuf;

use crate::models::ModelWrapper;
use crate::traits::Model;

#[path = "fixture_data.rs"]
mod data;

fn fixture(name: &str) -> Option<serde_json::Value> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "data",
        "hf",
        &format!("{name}.json"),
    ]
    .iter()
    .collect();
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(serde_json::from_str(&s).expect("fixture is valid JSON")),
        Err(_) => {
            println!(
                "skipping: {} not found (run scripts/fetch-hf-fixtures.sh)",
                path.display()
            );
            None
        }
    }
}

fn load_model(name: &str) -> Option<(ModelWrapper, serde_json::Value)> {
    let doc = fixture(name)?;
    let raw = doc["model"].clone();
    let model: ModelWrapper = serde_json::from_value(raw.clone())
        .unwrap_or_else(|e| panic!("{name}: failed to load model: {e}"));
    Some((model, raw))
}

/// What HF writes when it re-saves `raw`: an explicit `"type"`, every
/// option present, and merges as pairs.
fn hf_resave_shape(model: &ModelWrapper, mut raw: serde_json::Value) -> serde_json::Value {
    let obj = raw.as_object_mut().expect("model is an object");
    match model {
        ModelWrapper::Bpe(_) => {
            obj.insert("type".into(), "BPE".into());
            for (key, default) in [
                ("dropout", serde_json::Value::Null),
                ("unk_token", serde_json::Value::Null),
                ("continuing_subword_prefix", serde_json::Value::Null),
                ("end_of_word_suffix", serde_json::Value::Null),
                ("fuse_unk", false.into()),
                ("byte_fallback", false.into()),
                ("ignore_merges", false.into()),
            ] {
                obj.entry(key).or_insert(default);
            }
            let merges: Vec<serde_json::Value> = obj["merges"]
                .as_array()
                .expect("merges array")
                .iter()
                .map(|m| match m.as_str() {
                    Some(s) => {
                        let (a, b) = s.split_once(' ').expect("legacy merge");
                        serde_json::json!([a, b])
                    }
                    None => m.clone(),
                })
                .collect();
            obj.insert("merges".into(), merges.into());
        }
        ModelWrapper::WordPiece(_) => {
            obj.insert("type".into(), "WordPiece".into());
        }
        _ => {}
    }
    raw
}

const FIXTURES: &[&str] = &[
    "gpt2",
    "roberta-base",
    "gpt-neox-20b",
    "qwen2.5",
    "llama",
    "bert-base-uncased",
    "bert-base-cased",
    "minilm",
];

#[test]
fn real_models_round_trip_through_json() {
    for name in FIXTURES {
        let Some((model, raw)) = load_model(name) else {
            continue;
        };
        let ours = serde_json::to_value(&model).unwrap();
        assert_eq!(
            ours,
            hf_resave_shape(&model, raw),
            "{name}: re-serialization differs"
        );
        let again: ModelWrapper = serde_json::from_value(ours).unwrap();
        assert_eq!(again, model, "{name}: second load differs");
    }
}

/// Byte-exact comparison against HF's own re-serialization, when the
/// dumps are available (`MORPHEME_HF_MODEL_DUMPS=<dir>`, each file being
/// `json.loads(Tokenizer.from_file(p).to_str())["model"]` re-dumped with
/// `separators=(",", ":"), ensure_ascii=False`).
#[test]
fn real_models_serialize_byte_identically_to_hf() {
    let Ok(dir) = std::env::var("MORPHEME_HF_MODEL_DUMPS") else {
        return;
    };
    for name in FIXTURES {
        let Some((model, _)) = load_model(name) else {
            continue;
        };
        // Dumps are written with `separators=(",", ":")` and
        // `ensure_ascii=False`, so key order and formatting must match.
        let expected = std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap();
        let ours = serde_json::to_string(&model).unwrap();
        assert!(
            ours == expected,
            "{name}: serialization is not byte-identical"
        );
    }
}

#[test]
fn tokenize_matches_hf_model_tokenize() {
    let mut loaded: Vec<(&str, ModelWrapper)> = Vec::new();
    for (name, word, expected) in data::TOKENIZE_CASES {
        let model = match loaded.iter().find(|(n, _)| n == name) {
            Some((_, m)) => m,
            None => {
                let Some((m, _)) = load_model(name) else {
                    continue;
                };
                loaded.push((name, m));
                &loaded.last().unwrap().1
            }
        };
        let got: Vec<(u32, String, (usize, usize))> = model
            .tokenize(word)
            .unwrap_or_else(|e| panic!("{name} {word:?}: {e}"))
            .into_iter()
            .map(|t| (t.id, t.value, t.offsets))
            .collect();
        let want: Vec<(u32, String, (usize, usize))> = expected
            .iter()
            .map(|(id, v, o)| (*id, v.to_string(), *o))
            .collect();
        assert_eq!(got, want, "{name} {word:?}");
    }
}
