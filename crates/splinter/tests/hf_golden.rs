//! Golden interop tests against Hugging Face `tokenizers`.
//!
//! For every real `tokenizer.json` listed in `scripts/hf-fixtures.txt`,
//! splinter must reproduce the ids, tokens, char offsets, type ids,
//! masks, word ids and decoded strings recorded from the Python library
//! in `tests/golden/*.json` (see `scripts/gen_golden.py`), and must
//! serialize every pipeline component exactly as `tokenizers` does.
//!
//! The tokenizer files themselves are not committed; fetch them with
//! `scripts/fetch-hf-fixtures.sh`. Missing files fail the test unless
//! `SPLINTER_SKIP_HF_GOLDEN=1` is set.

use std::path::{Path, PathBuf};

use serde_json::Value;
use splinter::{Encoding, Tokenizer};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn fixtures() -> Vec<String> {
    let list = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/hf-fixtures.txt"),
    )
    .expect("scripts/hf-fixtures.txt");
    list.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().next().unwrap().to_owned())
        .collect()
}

fn load(name: &str) -> Option<(Tokenizer, Value)> {
    let tok_path = root().join("data/hf").join(format!("{name}.json"));
    if !tok_path.exists() {
        if std::env::var_os("SPLINTER_SKIP_HF_GOLDEN").is_some() {
            eprintln!("skipping {name}: {} missing", tok_path.display());
            return None;
        }
        panic!(
            "{} is missing; run scripts/fetch-hf-fixtures.sh (or set SPLINTER_SKIP_HF_GOLDEN=1)",
            tok_path.display()
        );
    }
    let tokenizer = Tokenizer::from_file(&tok_path)
        .unwrap_or_else(|e| panic!("{name}: failed to load tokenizer.json: {e}"));
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("golden").join(format!("{name}.json"))).unwrap(),
    )
    .unwrap();
    Some((tokenizer, golden))
}

/// Collects mismatches instead of stopping at the first one.
#[derive(Default)]
struct Report {
    failures: Vec<String>,
}

impl Report {
    fn check<T: PartialEq + std::fmt::Debug>(&mut self, ctx: &str, got: T, want: T) {
        if got != want {
            self.failures
                .push(format!("{ctx}\n    got:  {got:?}\n    want: {want:?}"));
        }
    }

    fn finish(self, name: &str) {
        if !self.failures.is_empty() {
            let shown: Vec<_> = self.failures.iter().take(25).cloned().collect();
            panic!(
                "{name}: {} mismatches (showing {}):\n{}",
                self.failures.len(),
                shown.len(),
                shown.join("\n")
            );
        }
    }
}

fn as_u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn compare_encoding(r: &mut Report, ctx: &str, got: &Encoding, want: &Value) {
    r.check(
        &format!("{ctx} ids"),
        got.ids().to_vec(),
        as_u32s(&want["ids"]),
    );
    let tokens: Vec<String> = want["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_owned())
        .collect();
    r.check(&format!("{ctx} tokens"), got.tokens().to_vec(), tokens);
    let offsets: Vec<(usize, usize)> = want["offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    r.check(&format!("{ctx} offsets"), got.offsets().to_vec(), offsets);
    r.check(
        &format!("{ctx} type_ids"),
        got.type_ids().to_vec(),
        as_u32s(&want["type_ids"]),
    );
    r.check(
        &format!("{ctx} special_tokens_mask"),
        got.special_tokens_mask().to_vec(),
        as_u32s(&want["special_tokens_mask"]),
    );
    r.check(
        &format!("{ctx} attention_mask"),
        got.attention_mask().to_vec(),
        as_u32s(&want["attention_mask"]),
    );
    let words: Vec<Option<u32>> = want["word_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_u64().map(|x| x as u32))
        .collect();
    r.check(&format!("{ctx} word_ids"), got.word_ids().to_vec(), words);
}

/// Feed `ids` one at a time and compare every step's output.
fn check_stream(
    r: &mut Report,
    ctx: &str,
    stream: &mut splinter::DecodeStream<'_>,
    ids: &[u32],
    want: &Value,
) {
    let want: Vec<Option<String>> = want
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().map(str::to_owned))
        .collect();
    let mut got = Vec::with_capacity(ids.len());
    for &id in ids {
        match stream.step(id) {
            Ok(chunk) => got.push(chunk),
            Err(e) => {
                r.failures
                    .push(format!("{ctx}: stream error at id {id}: {e}"));
                return;
            }
        }
    }
    r.check(ctx, got, want);
}

fn run_cases(r: &mut Report, label: &str, tok: &Tokenizer, golden: &Value) {
    for (i, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let input = case["input"].as_str().unwrap();
        let short: String = input.chars().take(30).collect();
        for (key, add_special) in [("with_special", true), ("without_special", false)] {
            let ctx = format!("[{label}] case {i} {short:?} {key}");
            match tok.encode_char_offsets(input, add_special) {
                Ok(enc) => compare_encoding(r, &ctx, &enc, &case[key]),
                Err(e) => r.failures.push(format!("{ctx}: encode error: {e}")),
            }
        }
        let ids = as_u32s(&case["with_special"]["ids"]);
        for (key, skip) in [
            ("decode_keep_special", false),
            ("decode_skip_special", true),
        ] {
            let ctx = format!("[{label}] case {i} {short:?} {key}");
            match tok.decode(&ids, skip) {
                Ok(s) => r.check(&ctx, s.as_str(), case[key].as_str().unwrap()),
                Err(e) => r.failures.push(format!("{ctx}: decode error: {e}")),
            }
        }
        for (key, skip) in [
            ("stream_keep_special", false),
            ("stream_skip_special", true),
        ] {
            let ctx = format!("[{label}] case {i} {short:?} {key}");
            let mut stream = tok.decode_stream(skip);
            check_stream(r, &ctx, &mut stream, &ids, &case[key]);
        }
        let half = ids.len() / 2;
        let ctx = format!("[{label}] case {i} {short:?} stream_prefill_half");
        let mut stream = tok.decode_stream(false).prefill(&ids[..half]);
        check_stream(
            r,
            &ctx,
            &mut stream,
            &ids[half..],
            &case["stream_prefill_half"],
        );
    }

    for (i, pair) in golden["pairs"].as_array().unwrap().iter().enumerate() {
        let (a, b) = (pair["a"].as_str().unwrap(), pair["b"].as_str().unwrap());
        let ctx = format!("[{label}] pair {i}");
        match tok.encode_char_offsets((a, b), true) {
            Ok(enc) => compare_encoding(r, &ctx, &enc, &pair["encoding"]),
            Err(e) => r.failures.push(format!("{ctx}: encode error: {e}")),
        }
    }

    let batch = &golden["batch"];
    let inputs: Vec<&str> = batch["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    match tok.encode_batch_char_offsets(inputs, true) {
        Ok(encs) => {
            for (i, (enc, want)) in encs
                .iter()
                .zip(batch["encodings"].as_array().unwrap())
                .enumerate()
            {
                compare_encoding(r, &format!("[{label}] batch {i}"), enc, want);
            }
        }
        Err(e) => r
            .failures
            .push(format!("[{label}] batch: encode error: {e}")),
    }
}

/// Every object key path, in document order (field order matters for
/// byte compatibility with `tokenizers`).
fn key_order(v: &Value, path: &str, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, child) in m {
                let p = format!("{path}.{k}");
                out.push(p.clone());
                key_order(child, &p, out);
            }
        }
        Value::Array(a) => {
            for (i, child) in a.iter().enumerate() {
                key_order(child, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

fn check_serialization(r: &mut Report, tok: &Tokenizer, golden: &Value) {
    let ours: Value = serde_json::from_str(&tok.to_json(false).unwrap()).unwrap();
    let hf = &golden["hf_serialized"];
    let top: Vec<&str> = ours
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    r.check(
        "serialized top-level key order",
        top,
        vec![
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        ],
    );
    for key in [
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
    ] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        key_order(&ours[key], key, &mut a);
        key_order(&hf[key], key, &mut b);
        r.check(&format!("serialized {key} field order"), a, b);
    }
    for key in [
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
    ] {
        r.check(&format!("serialized {key}"), &ours[key], &hf[key]);
    }
    let model = &ours["model"];
    let fp = &hf["model_fingerprint"];
    for (k, v) in fp.as_object().unwrap() {
        let got = match k.as_str() {
            "vocab_len" => model["vocab"]
                .as_object()
                .map(|m| m.len())
                .or_else(|| model["vocab"].as_array().map(|a| a.len()))
                .map(Value::from)
                .unwrap_or(Value::Null),
            "vocab_head" | "vocab_tail" => {
                let items: Vec<Value> = match &model["vocab"] {
                    Value::Object(m) => m
                        .iter()
                        .map(|(k, v)| Value::Array(vec![Value::from(k.clone()), v.clone()]))
                        .collect(),
                    Value::Array(a) => a.clone(),
                    _ => vec![],
                };
                let slice = if k == "vocab_head" {
                    items.iter().take(5).cloned().collect::<Vec<_>>()
                } else {
                    items[items.len().saturating_sub(5)..].to_vec()
                };
                Value::Array(slice)
            }
            "merges_len" => Value::from(model["merges"].as_array().map_or(0, |a| a.len())),
            "merges_head" => Value::Array(
                model["merges"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .take(5)
                    .cloned()
                    .collect(),
            ),
            "merges_tail" => {
                let m = model["merges"].as_array().unwrap();
                Value::Array(m[m.len().saturating_sub(5)..].to_vec())
            }
            other => model[other].clone(),
        };
        r.check(&format!("serialized model.{k}"), &got, v);
    }
    // Model keys must match exactly (no missing, no extra).
    let mut ours_keys: Vec<&String> = model.as_object().unwrap().keys().collect();
    let mut hf_keys: Vec<String> = fp
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| !k.ends_with("_len") && !k.ends_with("_head") && !k.ends_with("_tail"))
        .cloned()
        .chain(["vocab".to_owned()])
        .chain(fp.get("merges_len").map(|_| "merges".to_owned()))
        .collect();
    ours_keys.sort();
    hf_keys.sort();
    let hf_keys_ref: Vec<&String> = hf_keys.iter().collect();
    r.check("serialized model keys", ours_keys, hf_keys_ref);
}

fn golden(name: &str) {
    let Some((tok, golden)) = load(name) else {
        return;
    };
    let mut r = Report::default();
    r.check(
        "vocab size with added",
        tok.get_vocab_size(true) as u64,
        golden["vocab_size_with_added"].as_u64().unwrap(),
    );
    r.check(
        "vocab size without added",
        tok.get_vocab_size(false) as u64,
        golden["vocab_size_without_added"].as_u64().unwrap(),
    );
    run_cases(&mut r, "loaded", &tok, &golden);
    check_serialization(&mut r, &tok, &golden);

    // Save → load must not change behavior.
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap())
        .unwrap_or_else(|e| panic!("{name}: reloading our own JSON failed: {e}"));
    run_cases(&mut r, "reloaded", &reloaded, &golden);
    r.finish(name);
}

macro_rules! golden_tests {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(#[test] fn $test() { golden($name); })*

        #[test]
        fn every_fixture_has_a_test() {
            let covered = [$($name),*];
            for f in fixtures() {
                assert!(covered.contains(&f.as_str()), "fixture {f} has no golden test");
            }
        }
    };
}

golden_tests! {
    bert_base_uncased => "bert-base-uncased",
    bert_base_cased => "bert-base-cased",
    minilm => "minilm",
    gpt2 => "gpt2",
    roberta_base => "roberta-base",
    gpt_neox_20b => "gpt-neox-20b",
    qwen2_5 => "qwen2.5",
    llama => "llama",
    t5_small => "t5-small",
    albert_base_v2 => "albert-base-v2",
    xlm_roberta_base => "xlm-roberta-base",
}
