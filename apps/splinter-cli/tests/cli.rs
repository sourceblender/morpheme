//! End-to-end tests for the `splinter` binary.

use std::path::Path;
use std::process::{Command, Output};

fn splinter(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_splinter"))
        .args(args)
        .output()
        .expect("run splinter")
}

fn stdout(o: &Output) -> String {
    assert!(
        o.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn corpus() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/corpus.txt")
        .display()
        .to_string()
}

#[test]
fn train_encode_decode_inspect_every_model() {
    let dir = tempfile::tempdir().unwrap();
    for model in ["bpe", "wordpiece", "unigram", "wordlevel"] {
        let out = dir.path().join(format!("{model}.json"));
        let out = out.to_str().unwrap();
        stdout(&splinter(&[
            "train",
            "--model",
            model,
            "--vocab-size",
            "300",
            "--out",
            out,
            &corpus(),
        ]));

        let enc = stdout(&splinter(&[
            "encode",
            "-t",
            out,
            "--json",
            "the quick brown fox",
        ]));
        let v: serde_json::Value = serde_json::from_str(&enc).unwrap();
        let ids: Vec<String> = v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.to_string())
            .collect();
        assert!(!ids.is_empty(), "{model}: no ids");

        let mut args = vec!["decode", "-t", out, "--skip-special-tokens"];
        let joined = ids.join(",");
        args.push(&joined);
        let text = stdout(&splinter(&args));
        assert_eq!(text.trim(), "the quick brown fox", "{model}");

        let info = stdout(&splinter(&["inspect", "-t", out]));
        assert!(info.contains("vocab size"), "{model}: {info}");
    }
}

#[test]
fn bert_preset_adds_cls_and_sep() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("wp.json");
    let out = out.to_str().unwrap();
    stdout(&splinter(&[
        "train",
        "--model",
        "wordpiece",
        "--vocab-size",
        "300",
        "--out",
        out,
        &corpus(),
    ]));
    let enc = stdout(&splinter(&["encode", "-t", out, "--pair", "b", "a"]));
    assert!(enc.contains("\"[CLS]\""), "{enc}");
    assert!(enc.contains("types:   [0, 0, 0, 1, 1]"), "{enc}");
}

#[test]
fn errors_are_reported_with_nonzero_exit() {
    let o = splinter(&["encode", "-t", "/no/such/tokenizer.json", "hi"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("failed to load"));

    let o = splinter(&["train", "--out", "/tmp/x.json", "/no/such/corpus.txt"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("corpus.txt"));
}
