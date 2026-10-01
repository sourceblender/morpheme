//! End-to-end tests for the `morpheme` binary.

use std::path::Path;
use std::process::{Command, Output};

fn morpheme(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_morpheme"))
        .args(args)
        .output()
        .expect("run morpheme")
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
        stdout(&morpheme(&[
            "train",
            "--model",
            model,
            "--vocab-size",
            "300",
            "--out",
            out,
            &corpus(),
        ]));

        let enc = stdout(&morpheme(&[
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
        let text = stdout(&morpheme(&args));
        assert_eq!(text.trim(), "the quick brown fox", "{model}");

        let info = stdout(&morpheme(&["inspect", "-t", out]));
        assert!(info.contains("vocab size"), "{model}: {info}");
    }
}

#[test]
fn bert_preset_adds_cls_and_sep() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("wp.json");
    let out = out.to_str().unwrap();
    stdout(&morpheme(&[
        "train",
        "--model",
        "wordpiece",
        "--vocab-size",
        "300",
        "--out",
        out,
        &corpus(),
    ]));
    let enc = stdout(&morpheme(&["encode", "-t", out, "--pair", "b", "a"]));
    assert!(enc.contains("\"[CLS]\""), "{enc}");
    assert!(enc.contains("types:   [0, 0, 0, 1, 1]"), "{enc}");
}

#[test]
fn errors_are_reported_with_nonzero_exit() {
    let o = morpheme(&["encode", "-t", "/no/such/tokenizer.json", "hi"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("failed to load"));

    let o = morpheme(&["train", "--out", "/tmp/x.json", "/no/such/corpus.txt"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("corpus.txt"));
}

#[test]
fn revision_is_rejected_for_files() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/bpe.json");
    let o = morpheme(&[
        "inspect",
        "-t",
        file.to_str().unwrap(),
        "--revision",
        "main",
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--revision only applies"));
}

#[test]
#[cfg(feature = "hub")]
fn hub_ids_are_served_from_the_cache_offline() {
    // Pre-populate a Hugging Face cache, then load by model id with
    // HF_HUB_OFFLINE=1: no network involved.
    let cache = tempfile::tempdir().unwrap();
    let commit = "0123456789abcdef0123456789abcdef01234567";
    let repo = cache.path().join("models--example--tiny");
    let snapshot = repo.join("snapshots").join(commit);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/wordlevel.json"),
        snapshot.join("tokenizer.json"),
    )
    .unwrap();
    // Git blob hash for the copied snapshot: verified copied HF caches
    // retain the corresponding content-addressed blob.
    let output = Command::new("git")
        .args([
            "hash-object",
            snapshot.join("tokenizer.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let etag = String::from_utf8(output.stdout).unwrap();
    std::fs::create_dir_all(repo.join("blobs")).unwrap();
    std::fs::copy(
        snapshot.join("tokenizer.json"),
        repo.join("blobs").join(etag.trim()),
    )
    .unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/main"), commit).unwrap();

    let run = |id: &str, text: &str| {
        Command::new(env!("CARGO_BIN_EXE_morpheme"))
            .args(["encode", "-t", id, text])
            .env("HF_HUB_CACHE", cache.path())
            .env("HF_HUB_OFFLINE", "1")
            .output()
            .unwrap()
    };
    let out = stdout(&run("example/tiny", "the quick fox"));
    assert!(out.contains("\"the\""), "{out}");

    let o = run("example/not-cached", "hi");
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("HF_HUB_OFFLINE"));
}
