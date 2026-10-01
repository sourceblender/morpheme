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

fn configured_tokenizer(path: &Path) -> morpheme::Tokenizer {
    use morpheme::{AddedToken, PaddingParams, PaddingStrategy, TruncationParams};
    let vocab = [
        ("[UNK]", 0),
        ("a", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[PAD]", 4),
        ("b", 5),
        ("café", 6),
    ]
    .map(|(s, id)| (s.to_owned(), id))
    .into();
    let model = morpheme::models::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]")
        .build()
        .unwrap();
    let mut tokenizer = morpheme::Tokenizer::new(model)
        .with_pre_tokenizer(morpheme::pre_tokenizers::WhitespaceSplit)
        .with_post_processor(morpheme::processors::BertProcessing::new(
            ("[SEP]", 3),
            ("[CLS]", 2),
        ));
    tokenizer
        .add_special_tokens(&["[CLS]", "[SEP]", "[PAD]"].map(|s| AddedToken::new(s, true)))
        .unwrap();
    tokenizer
        .set_truncation(Some(TruncationParams {
            max_length: 4,
            ..Default::default()
        }))
        .unwrap();
    tokenizer.set_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(8),
        pad_id: 4,
        ..Default::default()
    }));
    tokenizer.save(path, false).unwrap();
    tokenizer
}

#[test]
fn counting_distinguishes_raw_budget_from_configured_limits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("configured.json");
    configured_tokenizer(&path);
    let path = path.to_str().unwrap();
    assert_eq!(
        stdout(&morpheme(&["count", "-t", path, "a a a a a"])).trim(),
        "7"
    );
    assert_eq!(
        stdout(&morpheme(&[
            "count",
            "-t",
            path,
            "--no-special-tokens",
            "a a a a a"
        ]))
        .trim(),
        "5"
    );
    assert_eq!(
        stdout(&morpheme(&[
            "count",
            "-t",
            path,
            "--use-tokenizer-settings",
            "a a a a a"
        ]))
        .trim(),
        "8"
    );
    let output = stdout(&morpheme(&[
        "count", "-t", path, "--pair", "b", "--json", "a a",
    ]));
    let value: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(value["count"], 6);
    assert_eq!(value["use_tokenizer_settings"], false);
    let info: serde_json::Value =
        serde_json::from_str(&stdout(&morpheme(&["inspect", "-t", path, "--json"]))).unwrap();
    assert_eq!(info["schema_version"], 1);
    assert_eq!(info["model"], "WordLevel");
    assert_eq!(info["truncation"]["max_length"], 4);
    assert_eq!(info["padding"]["pad_id"], 4);
    assert_eq!(info["special_tokens"].as_array().unwrap().len(), 3);
}

#[test]
fn jsonl_batches_keep_order_ids_pairs_offsets_and_decode_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("configured.json");
    configured_tokenizer(&path);
    let input = dir.path().join("input.jsonl");
    // CRLF, final record without newline, and an empty text are valid.
    std::fs::write(&input, "{\"id\":\"first\",\"text\":\"café\"}\r\n{\"id\":2,\"text\":\"a\",\"pair\":\"b\"}\n{\"text\":\"\"}").unwrap();
    let output = stdout(&morpheme(&[
        "encode-batch",
        "-t",
        path.to_str().unwrap(),
        "--input",
        input.to_str().unwrap(),
        "--batch-size",
        "2",
        "--char-offsets",
        "--ignore-tokenizer-settings",
    ]));
    let rows: Vec<serde_json::Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["id"], "first");
    assert_eq!(rows[1]["id"], 2);
    assert_eq!(rows[0]["encoding"]["offsets"][1], serde_json::json!([0, 4]));
    assert_eq!(
        rows[1]["encoding"]["sequence_ids"],
        serde_json::json!([null, 0, null, 1, null])
    );
    let decode_input = dir.path().join("decode.jsonl");
    let lines: Vec<String> = rows
        .iter()
        .map(|row| {
            let mut value = serde_json::json!({"ids": row["encoding"]["ids"]});
            if let Some(id) = row.get("id") {
                value["id"] = id.clone();
            }
            value.to_string()
        })
        .collect();
    std::fs::write(&decode_input, lines.join("\n")).unwrap();
    let decoded = stdout(&morpheme(&[
        "decode-batch",
        "-t",
        path.to_str().unwrap(),
        "--input",
        decode_input.to_str().unwrap(),
        "--batch-size",
        "1",
        "--skip-special-tokens",
    ]));
    let decoded: Vec<serde_json::Value> = decoded
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(decoded[0], serde_json::json!({"id":"first", "text":"café"}));
    assert_eq!(decoded[1], serde_json::json!({"id":2, "text":"a b"}));
    assert_eq!(decoded[2], serde_json::json!({"text":""}));
}

#[test]
fn jsonl_errors_identify_records_and_do_not_emit_the_failing_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("configured.json");
    configured_tokenizer(&path);
    let input = dir.path().join("input.jsonl");
    for (command, contents, extra, expected) in [
        (
            "encode-batch",
            "{\"text\":\"a\"}\nnot json\n",
            vec![],
            "record 2",
        ),
        (
            "encode-batch",
            "{\"text\":\"aaaaaaaaaaaaaaaa\"}\n",
            vec!["--max-record-bytes", "8"],
            "record 1 exceeds",
        ),
        (
            "decode-batch",
            "{\"ids\":[1]}\n{\"ids\":[999]}\n",
            vec![],
            "record 2: id 999",
        ),
    ] {
        std::fs::write(&input, contents).unwrap();
        let mut args = vec![
            command,
            "-t",
            path.to_str().unwrap(),
            "--input",
            input.to_str().unwrap(),
        ];
        args.extend(extra);
        let output = morpheme(&args);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
    let output = morpheme(&[
        "encode-batch",
        "-t",
        path.to_str().unwrap(),
        "--batch-size",
        "0",
    ]);
    assert!(!output.status.success());
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
    let contents = std::fs::read_to_string(snapshot.join("tokenizer.json")).unwrap();
    std::fs::write(
        snapshot.join("tokenizer.json"),
        contents.replace("\r\n", "\n").replace('\n', "\r\n"),
    )
    .unwrap();
    // Git blob hash for the copied snapshot: verified copied HF caches
    // retain the corresponding content-addressed blob.
    let output = Command::new("git")
        .args([
            "-c",
            "core.autocrlf=true",
            "hash-object",
            "--no-filters",
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
